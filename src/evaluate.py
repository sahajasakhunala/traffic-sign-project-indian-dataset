import os
import argparse
import collections
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import TrafficSignCNN

# ── Transforms & Dataset helpers ────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

def stratified_split(dataset: datasets.ImageFolder, val_split: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    class_indices = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs = class_indices[label]
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[i] for i in perm]
        n_val_c = max(1, int(len(idxs) * val_split))
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])

    return train_indices, val_indices

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate Indian Traffic Sign Model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset images folder')
    parser.add_argument('--model_type', type=str, default='resnet50', choices=['custom_cnn', 'resnet50'], help='Model type')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--log_path', type=str, default=None, help='Path to training log CSV')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save plots')
    parser.add_argument('--use_tta', action='store_true', help='Use Test-Time Augmentation (TTA)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    base_dataset = datasets.ImageFolder(root=args.data_dir)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    # Use same seed (42) and split (0.15) as train.py to get the exact validation set
    _, val_indices = stratified_split(base_dataset, val_split=0.15, seed=42)
    val_set = TransformSubset(Subset(base_dataset, val_indices), val_transform)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Validation set size: {len(val_indices)} samples across {num_classes} classes")

    # 2. Load Model
    if args.model_type == "resnet50":
        import torchvision.models as tv_models
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = TrafficSignCNN(num_classes=num_classes)

    model = model.to(device)

    print(f"Loading checkpoint weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    
    # Clean up keys if needed (in case of DataParallel or mismatched prefixes)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # 3. Predict on Validation Set
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            if args.use_tta:
                import torchvision.transforms.functional as F_t
                outputs1 = model(images)
                outputs2 = model(F_t.rotate(images, angle=3.0))
                outputs3 = model(F_t.rotate(images, angle=-3.0))
                outputs4 = model(F_t.affine(images, angle=0.0, translate=[0, 0], scale=1.05, shear=0.0))
                outputs5 = model(F_t.affine(images, angle=0.0, translate=[0, 0], scale=0.95, shear=0.0))
                
                probs = (
                    torch.softmax(outputs1, dim=1) +
                    torch.softmax(outputs2, dim=1) +
                    torch.softmax(outputs3, dim=1) +
                    torch.softmax(outputs4, dim=1) +
                    torch.softmax(outputs5, dim=1)
                ) / 5.0
                _, predicted = probs.max(1)
            else:
                outputs = model(images)
                _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_accuracy = (y_true == y_pred).mean() * 100.0
    print(f"\n==================================================")
    print(f"  Validation Accuracy: {overall_accuracy:.2f}%")
    print(f"==================================================\n")

    # 4. Generate & Save Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_txt = classification_report(y_true, y_pred, target_names=class_names)
    
    report_json_path = os.path.join(args.output_dir, f"{args.model_type}_classification_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {report_json_path}")

    # 5. Plot & Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {args.model_type.upper()} (Val Accuracy: {overall_accuracy:.2f}%)', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    cm_path = os.path.join(args.output_dir, f"{args.model_type}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix heatmap saved to: {cm_path}")

    # 6. Find and Print Top Confused Pairs
    confusion_pairs = collections.defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            confusion_pairs[(true, pred)] += 1

    sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Most Confusing Sign Pairs:")
    print("-" * 50)
    for (true_cls, pred_cls), count in sorted_confusion[:10]:
        true_name = class_names[true_cls]
        pred_name = class_names[pred_cls]
        print(f"Class {true_name} ↔ Predicted as Class {pred_name}: {count} times")
    print("-" * 50)

    # 7. Plot & Save Training Curves from CSV Log
    if args.log_path and os.path.exists(args.log_path):
        print(f"Plotting training curves from: {args.log_path}")
        df = pd.read_csv(args.log_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss Curve
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue', marker='o')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='darkorange', marker='s')
        ax1.set_title('Loss vs Epochs', fontsize=14)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=11)
        
        # Accuracy Curve
        ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='royalblue', marker='o')
        ax2.plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='darkorange', marker='s')
        ax2.set_title('Accuracy vs Epochs', fontsize=14)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=11)
        
        curves_path = os.path.join(args.output_dir, f"{args.model_type}_training_curves.png")
        plt.tight_layout()
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Training curves saved to: {curves_path}")
    else:
        print("No valid log_path CSV provided or file does not exist. Skipping curve plotting.")

if __name__ == "__main__":
    main()
