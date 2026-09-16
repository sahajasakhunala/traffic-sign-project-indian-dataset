import os
import argparse
import collections
import json
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from model import TrafficSignCNN


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, path


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


def load_class_names(csv_path: str) -> dict[int, str]:
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        c_col = [c for c in df.columns if c.strip().lower() in ['classid', 'class_id', 'id']][0]
        n_col = [c for c in df.columns if c.strip().lower() in ['name', 'class_name', 'label', 'description']][0]
        return {int(r[c_col]): str(r[n_col]).strip() for _, r in df.iterrows()}
    except Exception as e:
        print(f"[WARN] Failed loading CSV class names: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Extract Misclassified Validation Images into Gallery")
    parser.add_argument('--data_dir', type=str, default='data/Indian_Dataset_Clean', help='Path to clean dataset')
    parser.add_argument('--model_type', type=str, default='resnet50', choices=['custom_cnn', 'resnet50'], help='Model type')
    parser.add_argument('--checkpoint_path', type=str, default='models_clean/traffic_sign_resnet50_finetuned.pth', help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=128, help='Image resolution')
    parser.add_argument('--num_classes', type=int, default=58, help='Number of classes')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to traffic_sign.csv')
    parser.add_argument('--output_dir', type=str, default='results_clean_eval', help='Output directory for misclassified gallery')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve CSV Path
    csv_path = args.csv_path
    if not csv_path:
        candidate = os.path.join(args.data_dir, "traffic_sign.csv")
        if os.path.exists(candidate):
            csv_path = candidate

    class_name_map = load_class_names(csv_path) if csv_path else {}

    # Load Dataset
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = datasets.ImageFolder(root=args.data_dir)
    raw_class_names = base_dataset.classes
    num_classes = len(raw_class_names)

    _, val_indices = stratified_split(base_dataset, val_split=0.15, seed=42)
    val_set = TransformSubset(Subset(base_dataset, val_indices), val_transform)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    # Load Model
    if args.model_type == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = TrafficSignCNN(num_classes=num_classes)

    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    cleaned_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    gallery_dir = os.path.join(args.output_dir, "misclassified_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    misclassified_records = []
    confusion_pair_counts = collections.defaultdict(int)

    print("\nExtracting misclassified validation samples...")
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for path, true_idx, pred_idx, conf in zip(paths, labels.tolist(), preds.cpu().tolist(), confs.cpu().tolist()):
                if true_idx != pred_idx:
                    true_cname = raw_class_names[true_idx]
                    pred_cname = raw_class_names[pred_idx]
                    t_int = int(true_cname) if true_cname.isdigit() else true_idx
                    p_int = int(pred_cname) if pred_cname.isdigit() else pred_idx

                    t_label = class_name_map.get(t_int, f"Class_{true_cname}")
                    p_label = class_name_map.get(p_int, f"Class_{pred_cname}")

                    pair_folder_name = f"True_{true_cname}_{t_label.replace(' ', '_')}__VS__Pred_{pred_cname}_{p_label.replace(' ', '_')}"
                    pair_dir = os.path.join(gallery_dir, pair_folder_name)
                    os.makedirs(pair_dir, exist_ok=True)

                    fname = os.path.basename(path)
                    dst_path = os.path.join(pair_dir, fname)
                    shutil.copy2(path, dst_path)

                    confusion_pair_counts[(true_cname, pred_cname, t_label, p_label)] += 1

                    misclassified_records.append({
                        "original_path": path,
                        "gallery_path": os.path.relpath(dst_path, args.output_dir).replace("\\", "/"),
                        "true_class_id": true_cname,
                        "true_class_name": t_label,
                        "pred_class_id": pred_cname,
                        "pred_class_name": p_label,
                        "confidence": round(conf * 100, 2)
                    })

    # Export Misclassifications Summary Report
    summary_md_path = os.path.join(args.output_dir, "misclassified_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Misclassified Validation Images — Error Analysis Report\n\n")
        f.write(f"**Total Validation Samples**: `{len(val_indices)}`\n")
        f.write(f"**Total Misclassifications**: `{len(misclassified_records)}` (Accuracy: {(1 - len(misclassified_records)/len(val_indices))*100:.2f}%)\n")
        f.write(f"**Gallery Directory**: `{os.path.relpath(gallery_dir, args.output_dir)}`\n\n")

        f.write("## 1. Top Confusing Class Pairs (Ranked by Error Frequency)\n\n")
        f.write("| Rank | True Class | Predicted As | Error Count | % of Total Errors |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: |\n")

        sorted_pairs = sorted(confusion_pair_counts.items(), key=lambda x: x[1], reverse=True)
        for rank, ((t_id, p_id, t_name, p_name), count) in enumerate(sorted_pairs, 1):
            pct = (count / len(misclassified_records)) * 100.0
            f.write(f"| {rank:2d} | Class {t_id:2s} (`{t_name}`) | Class {p_id:2s} (`{p_name}`) | `{count}` | {pct:.1f}% |\n")

        f.write("\n---\n\n## 2. Sample Misclassifications List\n\n")
        for rec in misclassified_records[:30]:
            f.write(f"- True: **Class {rec['true_class_id']}** (`{rec['true_class_name']}`) ↔ Pred: **Class {rec['pred_class_id']}** (`{rec['pred_class_name']}`) | Conf: `{rec['confidence']}%` | Path: `{rec['gallery_path']}`\n")

    print("\n" + "="*70)
    print("  Misclassified Gallery Extraction Complete!")
    print(f"  Total Misclassifications : {len(misclassified_records)} / {len(val_indices)}")
    print(f"  Gallery Directory        : {gallery_dir}")
    print(f"  Error Summary Markdown   : {summary_md_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
