import os
import argparse
import collections
import shutil
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

from model import TrafficSignCNN

# ── Subset Wrapper ─────────────────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
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

def build_model(model_type: str, num_classes: int):
    if model_type == 'custom_cnn':
        return TrafficSignCNN(num_classes=num_classes)
    elif model_type == 'resnet50':
        import torchvision.models as tv_models
        m = tv_models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif model_type == 'efficientnet_b2':
        import torchvision.models as tv_models
        m = tv_models.efficientnet_b2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif model_type == 'efficientnet_b3':
        import torchvision.models as tv_models
        m = tv_models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif model_type == 'convnext_tiny':
        import torchvision.models as tv_models
        m = tv_models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def resolve_checkpoint_path(path: str) -> str:
    if os.path.exists(path):
        return path
    fname = os.path.basename(path)
    candidates = [
        path,
        os.path.join("/content/models_exp1", fname),
        os.path.join("/content/models_exp4_effnet_b2", fname),
        os.path.join("/content", fname),
        os.path.join("models_exp1", fname),
        os.path.join("results_baseline", fname),
        os.path.join("results_baseline", "models_exp1", fname),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            print(f"[INFO] Resolved checkpoint path: '{path}' -> '{cand}'")
            return cand
    raise FileNotFoundError(f"Checkpoint file not found: '{path}'. Checked candidates: {candidates}")

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_keys and model.state_dict()[k].shape == v.shape:
            filtered_state_dict[k] = v
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def evaluate_model_on_val(model: nn.Module, loader: DataLoader, device: torch.device):
    all_targets = []
    all_preds = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for inputs, targets, paths in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    return np.array(all_targets), np.array(all_preds), np.array(all_probs), all_paths

def main():
    parser = argparse.ArgumentParser(description="Target Class Diagnostic (Class 49 Focus)")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to clean dataset')
    parser.add_argument('--baseline_checkpoint', type=str, required=True, help='Baseline model checkpoint (.pth)')
    parser.add_argument('--baseline_model_type', type=str, default='resnet50')
    parser.add_argument('--baseline_image_size', type=int, default=128)
    parser.add_argument('--exp_checkpoint', type=str, required=True, help='Candidate experiment checkpoint (.pth)')
    parser.add_argument('--exp_model_type', type=str, default='efficientnet_b2')
    parser.add_argument('--exp_image_size', type=int, default=224)
    parser.add_argument('--target_class', type=int, default=49, help='Class ID to diagnose')
    parser.add_argument('--output_dir', type=str, default='results_exp4_effnet_b2/class49_diagnostic')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    base_dataset = datasets.ImageFolder(root=args.data_dir)
    num_classes = len(base_dataset.classes)
    train_indices, val_indices = stratified_split(base_dataset, val_split=0.15, seed=42)

    # Class sample counts in clean dataset
    total_class_samples = sum(1 for _, label in base_dataset.samples if label == args.target_class)
    train_class_samples = sum(1 for idx in train_indices if base_dataset.samples[idx][1] == args.target_class)
    val_class_samples   = sum(1 for idx in val_indices if base_dataset.samples[idx][1] == args.target_class)

    print(f"=== Class {args.target_class} Dataset Counts ===")
    print(f"  Total Clean Samples: {total_class_samples}")
    print(f"  Train Samples      : {train_class_samples}")
    print(f"  Validation Samples : {val_class_samples}")

    # Load Baseline Model & Evaluate
    transform_base = transforms.Compose([
        transforms.Resize((args.baseline_image_size, args.baseline_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_subset_base = Subset(base_dataset, val_indices)
    val_dataset_base = TransformSubset(val_subset_base, transform_base)
    loader_base = DataLoader(val_dataset_base, batch_size=64, shuffle=False)

    model_base = build_model(args.baseline_model_type, num_classes)
    model_base = load_checkpoint(model_base, args.baseline_checkpoint, device)
    targets_base, preds_base, probs_base, paths_base = evaluate_model_on_val(model_base, loader_base, device)

    # Load Exp Model & Evaluate
    transform_exp = transforms.Compose([
        transforms.Resize((args.exp_image_size, args.exp_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_subset_exp = Subset(base_dataset, val_indices)
    val_dataset_exp = TransformSubset(val_subset_exp, transform_exp)
    loader_exp = DataLoader(val_dataset_exp, batch_size=64, shuffle=False)

    model_exp = build_model(args.exp_model_type, num_classes)
    model_exp = load_checkpoint(model_exp, args.exp_checkpoint, device)
    targets_exp, preds_exp, probs_exp, paths_exp = evaluate_model_on_val(model_exp, loader_exp, device)

    # Function to compute target class metrics
    def compute_class_metrics(targets, preds, probs, c):
        actual_mask = (targets == c)
        pred_mask   = (preds == c)
        
        tp = np.sum(actual_mask & pred_mask)
        fp = np.sum((~actual_mask) & pred_mask)
        fn = np.sum(actual_mask & (~pred_mask))

        total_actual = np.sum(actual_mask)
        total_predicted = np.sum(pred_mask)

        recall = (tp / total_actual) if total_actual > 0 else 0.0
        precision = (tp / total_predicted) if total_predicted > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Distribution of predictions for actual class c samples
        actual_indices = np.where(actual_mask)[0]
        preds_for_actual = preds[actual_indices]
        probs_for_actual = probs[actual_indices]

        pred_counts = collections.Counter(preds_for_actual)
        
        return {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'total_actual': int(total_actual),
            'total_predicted': int(total_predicted),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'pred_counts': pred_counts,
            'actual_indices': actual_indices,
            'preds_for_actual': preds_for_actual,
            'probs_for_actual': probs_for_actual,
        }

    m_base = compute_class_metrics(targets_base, preds_base, probs_base, args.target_class)
    m_exp  = compute_class_metrics(targets_exp, preds_exp, probs_exp, args.target_class)

    # Build Diagnostic Report
    report_md = f"""# Diagnostic Report: Class {args.target_class} Prediction Breakdown

## 1. Dataset Class Statistics
- **Total Clean Samples**: {total_class_samples}
- **Train Samples**: {train_class_samples}
- **Validation Samples**: {val_class_samples}

---

## 2. Model Performance Comparison (Class {args.target_class})

| Metric | Baseline ({args.baseline_model_type}) | Candidate ({args.exp_model_type}) | Delta |
|---|:---:|:---:|:---:|
| **Total Actual Val Samples** | {m_base['total_actual']} | {m_exp['total_actual']} | 0 |
| **Total Predictions Made as Class {args.target_class}** | {m_base['total_predicted']} | {m_exp['total_predicted']} | {m_exp['total_predicted'] - m_base['total_predicted']} |
| **True Positives (TP)** | {m_base['tp']} | {m_exp['tp']} | {m_exp['tp'] - m_base['tp']} |
| **False Positives (FP)** | {m_base['fp']} | {m_exp['fp']} | {m_exp['fp'] - m_base['fp']} |
| **False Negatives (FN)** | {m_base['fn']} | {m_exp['fn']} | {m_exp['fn'] - m_base['fn']} |
| **Recall** | **{m_base['recall']*100:.2f}%** | **{m_exp['recall']*100:.2f}%** | **{(m_exp['recall'] - m_base['recall'])*100:+.2f}%** |
| **Precision** | **{m_base['precision']*100:.2f}%** | **{m_exp['precision']*100:.2f}%** | **{(m_exp['precision'] - m_base['precision'])*100:+.2f}%** |
| **F1-Score** | **{m_base['f1']*100:.2f}%** | **{m_exp['f1']*100:.2f}%** | **{(m_exp['f1'] - m_base['f1'])*100:+.2f}%** |

---

## 3. Destination Analysis: What were actual Class {args.target_class} samples predicted as?

### Baseline ({args.baseline_model_type}):
"""
    for pred_cls, count in m_base['pred_counts'].most_common():
        pct = (count / m_base['total_actual']) * 100
        report_md += f"- **Predicted as Class {pred_cls}**: {count} samples ({pct:.1f}%)\n"

    report_md += f"\n### Candidate ({args.exp_model_type}):\n"
    for pred_cls, count in m_exp['pred_counts'].most_common():
        pct = (count / m_exp['total_actual']) * 100
        report_md += f"- **Predicted as Class {pred_cls}**: {count} samples ({pct:.1f}%)\n"

    report_md += f"""\n---

## 4. Per-Sample Detailed Confidence Breakdown (Candidate: {args.exp_model_type})

| Sample Index | Image Path | Target Class | Predicted Class | Prob(Target) | Prob(Predicted) |
|---|---|:---:|:---:|:---:|:---:|
"""
    # Create gallery folders
    gallery_dir = os.path.join(args.output_dir, "class49_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    for i, idx in enumerate(m_exp['actual_indices']):
        img_path = paths_exp[idx]
        fname = os.path.basename(img_path)
        tgt_cls = targets_exp[idx]
        pred_cls = preds_exp[idx]
        prob_tgt = probs_exp[idx][args.target_class]
        prob_pred = probs_exp[idx][pred_cls]

        report_md += f"| {i+1} | `{fname}` | {tgt_cls} | {pred_cls} | {prob_tgt*100:.2f}% | {prob_pred*100:.2f}% |\n"

        # Categorized gallery copy
        if pred_cls == args.target_class:
            sub_folder = f"correct_class_{args.target_class}"
        else:
            sub_folder = f"predicted_as_{pred_cls}"
        
        sub_path = os.path.join(gallery_dir, sub_folder)
        os.makedirs(sub_path, exist_ok=True)
        dst_file = os.path.join(sub_path, f"sample_{i+1}_pred{pred_cls}_{fname}")
        shutil.copy(img_path, dst_file)

    report_path = os.path.join(args.output_dir, f"class{args.target_class}_diagnostic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[SUCCESS] Class {args.target_class} diagnostic report written to: {report_path}")
    print(f"[SUCCESS] Sample gallery exported to: {gallery_dir}")

if __name__ == '__main__':
    main()
