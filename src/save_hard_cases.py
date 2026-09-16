import os
import argparse
import collections
import json
import shutil
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
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


def compute_image_metrics(img_path: str) -> dict:
    """Computes quantitative quality diagnostics: dimensions, brightness, contrast, blur score."""
    with Image.open(img_path) as img:
        w, h = img.size
        img_gray = img.convert("L")
        arr = np.array(img_gray, dtype=np.float32)
        
    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))
    
    # Discrete 2D Laplacian operator for sharpness/blur estimation
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Simple 2D convolution via slicing
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        padded = np.pad(arr, ((1, 1), (1, 1)), mode='edge')
        lap = (
            padded[0:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
        )
        blur_score = float(np.var(lap))
    else:
        blur_score = 0.0

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
    }


# Targeted 5 Hard Confusion Families
HARD_FAMILIES = {
    ("23", "24"): "23_to_24",
    ("24", "23"): "24_to_23",
    ("36", "37"): "36_to_37",
    ("37", "36"): "37_to_36",
    ("42", "43"): "42_to_43",
    ("43", "42"): "43_to_42",
    ("47", "48"): "47_to_48",
    ("48", "47"): "48_to_47",
    ("49", "50"): "49_to_50",
    ("50", "49"): "50_to_49",
}


def main():
    parser = argparse.ArgumentParser(description="Extract & Analyze Hard Confusion Families for Phase 1")
    parser.add_argument('--data_dir', type=str, default='data/Indian_Dataset_Clean', help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='resnet50', help='Model type')
    parser.add_argument('--checkpoint_path', type=str, default='models_exp1/traffic_sign_resnet50_finetuned.pth', help='Path to checkpoint')
    parser.add_argument('--image_size', type=int, default=128, help='Image resolution')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to class mapping CSV')
    parser.add_argument('--output_dir', type=str, default='results_hard_cases', help='Output directory for hard cases')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = args.csv_path or os.path.join(args.data_dir, "traffic_sign.csv")
    class_name_map = load_class_names(csv_path)

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

    if args.model_type == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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

    # Create subdirectories for each hard confusion family
    family_dirs = {}
    for (t_id, p_id), folder in HARD_FAMILIES.items():
        fd = os.path.join(args.output_dir, folder)
        os.makedirs(fd, exist_ok=True)
        family_dirs[(t_id, p_id)] = fd

    hard_records = []
    total_val_errors = 0

    print("\nRunning hard-case quantitative extraction...")
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

            for path, true_idx, top3_p, top3_i in zip(paths, labels.tolist(), top3_probs.cpu().tolist(), top3_indices.cpu().tolist()):
                pred_idx = top3_i[0]
                if true_idx != pred_idx:
                    total_val_errors += 1
                    true_cname = raw_class_names[true_idx]
                    pred_cname = raw_class_names[pred_idx]

                    if (true_cname, pred_cname) in HARD_FAMILIES:
                        target_dir = family_dirs[(true_cname, pred_cname)]
                        fname = os.path.basename(path)
                        dst_path = os.path.join(target_dir, fname)
                        shutil.copy2(path, dst_path)

                        # Quantitative image metrics
                        img_metrics = compute_image_metrics(path)

                        # Top-3 predictions formatting
                        top3_desc = [
                            f"Class {raw_class_names[idx]} ({class_name_map.get(int(raw_class_names[idx]) if raw_class_names[idx].isdigit() else idx, '')}): {prob*100:.1f}%"
                            for prob, idx in zip(top3_p, top3_i)
                        ]

                        hard_records.append({
                            "family_folder": HARD_FAMILIES[(true_cname, pred_cname)],
                            "true_class_id": true_cname,
                            "true_class_name": class_name_map.get(int(true_cname) if true_cname.isdigit() else true_idx, f"Class {true_cname}"),
                            "pred_class_id": pred_cname,
                            "pred_class_name": class_name_map.get(int(pred_cname) if pred_cname.isdigit() else pred_idx, f"Class {pred_cname}"),
                            "confidence": round(top3_p[0] * 100, 2),
                            "top3": top3_desc,
                            "file_name": fname,
                            "gallery_rel_path": os.path.relpath(dst_path, args.output_dir).replace("\\", "/"),
                            **img_metrics
                        })

    # Generate Hard Case Diagnostic Summary Markdown
    summary_path = os.path.join(args.output_dir, "hard_case_summary.md")
    df_hard = pd.DataFrame(hard_records)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Hard Confusion Families — Quantitative Diagnostic Summary\n\n")
        f.write(f"**Total Validation Set Size**: `{len(val_indices)}` samples\n")
        f.write(f"**Total Validation Errors**: `{total_val_errors}`\n")
        f.write(f"**Hard Families Errors Analyzed**: `{len(hard_records)}` (**{len(hard_records)/total_val_errors*100:.1f}%** of all errors)\n\n")

        f.write("## 1. Confusion Family Breakdown & Quantitative Summary\n\n")
        f.write("| Family Folder | Confusion Pair | Sample Count | Avg Resolution | Avg Brightness | Avg Contrast | Avg Blur Score |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        family_counts = collections.defaultdict(list)
        for r in hard_records:
            family_counts[r['family_folder']].append(r)

        for (t_id, p_id), folder in HARD_FAMILIES.items():
            recs = family_counts[folder]
            count = len(recs)
            if count > 0:
                avg_w = np.mean([x['width'] for x in recs])
                avg_h = np.mean([x['height'] for x in recs])
                avg_bright = np.mean([x['brightness'] for x in recs])
                avg_contrast = np.mean([x['contrast'] for x in recs])
                avg_blur = np.mean([x['blur_score'] for x in recs])
                t_name = class_name_map.get(int(t_id) if t_id.isdigit() else 0, f"Class {t_id}")
                p_name = class_name_map.get(int(p_id) if p_id.isdigit() else 0, f"Class {p_id}")
                f.write(f"| `{folder}` | Class {t_id} (`{t_name}`) $\\rightarrow$ Class {p_id} (`{p_name}`) | `{count}` | `{avg_w:.0f}\\times{avg_h:.0f}` | `{avg_bright:.1f}` | `{avg_contrast:.1f}` | `{avg_blur:.1f}` |\n")
            else:
                f.write(f"| `{folder}` | Class {t_id} $\\rightarrow$ Class {p_id} | `0` | - | - | - | - |\n")

        f.write("\n---\n\n## 2. Quantitative Diagnostic Notes & Candidate Hypotheses\n\n")
        f.write("*(Note: All metrics report empirical measurements. Visual verification by domain reviewer recommended.)*\n\n")

        f.write("### Directional Arrow Signs (Classes 23 ↔ 24, 36 ↔ 37, 42 ↔ 43)\n")
        f.write("- **Primary Constraint**: Feature spatial resolution. Downsampling arrow features to 128x128 causes subtle arrow tip angles (e.g. left vs right inclination) to blur into symmetrical patches.\n")
        f.write("- **Hypothesis**: High-resolution scaling (192x192 / 224x224) provides sharper edge gradients along directional arrows.\n\n")

        f.write("### Countdown Markers (Classes 47 ↔ 48, 49 ↔ 50)\n")
        f.write("- **Primary Constraint**: Line bar counting. Distinguishing 3 diagonal bars vs 2 diagonal bars requires clear high-frequency line separation.\n")
        f.write("- **Hypothesis**: Resolution scaling combined with mild contrast jitter enhances line separation.\n\n")

        f.write("## 3. Individual Hard Case Inspection Registry\n\n")
        for r in hard_records[:40]:
            f.write(f"- **{r['family_folder']}** | File: `{r['file_name']}` | True: `{r['true_class_name']}` | Pred: `{r['pred_class_name']}` (`{r['confidence']}%`)\n")
            f.write(f"  - Metrics: Size=`{r['width']}x{r['height']}`, Brightness=`{r['brightness']}`, Contrast=`{r['contrast']}`, BlurScore=`{r['blur_score']}`\n")
            f.write(f"  - Top-3: `{', '.join(r['top3'])}` | Image Path: `{r['gallery_rel_path']}`\n\n")

    print("\n" + "=" * 70)
    print("  Phase 1 Hard Case Quantitative Gallery Extraction Complete!")
    print(f"  Hard Family Errors Extracted : {len(hard_records)} / {total_val_errors} total errors")
    print(f"  Gallery Directory            : {args.output_dir}")
    print(f"  Diagnostic Summary           : {summary_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
