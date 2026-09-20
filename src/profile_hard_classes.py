import os
import argparse
import collections
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import datasets

HARD_CLASSES = ["23", "24", "36", "37", "42", "43", "47", "48", "49", "50"]

HARD_PAIRS = [
    (("23", "24"), ("24", "23"), "23 ↔ 24 (Turn Left / Right)"),
    (("36", "37"), ("37", "36"), "36 ↔ 37 (Side Road Junction Left / Right)"),
    (("42", "43"), ("43", "42"), "42 ↔ 43 (Staggered Junction Left / Right)"),
    (("47", "48"), ("48", "47"), "47 ↔ 48 (Countdown Marker 3 vs 2 bars)"),
    (("49", "50"), ("50", "49"), "49 ↔ 50 (Countdown Marker 2 vs 1 bar)"),
]


def load_class_names(csv_path: str) -> dict[int, str]:
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        c_col = [c for c in df.columns if c.strip().lower() in ['classid', 'class_id', 'id']][0]
        n_col = [c for c in df.columns if c.strip().lower() in ['name', 'class_name', 'label', 'description']][0]
        return {int(r[c_col]): str(r[n_col]).strip() for _, r in df.iterrows()}
    except Exception as e:
        print(f"[WARN] Error loading class names from {csv_path}: {e}")
        return {}


def compute_img_stats(paths: list[str]) -> dict:
    if not paths:
        return {"avg_w": 0, "avg_h": 0, "avg_bright": 0.0, "avg_contrast": 0.0, "avg_blur": 0.0}

    widths, heights, brights, contrasts, blurs = [], [], [], [], []
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    for p in paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                gray = img.convert("L")
                arr = np.array(gray, dtype=np.float32)
                brights.append(float(np.mean(arr)))
                contrasts.append(float(np.std(arr)))
                if arr.shape[0] >= 3 and arr.shape[1] >= 3:
                    padded = np.pad(arr, ((1, 1), (1, 1)), mode='edge')
                    lap = (
                        padded[0:-2, 1:-1] + padded[2:, 1:-1] +
                        padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
                    )
                    blurs.append(float(np.var(lap)))
        except Exception as e:
            continue

    return {
        "avg_w": round(float(np.mean(widths)), 1) if widths else 0,
        "avg_h": round(float(np.mean(heights)), 1) if heights else 0,
        "avg_bright": round(float(np.mean(brights)), 1) if brights else 0.0,
        "avg_contrast": round(float(np.mean(contrasts)), 1) if contrasts else 0.0,
        "avg_blur": round(float(np.mean(blurs)), 1) if blurs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Targeted Hard-Class Data Profiler")
    parser.add_argument('--data_dir', type=str, default='data/Indian_Dataset_Clean', help='Clean dataset directory')
    parser.add_argument('--output_dir', type=str, default='results_hard_analysis', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.data_dir, "traffic_sign.csv")
    class_name_map = load_class_names(csv_path)

    base_dataset = datasets.ImageFolder(root=args.data_dir)
    class_names = base_dataset.classes

    # Map class_id string to file paths
    class_files = collections.defaultdict(list)
    for path, label in base_dataset.samples:
        c_str = class_names[label]
        class_files[c_str].append(path)

    print("=" * 80)
    print("  PHASE 4: TARGETED HARD-CLASS DATA PROFILING")
    print("=" * 80)

    profiles = {}
    for c_id in HARD_CLASSES:
        paths = class_files.get(c_id, [])
        c_name = class_name_map.get(int(c_id) if c_id.isdigit() else 0, f"Class {c_id}")
        stats = compute_img_stats(paths)
        profiles[c_id] = {
            "class_id": c_id,
            "class_name": c_name,
            "total_samples": len(paths),
            **stats
        }
        print(f"  Class {c_id:>2s} ({c_name:<35}) | Total: {len(paths):>4d} | Avg Size: {stats['avg_w']}x{stats['avg_h']} | Bright: {stats['avg_bright']} | Contrast: {stats['avg_contrast']} | Blur: {stats['avg_blur']}")

    print("-" * 80)

    # Save Markdown Profile Summary
    md_path = os.path.join(args.output_dir, "hard_class_profiles.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Targeted Hard-Class Data Profile Report\n\n")
        f.write("This report details image counts, average resolution, brightness, contrast, and Laplacian blur scores for the 10 classes in the 5 hard confusion families.\n\n")

        f.write("## 1. Class Distribution & Quality Metrics Table\n\n")
        f.write("| Class ID | Class Name | Total Samples | Avg Dimensions | Avg Brightness | Avg Contrast | Avg Blur Score |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for c_id in HARD_CLASSES:
            p = profiles[c_id]
            f.write(f"| `{p['class_id']}` | `{p['class_name']}` | `{p['total_samples']}` | `{p['avg_w']}\\times{p['avg_h']}` | `{p['avg_bright']}` | `{p['avg_contrast']}` | `{p['avg_blur']}` |\n")

        f.write("\n---\n\n## 2. Hard Confusion Pair Visual Contact Sheets\n\n")
        for pair_a, pair_b, title in HARD_PAIRS:
            f.write(f"### Pair: {title}\n")
            f.write(f"- **{pair_a[0]} $\\rightarrow$ {pair_a[1]}**: Class `{pair_a[0]}` ({class_name_map.get(int(pair_a[0]), '')}) vs Class `{pair_a[1]}` ({class_name_map.get(int(pair_a[1]), '')})\n")
            f.write(f"- **{pair_b[0]} $\\rightarrow$ {pair_b[1]}**: Class `{pair_b[0]}` ({class_name_map.get(int(pair_b[0]), '')}) vs Class `{pair_b[1]}` ({class_name_map.get(int(pair_b[1]), '')})\n\n")

    print(f"\n  Profile summary exported to: {md_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
