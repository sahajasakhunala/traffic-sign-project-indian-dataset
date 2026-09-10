import os
import sys
import json
import hashlib
import argparse
import collections
from PIL import Image
import numpy as np
import cv2
import pandas as pd


def compute_md5(file_path: str) -> str:
    """Computes MD5 hash of a file for exact duplicate matching."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Computes a 64-bit difference hash (dHash) for perceptual similarity comparison."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return decimal_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes Hamming distance between two 64-bit integers."""
    return bin(hash1 ^ hash2).count("1")


def compute_blur_and_contrast(img_np: np.ndarray) -> tuple[float, float]:
    """
    Computes Laplacian variance (blur metric) and RMS contrast.
    Low variance indicates blur; low contrast indicates washed out / dark images.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    return laplacian_var, contrast


def load_class_mapping(csv_path: str) -> dict[int, str]:
    """Reads class names from traffic_sign.csv dynamically."""
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        class_col, name_col = None, None
        for col in df.columns:
            c_low = col.strip().lower()
            if c_low in ["classid", "class_id", "id"]:
                class_col = col
            elif c_low in ["name", "class_name", "sign_name", "label", "description"]:
                name_col = col

        if class_col and name_col:
            return {int(row[class_col]): str(row[name_col]).strip() for _, row in df.iterrows()}
    except Exception as e:
        print(f"[WARN] Error reading class CSV: {e}", flush=True)
    return {}


def simulate_stratified_split(
    samples_by_class: dict[int, list[str]], val_split: float = 0.15, seed: int = 42
) -> tuple[set[str], set[str]]:
    """
    Simulates the exact deterministic stratified split used in train.py / evaluate.py
    to audit for train-vs-validation leakage.
    """
    rng = np.random.RandomState(seed)
    train_paths = set()
    val_paths = set()

    for class_id in sorted(samples_by_class.keys()):
        paths = list(samples_by_class[class_id])
        shuffled = rng.permutation(paths).tolist()
        n_val = max(1, int(len(shuffled) * val_split))
        val_paths.update(shuffled[:n_val])
        train_paths.update(shuffled[n_val:])

    return train_paths, val_paths


def main():
    parser = argparse.ArgumentParser(description="Audit Indian Traffic Sign Dataset Quality & Leakage")
    parser.add_argument("--data_dir", type=str, default="data/Indian_Dataset", help="Root folder of sorted class directories")
    parser.add_argument("--csv_path", type=str, default="data/Indian_Dataset/traffic_sign.csv", help="Path to class mapping CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to store audit reports")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio to check leakage")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for stratified split simulation")
    parser.add_argument("--blur_thresh", type=float, default=50.0, help="Laplacian variance threshold below which images are flagged as blurry")
    parser.add_argument("--contrast_thresh", type=float, default=18.0, help="RMS contrast threshold below which images are flagged as low contrast")
    parser.add_argument("--near_dup_thresh", type=int, default=3, help="Max Hamming distance to flag perceptual near-duplicates (0-64)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_map = load_class_mapping(args.csv_path)

    print("=" * 70, flush=True)
    print("  Indian Traffic Sign Dataset — Comprehensive Quality & Leakage Audit", flush=True)
    print("=" * 70, flush=True)
    print(f"Data directory      : {args.data_dir}", flush=True)
    print(f"Class mapping file  : {args.csv_path} ({len(class_map)} classes mapped)", flush=True)
    print(f"Simulated val split : {args.val_split * 100:.0f}% (seed={args.seed})", flush=True)
    print(f"Blur threshold      : {args.blur_thresh}", flush=True)
    print(f"Near-dup threshold  : {args.near_dup_thresh} bits", flush=True)
    print("-" * 70, flush=True)

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Directory not found: {args.data_dir}", flush=True)
        sys.exit(1)

    class_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    try:
        class_dirs.sort(key=lambda x: int(x))
    except ValueError:
        class_dirs.sort()

    print(f"Found {len(class_dirs)} class folders.", flush=True)

    samples_by_class: dict[int, list[str]] = collections.defaultdict(list)
    image_metadata: dict[str, dict] = {}
    corrupt_images: list[dict] = []
    anomalous_shapes: list[dict] = []
    channel_anomalies: list[dict] = []
    blurry_images: list[dict] = []
    low_contrast_images: list[dict] = []

    md5_to_paths: dict[str, list[str]] = collections.defaultdict(list)
    path_to_dhash: dict[str, int] = {}
    path_to_class: dict[str, int] = {}

    total_files_scanned = 0

    print("\n[Phase 1/4] Scanning image files, verifying integrity, and computing metrics...", flush=True)

    for c_idx, c_dir in enumerate(class_dirs, 1):
        try:
            class_id = int(c_dir)
        except ValueError:
            continue

        c_path = os.path.join(args.data_dir, c_dir)
        files = [f for f in os.listdir(c_path) if not f.startswith(".")]

        for fname in files:
            total_files_scanned += 1
            fpath = os.path.join(c_path, fname)
            rel_path = os.path.relpath(fpath, args.data_dir).replace("\\", "/")

            fsize = os.path.getsize(fpath)
            if fsize == 0:
                corrupt_images.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "reason": "Zero-byte file"
                })
                continue

            try:
                with Image.open(fpath) as pil_img:
                    pil_img.verify()
                with Image.open(fpath) as pil_img:
                    w, h = pil_img.size
                    mode = pil_img.mode
                    img_rgb = pil_img.convert("RGB")
                    dhash_val = compute_dhash(img_rgb)
                    np_img = np.array(img_rgb)
            except Exception as e:
                corrupt_images.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "reason": f"Corrupt image file: {str(e)}"
                })
                continue

            md5_val = compute_md5(fpath)
            md5_to_paths[md5_val].append(rel_path)
            path_to_dhash[rel_path] = dhash_val
            path_to_class[rel_path] = class_id
            samples_by_class[class_id].append(rel_path)

            aspect_ratio = w / float(h)
            if w < 16 or h < 16 or aspect_ratio < 0.33 or aspect_ratio > 3.0:
                anomalous_shapes.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "width": w, "height": h, "aspect_ratio": round(aspect_ratio, 2)
                })

            if mode not in ["RGB"]:
                channel_anomalies.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "mode": mode
                })

            lap_var, contrast = compute_blur_and_contrast(np_img)
            if lap_var < args.blur_thresh:
                blurry_images.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "laplacian_var": round(lap_var, 2)
                })

            if contrast < args.contrast_thresh:
                low_contrast_images.append({
                    "path": rel_path, "class_id": class_id,
                    "class_name": class_map.get(class_id, f"Class {class_id}"),
                    "contrast": round(contrast, 2)
                })

            image_metadata[rel_path] = {
                "class_id": class_id, "width": w, "height": h,
                "aspect_ratio": round(aspect_ratio, 2),
                "blur_var": round(lap_var, 2), "contrast": round(contrast, 2),
                "md5": md5_val, "dhash": dhash_val
            }

        if c_idx % 15 == 0 or c_idx == len(class_dirs):
            print(f"  Processed {c_idx}/{len(class_dirs)} classes ({total_files_scanned} images scanned)...", flush=True)

    print(f"Scanned {total_files_scanned} total files across {len(samples_by_class)} populated classes.", flush=True)

    # 2. Simulate Stratified Split
    print("\n[Phase 2/4] Simulating train/validation split and testing for leakage...", flush=True)
    train_paths, val_paths = simulate_stratified_split(samples_by_class, val_split=args.val_split, seed=args.seed)

    # 3. Duplicate Detection
    print("\n[Phase 3/4] Running multi-tiered duplicate analysis...", flush=True)

    # (A) Exact Duplicates (Same MD5)
    exact_duplicates_within_class = []
    exact_duplicates_cross_class = []
    exact_duplicates_train_val_leakage = []

    for md5_val, paths in md5_to_paths.items():
        if len(paths) > 1:
            classes = {path_to_class[p] for p in paths}
            splits = {("val" if p in val_paths else "train") for p in paths}

            dup_info = {
                "md5": md5_val,
                "paths": paths,
                "classes": [{
                    "id": c,
                    "name": class_map.get(c, f"Class {c}"),
                    "samples": [p for p in paths if path_to_class[p] == c]
                } for c in classes],
                "splits": {p: ("val" if p in val_paths else "train") for p in paths}
            }

            if len(classes) > 1:
                exact_duplicates_cross_class.append(dup_info)
            else:
                exact_duplicates_within_class.append(dup_info)

            if "train" in splits and "val" in splits:
                exact_duplicates_train_val_leakage.append(dup_info)

    # (B) Perceptual Near-Duplicates via Multi-Index Inverted Bucketing (Pigeonhole: 4 chunks of 16-bit)
    print("Indexing perceptual hashes (fast inverted index)...", flush=True)
    all_rel_paths = list(path_to_dhash.keys())
    chunk_tables = [collections.defaultdict(list) for _ in range(4)]

    for idx, p in enumerate(all_rel_paths):
        h = path_to_dhash[p]
        for c_i in range(4):
            chunk = (h >> (c_i * 16)) & 0xFFFF
            chunk_tables[c_i][chunk].append(idx)

    # Check candidate pairs that share at least one 16-bit block
    candidate_pairs = set()
    for c_i in range(4):
        for chunk, idx_list in chunk_tables[c_i].items():
            if len(idx_list) > 1:
                for a_i in range(len(idx_list)):
                    for b_i in range(a_i + 1, len(idx_list)):
                        i, j = idx_list[a_i], idx_list[b_i]
                        if i > j:
                            i, j = j, i
                        candidate_pairs.add((i, j))

    print(f"Filtered to {len(candidate_pairs):,} candidate near-duplicate pairs (from {len(all_rel_paths)*(len(all_rel_paths)-1)//2:,} total).", flush=True)

    perceptual_cross_class_review = []
    perceptual_train_val_leakage = []
    perceptual_within_class = []

    for i, j in candidate_pairs:
        p1, p2 = all_rel_paths[i], all_rel_paths[j]
        # Skip exact MD5
        if image_metadata[p1]["md5"] == image_metadata[p2]["md5"]:
            continue

        dist = hamming_distance(path_to_dhash[p1], path_to_dhash[p2])
        if dist <= args.near_dup_thresh:
            c1, c2 = path_to_class[p1], path_to_class[p2]
            split1 = "val" if p1 in val_paths else "train"
            split2 = "val" if p2 in val_paths else "train"

            match_info = {
                "distance": dist,
                "image_a": p1, "class_a": c1, "name_a": class_map.get(c1, f"Class {c1}"), "split_a": split1,
                "image_b": p2, "class_b": c2, "name_b": class_map.get(c2, f"Class {c2}"), "split_b": split2,
            }

            if c1 != c2:
                perceptual_cross_class_review.append(match_info)
            else:
                perceptual_within_class.append(match_info)

            if split1 != split2 and dist <= 1:
                perceptual_train_val_leakage.append(match_info)

    # 4. Class Imbalance Analysis
    print("\n[Phase 4/4] Profiling class distribution and sample counts...", flush=True)
    class_distribution = []
    for c_id in sorted(samples_by_class.keys()):
        total_c = len(samples_by_class[c_id])
        n_tr = sum(1 for p in samples_by_class[c_id] if p in train_paths)
        n_vl = sum(1 for p in samples_by_class[c_id] if p in val_paths)
        class_distribution.append({
            "class_id": c_id,
            "class_name": class_map.get(c_id, f"Class {c_id}"),
            "total_samples": total_c,
            "train_samples": n_tr,
            "val_samples": n_vl
        })

    class_distribution.sort(key=lambda x: x["total_samples"])
    minority_classes = [c for c in class_distribution if c["total_samples"] < 40]

    blurry_images.sort(key=lambda x: x["laplacian_var"])
    low_contrast_images.sort(key=lambda x: x["contrast"])
    perceptual_cross_class_review.sort(key=lambda x: x["distance"])

    # 5. Build Final JSON & Markdown Summary
    audit_report = {
        "summary": {
            "total_files_scanned": total_files_scanned,
            "total_valid_images": len(image_metadata),
            "total_classes": len(samples_by_class),
            "corrupt_images_count": len(corrupt_images),
            "exact_duplicate_groups_within_class": len(exact_duplicates_within_class),
            "exact_duplicate_groups_cross_class": len(exact_duplicates_cross_class),
            "exact_train_val_leakage_groups": len(exact_duplicates_train_val_leakage),
            "perceptual_train_val_leakage_matches": len(perceptual_train_val_leakage),
            "perceptual_cross_class_review_matches": len(perceptual_cross_class_review),
            "blurry_images_count": len(blurry_images),
            "low_contrast_images_count": len(low_contrast_images),
            "anomalous_shapes_count": len(anomalous_shapes),
            "channel_anomalies_count": len(channel_anomalies),
            "minority_classes_count (<40 samples)": len(minority_classes),
        },
        "critical_issues": {
            "corrupt_images": corrupt_images,
            "exact_cross_class_duplicates": exact_duplicates_cross_class,
            "exact_train_val_leakage": exact_duplicates_train_val_leakage,
            "severe_minority_classes": minority_classes
        },
        "high_priority_issues": {
            "perceptual_train_val_leakage": perceptual_train_val_leakage[:50],
            "blurry_images_sample": blurry_images[:50],
            "perceptual_cross_class_matches": perceptual_cross_class_review[:50]
        },
        "medium_priority_issues": {
            "low_contrast_sample": low_contrast_images[:50],
            "anomalous_shapes_sample": anomalous_shapes[:50],
            "exact_duplicates_within_class_count": len(exact_duplicates_within_class)
        },
        "class_distribution": class_distribution
    }

    json_report_path = os.path.join(args.output_dir, "dataset_audit_report.json")
    with open(json_report_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    md_report_path = os.path.join(args.output_dir, "dataset_audit_summary.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Indian Traffic Sign Dataset — Quality & Leakage Audit Report\n\n")
        f.write(f"**Audit Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Scanned Files**: `{total_files_scanned}` images across `{len(samples_by_class)}` classes\n\n")

        f.write("## 1. Executive Metric Summary\n\n")
        f.write("| Audit Dimension | Detected Count | Severity Level |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Corrupted / Zero-Byte Files** | `{len(corrupt_images)}` | 🔴 CRITICAL |\n")
        f.write(f"| **Cross-Class Exact Duplicates (Direct Label Conflicts)** | `{len(exact_duplicates_cross_class)}` | 🔴 CRITICAL |\n")
        f.write(f"| **Train ↔ Validation Exact Leakage** | `{len(exact_duplicates_train_val_leakage)}` | 🔴 CRITICAL |\n")
        f.write(f"| **Train ↔ Validation Perceptual Leakage** | `{len(perceptual_train_val_leakage)}` | 🟠 HIGH |\n")
        f.write(f"| **Cross-Class Perceptual Matches (Review Flags)** | `{len(perceptual_cross_class_review)}` | 🟠 HIGH |\n")
        f.write(f"| **Severe Blur Samples** (Laplacian var < {args.blur_thresh}) | `{len(blurry_images)}` | 🟠 HIGH |\n")
        f.write(f"| **Severe Class Imbalance (< 40 samples)** | `{len(minority_classes)} classes` | 🟠 HIGH |\n")
        f.write(f"| **Low Contrast Samples** ($\\text{{std}} < {args.contrast_thresh}$) | `{len(low_contrast_images)}` | 🟡 MEDIUM |\n")
        f.write(f"| **Anomalous Dimensions / Aspect Ratios** | `{len(anomalous_shapes)}` | 🟡 MEDIUM |\n")
        f.write(f"| **Intra-Class Exact Duplicate Groups** | `{len(exact_duplicates_within_class)}` | 🟡 MEDIUM |\n\n")

        f.write("---\n\n")
        f.write("## 2. Ranked Action List\n\n")

        f.write("### 🔴 CRITICAL ACTIONS\n")
        if exact_duplicates_cross_class:
            f.write(f"1. **Resolve {len(exact_duplicates_cross_class)} Cross-Class Label Conflicts**:\n")
            f.write("   - The exact same image file exists in two different class folders! This introduces contradictory gradients during training.\n")
            for item in exact_duplicates_cross_class[:15]:
                c_names = [f"Class {c['id']} ({c['name']})" for c in item["classes"]]
                f.write(f"   - MD5 `{item['md5'][:8]}...` shared between: **{' vs '.join(c_names)}** ({', '.join(item['paths'])})\n")
        else:
            f.write("1. **Cross-Class Label Conflicts**: None found ✅\n")

        if exact_duplicates_train_val_leakage:
            f.write(f"2. **Fix {len(exact_duplicates_train_val_leakage)} Train ↔ Validation Leakage Instances**:\n")
            f.write("   - The validation set contains exact copies of training images, artificially inflating validation accuracy.\n")
            for item in exact_duplicates_train_val_leakage[:10]:
                f.write(f"   - MD5 `{item['md5'][:8]}...` in both train and val splits ({len(item['paths'])} total copies).\n")
        else:
            f.write("2. **Train ↔ Validation Exact Leakage**: None found ✅\n")

        if corrupt_images:
            f.write(f"3. **Prune {len(corrupt_images)} Corrupt/Zero-Byte Files**:\n")
            for c in corrupt_images[:10]:
                f.write(f"   - `{c['path']}`: {c['reason']}\n")
        else:
            f.write("3. **Corrupted Images**: None found ✅\n")

        f.write("\n### 🟠 HIGH PRIORITY ACTIONS\n")
        if minority_classes:
            f.write(f"4. **Address Severe Class Imbalance ({len(minority_classes)} classes with < 40 samples)**:\n")
            f.write("   - The model cannot learn robust representations for classes with fewer than 30–40 samples.\n")
            f.write("   - Minority classes:\n")
            for c in minority_classes:
                f.write(f"     - Class {c['class_id']:2d} (`{c['class_name']}`): **{c['total_samples']} samples** (Train: {c['train_samples']}, Val: {c['val_samples']})\n")

        if blurry_images:
            f.write(f"5. **Review {len(blurry_images)} Severely Blurry Samples**:\n")
            f.write("   - Images with extremely low Laplacian variance are unreadable and corrupt arrow/number recognition.\n")
            f.write("   - Top 10 worst blur: " + ", ".join([f"`{b['path']}` ({b['laplacian_var']})" for b in blurry_images[:10]]) + "\n")

        if perceptual_cross_class_review:
            f.write(f"6. **Review {len(perceptual_cross_class_review)} Perceptual Cross-Class Matches**:\n")
            f.write("   - Visually near-identical signs assigned to different classes. Check if these are directional mirror mismatches:\n")
            for m in perceptual_cross_class_review[:15]:
                f.write(f"   - Dist `{m['distance']}`: `{m['image_a']}` ({m['name_a']}) ↔ `{m['image_b']}` ({m['name_b']})\n")

        f.write("\n### 🟡 MEDIUM PRIORITY ACTIONS\n")
        f.write(f"7. **Review {len(exact_duplicates_within_class)} Intra-Class Duplicate Groups**: Identical redundant copies inside the same class.\n")
        f.write(f"8. **Review {len(low_contrast_images)} Low Contrast Samples** and **{len(anomalous_shapes)} Extreme Aspect Ratio Crops**.\n\n")

        f.write("---\n\n")
        f.write("## 3. Class Distribution by Stratified Split\n\n")
        f.write("| Class ID | Class Name | Total Samples | Train (85%) | Val (15%) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: |\n")
        for c in class_distribution:
            f.write(f"| {c['class_id']} | {c['class_name']} | {c['total_samples']} | {c['train_samples']} | {c['val_samples']} |\n")

    print("\n" + "=" * 70, flush=True)
    print("  Audit Complete!", flush=True)
    print(f"  Structured JSON Report : {json_report_path}", flush=True)
    print(f"  Markdown Action Report : {md_report_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
