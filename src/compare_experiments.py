import os
import argparse
import json
import pandas as pd


HARD_FAMILIES = [
    (("23", "24"), ("24", "23"), "23 ↔ 24 (Turn Left / Right)"),
    (("36", "37"), ("37", "36"), "36 ↔ 37 (Side Road Junction Left / Right)"),
    (("42", "43"), ("43", "42"), "42 ↔ 43 (Staggered Junction Left / Right)"),
    (("47", "48"), ("48", "47"), "47 ↔ 48 (Countdown Marker 3 vs 2 bars)"),
    (("49", "50"), ("50", "49"), "49 ↔ 50 (Countdown Marker 2 vs 1 bar)"),
]


def load_metrics(exp_dir: str) -> dict:
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare Two Experiment Runs on 5 Hard Confusion Families")
    parser.add_argument('--base_dir', type=str, default='results_baseline', help='Baseline results directory')
    parser.add_argument('--exp_dir', type=str, required=True, help='New experiment results directory')
    args = parser.parse_args()

    m_base = load_metrics(args.base_dir)
    m_exp = load_metrics(args.exp_dir)

    if not m_base:
        print(f"[WARN] Could not load metrics from baseline: {args.base_dir}")
    if not m_exp:
        print(f"[WARN] Could not load metrics from experiment: {args.exp_dir}")

    print("=" * 80)
    print(f"  EXPERIMENT COMPARISON MATRIX")
    print(f"  Baseline   : {args.base_dir}")
    print(f"  Experiment : {args.exp_dir}")
    print("=" * 80)

    acc_base = m_base.get("accuracy", 0.0)
    acc_exp  = m_exp.get("accuracy", 0.0)
    f1_base   = m_base.get("macro_f1", 0.0)
    f1_exp    = m_exp.get("macro_f1", 0.0)
    err_base  = m_base.get("total_errors", 0)
    err_exp   = m_exp.get("total_errors", 0)

    print(f"  Overall Accuracy : {acc_base:.2f}%  →  {acc_exp:.2f}%  (Delta: {acc_exp - acc_base:+.2f}%)")
    print(f"  Macro-F1 Score   : {f1_base:.2f}%  →  {f1_exp:.2f}%  (Delta: {f1_exp - f1_base:+.2f}%)")
    print(f"  Total Errors     : {err_base}  →  {err_exp}  (Delta: {err_exp - err_base:+d})")
    print("-" * 80)

    print(f"\n  FIVE HARD CONFUSION FAMILIES BREAKDOWN:")
    print(f"  {'-'*76}")
    header = f"  {'Confusion Family':<42} | {'Baseline':<8} | {'Exp':<8} | {'Delta':<8} | {'% Change':<9}"
    print(header)
    print(f"  {'-'*76}")

    fam_base = m_base.get("family_errors", {})
    fam_exp  = m_exp.get("family_errors", {})

    total_f_base = 0
    total_f_exp = 0

    for pair_a, pair_b, label in HARD_FAMILIES:
        key_a = f"{pair_a[0]}->{pair_a[1]}"
        key_b = f"{pair_b[0]}->{pair_b[1]}"
        
        c_base = fam_base.get(key_a, 0) + fam_base.get(key_b, 0)
        c_exp  = fam_exp.get(key_a, 0) + fam_exp.get(key_b, 0)

        total_f_base += c_base
        total_f_exp  += c_exp

        delta = c_exp - c_base
        pct = (delta / c_base * 100.0) if c_base > 0 else 0.0
        print(f"  {label:<42} | {c_base:<8d} | {c_exp:<8d} | {delta:<+8d} | {pct:<+8.1f}%")

    print(f"  {'-'*76}")
    total_delta = total_f_exp - total_f_base
    total_pct = (total_delta / total_f_base * 100.0) if total_f_base > 0 else 0.0
    print(f"  {'TOTAL (5 Hard Families)':<42} | {total_f_base:<8d} | {total_f_exp:<8d} | {total_delta:<+8d} | {total_pct:<+8.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
