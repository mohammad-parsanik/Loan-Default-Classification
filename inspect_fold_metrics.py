"""
inspect_fold_metrics.py
=======================
Print the ranking comparison (pooled + per current-cat stratum) for every
model arm stored in a fold's stage pickles. Closes the Run-5 logging gap:
run.py only logged the DeepSets arm's per-stratum ranking.

Usage (on the server):
  python inspect_fold_metrics.py artifacts/20260708_135033/fold_01
"""

import argparse
import sys
from pathlib import Path

import joblib


def _fmt_ranking(label: str, rk: dict) -> None:
    if not rk or "pr_auc" not in rk:
        print(f"  {label:34s} | (no ranking block)")
        return
    parts = [
        f"{k[3:]}: R={v['recall']:.3f} P={v['precision']:.3f} lift={v['lift']:.1f}x"
        for k, v in rk.items()
        if k.startswith("at_")
    ]
    print(
        f"  {label:34s} | n={rk['n_ranked']:>7,} sev={rk['n_severe']:>6,} "
        f"base={rk['base_rate']:.4f} AP={rk['pr_auc']:.4f}\n"
        f"  {'':34s} | " + "  ".join(parts)
    )


def _print_arm(name: str, rk: dict) -> None:
    print(f"\n── {name} " + "─" * max(0, 66 - len(name)))
    _fmt_ranking("pooled (carved population)", rk)
    for slice_name, sub in (rk or {}).get("by_current_cat", {}).items():
        _fmt_ranking(f"  {slice_name}", sub)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold_dir", type=Path, help="e.g. artifacts/<run>/fold_01")
    args = parser.parse_args()

    stages = args.fold_dir / "stages"
    if not stages.exists():
        sys.exit(f"No stages/ directory under {args.fold_dir}")

    baseline = joblib.load(stages / "baseline.pkl") if (stages / "baseline.pkl").exists() else {}
    final    = joblib.load(stages / "evaluation.pkl") if (stages / "evaluation.pkl").exists() else {}

    _print_arm("Baseline: multiclass XGB, P(severe)", baseline.get("ranking", {}))
    _print_arm("Binary severe comparator", baseline.get("binary_ranking", {}))
    _print_arm("DeepSets+XGB, P(severe)", final.get("ranking", {}))


if __name__ == "__main__":
    main()
