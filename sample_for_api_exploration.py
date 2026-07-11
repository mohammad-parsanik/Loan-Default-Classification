"""
sample_for_api_exploration.py
==============================
Pull a spread of customers across risk levels from a predictions CSV, for
manual inspection (e.g. test API calls) before committing to a model.

Usage:
  python sample_for_api_exploration.py predictions/predictions_20260621.csv
  python sample_for_api_exploration.py predictions/predictions_20260621.csv --n 100 --bins 5
  python sample_for_api_exploration.py --self_check
"""

import argparse

import numpy as np
import pandas as pd


def sample_across_risk(
    df: pd.DataFrame,
    n: int = 100,
    bins: int = 5,
    include_flagged: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample by RISK_SCORE quantile bin, from the ranked queue."""
    queue = df[df["RULE_FLAG"] == ""].copy()

    # duplicates="drop": most customers cluster near RISK_SCORE ~0, so high
    # quantile counts can collapse to fewer distinct edges than requested.
    queue["RISK_BIN"] = pd.qcut(
        queue["RISK_SCORE"], bins, duplicates="drop"
    )
    actual_bins = queue["RISK_BIN"].nunique()
    if actual_bins < bins:
        print(f"Note: RISK_SCORE ties collapsed {bins} requested bins to {actual_bins}.")

    per_bin = max(n // actual_bins, 1)
    sample = pd.concat(
        [g.sample(min(per_bin, len(g)), random_state=seed) for _, g in queue.groupby("RISK_BIN", observed=True)],
        ignore_index=True,
    )

    if include_flagged:
        flagged = df[df["RULE_FLAG"] != ""]
        per_flag = max(n // 20, 3)
        flagged_sample = pd.concat(
            [g.sample(min(per_flag, len(g)), random_state=seed) for _, g in flagged.groupby("RULE_FLAG")],
            ignore_index=True,
        )
        sample = pd.concat([sample, flagged_sample], ignore_index=True)

    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "NATIONAL_CODE": [f"c{i}" for i in range(n)],
        "RISK_SCORE": np.concatenate([np.zeros(int(n * 0.8)), rng.random(int(n * 0.2))]),
        "RULE_FLAG": rng.choice(["", "", "", "ALREADY_SEVERE"], n),
    })
    out = sample_across_risk(df, n=100, bins=5)
    assert 0 < len(out) <= 105
    assert (out["RULE_FLAG"] == "").all()
    assert out["RISK_BIN"].nunique() >= 2   # spans more than one risk level
    print(f"self-check OK — {len(out)} rows across {out['RISK_BIN'].nunique()} risk bins")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=str, nargs="?")
    parser.add_argument("--n", type=int, default=100, help="total sample size")
    parser.add_argument("--bins", type=int, default=5, help="RISK_SCORE quantile bins")
    parser.add_argument("--output", type=str, default="api_exploration_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_flagged", action="store_true",
        help="also include a few ALREADY_SEVERE/SUPERSEDED/etc rows for contrast",
    )
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return
    if not args.predictions_csv:
        parser.error("predictions_csv is required (or pass --self_check)")

    df = pd.read_csv(args.predictions_csv)
    # RULE_FLAG="" round-trips through CSV as NaN (pandas reads empty
    # strings back as null) — restore the "unflagged" marker before filtering.
    df["RULE_FLAG"] = df["RULE_FLAG"].fillna("")
    sample = sample_across_risk(df, args.n, args.bins, args.include_flagged, args.seed)
    sample.to_csv(args.output, index=False)
    print(f"Wrote {len(sample):,} sampled rows -> {args.output}")
    print(sample["RISK_BIN"].value_counts().sort_index() if "RISK_BIN" in sample else "")


if __name__ == "__main__":
    main()
