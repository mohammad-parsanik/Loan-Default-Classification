"""
make_placeholder_bundle.py
==========================
Build a FAKE-but-structurally-real `model_bundle.pkl` so an integrating team
can wire up the scoring package before the real, server-trained bundle can be
exported off the training server.

The placeholder is produced by running the *actual* training path
(process_raw_data -> preprocessing pipeline -> build_features ->
multiclass arm -> StratifiedCalibrator) on synthetic data that carries the
real column schema. So it loads, scores, ranks and flags exactly like the real
bundle — only the numbers are meaningless. Swapping in the real bundle later is
a file copy, no code change.

Usage:
  python make_placeholder_bundle.py                        # -> placeholder/
  python make_placeholder_bundle.py --package scoring_package_placeholder
  python make_placeholder_bundle.py --self-check           # end-to-end smoke test
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import project_config as config

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("placeholder")

# The 64 per-loan feature columns of D_ANALYTICS.DPD_SAMPLE1 (column_changes.md,
# cross-checked against explore_output/iv_report.csv), in documented group
# order. NOTE: nothing in the scoring path validates feature NAMES — the
# preprocessing pipeline is positional, so what actually has to match at swap
# time is the column COUNT and ORDER of the caller's DataFrame.
FEATURE_COLUMNS = [
    # Group A — current DPD state
    "DPD_DAYS", "LOAN_CATEGORY", "DAYS_TO_NEXT_THRESHOLD",
    "PAYED_OVERDUE_INST_CNT", "UNPAYED_INST_CNT", "PAYED_OVERDUE_AMNT",
    "OVERDUE_RATIO", "ONTIME_RATIO", "IS_IN_WARNING_ZONE",
    "CNT_INSTALLMENT_WARNING_ZONE", "MATURED_INST_CNT", "UPCOMING_INST_CNT",
    "UPCOMING_AMNT",
    # Group B — DPD trajectory
    "DPD_DAYS_T1", "DPD_DAYS_T2", "DPD_DAYS_T3", "DPD_DAYS_T4", "DPD_DAYS_T5",
    "CATEGORY_T1", "CATEGORY_T2", "CATEGORY_T3",
    # Group C — trend & velocity
    "DPD_TREND_1M", "DPD_TREND_3M", "CATEGORY_TREND_1M", "CATEGORY_TREND_3M",
    "IS_DETERIORATING", "IS_IMPROVING", "IS_ACCELERATING",
    "MONTHS_IN_CURRENT_CATEGORY",
    # Group D — historical worst performance
    "HIST_MAX_DPD_DAYS", "HIST_MAX_CATEGORY", "HAS_EVER_BEEN_NPL",
    "HAS_EVER_BEEN_PRENPL", "HAS_RECOVERED_BEFORE", "CNT_RECOVERED_BEFORE",
    "COUNT_CATEGORY_CHANGES",
    # Group E — DPD event counts
    "COUNT_DPD_EVENTS_LAST_3M", "COUNT_DPD_EVENTS_LAST_6M",
    "COUNT_30PLUS_DPD_LAST_3M", "COUNT_60PLUS_DPD_LAST_3M",
    "COUNT_90PLUS_DPD_LAST_3M", "MAX_DPD_LAST_3M", "MAX_DPD_LAST_6M",
    "TOTAL_DPD_DAYS_LAST_3M", "TOTAL_DPD_DAYS_LAST_6M",
    "CONSECUTIVE_MONTHS_WITH_DPD", "DAYS_SINCE_LAST_DPD",
    "DAYS_SINCE_LAST_30_DPD", "DAYS_SINCE_LAST_60_DPD",
    "DAYS_SINCE_LAST_90_DPD",
    # Group F — cross-contract customer history
    "WORST_CLOSED_LOAN_DPD", "AVERAGE_CLOSE_LOAN_DPD", "MAX_DPD_ANY_PAST_LOAN",
    "AVG_DPD_OTHER_LOANS", "PRE_UPTO30_DPD_LOANS", "PRE_UPTO60_DPD_LOANS",
    "PRE_UPTO120_DPD_LOANS", "PRE_UPTO150_DPD_LOANS", "COUNT_ACTIVE_CONTRACTS",
    "COUNT_DELINQUENT_CONTRACTS",
    # Group G — contract maturity & structure
    "CONTRACT_AGE_MONTH", "PCT_COMPLETED", "REMAINING_INST_CNT",
    "REMAINING_AMNT",
]

MAX_LOANS = 2          # what MAX_LOANS_PER_CUSTOMER resolves to on real data
CAT_PROBS = [0.62, 0.22, 0.09, 0.05, 0.02]      # raw LOAN_CATEGORY 0..4 mix


# ── synthetic data with the real schema ───────────────────────────────────────

def _feature_values(name: str, latent: np.ndarray, cat: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Plausible values for one column, mildly driven by a shared latent risk."""
    n = len(latent)
    if name in config.BINARY_FEATURES:
        return (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(float)
    if name == "LOAN_CATEGORY":
        return cat.astype(float)
    if name.startswith("DAYS_SINCE"):
        return np.maximum(0.0, 900 - 200 * latent + rng.normal(0, 120, n)).round()
    if "RATIO" in name or name == "PCT_COMPLETED":
        return np.clip(0.5 + 0.15 * latent + rng.normal(0, 0.2, n), 0, 1)
    if "AMNT" in name:
        return np.exp(rng.normal(17, 1.2, n)).round(-3)
    if "CATEGORY" in name:                       # CATEGORY_T*, HIST_MAX_CATEGORY
        return np.clip(cat + rng.integers(-1, 2, n), 0, 4).astype(float)
    if "TREND" in name:
        return (0.4 * latent + rng.normal(0, 1, n)).round(2)
    if "DPD" in name:                            # DPD_DAYS, MAX_DPD_*, totals…
        return np.maximum(0.0, 30 * cat + 12 * latent + rng.normal(0, 15, n)).round()
    if name.startswith(("COUNT_", "CNT_")) or name.endswith(("_CNT", "_LOANS")):
        return rng.poisson(np.maximum(0.3, 1.5 + 0.6 * latent)).astype(float)
    return np.maximum(0.0, 24 + 6 * latent + rng.normal(0, 10, n)).round()


def synth_frame(n_rows: int, seed: int, snapshot: float = 20260601.0,
                with_target: bool = True) -> pd.DataFrame:
    """
    Loan-level frame with the real TRAIN_TABLE schema. Labels respect the
    defining property of WORST_FUTURE_CAT: label >= current LOAN_CATEGORY.
    """
    rng = np.random.default_rng(seed)
    cat = rng.choice(5, n_rows, p=CAT_PROBS)
    latent = rng.normal(0, 1, n_rows)                       # shared risk driver

    # ~20% of customers hold 2 loans (matches the real 99th-percentile of 2)
    n_customers = int(n_rows / 1.2)
    owner = rng.integers(0, n_customers, n_rows)

    df = pd.DataFrame({
        "LOAN_ID": np.arange(n_rows) + seed * 10_000_000,
        "CONTRACT_NUMBER": [f"CT{seed}{i:08d}" for i in range(n_rows)],
        "NATIONAL_CODE": [f"{seed}{c:09d}" for c in owner],
        "SNAPSHOT_DATE": snapshot,
    })
    for col in FEATURE_COLUMNS:
        df[col] = _feature_values(col, latent, cat, rng)

    if with_target:
        # future worsening driven by the same latent -> the arm has real signal
        steps = (rng.random(n_rows) < 1 / (1 + np.exp(-(latent - 0.8)))).astype(int) \
              + (rng.random(n_rows) < 0.12).astype(int)
        future = np.clip(cat + steps, cat, 4)
        df["WORST_FUTURE_CAT"] = future.astype(float)
        df["WORST_FUTURE_DPD"] = (future * 45 + rng.integers(0, 30, n_rows)).astype(float)
    return df


# ── build ─────────────────────────────────────────────────────────────────────

def build_bundle(out_path: Path, n_train: int, n_val: int, seed: int) -> Path:
    from src.baselines.aggregated_xgboost import ARM_BUILDERS, build_features
    from src.data.data_loader import DataLoader
    from src.data.preprocessing import create_preprocessing_pipeline
    from src.evaluation.calibration import StratifiedCalibrator
    from src.inference.model_loader import build_arm_bundle

    dl = DataLoader()
    train_inst, feats = dl.process_raw_data(synth_frame(n_train, seed), MAX_LOANS)
    val_inst, _ = dl.process_raw_data(synth_frame(n_val, seed + 1), MAX_LOANS)

    scaler = create_preprocessing_pipeline(feats, config.BINARY_FEATURES)
    scaler.fit([i["features"] for i in train_inst])
    for split in (train_inst, val_inst):
        for inst, x in zip(split, scaler.transform([i["features"] for i in split])):
            inst["features"] = x

    X_tr, y_tr = build_features(train_inst)
    X_v, y_v = build_features(val_inst)
    cat_tr = np.array([i["current_cat"] for i in train_inst])
    cat_v = np.array([i["current_cat"] for i in val_inst])

    arm = ARM_BUILDERS[config.DEPLOY_ARM]()
    arm.train(X_tr, y_tr, cat_tr, X_v, y_v, cat_v)
    cal = StratifiedCalibrator(config.CALIBRATION_MIN_STRATUM_N).fit(
        arm.predict_proba(X_v, cat_v), y_v, cat_v)

    bundle = build_arm_bundle(scaler, arm, cal, MAX_LOANS, feats)
    bundle["metadata"].update({
        "placeholder": True,
        "placeholder_note": (
            "SYNTHETIC MODEL — trained on random data with the real column "
            "schema, for integration testing only. Scores carry no business "
            "meaning. Replace with the server-trained model_bundle.pkl."
        ),
        "arm": config.DEPLOY_ARM,
        "built_with": _versions(),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


def _versions() -> dict:
    import sklearn
    import xgboost
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__}


# ── self-check ────────────────────────────────────────────────────────────────

def self_check(tmp_dir: Path) -> None:
    """Build a small placeholder and score with it, as the tech team would."""
    from src.inference.scoring import run_scoring
    from src.inference.scoring_params import ScoringParams

    bundle = build_bundle(tmp_dir / "model_bundle.pkl", 6_000, 3_000, seed=1)
    df = synth_frame(400, seed=99, snapshot=20260701.0, with_target=False)
    q = run_scoring(df, ScoringParams(bundle_path=bundle))

    # One scored row per unit of prediction: a loan, or a whole customer.
    expected_rows = (
        len(df) if config.PREDICTION_GRAIN == "loan"
        else df["NATIONAL_CODE"].nunique()
    )
    assert len(q) == expected_rows, (len(q), expected_rows)
    for col in ["RISK_RANK", "RISK_SCORE", "RULE_FLAG", "CURRENT_CAT",
                "P_NO_DELAY", "P_CURRENT", "P_PAST_DUE", "P_SEVERE_PAST_DUE",
                "PREDICTED_CLASS", "EXPECTED_COST", "CUSTOMER_MAX_RISK_SCORE"]:
        assert col in q.columns, col
    queued = q[q["RULE_FLAG"] == ""]
    assert queued["RISK_RANK"].tolist() == list(range(1, len(queued) + 1))
    assert queued["RISK_SCORE"].is_monotonic_decreasing
    assert queued["RISK_SCORE"].nunique() > 10, "scores are degenerate"
    # already-severe carve-out, and the monotone mask (label >= current_cat)
    assert (q.loc[q["CURRENT_CAT"] >= config.CARVE_CURRENT_CAT_GE,
                  "RULE_FLAG"] == "ALREADY_SEVERE").all()
    assert (q.loc[q["CURRENT_CAT"] >= 1, "P_NO_DELAY"] < 1e-6).all()
    print("self-check OK — placeholder bundle loads, scores and ranks")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=BASE_DIR / "placeholder",
                   help="Directory for the placeholder bundle + sample input "
                        "(default: ./placeholder/)")
    p.add_argument("--package", type=Path, default=None,
                   help="Also build a full scoring package here, with the "
                        "placeholder bundle inside.")
    p.add_argument("--n_train", type=int, default=120_000, help="synthetic train rows")
    p.add_argument("--n_val", type=int, default=50_000, help="synthetic val rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-check", action="store_true",
                   help="Build a small bundle in a temp dir and score with it.")
    args = p.parse_args()

    if args.self_check:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self_check(Path(td))
        return

    out_dir = args.output
    bundle = build_bundle(out_dir / "model_bundle.pkl", args.n_train, args.n_val, args.seed)

    sample = synth_frame(500, seed=args.seed + 7, snapshot=20260701.0, with_target=False)
    sample.to_csv(out_dir / "sample_input.csv", index=False)
    (out_dir / "PLACEHOLDER.md").write_text(PLACEHOLDER_README.format(
        versions=json.dumps(_versions(), indent=2), n_features=len(FEATURE_COLUMNS),
        arm=config.DEPLOY_ARM))

    size_mb = bundle.stat().st_size / 1e6
    print(f"\nPlaceholder bundle: {bundle}  ({size_mb:.1f} MB)")
    print(f"Sample input CSV:   {out_dir / 'sample_input.csv'} (500 loan rows, no labels)")
    print(f"Read me:            {out_dir / 'PLACEHOLDER.md'}")

    if args.package:
        subprocess.run([sys.executable, str(BASE_DIR / "build_scoring_package.py"),
                        "--bundle", str(bundle), "--output", str(args.package)],
                       check=True)
        shutil.copy2(out_dir / "sample_input.csv", args.package / "sample_input.csv")
        shutil.copy2(out_dir / "PLACEHOLDER.md", args.package / "PLACEHOLDER.md")
        # README_SCORING.md is what a recipient opens first — say up front that
        # the shipped bundle is not the trained model.
        readme = args.package / "README_SCORING.md"
        readme.write_text(PACKAGE_BANNER + readme.read_text())
        print(f"Sample input + PLACEHOLDER.md copied into {args.package}")


PACKAGE_BANNER = """\
> ⚠️ **The `model_bundle.pkl` in this folder is a PLACEHOLDER**, not the
> trained model — synthetic, for integration testing only. Its scores mean
> nothing. Read `PLACEHOLDER.md` first. Everything below describes the code,
> which is the real, final scoring code and does not change when the trained
> bundle replaces the placeholder.

"""

PLACEHOLDER_README = """\
# PLACEHOLDER model_bundle.pkl — READ BEFORE USE

`model_bundle.pkl` in this folder is **not the trained model**. It is a
synthetic stand-in produced by `make_placeholder_bundle.py`, trained on
randomly generated data that carries the real column schema ({n_features}
per-loan features + the ID/label columns).

**Its scores mean nothing.** It exists so the scoring code can be integrated,
run and tested in your pipeline before the real bundle — which is trained on
the training server — is exported.

## What it is faithful to

Everything except the numbers:

- same file format and loader path (`ModelLoader` / `load_bundle`, `kind="arm"`)
- same contents: fitted preprocessing pipeline + XGBoost `{arm}` arm +
  `StratifiedCalibrator` + metadata
- same expected input schema and same output columns (`RISK_RANK`,
  `RISK_SCORE`, `RULE_FLAG`, per-class probabilities, `PREDICTED_CLASS`,
  `EXPECTED_COST`)
- same rules: already-severe carve-out, supersede/dedup, recently-called,
  monotone masking, queue ranking

So integration code written against it works unchanged against the real one.

## Swapping in the real model

Replace `model_bundle.pkl` with the real file. Nothing else changes — no code,
no config. To confirm which one is loaded:

```python
import joblib
meta = joblib.load("model_bundle.pkl")["metadata"]
print(meta.get("placeholder", False))   # True = still the placeholder
```

Loading the placeholder also emits a `WARNING` log line naming it as such.

## Two things to verify at swap time

1. **Column order.** The preprocessing pipeline is positional: it applies
   per-column statistics by index, not by name. Your DataFrame must present
   the feature columns in the same order the model was trained on. The
   placeholder was built with the order in `column_changes.md`; the real
   bundle's order is whatever the training table returned — check
   `metadata["features"]` of the real bundle and order your columns to match.
2. **Library versions.** A pickle is version-sensitive. This placeholder was
   built with:

```json
{versions}
```

   If the real bundle was built on the server with different versions, install
   the server's versions rather than these.

## Sample input

`sample_input.csv` is 500 synthetic loan rows in the expected input schema
(no label columns) — enough to smoke-test the pipeline end to end:

```python
import pandas as pd
from src.inference.scoring import run_scoring
from src.inference.scoring_params import ScoringParams

queue = run_scoring(pd.read_csv("sample_input.csv"),
                    ScoringParams(bundle_path="model_bundle.pkl",
                                  output_path="queue.csv"))
```

Its values are random, so the resulting ranking is random too — what you are
testing is that it runs, not what it says.
"""


if __name__ == "__main__":
    main()
