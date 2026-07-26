"""
explore_risk_bands.py
=====================
Where do the true classes sit along the risk score, and what happens to the
API queue if you skim the very-risky top off it?

The `ranking` block answers "recall@K over the whole queue". It does NOT
answer the operating question: if customers above some risk threshold are
certain enough to act on directly (no API call — that is what
`CERTAINTY_ACT_THRESHOLD` would switch on), how much sooner does the API
reach the *middle-risk* customers who still go severe?

Reads the per-row dump `test_scores.csv.gz` written by a training run
(fold_dir), or any predictions CSV from `run.py predict` (no labels there,
so only the distribution/volume sections are produced).

Usage (server or Mac — needs only the CSV, no DB):
  python explore_risk_bands.py artifacts/<run>/fold_01
  python explore_risk_bands.py artifacts/<run>/fold_01 --slice 0 --bands 20
  python explore_risk_bands.py <run>/fold_01 --thresholds 0.5,0.7,0.8,0.9,0.95
  python explore_risk_bands.py --selfcheck
Outputs printed tables + CSVs (and PNGs) under --out.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import project_config as config

SEVERE = config.NUM_CLASSES - 1
RATE = config.API_RATE_PER_HOUR


# ── loading ──────────────────────────────────────────────────────────────────

def load_scores(path: Path) -> pd.DataFrame:
    """
    -> DataFrame with columns: risk, current_cat, [y_true], [pred_class].

    Accepts a fold directory, a test_scores.csv.gz, or a predictions CSV
    (whose column names differ — `Predictor` output uses RISK_SCORE /
    CURRENT_CAT / PREDICTED_CLASS and carries no label).
    """
    if path.is_dir():
        hits = [path / "test_scores.csv.gz", *sorted(path.glob("predictions/*.csv"))]
        path = next((p for p in hits if p.exists()), None) or _die(
            f"No test_scores.csv.gz or predictions/*.csv under {path}.\n"
            "A run from before this dump existed can produce it without retraining:\n"
            "  rm <fold_dir>/stages/deployed_eval.done && python run.py train --resume <run_dir>"
        )
    df = pd.read_csv(path)

    risk_col = next((c for c in ("RISK_SCORE", f"p{SEVERE}", "P_SEVERE_PAST_DUE")
                     if c in df.columns), None) or _die(
        f"No risk-score column in {path} (looked for RISK_SCORE / p{SEVERE} / P_SEVERE_PAST_DUE)")
    cat_col = next((c for c in ("current_cat", "CURRENT_CAT") if c in df.columns), None) or _die(
        f"No current-category column in {path}")

    out = pd.DataFrame({"risk": df[risk_col].to_numpy(float),
                        "current_cat": df[cat_col].to_numpy(int)})
    for src, dst in (("y_true", "y_true"), ("pred_class", "pred_class"),
                     ("PREDICTED_CLASS", "pred_class")):
        if src in df.columns:
            out[dst] = df[src].to_numpy(int)
    print(f"Loaded {len(out):,} rows from {path}"
          + ("" if "y_true" in out else "  (no labels — recall sections skipped)"))
    return out


def _die(msg):
    raise SystemExit(msg)


# ── sections ─────────────────────────────────────────────────────────────────

def population_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Carve-out accounting: who is ranked, who is rule-flagged, at what cost."""
    carved = df["current_cat"] >= config.CARVE_CURRENT_CAT_GE
    rows = []
    for name, sub in (("rule-flagged (ALREADY_SEVERE)", df[carved]),
                      ("ranked queue", df[~carved]),
                      ("all rows", df)):
        r = {"population": name, "n": len(sub),
             "hours_to_call_all": len(sub) / RATE}
        if "y_true" in df:
            r["n_severe_label"] = int((sub["y_true"] == SEVERE).sum())
            r["severe_rate"] = r["n_severe_label"] / len(sub) if len(sub) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def class_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """True class x predicted class, on the ranked queue only."""
    q = df[df["current_cat"] < config.CARVE_CURRENT_CAT_GE]
    ct = pd.crosstab(q["y_true"], q["pred_class"], dropna=False)
    ct.index.name, ct.columns.name = "true_class", "pred_class"
    ct["n"] = ct.sum(axis=1)
    ct["mean_risk"] = q.groupby("y_true")["risk"].mean()
    for p in (50, 90, 99):
        ct[f"risk_p{p}"] = q.groupby("y_true")["risk"].quantile(p / 100)
    return ct


def band_table(df: pd.DataFrame, n_bands: int) -> pd.DataFrame:
    """
    Equal-count bands of the ranked queue, riskiest first. `cum_recall` is
    the queue-wide recall you get by calling everything down to this band,
    `cum_hours` how long that takes at API_RATE_PER_HOUR.
    """
    q = df[df["current_cat"] < config.CARVE_CURRENT_CAT_GE].sort_values(
        "risk", ascending=False, kind="stable")
    sev = (q["y_true"] == SEVERE).to_numpy() if "y_true" in q else np.zeros(len(q), bool)
    total_sev = max(int(sev.sum()), 1)
    base = sev.mean() if len(q) else np.nan

    rows, seen, seen_sev = [], 0, 0
    for i, idx in enumerate(np.array_split(np.arange(len(q)), n_bands), start=1):
        if not len(idx):
            continue
        b_sev, r = int(sev[idx].sum()), q["risk"].to_numpy()
        seen, seen_sev = seen + len(idx), seen_sev + b_sev
        rows.append({
            "band": i, "risk_hi": r[idx[0]], "risk_lo": r[idx[-1]], "n": len(idx),
            "n_severe": b_sev, "precision": b_sev / len(idx),
            "lift": (b_sev / len(idx)) / base if base else np.nan,
            "share_of_severe": b_sev / total_sev,
            "cum_recall": seen_sev / total_sev, "cum_hours": seen / RATE,
        })
    return pd.DataFrame(rows)


def skim_simulation(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    """
    THE question: flag everyone with risk >= t as act-directly (no API call),
    then measure how the API does on what is left.

    Two denominators, kept separate on purpose:
      `acted_recall`  — severe customers handled without spending a call
      `api_recall_*`  — of the severe still IN the queue, share reached in
                        that window (this is the "middle risk but will be
                        delinquent" number the skim is meant to improve)
      `combined_*`    — (acted severe + API hits) / all severe in the queue
    """
    q = df[df["current_cat"] < config.CARVE_CURRENT_CAT_GE].sort_values(
        "risk", ascending=False, kind="stable")
    risk = q["risk"].to_numpy()
    sev = (q["y_true"] == SEVERE).to_numpy()
    total_sev = max(int(sev.sum()), 1)

    rows = []
    for t in thresholds:
        n_act = int((risk >= t).sum())
        rest_sev = sev[n_act:]
        rest_total = max(int(rest_sev.sum()), 1)
        hits = np.cumsum(rest_sev)
        acted_sev = int(sev[:n_act].sum())

        row = {
            "threshold": t, "n_acted": n_act,
            "acted_precision": acted_sev / n_act if n_act else np.nan,
            "hours_saved": n_act / RATE,
            "acted_recall": acted_sev / total_sev,
            "n_queue_after": len(risk) - n_act,
            "severe_left": int(rest_sev.sum()),
        }
        for name, hours in config.RANKING_REF_WINDOWS.items():
            k = min(int(RATE * hours), len(rest_sev))
            api = hits[k - 1] / rest_total if k else 0.0
            row[f"api_recall_{name}"] = api
            row[f"combined_{name}"] = (acted_sev + (hits[k - 1] if k else 0)) / total_sev
        for pct in (0.5, 0.8):
            reach = np.searchsorted(hits, pct * rest_total) + 1
            row[f"hours_to_{int(pct * 100)}pct_left"] = (
                reach / RATE if reach <= len(rest_sev) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, bands: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = df[df["current_cat"] < config.CARVE_CURRENT_CAT_GE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    edges = np.linspace(0, max(q["risk"].max(), 1e-6), 60)
    for c in sorted(q["y_true"].unique()):
        a1.hist(q.loc[q["y_true"] == c, "risk"], bins=edges, histtype="step",
                label=f"true class {c}", linewidth=1.5)
    a1.set(yscale="log", xlabel="calibrated P(severe)", ylabel="customers (log)",
           title="Risk score by true class — ranked queue")
    a1.legend()

    a2.plot(bands["cum_hours"], bands["cum_recall"], marker="o", ms=3)
    for name, hours in config.RANKING_REF_WINDOWS.items():
        if hours <= bands["cum_hours"].max():
            a2.axvline(hours, ls="--", lw=1, color="grey")
            # axes-fraction y: recall does not start at 0, so a data-space
            # y would drop the label off the bottom of the plot.
            a2.text(hours, 0.02, f" {name}", fontsize=8, color="grey",
                    transform=a2.get_xaxis_transform())
    a2.set(xscale="log", xlabel=f"hours of calling @ {RATE}/h", ylabel="cumulative recall",
           title="Capture curve (severe customers reached)")
    a2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out / "risk_bands.png", dpi=130)
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path,
                    help="fold dir, test_scores.csv.gz, or a predictions CSV")
    ap.add_argument("--slice", default="all",
                    help="'all' or a current_cat value (0/1/2) to restrict to")
    ap.add_argument("--bands", type=int, default=20, help="equal-count bands (default 20 = 5%%)")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated risk cutoffs; default = queue score quantiles")
    ap.add_argument("--out", type=Path, default=Path("explore_output/risk_bands"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="run the built-in math check and exit")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()
    if args.path is None:
        ap.error("path is required (or pass --selfcheck)")

    df = load_scores(args.path)
    if args.slice != "all":
        df = df[df["current_cat"] == int(args.slice)]
        print(f"Slice current_cat == {args.slice}: {len(df):,} rows")
    args.out.mkdir(parents=True, exist_ok=True)

    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        pop = population_summary(df)
        print("\n── Population / carve-out " + "─" * 50)
        print(pop.to_string(index=False))
        pop.to_csv(args.out / "population.csv", index=False)

        if "y_true" not in df:
            print("\nNo labels in this file — stopping after the volume summary.")
            return

        if "pred_class" in df:
            ct = class_crosstab(df)
            print("\n── True class x predicted class, and risk spread " + "─" * 30)
            print(ct.to_string())
            ct.to_csv(args.out / "class_crosstab.csv")

        bands = band_table(df, args.bands)
        print(f"\n── Ranked queue in {args.bands} equal-count bands " + "─" * 35)
        print(bands.to_string(index=False))
        bands.to_csv(args.out / "bands.csv", index=False)

        q = df.loc[df["current_cat"] < config.CARVE_CURRENT_CAT_GE, "risk"]
        thr = ([float(t) for t in args.thresholds.split(",")] if args.thresholds
               else sorted(q.quantile([.5, .8, .9, .95, .99, .995, .999]).unique()))
        skim = skim_simulation(df, thr)
        print("\n── Skim the top: act directly above the threshold, API below " + "─" * 15)
        print(skim.to_string(index=False))
        skim.to_csv(args.out / "skim_thresholds.csv", index=False)

    if not args.no_plots:
        plot(df, bands, args.out)
    print(f"\nWrote CSVs{'' if args.no_plots else ' + risk_bands.png'} to {args.out}/")


def _selfcheck() -> None:
    """Hand-built queue: the band/skim arithmetic must reproduce it exactly."""
    # 10 customers, riskiest first. Severe (class 3) at positions 0, 1, 5.
    # One already-severe customer (current_cat 3) must be carved out entirely.
    df = pd.DataFrame({
        "risk":        [.95, .90, .80, .70, .60, .50, .40, .30, .20, .10, .99],
        "y_true":      [  3,   3,   0,   1,   2,   3,   0,   1,   0,   0,   3],
        "current_cat": [  0,   0,   0,   1,   2,   0,   0,   1,   0,   0,   3],
        "pred_class":  [  3,   3,   0,   0,   2,   0,   0,   0,   0,   0,   3],
    })
    pop = population_summary(df).set_index("population")
    assert pop.loc["ranked queue", "n"] == 10, "carve must drop the current_cat=3 row"
    assert pop.loc["rule-flagged (ALREADY_SEVERE)", "n_severe_label"] == 1

    b = band_table(df, n_bands=5)                     # 5 bands of 2
    assert b["n"].tolist() == [2] * 5
    assert b.loc[0, "n_severe"] == 2                  # top band holds 2 of 3 severe
    assert np.isclose(b.loc[0, "cum_recall"], 2 / 3)
    assert np.isclose(b["cum_recall"].iloc[-1], 1.0)
    assert np.isclose(b.loc[0, "cum_hours"], 2 / RATE)
    assert np.isclose(b.loc[0, "precision"], 1.0)
    assert np.isclose(b.loc[0, "lift"], 1.0 / 0.3)

    # Skim at 0.85: the top 2 (both severe) are acted on, 8 left holding 1 severe.
    s = skim_simulation(df, [0.85]).iloc[0]
    assert s["n_acted"] == 2 and s["acted_precision"] == 1.0
    assert np.isclose(s["acted_recall"], 2 / 3) and s["severe_left"] == 1
    assert np.isclose(s["hours_saved"], 2 / RATE)
    # That last severe sits 4th among the 8 remaining -> 4/RATE hours, and
    # every reference window is far longer than 8 customers of calling, so
    # the API reaches all of what is left and combined recall is total.
    assert np.isclose(s["hours_to_50pct_left"], 4 / RATE)
    for name in config.RANKING_REF_WINDOWS:
        assert np.isclose(s[f"api_recall_{name}"], 1.0)
        assert np.isclose(s[f"combined_{name}"], 1.0)

    # No skim = plain queue behaviour: nothing acted, recall denominators whole.
    s0 = skim_simulation(df, [1.01]).iloc[0]
    assert s0["n_acted"] == 0 and s0["acted_recall"] == 0.0 and s0["severe_left"] == 3
    assert np.isclose(s0["hours_to_50pct_left"], 2 / RATE)
    print("explore_risk_bands.py self-check OK")


if __name__ == "__main__":
    main()
