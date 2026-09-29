"""
Monthly monitoring: how did the predictions we actually made turn out?

Joins the inference engine's prediction table (PRED_ARCHIVE_TABLE — one row
per scored loan x snapshot, `score_instances` columns) to the feed's outcomes
and recomputes every cohort from the DB on each run:

  - matured cohort  (LABEL_HORIZON_DATE <= asof): the label is final — ranking
    quality at the API budget, exposure caught, calibration, migration.
  - interim cohort  (horizon not reached): WORST_FUTURE_CAT holds the worst
    category reached SO FAR (contract §2), a lower bound on the final label.
    Reported only as "flagged loans deteriorated at N x the rate of the rest",
    never mixed into matured metrics.

Evaluation is per snapshot, on the ranked population (current_cat <
CARVE_CURRENT_CAT_GE), ordered by RISK_SCORE with LOAN_ID as tie-break —
the same order the production queue uses.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import project_config as config
from src.evaluation.ranking import SEVERE_CLASS, _order, capture_curve, ranking_metrics

logger = logging.getLogger(__name__)

PRED_COLS = ["SNAPSHOT_DATE", "LOAN_ID", "NATIONAL_CODE", "CURRENT_CAT", "RISK_SCORE"]
OUTCOME_COLS = ["SNAPSHOT_DATE", "LOAN_ID", "WORST_FUTURE_CAT", "LABEL_HORIZON_DATE",
                "REMAINING_AMNT"]
PSI_REF_SNAPSHOTS = 6


# ── Loading ───────────────────────────────────────────────────────────────────

def load_frames(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions from PRED_ARCHIVE_TABLE, outcomes for the same snapshots from TRAIN_TABLE."""
    table = config.PRED_ARCHIVE_TABLE
    if not table:
        raise ValueError("project_config.PRED_ARCHIVE_TABLE is not set.")
    preds = conn.read_sql(f"SELECT * FROM {table}")
    snaps = sorted(int(s) for s in preds["SNAPSHOT_DATE"].unique())
    outcomes = pd.concat(
        [conn.read_sql(
            f"SELECT {', '.join(OUTCOME_COLS)} FROM {config.TRAIN_TABLE} WHERE SNAPSHOT_DATE = ?",
            (s,),
        ) for s in snaps],
        ignore_index=True,
    ) if snaps else pd.DataFrame(columns=OUTCOME_COLS)
    return preds, outcomes


def prepare(preds: pd.DataFrame, outcomes: pd.DataFrame, asof: int) -> tuple[pd.DataFrame, dict]:
    """Validate, dedupe and join. Returns (frame, data-quality dict)."""
    missing = set(PRED_COLS) - set(preds.columns)
    if missing:
        raise ValueError(f"{config.PRED_ARCHIVE_TABLE} lacks required columns {sorted(missing)}")

    p = preds.copy()
    p["SNAPSHOT_DATE"] = p["SNAPSHOT_DATE"].astype(float).astype(int)
    # A snapshot scored twice leaves two rows per loan. Prefer the newest run
    # when the table says which that is, else the higher score — either way a
    # stated, deterministic choice.
    order_col = "SCORED_AT" if "SCORED_AT" in p.columns else "RISK_SCORE"
    key = ["SNAPSHOT_DATE", "LOAN_ID"]
    dup = p.duplicated(key, keep=False)
    n_dup_conflict = int(
        (p[dup].groupby(key)["RISK_SCORE"].nunique() > 1).sum()) if dup.any() else 0
    p = (p.sort_values(key + [order_col], ascending=[True, True, False])
          .drop_duplicates(key, keep="first"))

    o = outcomes.copy()
    o["SNAPSHOT_DATE"] = o["SNAPSHOT_DATE"].astype(float).astype(int)
    df = p[PRED_COLS].merge(o, on=key, how="left", validate="one_to_one")

    unmatched = df["WORST_FUTURE_CAT"].isna()
    quality = {
        "n_prediction_rows": int(len(preds)),
        "n_duplicates_dropped": int(len(preds) - len(p)),
        "n_duplicate_keys_with_different_scores": n_dup_conflict,
        "dedup_rule": f"keep highest {order_col}",
        "unmatched_by_snapshot": {
            int(s): int(n) for s, n in df[unmatched].groupby("SNAPSHOT_DATE").size().items()
        },
    }
    if unmatched.any():
        logger.warning(f"{int(unmatched.sum()):,} predicted loans have no row in "
                       f"{config.TRAIN_TABLE} — excluded from outcome metrics.")
    df = df[~unmatched].copy()
    df["OUTCOME"] = np.minimum(df["WORST_FUTURE_CAT"].astype(int), config.NUM_CLASSES - 1)
    df["CURRENT_CAT"] = df["CURRENT_CAT"].astype(int)
    df["MATURED"] = df["LABEL_HORIZON_DATE"].astype(float).astype(int) <= asof
    return df, quality


# ── Metrics ───────────────────────────────────────────────────────────────────

def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def _ranked(cohort: pd.DataFrame) -> pd.DataFrame:
    """Ranked population in queue order, with a 1-based QUEUE_POS."""
    r = cohort[cohort["CURRENT_CAT"] < config.CARVE_CURRENT_CAT_GE]
    r = r.iloc[_order(r["RISK_SCORE"].to_numpy(), r["LOAN_ID"].to_numpy())].copy()
    r["QUEUE_POS"] = np.arange(1, len(r) + 1)
    return r


def _k(window: str) -> int:
    return int(config.API_RATE_PER_HOUR * config.RANKING_REF_WINDOWS[window])


def matured_metrics(cohort: pd.DataFrame) -> dict:
    y, cc = cohort["OUTCOME"].to_numpy(), cohort["CURRENT_CAT"].to_numpy()
    out = ranking_metrics(y, cohort["RISK_SCORE"].to_numpy(), strata=cc,
                          tie_break=cohort["LOAN_ID"].to_numpy())
    r = _ranked(cohort)
    severe = r["OUTCOME"] == SEVERE_CLASS
    if not severe.any():
        return out

    for w in config.RANKING_REF_WINDOWS:
        block = out[f"at_{w}"]
        top = r["QUEUE_POS"] <= block["k"]
        block["recall_ci"] = wilson(int((top & severe).sum()), int(severe.sum()))
        sev_amt = r.loc[severe, "REMAINING_AMNT"].sum()
        block["exposure_caught"] = (float(r.loc[top & severe, "REMAINING_AMNT"].sum() / sev_amt)
                                    if sev_amt > 0 else float("nan"))

    # Per current_cat, inside the ONE pooled queue. ranking_metrics'
    # by_current_cat re-ranks each stratum alone with the full K, which reads
    # 100% recall for any stratum smaller than K — not what the queue does.
    by_cat = out.get("by_current_cat", {})
    out["queue_by_cat"] = {}
    for cat, sub in r.groupby("CURRENT_CAT"):
        sev = sub["OUTCOME"] == SEVERE_CLASS
        row = {"n": int(len(sub)), "n_severe": int(sev.sum()), "base_rate": float(sev.mean()),
               "pr_auc": by_cat.get(f"current_cat_{cat}", {}).get("pr_auc", float("nan"))}
        for w in config.RANKING_REF_WINDOWS:
            top = sub["QUEUE_POS"] <= _k(w)
            row[f"share_of_top_{w}"] = float(top.sum() / min(_k(w), len(r)))
            row[f"recall_{w}"] = float((top & sev).sum() / sev.sum()) if sev.any() else float("nan")
        out["queue_by_cat"][int(cat)] = row

    out["predicted_severe"] = float(r["RISK_SCORE"].sum())
    # Rank-based deciles: isotonic scores tie in blocks, so qcut on the score
    # itself would collapse bins.
    dec = pd.qcut(r["QUEUE_POS"], 10, labels=False)
    cal = r.groupby(dec).agg(mean_score=("RISK_SCORE", "mean"),
                             severe_rate=("OUTCOME", lambda s: (s == SEVERE_CLASS).mean()),
                             n=("OUTCOME", "size"))
    out["calibration_deciles"] = cal.reset_index(drop=True).to_dict("list")

    r["GROUP"] = np.where(r["QUEUE_POS"] <= _k("1_week"), "top_1_week", "rest")
    out["migration"] = {
        g: pd.crosstab(sub["CURRENT_CAT"], sub["OUTCOME"])
             .reindex(columns=range(config.NUM_CLASSES), fill_value=0)
             .to_dict("index")
        for g, sub in r.groupby("GROUP")
    }
    out["capture_curve"] = capture_curve(r["OUTCOME"].to_numpy(), r["RISK_SCORE"].to_numpy(),
                                         tie_break=r["LOAN_ID"].to_numpy())
    return out


def interim_metrics(cohort: pd.DataFrame, snapshot: int, asof: int) -> dict:
    r = _ranked(cohort)
    worse = r["OUTCOME"] > r["CURRENT_CAT"]
    severe = r["OUTCOME"] == SEVERE_CLASS
    months = (pd.Timestamp(str(asof)) - pd.Timestamp(str(snapshot))).days / 30.44
    out = {"months_elapsed": round(months, 1), "n_ranked": int(len(r)),
           "deteriorated_rate": float(worse.mean()) if len(r) else float("nan"),
           "severe_so_far_rate": float(severe.mean()) if len(r) else float("nan")}
    rest = r["QUEUE_POS"] > _k("1_week")
    for w in config.MONITOR_HEADLINE_WINDOWS:
        top = r["QUEUE_POS"] <= _k(w)
        out[f"top_{w}"] = {"n": int(top.sum()),
                           "deteriorated_rate": float(worse[top].mean()),
                           "severe_so_far_rate": float(severe[top].mean())}
    out["rest"] = {"n": int(rest.sum()),
                   "deteriorated_rate": float(worse[rest].mean()) if rest.any() else float("nan"),
                   "severe_so_far_rate": float(severe[rest].mean()) if rest.any() else float("nan")}
    rest_rate = out["rest"]["deteriorated_rate"]
    for w in config.MONITOR_HEADLINE_WINDOWS:
        out[f"top_{w}"]["lift_vs_rest"] = (out[f"top_{w}"]["deteriorated_rate"] / rest_rate
                                           if rest_rate else float("nan"))
    return out


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def drift(preds: pd.DataFrame) -> list[dict]:
    """Per-snapshot population and score-distribution stats, straight from the prediction table."""
    p = preds.copy()
    p["SNAPSHOT_DATE"] = p["SNAPSHOT_DATE"].astype(float).astype(int)
    p = p.drop_duplicates(["SNAPSHOT_DATE", "LOAN_ID"])
    snaps = sorted(p["SNAPSHOT_DATE"].unique())
    ranked = p[p["CURRENT_CAT"] < config.CARVE_CURRENT_CAT_GE]
    rows = []
    for i, s in enumerate(snaps):
        cur = ranked.loc[ranked["SNAPSHOT_DATE"] == s, "RISK_SCORE"].to_numpy()
        ref_snaps = snaps[max(0, i - PSI_REF_SNAPSHOTS):i]
        ref = ranked.loc[ranked["SNAPSHOT_DATE"].isin(ref_snaps), "RISK_SCORE"].to_numpy()
        mix = p.loc[p["SNAPSHOT_DATE"] == s, "CURRENT_CAT"].value_counts(normalize=True)
        rows.append({
            "snapshot": int(s),
            "n_rows": int((p["SNAPSHOT_DATE"] == s).sum()),
            "n_ranked": int(len(cur)),
            **{f"cat_{c}_share": float(mix.get(c, 0.0)) for c in range(config.NUM_CLASSES)},
            "score_mean": float(cur.mean()) if len(cur) else float("nan"),
            "score_p90": float(np.quantile(cur, 0.9)) if len(cur) else float("nan"),
            "score_p99": float(np.quantile(cur, 0.99)) if len(cur) else float("nan"),
            "score_psi": psi(ref, cur) if len(ref) and len(cur) else float("nan"),
        })
    return rows


# ── Orchestration ─────────────────────────────────────────────────────────────

def compute(preds: pd.DataFrame, outcomes: pd.DataFrame, asof: int) -> dict:
    df, quality = prepare(preds, outcomes, asof)
    matured, interim = {}, {}
    for s, cohort in df.groupby("SNAPSHOT_DATE"):
        s = int(s)
        if cohort["MATURED"].all():
            matured[s] = matured_metrics(cohort)
        else:
            if cohort["MATURED"].any():
                logger.warning(f"Snapshot {s} is partly mature — treated as interim.")
            interim[s] = interim_metrics(cohort, s, asof)
    return {"asof": asof, "generated_at": datetime.now().isoformat(timespec="seconds"),
            "matured": matured, "interim": interim, "drift": drift(preds), "quality": quality}


def history_frame(metrics: dict) -> pd.DataFrame:
    """One row per matured cohort — the headline numbers, Excel-friendly."""
    rows = []
    for s, m in sorted(metrics["matured"].items()):
        row = {"snapshot": s, "n_ranked": m["n_ranked"], "n_severe": m["n_severe"],
               "base_rate": m["base_rate"], "pr_auc": m.get("pr_auc"),
               "predicted_severe": m.get("predicted_severe")}
        cat0 = m.get("by_current_cat", {}).get("current_cat_0", {})
        row["cat0_pr_auc"], row["cat0_base_rate"] = cat0.get("pr_auc"), cat0.get("base_rate")
        for w in config.MONITOR_HEADLINE_WINDOWS:
            b = m.get(f"at_{w}", {})
            lo, hi = b.get("recall_ci", (None, None))
            row.update({f"recall_{w}": b.get("recall"), f"recall_{w}_lo": lo,
                        f"recall_{w}_hi": hi, f"precision_{w}": b.get("precision"),
                        f"lift_{w}": b.get("lift"),
                        f"exposure_caught_{w}": b.get("exposure_caught")})
        rows.append(row)
    return pd.DataFrame(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_monitor(asof: Optional[int] = None, output_dir: Optional[str] = None,
                conn=None) -> Path:
    """Load, compute, write metrics.json + history.csv + the HTML report. Returns the report path."""
    from src.monitoring.report import build_report

    asof = asof or int(date.today().strftime("%Y%m%d"))
    if conn is None:
        from src.db.mssql_connection import MSSQLConnector
        conn = MSSQLConnector()
    preds, outcomes = load_frames(conn)
    metrics = compute(preds, outcomes, asof)

    out = Path(output_dir or config.MONITOR_OUTPUT_DIR)
    run_dir = out / str(asof)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, default=_json_default, indent=1))
    history_frame(metrics).to_csv(out / "history.csv", index=False)
    report = build_report(metrics, run_dir)
    logger.info(f"Monitoring: {len(metrics['matured'])} matured, {len(metrics['interim'])} "
                f"interim cohorts -> {report}")
    return report
