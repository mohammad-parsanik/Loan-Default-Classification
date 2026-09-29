"""
Monthly monitoring: how did the predictions we actually made turn out?

The inference engine (EDP-inference-engine repo) records each monthly run in
OperationLog (one operation_id per run, keyed to a snapshot_date) and writes
one InferenceOutput row per scored loan (p0..p3, dpd_cat, dpd,
remain_of_account), plus a ContractTracking row saying whether its exception
rule sent the loan to enrichment. It scores LOAN_CATEGORY 0-2 only — exactly
the ranked population. A snapshot can hold several runs (--force); the newest
SUCCESS/PARTIAL_SUCCESS one counts, the same rule as the engine's
db/analysis/01_backtest_base.sql.

Outcomes come from the feed's WORST_FUTURE_CAT on the scored snapshot's own
row, and every cohort is recomputed from the DB on each run:

  - matured cohort  (LABEL_HORIZON_DATE <= asof): the label is final — how
    many severe loans the top of the list caught (MONITOR_CUTOFFS: a share of
    the list or a fixed count), exposure caught, calibration, migration, and
    how the engine's enrichment rule did.
  - interim cohort  (horizon not reached): the ETL refreshes WORST_FUTURE_CAT
    on the newest 7 snapshots every load, so it holds the worst category
    reached SO FAR — a lower bound on the final label. Reported only as
    "flagged loans deteriorated at N x the rate of the rest", never mixed
    into matured metrics.

Evaluation is per snapshot, ordered by p3 with LOAN_ID as tie-break. The
engine does not rank (its queue columns are deliberately not persisted), so
"the list" here is that ordering, cut at MONITOR_CUTOFFS.
"""

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import project_config as config
from src.evaluation.ranking import SEVERE_CLASS, _average_precision, _order, capture_curve

logger = logging.getLogger(__name__)

# Internal names, shared with the evaluation code; engine columns are mapped onto them.
PRED_COLS = ["SNAPSHOT_DATE", "LOAN_ID", "NATIONAL_CODE", "CURRENT_CAT", "RISK_SCORE",
             "REMAINING_AMNT", "FLAGGED"]
OUTCOME_COLS = ["SNAPSHOT_DATE", "LOAN_ID", "WORST_FUTURE_CAT", "LABEL_HORIZON_DATE"]
USABLE_RUN_STATUSES = ("SUCCESS", "PARTIAL_SUCCESS")
PSI_REF_SNAPSHOTS = 6


# ── Loading ───────────────────────────────────────────────────────────────────

def pick_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Newest usable run per snapshot, with how many usable runs the snapshot had."""
    r = runs[runs["operation_status"].isin(USABLE_RUN_STATUSES)
             & runs["snapshot_date"].notna()].copy()
    r["snapshot_date"] = r["snapshot_date"].astype(int)
    r["n_usable_runs"] = r.groupby("snapshot_date")["operation_id"].transform("size")
    return (r.sort_values(["snapshot_date", "operation_date"], ascending=[True, False])
             .drop_duplicates("snapshot_date").reset_index(drop=True))


def engine_to_preds(output: pd.DataFrame, snapshot: int) -> pd.DataFrame:
    """One run's InferenceOutput (+ ContractTracking status) -> internal prediction frame."""
    status = output["operation_status_enrichment"]
    return pd.DataFrame({
        "SNAPSHOT_DATE": int(snapshot),
        "LOAN_ID": output["loan_id"].astype("int64"),
        "NATIONAL_CODE": output["national_code"],
        # dpd_cat is the raw 0-4 feed band; the model's classes are 0-3
        "CURRENT_CAT": np.minimum(output["dpd_cat"].astype(int), config.NUM_CLASSES - 1),
        "RISK_SCORE": output["p3"].astype(float),
        "REMAINING_AMNT": output["remain_of_account"].astype(float),
        # PENDING_ENRICHMENT when written; later enrichment statuses overwrite it
        "FLAGGED": status.notna() & (status != "NOT_APPLICABLE"),
    })


def load_frames(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(predictions, outcomes, chosen runs) for every snapshot the engine has scored."""
    statuses = ", ".join(f"'{s}'" for s in USABLE_RUN_STATUSES)
    runs = pick_runs(conn.read_sql(
        "SELECT operation_id, snapshot_date, operation_status, operation_date, "
        f"inference_failures FROM {config.INFERENCE_OPLOG_TABLE} "
        f"WHERE operation_status IN ({statuses}) AND snapshot_date IS NOT NULL"))
    if runs.empty:
        raise ValueError(f"No {'/'.join(USABLE_RUN_STATUSES)} run in {config.INFERENCE_OPLOG_TABLE}.")

    preds, outcomes = [], []
    for run in runs.itertuples():
        out = conn.read_sql(
            "SELECT o.loan_id, o.national_code, o.p3, o.dpd_cat, o.remain_of_account, "
            "t.operation_status_enrichment "
            f"FROM {config.INFERENCE_OUTPUT_TABLE} o "
            f"LEFT JOIN {config.INFERENCE_TRACKING_TABLE} t "
            "ON t.operation_id = o.operation_id AND t.loan_id = o.loan_id "
            "WHERE o.operation_id = ?", (run.operation_id,))
        preds.append(engine_to_preds(out, run.snapshot_date))
        outcomes.append(conn.read_sql(
            f"SELECT {', '.join(OUTCOME_COLS)} FROM {config.TRAIN_TABLE} WHERE SNAPSHOT_DATE = ?",
            (int(run.snapshot_date),)))
    return pd.concat(preds, ignore_index=True), pd.concat(outcomes, ignore_index=True), runs


def prepare(preds: pd.DataFrame, outcomes: pd.DataFrame, asof: int) -> tuple[pd.DataFrame, dict]:
    """Join predictions to outcomes. Returns (frame, data-quality dict)."""
    key = ["SNAPSHOT_DATE", "LOAN_ID"]
    o = outcomes.copy()
    o["SNAPSHOT_DATE"] = o["SNAPSHOT_DATE"].astype(float).astype(int)
    o["LOAN_ID"] = o["LOAN_ID"].astype("int64")
    df = preds[PRED_COLS].merge(o, on=key, how="left", validate="one_to_one")

    unmatched = df["WORST_FUTURE_CAT"].isna()
    quality = {
        "n_prediction_rows": int(len(preds)),
        "unmatched_by_snapshot": {
            int(s): int(n) for s, n in df[unmatched].groupby("SNAPSHOT_DATE").size().items()
        },
    }
    if unmatched.any():
        logger.warning(f"{int(unmatched.sum()):,} scored loans have no row in "
                       f"{config.TRAIN_TABLE} — excluded from outcome metrics.")
    df = df[~unmatched].copy()
    df["OUTCOME"] = np.minimum(df["WORST_FUTURE_CAT"].astype(int), config.NUM_CLASSES - 1)
    df["CURRENT_CAT"] = df["CURRENT_CAT"].astype(int)
    df["MATURED"] = df["LABEL_HORIZON_DATE"].astype(float).astype(int) <= asof
    return df, quality


# ── Label archive ─────────────────────────────────────────────────────────────

def reconcile_labels(outcomes: pd.DataFrame, archive_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Guard the outcome against upstream deletion.

    Every month the ETL rebuilds the newest snapshots, and a rebuilt label can
    only have seen MORE installments than the last one, so within a snapshot a
    loan's WORST_FUTURE_CAT never legitimately goes down. It goes down, or the
    loan vanishes, when its installments were deleted between builds (§24) —
    and that happens to exactly the loans that went NPL. So keep, per snapshot,
    the highest label ever read for each loan, use that, and count every
    lowered label and vanished loan so the report shows whether it happened.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    frames, stats = [], {}
    for snap, cur in outcomes.groupby("SNAPSHOT_DATE"):
        snap = int(snap)
        cur = cur.assign(LOAN_ID=cur["LOAN_ID"].astype("int64"),
                         WORST_FUTURE_CAT=cur["WORST_FUTURE_CAT"].astype(int),
                         LABEL_HORIZON_DATE=cur["LABEL_HORIZON_DATE"].astype(float).astype(int))
        path = archive_dir / f"{snap}.npz"
        st = {"n_label_lowered": 0, "n_vanished": 0, "first_read": not path.exists()}
        if path.exists():
            with np.load(path) as z:
                prev = pd.DataFrame({"LOAN_ID": z["loan_id"], "PREV": z["label"],
                                     "PREV_HORIZON": z["horizon"]})
            m = cur.merge(prev, on="LOAN_ID", how="outer", indicator=True)
            both = m["_merge"] == "both"
            gone = m["_merge"] == "right_only"
            st["n_label_lowered"] = int((both & (m["WORST_FUTURE_CAT"] < m["PREV"])).sum())
            st["n_vanished"] = int(gone.sum())
            m["WORST_FUTURE_CAT"] = np.fmax(m["WORST_FUTURE_CAT"], m["PREV"]).astype(int)
            m.loc[gone, "LABEL_HORIZON_DATE"] = m.loc[gone, "PREV_HORIZON"]
            m["SNAPSHOT_DATE"] = snap
            cur = m[OUTCOME_COLS].astype({"LABEL_HORIZON_DATE": int})
        np.savez(path, loan_id=cur["LOAN_ID"].to_numpy("int64"),
                 label=cur["WORST_FUTURE_CAT"].to_numpy("int8"),
                 horizon=cur["LABEL_HORIZON_DATE"].to_numpy("int64"))
        if st["n_label_lowered"] or st["n_vanished"]:
            logger.warning(f"Snapshot {snap}: {st['n_label_lowered']:,} labels lower than last read, "
                           f"{st['n_vanished']:,} loans gone from the feed — kept the archived values.")
        stats[snap] = st
        frames.append(cur)
    return pd.concat(frames, ignore_index=True), stats


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
    """Ranked population in list order, with a 1-based QUEUE_POS."""
    r = cohort[cohort["CURRENT_CAT"] < config.CARVE_CURRENT_CAT_GE]
    r = r.iloc[_order(r["RISK_SCORE"].to_numpy(), r["LOAN_ID"].to_numpy())].copy()
    r["QUEUE_POS"] = np.arange(1, len(r) + 1)
    return r


def cutoffs(n: int) -> dict[str, int]:
    """MONITOR_CUTOFFS resolved to row counts for a list of n loans (float = share, int = count)."""
    return {name: min(n, math.ceil(v * n) if isinstance(v, float) else int(v))
            for name, v in config.MONITOR_CUTOFFS.items()}


def cutoff_label(name: str) -> str:
    v = config.MONITOR_CUTOFFS[name]
    return f"top {100 * v:g}%" if isinstance(v, float) else f"top {v:,}"


def _ap(severe: pd.Series, scores: pd.Series) -> float:
    return _average_precision(severe.to_numpy(np.int32), scores.to_numpy()) if severe.any() else float("nan")


def matured_metrics(cohort: pd.DataFrame) -> dict:
    r = _ranked(cohort)
    severe = r["OUTCOME"] == SEVERE_CLASS
    n_sev = int(severe.sum())
    out = {"n_ranked": int(len(r)), "n_severe": n_sev,
           "base_rate": float(n_sev / len(r)) if len(r) else float("nan")}
    if not n_sev:
        return out
    out["pr_auc"] = _ap(severe, r["RISK_SCORE"])

    sev_amt = r.loc[severe, "REMAINING_AMNT"].sum()
    ks = cutoffs(len(r))
    out["cutoffs"] = {}
    for name, k in ks.items():
        top = r["QUEUE_POS"] <= k
        hits = int((top & severe).sum())
        out["cutoffs"][name] = {
            "k": k, "recall": hits / n_sev, "recall_ci": wilson(hits, n_sev),
            "precision": hits / k, "lift": hits / k / out["base_rate"],
            "exposure_caught": (float(r.loc[top & severe, "REMAINING_AMNT"].sum() / sev_amt)
                                if sev_amt > 0 else float("nan")),
        }

    # Per current_cat, inside the ONE pooled list — not re-ranked per stratum.
    out["queue_by_cat"] = {}
    for cat, sub in r.groupby("CURRENT_CAT"):
        sev = sub["OUTCOME"] == SEVERE_CLASS
        row = {"n": int(len(sub)), "n_severe": int(sev.sum()), "base_rate": float(sev.mean()),
               "pr_auc": _ap(sev, sub["RISK_SCORE"])}
        for name, k in ks.items():
            top = sub["QUEUE_POS"] <= k
            row[f"share_of_{name}"] = float(top.sum() / k)
            row[f"recall_{name}"] = float((top & sev).sum() / sev.sum()) if sev.any() else float("nan")
        out["queue_by_cat"][int(cat)] = row

    out["predicted_severe"] = float(r["RISK_SCORE"].sum())
    # Rank-based deciles: isotonic scores tie in blocks, so qcut on the score
    # itself would collapse bins.
    dec = pd.qcut(r["QUEUE_POS"], 10, labels=False)
    cal = r.groupby(dec).agg(mean_score=("RISK_SCORE", "mean"),
                             severe_rate=("OUTCOME", lambda s: (s == SEVERE_CLASS).mean()),
                             n=("OUTCOME", "size"))
    out["calibration_deciles"] = cal.reset_index(drop=True).to_dict("list")

    r["GROUP"] = np.where(r["QUEUE_POS"] <= ks[config.MONITOR_MAIN_CUTOFF], "top", "rest")
    out["migration"] = {
        g: pd.crosstab(sub["CURRENT_CAT"], sub["OUTCOME"])
             .reindex(columns=range(config.NUM_CLASSES), fill_value=0)
             .to_dict("index")
        for g, sub in r.groupby("GROUP")
    }
    out["capture_curve"] = capture_curve(r["OUTCOME"].to_numpy(), r["RISK_SCORE"].to_numpy(),
                                         tie_break=r["LOAN_ID"].to_numpy())

    # The engine's exception rule (remain > 50M, p3 > 0.5, dpd > 120) is what
    # actually goes to enrichment. It can only reach cat-2 loans at DPD 121-155,
    # so its hit rate is largely mechanical accrual; the same-size column says
    # what taking that many loans from the top of the list would have caught.
    fl = r["FLAGGED"].to_numpy(bool)
    n_fl = int(fl.sum())
    out["enrichment_rule"] = {
        "n_flagged": n_fl,
        "n_severe_flagged": int((fl & severe).sum()),
        "precision": float(severe[fl].mean()) if n_fl else float("nan"),
        "recall": float((fl & severe).sum() / n_sev),
        "recall_top_same_size": float(((r["QUEUE_POS"] <= n_fl) & severe).sum() / n_sev),
    }
    return out


def interim_metrics(cohort: pd.DataFrame, snapshot: int, asof: int) -> dict:
    r = _ranked(cohort)
    worse = r["OUTCOME"] > r["CURRENT_CAT"]
    severe = r["OUTCOME"] == SEVERE_CLASS
    months = (pd.Timestamp(str(asof)) - pd.Timestamp(str(snapshot))).days / 30.44
    out = {"months_elapsed": round(months, 1), "n_ranked": int(len(r)),
           "deteriorated_rate": float(worse.mean()) if len(r) else float("nan"),
           "severe_so_far_rate": float(severe.mean()) if len(r) else float("nan")}

    def rates(mask):
        return {"n": int(mask.sum()),
                "deteriorated_rate": float(worse[mask].mean()) if mask.any() else float("nan"),
                "severe_so_far_rate": float(severe[mask].mean()) if mask.any() else float("nan")}

    ks = cutoffs(len(r))
    out["rest"] = rates(r["QUEUE_POS"] > ks[config.MONITOR_MAIN_CUTOFF])
    rest_rate = out["rest"]["deteriorated_rate"]
    out["cutoffs"] = {}
    for name, k in ks.items():
        g = rates(r["QUEUE_POS"] <= k)
        g["lift_vs_rest"] = g["deteriorated_rate"] / rest_rate if rest_rate else float("nan")
        out["cutoffs"][name] = g
    out["enrichment_rule"] = rates(r["FLAGGED"].to_numpy(bool))
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

def compute(preds: pd.DataFrame, outcomes: pd.DataFrame, asof: int,
            runs: Optional[pd.DataFrame] = None, label_stats: Optional[dict] = None) -> dict:
    df, quality = prepare(preds, outcomes, asof)
    if label_stats is not None:
        quality["labels"] = label_stats
    if runs is not None:
        quality["runs"] = [
            {"snapshot": int(r.snapshot_date), "operation_id": str(r.operation_id),
             "status": r.operation_status, "operation_date": str(r.operation_date),
             "inference_failures": int(r.inference_failures), "n_usable_runs": int(r.n_usable_runs)}
            for r in runs.itertuples()]
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
               "cat0_pr_auc": m.get("queue_by_cat", {}).get(0, {}).get("pr_auc"),
               "predicted_severe": m.get("predicted_severe")}
        for name, b in m.get("cutoffs", {}).items():
            lo, hi = b["recall_ci"]
            row.update({f"{name}_k": b["k"], f"{name}_recall": b["recall"],
                        f"{name}_recall_lo": lo, f"{name}_recall_hi": hi,
                        f"{name}_precision": b["precision"], f"{name}_lift": b["lift"],
                        f"{name}_exposure_caught": b["exposure_caught"]})
        rule = m.get("enrichment_rule", {})
        row.update({"rule_n_flagged": rule.get("n_flagged"), "rule_precision": rule.get("precision"),
                    "rule_recall": rule.get("recall"),
                    "rule_recall_top_same_size": rule.get("recall_top_same_size")})
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
    out = Path(output_dir or config.MONITOR_OUTPUT_DIR)
    preds, outcomes, runs = load_frames(conn)
    outcomes, label_stats = reconcile_labels(outcomes, out / "labels")
    metrics = compute(preds, outcomes, asof, runs, label_stats)

    run_dir = out / str(asof)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, default=_json_default, indent=1))
    history_frame(metrics).to_csv(out / "history.csv", index=False)
    report = build_report(metrics, run_dir)
    logger.info(f"Monitoring: {len(metrics['matured'])} matured, {len(metrics['interim'])} "
                f"interim cohorts -> {report}")
    return report
