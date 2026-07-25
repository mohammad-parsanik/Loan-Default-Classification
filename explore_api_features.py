"""
explore_api_features.py
=======================
Standalone explorer for the enrichment-API responses we pulled for a spread of
customers (~2,400 NATIONAL_CODEs sampled across risk levels). The API returns
data from THREE endpoints; the sample was joined against the model predictions
and saved as a single long-format parquet where each row is one
(NATIONAL_CODE, endpoint) response and carries the customer's CURRENT_CAT.

Goal: for every feature returned by every endpoint, see how its value
distribution differs across the customer's CURRENT class (0 No Delay .. 3 Severe)
so we can hand-write threshold rules. We split by the *current* category, not the
predicted one.

The module is schema-agnostic — it discovers the endpoints, the feature columns
of each, and their types (numeric vs categorical) from the data itself. Nothing
about the specific API fields is hard-coded, so it keeps working as the API
schema changes.

Some fields are lists of records (variable count per customer — e.g. scoreCodes,
bounced cheques, external loan-registry rows). Each such column is automatically
exploded two ways and fed through the SAME pipeline as a real endpoint, as a
pseudo-endpoint named "<endpoint>::<column>[item]" (one row per list item — do
specific codes/values concentrate in one class) and "...[agg]" (one row per
customer — count of items + sum/max of amount-like subfields — does HAVING
several/large flagged items correlate with class). A paired code+description
field (like scoreCodes) is additionally split into its own reference CSV instead
of being plotted as a redundant many-level categorical. Columns that are
themselves personal identifiers (e.g. a phone-number list) are never exploded.

Outputs (in --output_dir, default `api_exploration/`):
  schema.csv                        -- discovered endpoints x features x type x null-rate
                                        (includes the exploded JSON pseudo-endpoints)
  summary_stats.csv                 -- per feature x CURRENT_CAT: count/mean/median/quantiles/nulls
  separation.csv                    -- per feature: how well it separates severe from the rest (AUC / Cramer's V)
  threshold_suggestions.csv         -- per numeric feature: best severe-vs-rest cut (threshold, dir, precision/recall/F1/lift)
  category_rates.csv                -- per categorical feature x level: severe-rate and support
  <endpoint>_<column>_code_reference.csv -- code -> description mapping for paired code/description JSON fields
  plots/<endpoint or pseudo-endpoint>/<feature>.png  -- distribution split by CURRENT_CAT (+ box-by-class panel)

Usage (run on the server, where the parquet lives):
  # 1. sanity-check what the module sees before generating everything:
  python explore_api_features.py --data api_joined.parquet --inspect

  # 2. full run (tables + plots):
  python explore_api_features.py --data api_joined.parquet

  # narrow to one endpoint / a few features while iterating:
  python explore_api_features.py --data api_joined.parquet --endpoints credit_bureau --max_features 20

  # verify the module runs end-to-end on synthetic data (no server needed):
  python explore_api_features.py --self_check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("explore_api")

# Mirror the project's class vocabulary (see explore_umap.py / project_config).
CLASS_NAMES = ["No Delay", "Current", "Past Due+", "Severe Past Due"]
CLASS_COLORS = ["#4C9BE8", "#F0A500", "#E84C4C", "#9C27B0"]

# Columns that are model/meta output, never API features. Matched case-insensitively.
META_COLUMNS = {
    "national_code", "snapshot_date", "n_loans_in_portfolio", "current_cat",
    "endpoint", "risk_rank", "risk_score", "expected_cost", "predicted_class",
    "rule_flag", "risk_bin", "p_no_delay", "p_current", "p_past_due",
    "p_severe_past_due", "worst_future_cat", "worst_future_dpd", "dpd_days",
    "called_at",
}

# Column-name substrings (case-insensitive) that mark direct personal identifiers.
# These are excluded from stats/plots regardless of cardinality — a common surname
# or shared address prefix could survive a cardinality guard but still identify
# someone. Generic on purpose so it travels to other endpoints/APIs; extend via
# --pii_substrings if a schema has identifiers this list doesn't catch.
PII_SUBSTRINGS = [
    "name", "family", "address", "dateofbirth", "birthdate",
    "phone", "mobile", "email", "nationalid", "nationalcode", "iban", "contact",
]


# ── Column resolution ─────────────────────────────────────────────────────────

def _resolve(df: pd.DataFrame, wanted: str, aliases: list[str], required: bool) -> str | None:
    """Find a column by exact name, else case-insensitively among aliases."""
    if wanted in df.columns:
        return wanted
    lower = {c.lower(): c for c in df.columns}
    for a in [wanted, *aliases]:
        if a.lower() in lower:
            return lower[a.lower()]
    if required:
        raise SystemExit(
            f"Could not find the '{wanted}' column. Tried {[wanted, *aliases]}. "
            f"Available columns: {list(df.columns)}\n"
            f"Pass it explicitly, e.g. --class_col <name>."
        )
    return None


# ── Type inference ────────────────────────────────────────────────────────────
#
# Only "numeric" and "categorical" get stats/plots. Everything else is still
# listed in schema.csv (so nothing silently vanishes) but skipped, with a type
# name that says why:
#   pii_excluded  — column name matches a personal-identifier pattern
#   json_skipped  — value is a list/dict (variable-length nested JSON)
#   constant      — a single distinct value; zero separation signal possible
#   id_like       — too many distinct values to be a feature (reference/date ids)
#   sparse        — too few non-null values to compare across classes
#   empty         — endpoint returned this column but every value is null here

def _is_pii(col: str, pii_substrings: list[str]) -> bool:
    name_l = col.lower()
    return any(p in name_l for p in pii_substrings)


def infer_type(
    s: pd.Series, max_cat_levels: int, max_id_cardinality: int, min_n_present: int,
    min_avg_per_level: float = 5.0,
) -> str:
    """Auto-detect type from values alone (PII/override handled by the caller)."""
    s = s.dropna()
    if s.empty:
        return "empty"
    if s.dtype == object:
        # Object dtype here means "not a uniform scalar type" — could be True/False
        # with gaps, or could be numpy arrays of dicts (nested JSON). Only the
        # latter is unhashable / unanalyzable, so check the actual value shape.
        if isinstance(s.iloc[0], (list, dict, np.ndarray)):
            return "json_skipped"
    if len(s) < min_n_present:
        return "sparse"
    try:
        nun = s.nunique()
    except TypeError:
        return "json_skipped"  # unhashable values that slipped past the sample check
    if nun <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        integer_like = np.allclose(s.to_numpy(dtype=float) % 1, 0)
        if nun <= max_cat_levels and integer_like:
            return "categorical"
        return "numeric"
    # Absolute cap AND the standard chi-square rule-of-thumb (avg >=5 obs/level):
    # a raised max_id_cardinality (e.g. for item-level pseudo-endpoints, to admit
    # a 68-code vocabulary) can otherwise also admit small-sample identifier-like
    # columns — e.g. 70 bounced-cheque records with 90+ distinct branch codes,
    # ~3 items each — where Cramer's V/chi2 are unreliable (most levels seen
    # only once or twice, so "only one class observed" is chance, not signal)
    # even though the raw distinct-value count is under the cap.
    if nun <= max_id_cardinality and len(s) / nun >= min_avg_per_level:
        return "categorical"
    return "id_like"


def discover(
    df: pd.DataFrame, endpoint_col: str, exclude: set[str], max_cat_levels: int,
    max_id_cardinality: int, min_n_present: int, pii_substrings: list[str],
    numeric_override: set[str], categorical_override: set[str],
) -> pd.DataFrame:
    """Per (endpoint, feature): inferred type, null-rate, n-unique — the schema."""
    rows = []
    for ep, g in df.groupby(endpoint_col, observed=True):
        for col in df.columns:
            if col.lower() in exclude or col == endpoint_col:
                continue
            s = g[col]
            if s.notna().sum() == 0:  # feature not returned by this endpoint
                continue
            if col in numeric_override:
                t = "numeric"
            elif col in categorical_override:
                t = "categorical"
            elif _is_pii(col, pii_substrings):
                t = "pii_excluded"
            else:
                t = infer_type(s, max_cat_levels, max_id_cardinality, min_n_present)
            try:
                n_unique = int(s.nunique())
            except TypeError:
                n_unique = -1  # unhashable (json) values — cardinality not defined
            rows.append({
                "endpoint": ep, "feature": col, "type": t,
                "n_present": int(s.notna().sum()),
                "null_rate": round(float(s.isna().mean()), 4),
                "n_unique": n_unique,
            })
    return pd.DataFrame(rows).sort_values(["endpoint", "type", "feature"]).reset_index(drop=True)


# ── Per-class statistics ──────────────────────────────────────────────────────

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def summarize_numeric(df: pd.DataFrame, feature: str, class_col: str) -> pd.DataFrame:
    rows = []
    for cat, g in df.groupby(class_col, observed=True):
        s = pd.to_numeric(g[feature], errors="coerce").dropna()
        rec = {
            "feature": feature, "current_cat": int(cat),
            "class_name": CLASS_NAMES[int(cat)] if int(cat) < len(CLASS_NAMES) else str(cat),
            "n": int(len(s)), "null_rate": round(float(g[feature].isna().mean()), 4),
            "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
        }
        qs = s.quantile(QUANTILES) if len(s) else pd.Series({q: np.nan for q in QUANTILES})
        for q in QUANTILES:
            rec[f"q{int(q * 100):02d}"] = qs.loc[q]
        rows.append(rec)
    return pd.DataFrame(rows)


def separation_numeric(df: pd.DataFrame, feature: str, class_col: str, severe_ge: int) -> dict:
    """ROC-AUC of the raw feature vs positive=(current_cat >= severe_ge)."""
    from sklearn.metrics import roc_auc_score

    s = pd.to_numeric(df[feature], errors="coerce")
    y = (df[class_col] >= severe_ge).astype(int)
    mask = s.notna()
    s, y = s[mask], y[mask]
    if y.nunique() < 2 or len(s) < 10:
        return {"feature": feature, "type": "numeric", "metric": "auc", "value": np.nan, "direction": ""}
    auc = roc_auc_score(y, s)
    # auc<0.5 => the feature separates in the inverse direction; report strength >=0.5.
    direction = "high=severe" if auc >= 0.5 else "low=severe"
    return {"feature": feature, "type": "numeric", "metric": "auc",
            "value": round(float(max(auc, 1 - auc)), 4), "direction": direction}


def suggest_threshold(df: pd.DataFrame, feature: str, class_col: str, severe_ge: int) -> dict | None:
    """Scan candidate cuts; return the one best separating severe from the rest (max F1)."""
    s = pd.to_numeric(df[feature], errors="coerce")
    y = (df[class_col] >= severe_ge).astype(int)
    mask = s.notna()
    s, y = s[mask].to_numpy(), y[mask].to_numpy()
    if y.sum() == 0 or y.sum() == len(y) or len(s) < 10:
        return None
    base_rate = y.mean()
    cands = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 19)))
    best = None
    for t in cands:
        for direction, pred in (("high=severe", s >= t), ("low=severe", s <= t)):
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if best is None or f1 > best["f1"]:
                best = {"feature": feature, "direction": direction, "threshold": float(t),
                        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                        "lift": round(prec / base_rate, 3) if base_rate else np.nan,
                        "n_flagged": int(pred.sum()), "base_rate": round(float(base_rate), 4)}
    return best


def category_rates(df: pd.DataFrame, feature: str, class_col: str, severe_ge: int) -> pd.DataFrame:
    """Per level: severe-rate, support, lift over base — candidate categorical rules."""
    y = (df[class_col] >= severe_ge).astype(int)
    base = y.mean()
    rows = []
    for level, idx in df.groupby(feature, observed=True).groups.items():
        yy = y.loc[idx]
        rate = float(yy.mean())
        rows.append({"feature": feature, "level": level, "n": int(len(yy)),
                     "severe_rate": round(rate, 4), "base_rate": round(float(base), 4),
                     "lift": round(rate / base, 3) if base else np.nan})
    return pd.DataFrame(rows).sort_values("severe_rate", ascending=False)


def separation_categorical(df: pd.DataFrame, feature: str, class_col: str, severe_ge: int) -> dict:
    """Cramer's V between the feature and severe/not-severe."""
    from scipy.stats import chi2_contingency

    y = (df[class_col] >= severe_ge).astype(int)
    tab = pd.crosstab(df[feature], y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"feature": feature, "type": "categorical", "metric": "cramers_v", "value": np.nan, "direction": ""}
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return {"feature": feature, "type": "categorical", "metric": "cramers_v",
            "value": round(float(v), 4), "direction": ""}


# ── JSON list-column exploration ──────────────────────────────────────────────
#
# scoreCodes / cheques.bouncedChequeCustomerModel / estelamAsliRows.estelamAsliRow
# are lists of records (0..N per customer) — the "json_skipped" columns. Rather
# than hand-write per-column logic, every such column (except ones that are
# themselves PII, e.g. personInformation.contacts = phone numbers) is expanded
# two ways and fed through the SAME stats/plot pipeline as a real endpoint:
#   [item]  one row per list item (e.g. one row per bounced cheque) — answers
#           "which codes/values appear, and do they concentrate in one class".
#   [agg]   one row per customer — count of items + sum/max of amount-like
#           subfields (name starts with "am" or contains "amount", matching
#           this API's own naming convention) — answers "does HAVING several
#           /large flagged items correlate with class".

def _avro_unwrap(v):
    """Nullable/union fields arrive as {'int': array([x])} — unwrap to a scalar.

    A field is (almost) always single-valued but rarely carries 2+ codes in one
    record. Multi-value rows are joined into a "402|403" string rather than kept
    as a tuple — mixing bare ints and tuples in the same column crashes pandas'
    crosstab/groupby sort (unorderable types). These wrapper fields are codes,
    not continuous numbers, so always-string loses nothing semantically.
    """
    if isinstance(v, dict) and len(v) == 1:
        inner = next(iter(v.values()))
        if isinstance(inner, np.ndarray):
            return str(inner[0]) if inner.size == 1 else "|".join(map(str, inner.tolist()))
        return inner
    return v


def _is_amount_like(key: str) -> bool:
    kl = key.lower()
    return kl.startswith("am") or "amount" in kl


def explode_list_column(df: pd.DataFrame, col: str, id_col: str | None, class_col: str) -> pd.DataFrame:
    """One row per list item; item dict keys become columns; unions unwrapped."""
    keep = [c for c in (id_col, class_col) if c]
    records = []
    for _, row in df.loc[df[col].notna()].iterrows():
        items = row[col]
        if not isinstance(items, (list, np.ndarray)):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = {c: row[c] for c in keep}
            rec.update({k: _avro_unwrap(v) for k, v in item.items()})
            records.append(rec)
    return pd.DataFrame.from_records(records)


def aggregate_list_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-customer derived features: item count, sum/max of amount-like
    subfields, and max-value-seen for every OTHER numeric/coded subfield too
    (e.g. a per-loan status code) — "the worst value across this customer's
    records". A status/severity code is rarely named like an amount, so without
    this a genuinely strong per-loan signal (e.g. status 4/5 = written-off/
    default) would only ever surface at the per-loan-item level, never rolled up
    to the customer level where the actual decision gets made. Whether "higher
    is worse" is left for separation/threshold analysis downstream to judge —
    junk candidates (e.g. a max'd branch-code identifier) fall out naturally via
    the same id_like/cardinality checks applied everywhere else in the pipeline.
    """
    lists = df[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [])
    n_items = lists.apply(len)
    out = pd.DataFrame({"n_items": n_items, "has_items": n_items > 0}, index=df.index)

    keys = set()
    for items in lists:
        for item in items:
            if isinstance(item, dict):
                keys.update(item.keys())

    def _numeric_values(field):
        def _values(items, f=field):
            vals = []
            for it in items:
                if isinstance(it, dict) and it.get(f) is not None:
                    v = _avro_unwrap(it[f])
                    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                        vals.append(float(v))
            return vals
        return lists.apply(_values)

    for field in sorted(k for k in keys if _is_amount_like(k)):
        per_row = _numeric_values(field)
        out[f"sum_{field}"] = per_row.apply(lambda v: sum(v) if v else np.nan)
        out[f"max_{field}"] = per_row.apply(lambda v: max(v) if v else np.nan)

    for field in sorted(k for k in keys if not _is_amount_like(k) and "date" not in k.lower()):
        per_row = _numeric_values(field)
        if per_row.apply(len).sum() == 0:
            continue  # never numeric for this field (text/id/etc) — nothing to aggregate
        out[f"max_{field}"] = per_row.apply(lambda v: max(v) if v else np.nan)
    return out


def _split_code_reference(exploded: pd.DataFrame, item_schema: pd.DataFrame, out: Path, label: str) -> pd.DataFrame:
    """If the exploded records pair a 'code' with a free-text 'description' (e.g.
    scoreCodes), write the unique code->description mapping to its own CSV and
    retype 'description' out of the analyzable set — otherwise it's plotted as a
    redundant, unreadable N-level categorical duplicating what 'code' already shows."""
    cols_l = {c.lower(): c for c in exploded.columns}
    code_col, desc_col = cols_l.get("code"), cols_l.get("description")
    if not code_col or not desc_col:
        return item_schema
    ref = (
        exploded[[code_col, desc_col]].dropna().drop_duplicates(subset=[code_col])
        .sort_values(code_col).reset_index(drop=True)
    )
    _write(out / f"{_safe(label)}_code_reference.csv", ref)
    item_schema = item_schema.copy()
    item_schema.loc[item_schema["feature"] == desc_col, "type"] = "reference"
    return item_schema


def prepare_json_extensions(
    df: pd.DataFrame, endpoint_col: str, class_col: str, id_col: str | None,
    schema: pd.DataFrame, pii_substrings: list[str], out: Path, args,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Explode+aggregate every eligible json_skipped column; each becomes a
    pseudo-endpoint that the normal discover()/stats loop can process unchanged."""
    extensions, extra_schema = [], []
    for _, srow in schema[schema["type"] == "json_skipped"].iterrows():
        col, owning_ep = srow["feature"], srow["endpoint"]
        if _is_pii(col, pii_substrings):
            log.info("JSON column '%s' skipped — name matches a PII pattern.", col)
            continue
        ep_df = df[df[endpoint_col] == owning_ep]

        exploded = explode_list_column(ep_df, col, id_col, class_col).drop_duplicates()
        if not exploded.empty:
            label = f"{owning_ep}::{col}[item]"
            exploded["endpoint"] = label
            exclude = {class_col.lower()} | ({id_col.lower()} if id_col else set())
            item_schema = discover(
                exploded, "endpoint", exclude, args.max_cat_levels,
                max(args.max_id_cardinality, 100),  # code vocabularies run higher than typical categoricals
                args.min_n_present, pii_substrings, set(), set(),
            )
            item_schema = _split_code_reference(exploded, item_schema, out, label)
            extra_schema.append(item_schema)
            extensions.append({"label": label, "df": exploded, "schema": item_schema})

        agg = aggregate_list_column(ep_df, col)
        if not agg.empty and agg.drop(columns=["has_items"]).notna().any().any():
            label = f"{owning_ep}::{col}[agg]"
            merged = pd.concat([ep_df[[class_col]].reset_index(drop=True), agg.reset_index(drop=True)], axis=1)
            agg_schema = discover(
                merged.assign(endpoint=label), "endpoint", {class_col.lower()},
                args.max_cat_levels, args.max_id_cardinality, args.min_n_present,
                pii_substrings, set(), set(),
            )
            merged = merged.rename(columns={c: f"{col}.{c}" for c in agg.columns})
            agg_schema["feature"] = col + "." + agg_schema["feature"]
            merged["endpoint"] = label
            extra_schema.append(agg_schema)
            extensions.append({"label": label, "df": merged, "schema": agg_schema})
    return extensions, extra_schema


# ── Plots ─────────────────────────────────────────────────────────────────────

def _class_color(cat: int) -> str:
    return CLASS_COLORS[cat] if cat < len(CLASS_COLORS) else "#777777"


def plot_numeric(df: pd.DataFrame, feature: str, class_col: str, out: Path) -> None:
    s_all = pd.to_numeric(df[feature], errors="coerce")
    finite = s_all.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    # Clip the display range to [1,99] pct so a few outliers don't flatten the plot
    # (mirrors the project's preprocessing clip). Shared bins across classes.
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        lo, hi = finite.min(), finite.max()
    if lo == hi:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 40)

    cats = sorted(int(c) for c in df[class_col].dropna().unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    box_data, box_labels, box_colors = [], [], []
    for cat in cats:
        s = pd.to_numeric(df.loc[df[class_col] == cat, feature], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        name = CLASS_NAMES[cat] if cat < len(CLASS_NAMES) else str(cat)
        ax1.hist(s.clip(lo, hi), bins=bins, density=True, histtype="step",
                 linewidth=1.8, color=_class_color(cat), label=f"{cat} {name} (n={len(s)})")
        box_data.append(s.to_numpy())
        box_labels.append(f"{cat}")
        box_colors.append(_class_color(cat))

    ax1.set_title(f"{feature} — density by CURRENT_CAT")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("density")
    ax1.legend(fontsize=8)

    bp = ax2.boxplot(box_data, patch_artist=True, showfliers=False, tick_labels=box_labels)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax2.set_title(f"{feature} — by class (fliers hidden)")
    ax2.set_xlabel("CURRENT_CAT")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_categorical(df: pd.DataFrame, feature: str, class_col: str, out: Path, max_levels: int = 20) -> None:
    top = df[feature].value_counts().head(max_levels).index
    sub = df[df[feature].isin(top)]
    # Normalized composition: within each class, what fraction falls in each level.
    ct = pd.crosstab(sub[class_col], sub[feature], normalize="index")
    # .reindex (not list-indexing) — a plain [True, False] column list is
    # otherwise misread by pandas as a boolean row/column mask, not labels.
    ct = ct.reindex(columns=[c for c in top if c in ct.columns])
    cats = sorted(int(c) for c in ct.index)
    fig, ax = plt.subplots(figsize=(max(7, len(top) * 0.7), 4.4))
    x = np.arange(len(ct.columns))
    w = 0.8 / max(len(cats), 1)
    for i, cat in enumerate(cats):
        if cat not in ct.index:
            continue
        name = CLASS_NAMES[cat] if cat < len(CLASS_NAMES) else str(cat)
        ax.bar(x + i * w, ct.loc[cat].to_numpy(), width=w, color=_class_color(cat), label=f"{cat} {name}")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([str(c) for c in ct.columns], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction within class")
    ax.set_title(f"{feature} — composition by CURRENT_CAT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def process_features(
    ep_df: pd.DataFrame, ep_schema: pd.DataFrame, ep_label: str, class_col: str,
    severe_ge: int, plot_dir: Path | None, max_features: int | None,
) -> tuple[list, list, list, list, int]:
    """Stats/separation/thresholds/plots for one (pseudo-)endpoint's analyzable
    features. Shared by real endpoints and by exploded/aggregated JSON columns."""
    stats_all, sep_all, thr_all, catrate_all = [], [], [], []
    analyzable = ep_schema[ep_schema["type"].isin(["numeric", "categorical"])]
    if max_features:
        analyzable = analyzable.head(max_features)
    n_done = 0
    for _, srow in analyzable.iterrows():
        feature, ftype = srow["feature"], srow["type"]
        if ftype == "numeric":
            stats_all.append(summarize_numeric(ep_df, feature, class_col))
            sep_all.append(separation_numeric(ep_df, feature, class_col, severe_ge))
            thr = suggest_threshold(ep_df, feature, class_col, severe_ge)
            if thr:
                thr["endpoint"] = ep_label
                thr_all.append(thr)
            if plot_dir is not None:
                plot_numeric(ep_df, feature, class_col, plot_dir / f"{_safe(feature)}.png")
        elif ftype == "categorical":
            sep_all.append(separation_categorical(ep_df, feature, class_col, severe_ge))
            cr = category_rates(ep_df, feature, class_col, severe_ge)
            cr.insert(0, "endpoint", ep_label)
            catrate_all.append(cr)
            if plot_dir is not None:
                plot_categorical(ep_df, feature, class_col, plot_dir / f"{_safe(feature)}.png")
        n_done += 1
    return stats_all, sep_all, thr_all, catrate_all, n_done


# ── Driver ────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, args) -> None:
    class_col = _resolve(df, args.class_col, ["current_cat", "CURRENT_CAT"], required=True)
    endpoint_col = _resolve(df, args.endpoint_col, ["endpoint", "ENDPOINT", "source"], required=True)
    id_col = _resolve(df, args.id_col, ["national_code", "NATIONAL_CODE"], required=False)

    df = df.copy()
    df[class_col] = pd.to_numeric(df[class_col], errors="coerce")
    before = len(df)
    df = df[df[class_col].notna()]
    df[class_col] = df[class_col].astype(int)
    if len(df) < before:
        log.warning("Dropped %d rows with missing/non-numeric %s.", before - len(df), class_col)

    exclude = {c.lower() for c in META_COLUMNS} | {class_col.lower()}
    if id_col:
        exclude.add(id_col.lower())
    numeric_override = set(args.numeric.split(",")) if args.numeric else set()
    categorical_override = set(args.categorical.split(",")) if args.categorical else set()
    pii_substrings = list(PII_SUBSTRINGS) + (args.pii_substrings.split(",") if args.pii_substrings else [])

    schema = discover(
        df, endpoint_col, exclude, args.max_cat_levels, args.max_id_cardinality,
        args.min_n_present, pii_substrings, numeric_override, categorical_override,
    )
    if args.endpoints:
        keep = set(args.endpoints.split(","))
        schema = schema[schema["endpoint"].isin(keep)]

    log.info("Endpoints: %s", sorted(df[endpoint_col].dropna().unique().tolist()))
    log.info("Class split '%s' counts:\n%s", class_col, df[class_col].value_counts().sort_index().to_string())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extensions = []
    if not args.no_json_explode:
        extensions, extra_schema = prepare_json_extensions(
            df, endpoint_col, class_col, id_col, schema, pii_substrings, out, args,
        )
        if extra_schema:
            schema = pd.concat([schema, *extra_schema], ignore_index=True)
            log.info(
                "Exploded %d JSON list column(s) into %d item/aggregate pseudo-endpoint(s): %s",
                schema.loc[schema["type"] == "json_skipped", "feature"].nunique(),
                len(extensions), [e["label"] for e in extensions],
            )

    log.info(
        "Discovered %d (endpoint, feature) pairs total. By type:\n%s",
        len(schema), schema["type"].value_counts().to_string(),
    )
    skipped = schema[~schema["type"].isin(["numeric", "categorical"])]
    if not skipped.empty:
        log.info(
            "Skipped from stats/plots (see schema.csv for the full reason breakdown): %s",
            ", ".join(f"{t}={n}" for t, n in skipped["type"].value_counts().items()),
        )

    schema.to_csv(out / "schema.csv", index=False)
    log.info("Wrote %s", out / "schema.csv")

    if args.inspect:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print("\n=== DISCOVERED SCHEMA (real endpoints + exploded JSON pseudo-endpoints) ===")
            print(schema.to_string(index=False))
        print("\nInspect only — rerun without --inspect to generate stats and plots.")
        return

    stats_all, sep_all, thr_all, catrate_all = [], [], [], []
    n_done = 0

    def _process(label: str, data: pd.DataFrame, feat_schema: pd.DataFrame) -> None:
        nonlocal n_done
        plot_dir = None
        if not args.no_plots:
            plot_dir = out / "plots" / _safe(label)
            plot_dir.mkdir(parents=True, exist_ok=True)
        s, sp, t, c, n = process_features(data, feat_schema, label, class_col, args.severe_ge, plot_dir, args.max_features)
        stats_all.extend(s)
        sep_all.extend(sp)
        thr_all.extend(t)
        catrate_all.extend(c)
        n_done += n
        if n_done % 25 == 0:
            log.info("… processed %d features", n_done)

    for ep, ep_schema in schema[schema["endpoint"].isin(df[endpoint_col].unique())].groupby("endpoint", observed=True):
        _process(str(ep), df[df[endpoint_col] == ep], ep_schema)
    for ext in extensions:
        _process(ext["label"], ext["df"], ext["schema"])

    _write(out / "summary_stats.csv", pd.concat(stats_all, ignore_index=True) if stats_all else None)
    if sep_all:
        sep = pd.DataFrame(sep_all).sort_values("value", ascending=False, na_position="last")
        _write(out / "separation.csv", sep)
    _write(out / "threshold_suggestions.csv",
           pd.DataFrame(thr_all).sort_values("f1", ascending=False) if thr_all else None)
    _write(out / "category_rates.csv", pd.concat(catrate_all, ignore_index=True) if catrate_all else None)

    log.info("Done. %d features processed. Tables + plots in %s", n_done, out.resolve())
    log.info("Start with separation.csv (strongest signals first) and threshold_suggestions.csv.")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]


def _write(path: Path, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(df))


# ── Self-check ────────────────────────────────────────────────────────────────

def _self_check() -> None:
    """Synthetic long-format frame across 3 endpoints; exercise the full pipeline."""
    import tempfile

    rng = np.random.default_rng(0)
    n = 800
    cats = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.12, 0.08])
    codes = [f"c{i}" for i in range(n)]
    frames = []
    # endpoint A: a numeric that rises with severity + a noise numeric
    frames.append(pd.DataFrame({
        "NATIONAL_CODE": codes, "endpoint": "bureau", "CURRENT_CAT": cats,
        "utilization": np.clip(cats * 0.15 + rng.normal(0.3, 0.1, n), 0, 1),
        "noise_a": rng.normal(0, 1, n),
    }))
    # endpoint B: a categorical whose 'bad' level concentrates in severe classes,
    # plus JSON list columns (one code+amount per item, avro-union field, a PII
    # phone-list column that must stay unexploded) and a PII-ish name column —
    # mirrors the real API data (scoreCodes, bounced cheques, person.firstName).
    lvl = np.where(rng.random(n) < (cats >= 2) * 0.6 + 0.1, "flagged", "clean")

    def _events(cat):
        n_items = rng.poisson(cat * 0.7)
        out = []
        for _ in range(n_items):
            code = "BAD" if rng.random() < 0.2 + 0.2 * cat else "OK"
            out.append({
                "code": code,
                "description": "customer has a bad history" if code == "BAD" else "no issues found",
                "amFlagged": float(rng.integers(100, 1000)),
                # occasionally a multi-value union (array([402, 403])) — must stay
                # hashable (tuple) after unwrap, not a plain list, or drop_duplicates() breaks.
                "reasonCode": {"int": np.array([402, 403] if rng.random() < 0.05 else [402])},
                # a per-item coded field, not amount-named — the max-across-items
                # aggregate is what should surface this at the customer level.
                "statusCode": int(rng.integers(1, 6)) if rng.random() < 0.2 + 0.2 * cat else 1,
            })
        return out

    frames.append(pd.DataFrame({
        "NATIONAL_CODE": codes, "endpoint": "fraud", "CURRENT_CAT": cats,
        "status": lvl, "n_alerts": rng.poisson(cats + 0.5),
        "events": [_events(c) for c in cats],
        "phone.contacts": [[{"title": "mobile", "value": f"555-{i:04d}"}] for i in range(n)],
        "person.firstName": [f"person{i}" for i in range(n)],
        "notes": [""] * n,  # constant
        "has_block": rng.choice([True, False], n),  # bool categorical — pandas col-select gotcha
    }))
    # endpoint C: overlaps id space, own numeric, plus a near-unique reference id
    frames.append(pd.DataFrame({
        "NATIONAL_CODE": codes, "endpoint": "income", "CURRENT_CAT": cats,
        "monthly_income": np.clip(rng.normal(5000 - cats * 400, 800, n), 0, None),
        "report_ref": [f"ref-{i}" for i in range(n)],
    }))
    df = pd.concat(frames, ignore_index=True)

    with tempfile.TemporaryDirectory() as d:
        args = argparse.Namespace(
            class_col="CURRENT_CAT", endpoint_col="endpoint", id_col="NATIONAL_CODE",
            output_dir=str(Path(d) / "out"), max_cat_levels=12, max_id_cardinality=50,
            min_n_present=20, pii_substrings="", severe_ge=3,
            numeric="", categorical="", endpoints=None, max_features=None,
            inspect=False, no_plots=False, no_json_explode=False,
        )
        run(df, args)
        out = Path(args.output_dir)
        for f in ["schema.csv", "summary_stats.csv", "separation.csv",
                  "threshold_suggestions.csv", "category_rates.csv"]:
            assert (out / f).exists(), f"missing {f}"
        schema = pd.read_csv(out / "schema.csv")
        types = dict(zip(schema["feature"], schema["type"]))
        assert types["events"] == "json_skipped", types
        assert types["person.firstName"] == "pii_excluded", types
        assert types["phone.contacts"] == "pii_excluded", types  # PII json col — must NOT get exploded
        assert types["notes"] == "constant", types
        assert types["report_ref"] == "id_like", types
        # events[item]: 'code' and 'reasonCode' (avro-union-unwrapped) should appear as their own features;
        # 'description' pairs with 'code' so it's retyped 'reference' and split into its own CSV, not plotted.
        item_rows = schema[schema["endpoint"] == "fraud::events[item]"].set_index("feature")
        assert set(item_rows.index) >= {"code", "reasonCode", "amFlagged", "description"}, item_rows
        assert item_rows.loc["reasonCode", "type"] == "categorical"
        assert item_rows.loc["description", "type"] == "reference"
        ref_csv = next(out.glob("*events*item*code_reference.csv"))
        ref = pd.read_csv(ref_csv)
        assert set(ref["code"]) == {"BAD", "OK"}, ref
        # events[agg]: per-customer n_items/has_items + sum/max of the amount-like subfield
        # + max of the non-amount coded subfield ("worst status across this customer's items").
        agg_rows = schema[schema["endpoint"] == "fraud::events[agg]"]
        assert set(agg_rows["feature"]) >= {
            "events.n_items", "events.has_items", "events.sum_amFlagged", "events.max_amFlagged",
            "events.max_statusCode",
        }, agg_rows
        # 'phone.contacts' must never be exploded into item/agg pseudo-endpoints (it's PII).
        assert not schema["endpoint"].astype(str).str.startswith("fraud::phone.contacts").any()
        sep = pd.read_csv(out / "separation.csv")
        # 'utilization' was built to track severity — it should top the separation table.
        top_feat = sep.iloc[0]["feature"]
        assert top_feat in {"utilization", "status", "n_alerts", "code"}, f"weak signal on top: {top_feat}"
        plots = list((out / "plots").rglob("*.png"))
        assert len(plots) >= 4, f"expected several plots, got {len(plots)}"
    print(f"self-check OK — schema/stats/separation/thresholds written, {len(plots)} plots, "
          f"top separating feature = {top_feat!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, help="path to the joined API parquet (or .csv)")
    p.add_argument("--output_dir", type=str, default="api_exploration")
    p.add_argument("--class_col", type=str, default="CURRENT_CAT", help="the current-class column to split by")
    p.add_argument("--endpoint_col", type=str, default="endpoint")
    p.add_argument("--id_col", type=str, default="NATIONAL_CODE")
    p.add_argument("--severe_ge", type=int, default=3,
                   help="a customer is 'severe' (positive) when CURRENT_CAT >= this (default 3)")
    p.add_argument("--max_cat_levels", type=int, default=12,
                   help="integer numerics with <= this many distinct values are treated as categorical")
    p.add_argument("--max_id_cardinality", type=int, default=50,
                   help="categorical columns with more distinct values than this are treated as "
                        "identifiers/dates ('id_like') and skipped, not plotted as a 50-bar chart")
    p.add_argument("--min_n_present", type=int, default=20,
                   help="columns with fewer non-null values than this are too sparse to compare "
                        "across classes ('sparse') and are skipped")
    p.add_argument("--pii_substrings", type=str, default="",
                   help="comma-list of extra column-name substrings to exclude as personal identifiers, "
                        f"added to the built-in list ({', '.join(PII_SUBSTRINGS)})")
    p.add_argument("--numeric", type=str, default="", help="comma-list of columns to force-treat as numeric")
    p.add_argument("--categorical", type=str, default="", help="comma-list of columns to force-treat as categorical")
    p.add_argument("--endpoints", type=str, default=None, help="comma-list to restrict to specific endpoints")
    p.add_argument("--max_features", type=int, default=None, help="cap features per endpoint (for quick iterations)")
    p.add_argument("--no_plots", action="store_true", help="tables only, skip PNGs")
    p.add_argument("--no_json_explode", action="store_true",
                   help="skip exploding json_skipped list columns (scoreCodes, bounced cheques, etc) "
                        "into item/aggregate pseudo-endpoints — just report them as json_skipped")
    p.add_argument("--inspect", action="store_true", help="print discovered schema and exit (no stats/plots)")
    p.add_argument("--self_check", action="store_true", help="run on synthetic data; no server/data needed")
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return
    if not args.data:
        p.error("--data is required (or pass --self_check)")

    path = Path(args.data)
    if not path.exists():
        sys.exit(f"No such file: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    log.info("Loaded %s — %d rows x %d cols", path.name, len(df), df.shape[1])
    run(df, args)


if __name__ == "__main__":
    main()
