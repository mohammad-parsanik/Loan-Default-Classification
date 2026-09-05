# Data & Model Exploration Tools

Four standalone scripts, kept permanently in the project root (not a
one-off exploration phase — re-run them each retraining cycle, especially
after an ETL change or a new snapshot). They answer:

1. **Do individual features actually carry predictive signal?** → `explore_iv_woe.py`
2. **Is there a separable manifold in the feature space?** → `explore_umap.py`
3. **Is the deployed model relying on economically sensible features?** → `explore_shap.py`
4. **Is preprocessing destroying signal before the model sees it?** → `explore_clip_impact.py`

All four read from a local cache or a lightweight CSV — no live DB
connection needed for any of them.

---

## Script 1: `explore_iv_woe.py` — Information Value & Weight of Evidence

Computes **Information Value (IV)** and **Weight of Evidence (WoE)** for
every feature using a **One-vs-Rest (OvR)** strategy across all **4**
target classes:

| OvR Problem | Binary Question |
|---|---|
| **OvR-0** | "No Delay" vs {Current, Past Due+, Severe} |
| **OvR-1** | "Current" vs {No Delay, Past Due+, Severe} ← usually the hardest |
| **OvR-2** | "Past Due+" vs {No Delay, Current, Severe} |
| **OvR-3** | "Severe Past Due" vs rest — **the class the ranked queue is built on** |

### IV Rule of Thumb

| IV Range | Interpretation |
|---|---|
| `< 0.02` | Useless — feature carries no signal |
| `0.02 – 0.10` | Weak predictor |
| `0.10 – 0.30` | Medium predictor |
| `0.30 – 0.50` | Strong predictor |
| `> 0.50` | Very strong — check it isn't a leak, but see note below |

**Note on high IV / "constant" features:** an earlier IV binning bug made
several genuinely-predictive features (all binary flags, rare-event
counts) look constant (IV = 0.0). The binning was fixed July 2026 — if a
feature reports IV = 0.0, verify it's actually constant in the data before
trusting the report (a quick `df[col].nunique()` check). Conversely, very
high IV on point-in-time features like `LOAN_CATEGORY`/`DPD_DAYS` is
expected, not leakage — they're observed at the same instant as the other
features, just strongly correlated with the label by construction (the
label window includes the current month).

### Inputs / Outputs

- Input: the per-snapshot cache `data/snapshots/train_<key>/` (run
  `python run.py train` once to generate it; **matured snapshots only** —
  the training cache holds nothing else). Override with `--cache_dir`.
- Output: `explore_output/iv_report.csv`, `iv_chart.png`,
  `woe_detail_<feature>.png` (per top-N feature).

```bash
python explore_iv_woe.py                        # defaults: 10 bins, top 20
python explore_iv_woe.py --n_bins 15 --top_n 30
```

### How to read it

1. Sort `iv_report.csv` by `iv_ovr3` — that's the column tied to the
   deployed model's target (P(severe)).
2. Cross-check against the deployed model's SHAP output (Script 3) — a
   feature with high IV that never shows up in SHAP top-N is worth a
   second look (may indicate the model isn't fully using available signal).

---

## Script 2: `explore_umap.py` — UMAP Dimensionality Reduction

Projects high-dimensional data to 2D/3D via UMAP, points colored by
`WORST_FUTURE_CAT` (4 classes: No Delay / Current / Past Due+ / Severe).

| Mode | Data source | Status |
|---|---|---|
| `raw` | NPZ cache → mean-pooled portfolio vectors | **Primary** — always available, model-independent |
| `embeddings` | `.npy` file from a training run | **Legacy** — only meaningful for the DeepSets encoder (`DEEPSETS_ENABLED=True`); the deployed XGBoost arms don't produce a learned embedding space to visualize |

```bash
python explore_umap.py --mode raw
python explore_umap.py --mode raw --n_neighbors 15 --min_dist 0.05   # tighter clusters
python explore_umap.py --mode raw --sample_size 50000                # more global structure
```

### Tuning Guide

| Observation | Suggested adjustment |
|---|---|
| Classes completely overlap | Increase `n_neighbors`, or try `--metric cosine` |
| Plot too noisy / scattered | Decrease `min_dist` (0.01–0.05) |
| Clusters too few/large | Decrease `n_neighbors` |

If classes overlap heavily in `raw` mode, that's consistent with the
project's own finding: the true early-warning task (currently-clean
customers who will later go severe) is genuinely hard in raw feature
space — see the `current_cat_0` slice discussion in `AGENT_HANDOFF.md`.
It does not by itself mean the model is broken; check the ranking metrics
(`src/evaluation/ranking.py`) before concluding anything from this plot alone.

---

## Script 3: `explore_shap.py` — SHAP Feature Attribution

Explains the deployed model's predictions with **SHAP TreeExplainer**.

| Mode | Use when |
|---|---|
| `--bundle <model_bundle.pkl> --data <snapshot.csv>` | **Primary.** Current arm-based model. Feature names are real (`MIN_DPD_DAYS`, `MEAN_LOAN_CATEGORY`, ...), not opaque embedding dimensions — this is genuinely more interpretable than the old DeepSets setup. |
| `--model_path ... --embeddings_path ... --labels_path ...` | **Legacy.** Only for a `DEEPSETS_ENABLED=True` run; features are `embed_0 … embed_63` and carry no individual meaning. |

Bundle mode only supports single-model arms (`multiclass`, `binary`) —
`ordinal`/`per_cat` compose several boosters and aren't representable as
one TreeExplainer target. This isn't a practical limitation:
`DEPLOY_ARM` is locked to `"multiclass"` after the Run-6 shootout, so the
model you're actually shipping is always explainable this way.

```bash
# Primary: no DB, no DeepSets — just the bundle + a CSV of raw rows
# (same columns as TRAIN_TABLE — see contract/columns.json; any order)
python explore_shap.py --bundle artifacts/<ts>_final/fold_01/model_bundle.pkl \
                       --data snapshot_sample.csv
```

### Outputs

| File | Description |
|---|---|
| `shap_summary_class2.png`, `shap_summary_class3.png` | Beeswarm plots for Past Due+ and Severe |
| `shap_class<N>_bar.png` | Mean \|SHAP\| bar chart per class |
| `shap_class<N>_dependence.png` | Dependence plots for top features per class — monotonic trend = clean signal; random scatter = noisy/unused |

### How to read it jointly with Script 1

| Observation | Likely meaning | Action |
|---|---|---|
| High IV on `iv_ovr3` but feature absent from `shap_class3_bar.png` | Model underusing a real signal | Consider Optuna tuning (`ARM_OPTUNA_TRIALS`) or feature interactions |
| SHAP dependence plot for a feature is a clean monotonic line | Model learned an intuitive relationship | Good — safe to explain to stakeholders/auditors |
| SHAP shows an unexpected feature direction (e.g., more overdue → lower risk) | Possible leakage, encoding bug, or genuine surprise worth investigating | Trace back to the raw ETL definition in `etl_integration/CONSUMER_CONTRACT.md` §5 (local-only), or `column_changes.md`. Check the sentinel columns first: `DAYS_SINCE_LAST_*` run worst→best, so a *high* value is the safest state, not the riskiest. |

---

---

## Script 4: `explore_clip_impact.py` — does the [p1, p99] clip destroy signal?

Answers a question the other three can't: whether `OutlierClipper` is
throwing away risk before the model ever sees it.

**Why clipping specifically.** Of the three preprocessing steps, it's the
only one XGBoost can't shrug off. Median imputation and
`PortfolioRobustScaler` are monotone, and trees split on order — a scaled
feature produces identical splits. Clipping is different: it merges every
value above p99 into one number, so any ordering up there is gone
permanently. That's correct when the tail is noise and harmful when the
tail is the risk, and nothing in the pipeline distinguishes the two cases.

Run it after any change that moves a feature's distribution — a population
change most of all. The immediate motive was the Sept 2, 2026 widening of
the ETL scope filter from contract amount ≤ 700M to ≤ 7B
(`AGENT_HANDOFF.md` §21), which stretched the right tail of the three
monetary features by an order of magnitude and put the newly admitted
loans exactly where the clip bites.

### Output columns

| Column | Meaning |
|---|---|
| `p1`, `p99`, `max` | the bounds the clipper would fit, and the real maximum |
| `pct_hi` / `pct_lo` | share of loan-rows pinned to each bound (~1% by construction, unless the column is a `*RATIO*`) |
| `n_merged` | distinct values above p99 collapsed into a single number |
| `tail_span` | `(max − p99) / (p99 − p1)` — how much of the feature's range is discarded |
| **`tail_lift`** | `P(severe \| x > p99) / P(severe)` — **this is the number to read** |

`tail_lift > 1` means severe events concentrate above the clip bound, i.e.
the clip is merging away a region that predicts the label. `≈ 1` means the
tail is indistinguishable from the body and clipping costs nothing.
`< 1` means the tail is *safer* than average and clipping is, if anything,
helping. The script warns on any feature above `--lift_threshold`
(default 1.5).

It mirrors the clipper rather than reimplementing it: `BINARY_FEATURES`
and the contract's `NO_CLIP` columns are skipped exactly as
`OutlierClipper.fit` skips them, and `*RATIO*` columns use the same
`[0, 1]` bounds instead of percentiles. Only mature snapshots are counted
(`filter_mature_snapshots` over the manifest's `label_horizons`), because
immature rows carry a systematically optimistic label that would bias every
`tail_lift`.

### Inputs / Outputs

Reads the per-snapshot cache `data/snapshots/train_<key>/` (override with
`--cache`). No DB, no model, no bundle. Writes `explore_output/clip_impact.csv` and prints the
same table.

```bash
python explore_clip_impact.py
python explore_clip_impact.py --only REMAINING_AMNT,UPCOMING_AMNT,PAYED_OVERDUE_AMNT
python explore_clip_impact.py --baseline data/cache_700m.npz
```

`--baseline` diffs `p99` and `max` against an older cache and adds a
`p99_ratio` column — how far the bound moved when the population changed.
**Copy the old cache aside before rebuilding**; the path is reused, so a
rebuild overwrites the only copy of the comparison.

### How to act on it

| Observation | Meaning | Action |
|---|---|---|
| `tail_lift ≈ 1`, small `tail_span` | clip bound is benign | leave it alone |
| `tail_lift > 1.5`, large `n_merged` | the clip is merging a risk-bearing tail into one value | candidate for `clip: false` in `contract/columns.json` — **then A/B it on validation lift@K**, don't ship on this number alone |
| `tail_lift < 1` | large values are the *safe* end | leave it alone; this is the `DAYS_SINCE_LAST_*` shape, where high = healthy |
| `p99_ratio` ≫ 1 under `--baseline` | the population change moved the bound | expected after a scope-filter change; read `tail_lift` on the new cache to decide whether it matters |

A high `tail_lift` is a reason to investigate, not a verdict. Exempting a
column from clipping hands the model an unbounded feature, which is what
clipping was there to prevent — the measurement says the tail *carries*
signal, not that keeping it will net out positive. Validation lift@K
decides that.

Changing a `clip` flag edits `contract/columns.json`, which bumps
`contract_version` and invalidates the NPZ cache on its own — no
`DATA_VERSION` bump needed for that particular change.

## Recommended workflow (each retraining cycle)

```
1. python explore_iv_woe.py
     → sort by iv_ovr3; note the top signals for the severe-class question

2. python explore_umap.py --mode raw
     → sanity-check separability hasn't degraded with new data

3. python explore_shap.py --bundle <latest model_bundle.pkl> --data <a recent snapshot>
     → confirm the shipped model's top features are still the expected,
       economically sensible ones (DPD trend, category history, etc.)

4. python explore_clip_impact.py
     → read tail_lift; confirm preprocessing isn't merging away a
       risk-bearing tail. Run this BEFORE step 3 after any population
       or distribution change — no point explaining a model that was
       fitted on truncated inputs.
```

None of these scripts modify the pipeline or its artifacts — they're
read-only diagnostics. Keep them; don't delete after a single run, since
each new snapshot or ETL change is worth re-checking against.
