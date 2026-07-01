# Data Reliability Exploration Tools

This document describes the three standalone exploratory scripts added to the project root for evaluating whether the dataset is **reliable and separable enough** to train the Loan Default Classification model.

These scripts are **self-contained and non-invasive**: they do not modify any existing source files and can be run independently, moved inside the project structure, or deleted after the exploration phase.

---

## Why These Tools?

Before investing further in model tuning, it is important to answer:

1. **Do individual features actually carry predictive signal?** → IV/WoE
2. **Is there a separable manifold in the feature space?** → UMAP
3. **Is the model learning economically sensible patterns?** → SHAP

Traditional approaches like PCA (linear only) and t-SNE (too slow for 4M rows, no global structure) are not suitable for this dataset. The tools below are purpose-built for tabular, high-cardinality credit risk data.

---

## Prerequisites

All three scripts run with the existing project virtual environment (`.venv`). No new dependencies are needed — `umap-learn`, `xgboost`, `shap`, `matplotlib`, `numpy`, and `pandas` are already in `requirements.txt`.

```bash
source /Users/mohammad/.venv/bin/activate
```

Outputs from all scripts are written to `explore_output/` by default (created automatically).

---

## Script 1: `explore_iv_woe.py` — Information Value & Weight of Evidence

### What it does

Computes **Information Value (IV)** and **Weight of Evidence (WoE)** for every feature using a **One-vs-Rest (OvR)** strategy across all three target classes:

| OvR Problem | Binary Question |
|---|---|
| **OvR-0** | "No Delay" vs {Current, Past Due+} |
| **OvR-1** | "Current" vs {No Delay, Past Due+} ← the diagnostic one |
| **OvR-2** | "Past Due+" vs {No Delay, Current} |

The OvR approach avoids a single binary collapse and reveals *per-class* feature power. If OvR-1 IVs are universally low, that is a **data signal** (the "Current" state is genuinely ambiguous in the features), not a model failure.

### IV Rule of Thumb

| IV Range | Interpretation |
|---|---|
| `< 0.02` | Useless — feature carries no signal |
| `0.02 – 0.10` | Weak predictor |
| `0.10 – 0.30` | Medium predictor |
| `0.30 – 0.50` | Strong predictor |
| `> 0.50` | Suspicious — possible data leakage |

### Inputs

- `data/train_portfolios_cache.npz` + `data/train_portfolios_cache.manifest.json`
- **No database connection required.** The NPZ cache must exist (run `python run.py train` once to generate it).

### Outputs

| File | Description |
|---|---|
| `explore_output/iv_report.csv` | IV table: one row per feature, columns `iv_ovr0`, `iv_ovr1`, `iv_ovr2`, `iv_max`, `iv_mean` |
| `explore_output/iv_chart.png` | Horizontal grouped bar chart, sorted by `iv_max` |
| `explore_output/woe_detail_<feature>.png` | 3-subplot WoE bin chart for each of the top N features |

### Usage

```bash
# Quick run with defaults (10 bins, top 20 WoE detail plots):
python explore_iv_woe.py

# More bins for smoother WoE curves, more detail plots:
python explore_iv_woe.py --n_bins 15 --top_n 30

# Custom output directory:
python explore_iv_woe.py --output_dir results/iv_analysis/
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data/` | Directory containing the NPZ cache |
| `--output_dir` | `./explore_output/` | Output directory |
| `--n_bins` | `10` | Number of quantile bins per feature |
| `--top_n` | `20` | Number of top features to generate WoE detail plots for |
| `--chart_top_n` | `40` | Number of features shown in the summary bar chart |

### How to read the output

1. Open `iv_report.csv` and sort by `iv_ovr1`.
2. If `iv_ovr1` is consistently `< 0.02` across all features, the "Current" class is inherently hard to distinguish from raw feature values — this suggests the model needs to rely on portfolio-level context (which the DeepSets architecture provides).
3. Check `iv_ovr2` for the "Past Due+" class — it should be high (strong separation is expected).
4. Any feature with `iv_max > 0.50` should be inspected for leakage.

---

## Script 2: `explore_umap.py` — UMAP Dimensionality Reduction

### What it does

Projects high-dimensional data to 2D (or 3D) using **UMAP** (Uniform Manifold Approximation and Projection), with points coloured by `WORST_FUTURE_CAT`. Supports two modes:

| Mode | Data source | When to use |
|---|---|---|
| `raw` | NPZ cache → mean-pooled portfolio vectors | No model artifacts needed; tests raw feature separability |
| `embeddings` | `.npy` file from a training run | Highest signal; tests separability at the model's learned representation |

All UMAP hyperparameters are exposed as CLI flags so you can **tune iteratively without touching code**.

### Inputs

- **Mode `raw`**: `data/train_portfolios_cache.npz` (no model needed)
- **Mode `embeddings`**: Two `.npy` files from the training server's artifact directory

### Outputs

| File | Description |
|---|---|
| `explore_output/umap_raw_<timestamp>.png` | 2D/3D scatter plot, classes colour-coded |
| `explore_output/umap_embeddings_<timestamp>.png` | Same, from model embeddings |

### Usage

```bash
# Raw features, defaults (10k sample, n_neighbors=30, min_dist=0.1):
python explore_umap.py --mode raw

# Tighter clusters, more local structure:
python explore_umap.py --mode raw --n_neighbors 15 --min_dist 0.05

# Larger sample for more global structure (slower):
python explore_umap.py --mode raw --sample_size 50000

# Cosine metric (better for high-dim normalized embeddings):
python explore_umap.py --mode raw --metric cosine

# 3D plot:
python explore_umap.py --mode raw --n_components 3

# Embeddings from model (copy .npy files from server first):
python explore_umap.py \
  --mode embeddings \
  --embeddings_path artifacts/20260701_124055/test_embeddings.npy \
  --labels_path artifacts/20260701_124055/test_labels.npy
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `raw` | `raw` or `embeddings` |
| `--data_dir` | `./data/` | Cache directory (mode `raw`) |
| `--embeddings_path` | — | `.npy` embeddings file (mode `embeddings`) |
| `--labels_path` | — | `.npy` labels file (mode `embeddings`) |
| `--output_dir` | `./explore_output/` | Output directory |
| `--sample_size` | `10000` | Stratified sample size (preserves class balance) |
| `--n_neighbors` | `30` | UMAP: controls local vs global structure balance |
| `--min_dist` | `0.1` | UMAP: smaller = tighter clusters, more separated |
| `--metric` | `euclidean` | UMAP: distance metric (`euclidean`, `cosine`, `manhattan`) |
| `--n_components` | `2` | Output dimensionality (`2` or `3`) |
| `--random_seed` | `42` | Reproducibility seed |

### Tuning Guide

| Observation | Suggested adjustment |
|---|---|
| Classes completely overlap | Increase `n_neighbors` (more global) or try `--metric cosine` |
| Plot is too noisy / scattered | Decrease `min_dist` (0.01–0.05) |
| Class 1 blends with 0 and 2 | Expected — try embedding mode for a cleaner picture |
| Clusters are too few/large | Decrease `n_neighbors` (more local structure) |

---

## Script 3: `explore_shap.py` — SHAP Feature Attribution

### What it does

Uses **SHAP TreeExplainer** on the trained XGBoost meta-learner to explain predictions. Generates:

- **Global beeswarm summary** — which embedding dimensions most influence Class 2 (Past Due+) predictions
- **Per-class mean |SHAP| bar charts** — feature importance ranked separately for each OvR problem
- **Dependence plots** — reveals whether the relationship between a feature and its SHAP value is monotonic, threshold-based, or noisy

Because XGBoost operates on **DeepSets embeddings** (64-dim), the "features" here are `embed_0` … `embed_63`. The dependence plots are the key diagnostic: if SHAP values show a clean monotonic trend, the embedding is carrying structured signal.

### Inputs

These files must be **copied from the training server's artifact directory**:

| File | Shape | Notes |
|---|---|---|
| `xgb_model.json` | — | Saved XGBoost model |
| `test_embeddings.npy` | `(N, 64)` | Test-set DeepSets embeddings |
| `test_labels.npy` | `(N,)` | Test-set ground-truth labels |

#### How to export embeddings from the server

Add the following snippet to `run.py` **after Stage 8 (XGBoost training)** and re-run:

```python
import numpy as np
emb, lbl = meta_learner._extract_embeddings(test_dataset)
np.save(run_dir / "test_embeddings.npy", emb)
np.save(run_dir / "test_labels.npy", lbl)
```

### Outputs

| File | Description |
|---|---|
| `explore_output/shap_summary_class2.png` | Beeswarm plot for Past Due+ class |
| `explore_output/shap_class0_bar.png` | Mean \|SHAP\| bar chart — No Delay |
| `explore_output/shap_class1_bar.png` | Mean \|SHAP\| bar chart — Current |
| `explore_output/shap_class2_bar.png` | Mean \|SHAP\| bar chart — Past Due+ |
| `explore_output/shap_class0_dependence.png` | Dependence plots for top features, Class 0 |
| `explore_output/shap_class1_dependence.png` | Dependence plots for top features, Class 1 |
| `explore_output/shap_class2_dependence.png` | Dependence plots for top features, Class 2 |

### Usage

```bash
python explore_shap.py \
  --model_path artifacts/20260701_124055/xgb_model.json \
  --embeddings_path artifacts/20260701_124055/test_embeddings.npy \
  --labels_path artifacts/20260701_124055/test_labels.npy

# Show more features:
python explore_shap.py ... --max_display 30

# Use a larger SHAP sample (slower but more accurate):
python explore_shap.py ... --sample_size 10000
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to saved XGBoost model |
| `--embeddings_path` | *(required)* | Path to `.npy` test embeddings |
| `--labels_path` | *(required)* | Path to `.npy` test labels |
| `--output_dir` | `./explore_output/` | Output directory |
| `--max_display` | `20` | Max features shown in summary/bar plots |
| `--sample_size` | `5000` | Max samples used for SHAP computation |

---

## Recommended Workflow

Run the tools in this order:

```
1.  python explore_iv_woe.py
       → inspect iv_report.csv
       → how strong is OvR-1? Are features actually predictive for "Current"?

2.  python explore_umap.py --mode raw
       → visual check: do the three classes form separable regions?
       → iterate with --n_neighbors / --min_dist until the picture is clear

3.  (copy .npy files from server)
    python explore_umap.py --mode embeddings
       → cleaner signal: the 64-dim DeepSets space vs the raw 64-feature space

4.  (copy model + .npy files from server)
    python explore_shap.py --model_path ... --embeddings_path ... --labels_path ...
       → do the dependence plots show monotonic / interpretable patterns?
       → are the top SHAP features economically sensible?
```

### Interpreting results jointly

| Observation | Likely cause | Action |
|---|---|---|
| OvR-1 IVs all `< 0.02` | "Current" state is ambiguous in raw features | Expected; trust the DeepSets portfolio context |
| OvR-2 IVs all `< 0.10` | "Past Due+" is not separable from raw data | Data quality / label issue — investigate |
| UMAP raw: all 3 classes overlap entirely | Features carry no signal | Review feature engineering |
| UMAP embeddings: 3 clear clusters | Model learned good representations | Data is reliable; model is working |
| SHAP dependence plot: random scatter | That embedding dimension is uninformative | May indicate unused capacity in DeepSets |
| SHAP shows unexpected feature direction | Possible leakage or spurious correlation | Trace back to raw feature definition |

---

## Cleanup

Once exploration is complete, these files can be removed without affecting the pipeline:

```bash
rm explore_iv_woe.py explore_umap.py explore_shap.py EXPLORATION.md
rm -rf explore_output/
```
