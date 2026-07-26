# Configuration Reference (`project_config.py`)

Every setting in `project_config.py`, grouped by what it controls. Defaults
shown are the **current shipped values**. See `DEPLOYMENT.md` §0 for the
subset you should double-check before a production `--final` run.

---

## Database & table

| Setting | Default | Notes |
|---|---|---|
| `MSSQL_SERVER` / `MSSQL_DATABASE` / `MSSQL_USER` / `MSSQL_PASSWORD` | `localhost` / `EDP` / `sa` / `password` | Read from env vars (`MSSQL_SERVER` etc.) with these as fallbacks. Set real values via environment, not by editing the file. |
| `TRAIN_TABLE` | `"EDP_Feature_Train"` | The **only** live table (aliases `D_ANALYTICS.DPD_SAMPLE1`). Holds both matured snapshots (training) and the newest immature ones (prediction) — there is no separate prediction table. |
| `ID_COL`, `CONTRACT_COL`, `CUSTOMER_COL`, `SNAPSHOT_COL`, `TARGET_COL` | `LOAN_ID`, `CONTRACT_NUMBER`, `NATIONAL_CODE`, `SNAPSHOT_DATE`, `WORST_FUTURE_CAT` | Column name constants — change only if the upstream ETL renames columns. |
| `META_COLS` | derived from the above + `WORST_FUTURE_DPD` | Columns excluded from the feature set. Everything else in the table is a feature. |

## Prediction grain

| Setting | Default | Notes |
|---|---|---|
| `PREDICTION_GRAIN` | `"loan"` | What one scored row IS. `"loan"` = one row per (`LOAN_ID`, `SNAPSHOT_DATE`), the ETL's native grain — `WORST_FUTURE_CAT` is already per-loan upstream (`MAX(LABEL_DPD) GROUP BY LOAN_ID`). `"portfolio"` = one row per (`NATIONAL_CODE`, `SNAPSHOT_DATE`), label = max over the customer's loans, features aggregated to min/max/mean/std + count — the pre-July-2026 behaviour that `results_1`…`results_6` were measured under. Changes `process_raw_data`, `build_features`, and what the queue ranks; part of the NPZ cache key, so the two grains cache separately. `DEEPSETS_ENABLED` requires `"portfolio"`. |

## Label & classes

| Setting | Default | Notes |
|---|---|---|
| `NUM_CLASSES` | `4` | `{0: No Delay, 1: Current, 2: Past Due+, 3: Severe Past Due}`. Raw ETL categories 0/1/2 pass through 1:1; raw 3/4 collapse into class 3. Changing this requires a full retrain (label semantics change) and a `DATA_VERSION` bump. |
| `LABEL_HORIZON_MONTHS` | `6` | Forward window for `WORST_FUTURE_CAT`. Also the minimum gap enforced between train/val/test snapshots to avoid label-window overlap. |

## Cost matrix

| Setting | Default | Notes |
|---|---|---|
| `COST_MATRIX` | 4×4, `COST_MATRIX[true][predicted]` | Single source of truth for cost, used by the expected-cost decision rule (`PREDICTED_CLASS`/`EXPECTED_COST` — secondary output columns) and `avg_cost` metric. **Never drives `RISK_SCORE` or the queue ranking** — that's calibrated P(severe), cost-free by design (see `MODEL_EVALUATION.md`). Business only confirmed the 4× anchor (missing class-2 costs 4× a false alarm); the class-3 row/column is an unvalidated extrapolation — treat any cost-based number as a rough diagnostic, not a target. |
| `BASELINE_COST_WEIGHTS` | `False` | Whether the deployed arm's training uses cost-scaled sample weights. Off because it measurably hurt ranking (Run 6: +1pt AP from turning it off). Leave off unless re-validated. |

## Feature typing

| Setting | Default | Notes |
|---|---|---|
| `BINARY_FEATURES` | 7 named flags (`IS_IN_WARNING_ZONE`, `IS_DETERIORATING`, ...) | These skip outlier clipping/scaling in preprocessing (already 0/1). Everything else is treated as continuous. |

## Split & validation

| Setting | Default | Notes |
|---|---|---|
| `WALK_FORWARD_ENABLED` | `False` | `True` = train/evaluate across every valid rolling-window fold instead of one static split. Off by default — routine runs use the single static split; flip on only for a periodic stability check (`DEPLOYMENT.md` §2), and revert after. |
| `MIN_TRAIN_SNAPSHOTS` | `1` | Minimum training snapshots for a walk-forward fold to be generated at all. Folds with very few snapshots are unstable — see `analyze_walk_forward.py`'s `--min_train_snaps` filter for post-hoc analysis rather than raising this (raising it would silently drop early folds instead of showing you why they're unstable). |
| `OPTIMIZE_ON_VALIDATION` | `True` | `True` = carve a validation set for early-stopping/tuning/calibration. `False` = no validation set at all (only meaningful for the legacy DeepSets path). |
| `VAL_SPLIT_MODE` | `"customer"` | `"customer"` = in-time, customer-disjoint holdout (stable hash of `NATIONAL_CODE`) carved from the training snapshots — never overlaps the test label window. `"temporal"` is the legacy mode (val = a distinct calendar snapshot) — leaky when val/test label windows overlap; don't use for the deployed arm. |
| `CUSTOMER_VAL_FRACTION` | `0.20` | Fraction of customers held out for validation under `VAL_SPLIT_MODE="customer"`. |

## Model arms

| Setting | Default | Notes |
|---|---|---|
| `MODEL_ARMS` | `["multiclass", "binary", "ordinal", "per_cat"]` | Which arms an **evaluation** run (`python run.py train`, no `--final`) trains and compares. See `README.md`'s architecture section for what each arm is. Trim to `["multiclass", "binary"]` for cheaper walk-forward runs. |
| `DEPLOY_ARM` | `"multiclass"` | Which arm becomes the shipped model. `"auto"` = pick the best pooled ranking AP among full-distribution arms (needs a test set — invalid for `--final`, which requires an explicit name). Locked to `"multiclass"` after the Run-6 shootout; see `MODEL_EVALUATION.md` for the evidence. |
| `ARM_OPTUNA_TRIALS` | `0` | `0` = train the deployed arm with `XGB_DEFAULTS` (fixed hyperparameters). `>0` = tune it via Optuna, maximizing validation PR-AUC of P(severe) — the real objective, not macro-F1. Only the arm(s) that could actually deploy are tuned; diagnostics always use defaults. |
| `DEEPSETS_ENABLED` | `False` | Legacy neural (DeepSets encoder + XGBoost meta-learner) path. Lost the Run-5/6 shootouts on every ranking slice; kept for reproducibility only. Leave off. |

## Prediction & the ranked queue

| Setting | Default | Notes |
|---|---|---|
| `RECALIBRATE_ON_PREDICT` | `True` | At `predict` time, refit the probability calibrator on the newest matured snapshot before scoring (tracks base-rate drift). Requires cache/DB access to the training table. |
| `PRED_SNAPSHOT_DATES` | `None` | Which snapshot(s) `predict` scores when `--snapshot_date` isn't passed. `None` = every currently-immature snapshot in `TRAIN_TABLE`. |
| `PRED_DEDUP_LATEST` | `True` | When several snapshots are scored in one call, keep only each customer's newest row in the ranked queue (older rows flagged `SUPERSEDED`) — the enrichment API can't be queried "as of" the past, so an old score wastes budget. Keyed on `NATIONAL_CODE` at both grains, deliberately: the API is customer-keyed, so any newer row makes that customer's older rows stale for calling purposes. |
| `CARVE_CURRENT_CAT_GE` | `NUM_CLASSES - 1` (i.e. `3`) | Rows with `current_cat` at or above this are `ALREADY_SEVERE` — rule-flagged, never ranked. Defines the population the ranking metrics are computed on. At loan grain this flags the *severe loan*, leaving its healthy siblings in the queue; at portfolio grain `current_cat` was the portfolio max, so one severe loan carved out the whole customer. |
| `API_RATE_PER_HOUR` | `240` | The enrichment API's call budget. Ranking metrics are reported at `K = API_RATE_PER_HOUR × window_hours`. |
| `RANKING_REF_WINDOWS` | `{"1_day": 24, "1_week": 168, "1_month": 720}` | Reference call-budget windows (hours) the ranking metrics report recall/lift at. |
| `CALIBRATION_MIN_STRATUM_N` | `5000` | Minimum validation samples in a `current_cat` stratum to fit its own isotonic calibrator; smaller strata fall back to the pooled calibrator. |
| `API_DATA_TTL_DAYS` | `30` | How long an enrichment result stays "fresh" (business-confirmed). Customers called within this window are flagged `RECENTLY_CALLED` and skipped, via the call ledger (`--called_log`). |
| `API_CALL_LOG` | `None` | Default path to the call ledger CSV (`NATIONAL_CODE, CALLED_AT`), used if `--called_log`/`called_log_path` isn't passed explicitly. |
| `CERTAINTY_ACT_THRESHOLD` | `None` | When set (e.g. `0.9`), queue rows with `RISK_SCORE` at or above this are flagged `PREDICTED_SEVERE` — treated as certain enough to act on directly without spending an API call. **Pending business sign-off** — leave `None` until confirmed. |

## Cache

| Setting | Default | Notes |
|---|---|---|
| `DATA_VERSION` | `"v1.3"` | Bump this string whenever the upstream ETL schema or label semantics change, to force the NPZ cache (`data/train_portfolios_cache.npz`) to rebuild. A stale cache after an ETL change will silently train on the old schema. |

## Legacy DeepSets hyperparameters

Only relevant if `DEEPSETS_ENABLED = True`.

| Setting | Default |
|---|---|
| `MAX_LOANS_PER_CUSTOMER` | `None` (computed as the 99th percentile at runtime — resolves to `2`) |
| `DEEPSETS_HIDDEN_DIM` / `DEEPSETS_EMBED_DIM` | `128` / `64` |
| `DROPOUT` | `0.15` |
| `BATCH_SIZE` | `512` |
| `LEARNING_RATE` / `WEIGHT_DECAY` | `5e-4` / `1e-4` |
| `EPOCHS` / `FIXED_EPOCHS` / `PATIENCE` | `80` / `15` (used when `OPTIMIZE_ON_VALIDATION=False`) / `10` |
| `NUM_WORKERS` | `2` |

## Reproducibility & runtime

| Setting | Default | Notes |
|---|---|---|
| `RANDOM_SEED` | `42` | Seeds `random`, `numpy`, and `torch` (legacy path). XGBoost arms are otherwise deterministic given identical data and hyperparameters. |
| `TORCH_NUM_THREADS` / `TORCH_NUM_INTEROP` | `12` / `4` | CPU thread budget for the legacy DeepSets path on the 20-core training server. Irrelevant to the deployed XGBoost arms. |

## Paths

| Setting | Default | Notes |
|---|---|---|
| `BASE_DIR` / `ARTIFACT_DIR` / `DATA_DIR` | repo root / `artifacts/` / `data/` | `ARTIFACT_DIR` and `DATA_DIR` are created automatically on import if missing. |

---

## Changing a config for one call without editing the file

Most inference-time knobs above (`cost_matrix`, `carve_current_cat_ge`,
`certainty_act_threshold`, `calibration_min_stratum_n`, `api_data_ttl_days`,
`pred_dedup_latest`, `called_log_path`) can be overridden **per call**
without touching `project_config.py`, via `ScoringParams` — see
`src/inference/scoring_params.py` and `DEPLOYMENT.md` §4 (pattern B). This
is the recommended way to experiment with a knob (e.g. a candidate
`certainty_act_threshold`) without affecting other callers or requiring a
code change.
