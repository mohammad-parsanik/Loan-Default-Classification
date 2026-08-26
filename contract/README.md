# The column contract

`columns.json` is the single source in this repo for **what the upstream feed
contains and what each column may be done to**. Everything downstream keys on
the names in it, never on the order a `SELECT *` happened to return.

It is authored on the ETL side, beside the SQL that produces the table, and
vendored here. **Do not hand-edit it** — refresh it from upstream instead
(see "Refreshing" below).

## What it holds

One entry per column, in the feed's own ordinal order:

| Field | On | Meaning |
|---|---|---|
| `ordinal` | all | 1-based position in the feed. New columns are appended at the end, never inserted. |
| `name` | all | The column name. This is what code matches on. |
| `type` | all | SQL type, for reference. |
| `role` | all | `key` / `feature` / `label` / `meta`. Features are what the model sees; everything else is a meta column. |
| `nullable` | all | Whether the feed defends the column against NULL. |
| `binary` | features | 0/1 flag — exempt from clipping and scaling. |
| `sentinel` | some features | A value that is a *code*, not a quantity. |
| `clip` / `scale` | some features | `false` exempts the column from `OutlierClipper` / `PortfolioRobustScaler`. |

It deliberately carries **no column semantics** — no prose describing what a
column means or how the ETL derives it. That belongs to the upstream project;
this repo is public and its working copy of that documentation
(`etl_integration/`, `column_changes.md`) is gitignored.

## What reads it

- [`src/data/column_contract.py`](../src/data/column_contract.py) loads and
  validates it at import, exposing `FEATURE_ORDER`, `META_COLS`,
  `BINARY_FEATURES`, `NO_CLIP`, `NO_SCALE`, `SENTINELS`. A malformed contract
  fails there, not later at fit time.
- `project_config.py` re-exports `META_COLS` / `BINARY_FEATURES` from it
  rather than maintaining its own copies, so the two cannot drift.
- `DataLoader.project_features` projects every incoming frame to it **by
  name**: a missing feature raises and names itself; a column the model does
  not know is dropped with a warning.
- `DataLoader._cache_key` hashes `contract_version` and the feature list *in
  order*, so a contract change invalidates every NPZ cache.

## Why by name and not by position

The preprocessing transformers receive bare arrays, so every statistic they
hold — fill values, clip bounds, scaler indices — is keyed by **column
position**. Feed them the same 64 columns in a different order and each
column is silently transformed with its neighbour's parameters: same width,
no error, wrong answers. Projecting by name upstream is what makes those
integer indices mean what the fitted transformer thinks they mean, and
`assert_pipeline_features` checks the assumption rather than trusting it.

`tests/test_order_independence.py` is the regression suite for this.

## Refreshing

When the feed changes, the upstream project updates its own `columns.json` in
the same commit as the SQL. To bring it across:

1. Copy the upstream file to `etl_integration/columns.json` (local-only).
2. Re-project it into this file, keeping the code fields and dropping the
   prose:

   ```bash
   python - <<'PY'
   import json, collections
   src = json.load(open("etl_integration/columns.json"))
   keep = ["ordinal", "name", "type", "role", "nullable",
           "binary", "sentinel", "clip", "scale"]
   doc = json.load(open("contract/columns.json"))
   doc["contract_version"] = src["contract_version"]
   doc["table"] = src["table"]
   doc["etl_commit"] = src["etl_commit"]
   doc["columns"] = [collections.OrderedDict((k, c[k]) for k in keep if k in c)
                     for c in src["columns"]]
   json.dump(doc, open("contract/columns.json", "w"), indent=2)
   PY
   ```

3. Bump `DATA_VERSION` in `project_config.py` — caches built against the old
   contract are stale.
4. Run `python -m src.data.column_contract` and `pytest tests/`.

`column_contract.py` cross-checks this file against
`etl_integration/columns.json` on import whenever that folder is present, and
warns if the two disagree on any code field — so a forgotten step 2 shows up
in the logs rather than silently.

## A change that the contract cannot catch

A column whose *meaning* changes while its name, position and type stay the
same is invisible here and to any schema check. Those are recorded upstream
and land here as a `DATA_VERSION` bump with a comment. The current feed
included several — see the `DATA_VERSION = "v1.4"` note in
`project_config.py`, and `etl_integration/CONSUMER_CONTRACT.md` §8 for the
list.
