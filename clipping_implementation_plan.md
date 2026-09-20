# Clipping Analysis: Features That Shouldn't Be Clipped (Contract v2 → v3)

> [!WARNING]
> **Superseded in part — corrected 2026-09-20. See `AGENT_HANDOFF.md` §25.**
> This document is kept because the reasoning is the useful part of the record,
> including where it was wrong. Three corrections:
>
> 1. **`PCT_COMPLETED` is not bounded by the feed.** Category 3 item 3 below
>    asserted it is. `etl_integration/CONSUMER_CONTRACT.md` col 64 says the
>    opposite. Corrected in place below; the column now carries
>    `clip_bounds: [0, 1]` instead of `clip: false`.
> 2. **The `head_lift` figures were measured before imputation**, but the
>    pipeline clips *after* it. For a nullable column the `p1` quoted here may
>    never have been the clipper's `p1` — which affects `HIST_MAX_DPD_DAYS`
>    and `PCT_COMPLETED`, both nullable. `PAYED_OVERDUE_INST_CNT` is not.
> 3. **The `tail_lift` figures are composition, not signal.** `--ranked_only`
>    removes cat_3 but not cat_2, and at the observed rates a tail made purely
>    of cat_2 rows reports ~11.7 with zero incremental signal. The "Features
>    Retained" section's reasoning stands on a different footing than it
>    claims, and the DPD A/B it earmarks is now gated on a re-measurement.

## How Clipping Works in This Pipeline

`OutlierClipper` clips non-binary, non-exempt features to `[p1, p99]` (fitted on train). XGBoost splits on order, so clipping is the **only** preprocessing step the model can feel — it merges all values outside the bounds into one number, permanently destroying the ordering in those regions.

A feature should be exempt (`clip: false`) when:
1. It is **bounded** and p99 == max (clip is a literal no-op — CPU waste).
2. It has **few discrete values** and the clip merges meaningful categories/deterioration jumps.
3. The clip **destroys signal** in the direction that matters for the early-warning task (especially on the head/p1 side for pristine borrowers or failed cures).

## Currently: 44 Features Clipped

Of 64 features: 7 binary (exempt), 4 sentinel (exempt), 9 bounded counts (already `clip: false` from Contract v2). That leaves **44 still clipped**.

Analysis of `results_8/clip_impact_ranked.csv` (scored on `current_cat < 3`) reveals critical issues on **both the head side ($p_1$) and the tail side ($p_{99}$)**.

---

## Category 1: Bounded No-Ops (10 features) — `clip: false`

These have `p1 == 0` and `p99 == max` (or `p99 == max` via bounded range). The clip **literally does nothing** — `n_merged == 0`, no value is ever moved. Setting `clip: false` is free: identical model, no risk.

| Feature | p1 | p99 | max | Values | Why no-op |
|---|---|---|---|---|---|
| `LOAN_CATEGORY` | 0 | 4 | 4 | 0–4 | Ordinal category, bounded |
| `OVERDUE_RATIO` | 0 | 1 | 1 | [0,1] ratio | Bounded ratio |
| `ONTIME_RATIO` | 0 | 1 | 1 | [0,1] ratio | Bounded ratio |
| `CATEGORY_T1` | 0 | 4 | 4 | 0–4 | Historical category lag |
| `CATEGORY_T2` | 0 | 4 | 4 | 0–4 | Historical category lag |
| `CATEGORY_T3` | 0 | 4 | 4 | 0–4 | Historical category lag |
| `HIST_MAX_CATEGORY` | 0 | 4 | 4 | 0–4 | Bounded category |
| `MONTHS_IN_CURRENT_CATEGORY` | 1 | 6 | 6 | 1–6 | Bounded by design |
| `COUNT_DPD_EVENTS_LAST_3M` | 0 | 3 | 3 | 0–3 | Bounded count |
| `COUNT_DPD_EVENTS_LAST_6M` | 0 | 6 | 6 | 0–6 | Bounded count |

> [!NOTE]
> `COUNT_DPD_EVENTS_LAST_3M` and `COUNT_DPD_EVENTS_LAST_6M` are the same family as the 9 counts fixed in Contract v2 — they were missed because their $p_1 \neq p_{99}$ (not degenerate), but they are bounded integer counts with zero tail risk.

---

## Category 2: Directional Trend Signals (4 features) — `clip: false`

### `CATEGORY_TREND_1M` and `CATEGORY_TREND_3M`

| Feature | p1 | p99 | max | n_merged | tail_lift | head_lift |
|---|---|---|---|---|---|---|
| `CATEGORY_TREND_1M` | -1 | 1 | **2** | 1 | — | **8.80** |
| `CATEGORY_TREND_3M` | -1 | 1 | **2** | 1 | **6.85** | **8.96** |

These take only **4 discrete values**: {-1, 0, 1, 2}. The clip merges `value=2` ("jumped 2 categories in the window") into `value=1`. This is a **deterioration signal** — the exact kind the cat_0 early-warning slice is starved for. Both `head_lift` and `tail_lift` are extremely high on the ranked population, confirming signal on both ends.

There is **zero unbounded-tail risk** — the feature has 4 values. `clip: false` is unambiguously correct.

### `DPD_TREND_1M` and `DPD_TREND_3M`

| Feature | p1 | p99 | max | n_merged | tail_lift | head_lift |
|---|---|---|---|---|---|---|
| `DPD_TREND_1M` | -61 | 31 | 31 | 0 | — | **5.22** |
| `DPD_TREND_3M` | -93 | 93 | 93 | 0 | — | **7.47** |

The p99 side is a no-op ($p_{99} == \text{max}$). But the **p1 clip destroys the head**: `DPD_TREND_1M` clips at -61, and `DPD_TREND_3M` clips at -93.
Loans below p1 have `head_lift` of 5.22 and 7.47 (5–7× more likely to go severe than average). These are volatile accounts with rapid drops from severe past delinquency. The clip collapses these volatile borrowers into one bin.

---

## Category 3: Critical Head-Side & Bounded Distortions (3 features) — `clip: false`

### 1. `HIST_MAX_DPD_DAYS`: Preserving the Pristine 0-DPD Population
* In `clip_impact_ranked.csv`: $p_1 = 1.0$, `pct_lo = 0.83%` (~270,000 loans), **`head_lift = 0.00014`**!
* Values below $p_1$ are strictly `0.0` — borrowers who have literally **never missed a payment in their entire history**.
* Because `clip: true`, `OutlierClipper` pushes `0.0` **up to `1.0`**, falsely injecting delinquency into 270k pristine borrowers and erasing the single cleanest negative indicator for the early-warning slice (`current_cat_0`).
* **Action:** Set `clip: false`.

### 2. `PAYED_OVERDUE_INST_CNT`: Preserving the Cure-Failure Signal
* In `clip_impact_ranked.csv`: $p_1 = 3.0$, `pct_lo = 0.91%`, **`head_lift = 16.26`** (the highest head lift in the entire dataset!).
* Borrowers with $< 3$ paid overdue installments (0, 1, 2) have a 16.3× higher default rate because they failed to cure past-due installments.
* `OutlierClipper` pushes them **up to 3.0**, blending cure failures directly into the cured population.
* **Action:** Set `clip: false`.

### 3. `PCT_COMPLETED`: Bounded by Definition, Not by the Feed — CORRECTED
* ~~`PCT_COMPLETED` is naturally bounded between $0.0$ and $1.0$.~~ **Wrong.**
  `etl_integration/CONSUMER_CONTRACT.md` col 64: it is `MATURED_INST_CNT / INSTALLMENT_COUNT`,
  explicitly nullable, and carries "a known, unquantified error — the dimension's
  `INSTALLMENT_COUNT` disagrees with the fact table in both directions". Group G states
  plainly: **"`PCT_COMPLETED` can exceed 1.0. Do not assume it is bounded."** The
  `max = 0.9958` in `clip_impact_ranked.csv` is one cache's observed max, not a bound.
* Because its name doesn't contain `"RATIO"`, it was clipped to $[p_1, p_{99}] = [0.067, 0.850]$, capping loans that are 99% complete down to 85%, and new loans at 1% up to 6.7%.
* **Action (corrected):** `clip_bounds: [0.0, 1.0]`, **not** `clip: false`. Its range comes
  from its *definition* even though the feed does not enforce it, so a declared bound is
  right where both a percentile and an exemption are wrong: `hi = 1.0` clamps the upstream
  defect that $p_{99} = 0.850$ was also clamping, while `lo = 0.0` leaves a brand-new loan
  alone, which $p_1 = 0.067$ did not. Setting `clip: false` would have been the only one of
  the 17 exemptions to genuinely un-bound a column — and on the column with a documented
  data-quality defect.
* Note its nulls impute to `0.0` and now land exactly on the new lower bound, conflating
  "unknown completion" with "unstarted loan". Whether that matters depends on the null
  rate, which `pct_null` now reports.

---

## Features Retained with `clip: true`

### 1. Amount Features (`REMAINING_AMNT`, `UPCOMING_AMNT`, `PAYED_OVERDUE_AMNT`)
* `tail_lift < 1.0` (0.39 – 0.62) — large loans in this population are safer.
* Leaving `clip: true` protects against unbounded monetary outliers from the $\le 7\text{B}$ widening.

### 2. Loan History Counts (`CNT_INSTALLMENT_WARNING_ZONE`, `CNT_RECOVERED_BEFORE`, etc.)
* Moderate tail lift (2.3–4.7), can grow with loan maturity. Retained as clipped.

### 3. Continuous DPD Tails (`TOTAL_DPD_DAYS_LAST_6M`, `MAX_DPD_LAST_3M`, etc.)
* Kept clipped in Contract v3 baseline, but earmarked for a targeted A/B validation test on Lift@K to evaluate whether unclipping them further improves queue ranking.

---

## Contract v2 → v3 Summary

| Category | Count | Features |
|---|---|---|
| Bounded no-ops | 8 | `LOAN_CATEGORY`, `CATEGORY_T1`, `CATEGORY_T2`, `CATEGORY_T3`, `HIST_MAX_CATEGORY`, `MONTHS_IN_CURRENT_CATEGORY`, `COUNT_DPD_EVENTS_LAST_3M`, `COUNT_DPD_EVENTS_LAST_6M` (the two ratios moved to declared bounds) |
| Trend signals | 4 | `CATEGORY_TREND_1M`, `CATEGORY_TREND_3M`, `DPD_TREND_1M`, `DPD_TREND_3M` |
| Critical head & boundary fixes | 2 | `HIST_MAX_DPD_DAYS`, `PAYED_OVERDUE_INST_CNT` |
| **Total new exemptions** | **14** | (was 17 as committed; `PCT_COMPLETED` corrected to `clip_bounds`) |
| Declared bounds `[0, 1]` | 3 | `PCT_COMPLETED`, `OVERDUE_RATIO`, `ONTIME_RATIO` |

### Code Changes
1. **`contract/columns.json`**: Add `"clip": false` to all 17 features; bump `contract_version` to 3.
2. **`src/data/column_contract.py`**: Add all 17 feature names to `_LOCAL_OVERRIDES`.
3. **Tests**: Run `pytest tests/test_order_independence.py` and `pytest tests/test_pipeline_changes.py` to verify contract validation and invariant checks.
