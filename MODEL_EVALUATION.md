# How to Judge a Trained Model

A guide for reading a training run's output and deciding whether the
model is good — written for someone who didn't build the pipeline and
needs to know what to look at and what "good" looks like.

**The one-sentence version:** the deliverable is a ranked queue, so the
headline number is **ranking quality on the currently-clean-customer
slice**, not aggregate classification accuracy. If you only read one
section below, read "Where to look" and "The trap: aggregate accuracy."

---

## Where to look

An **evaluation run** (`python run.py train`, no `--final`) writes, per
fold (usually just `fold_01`):

| File | What it has |
|---|---|
| `fold_01/arms_metrics.json` | Full metrics for every trained arm — this is the primary source, read it first. |
| `fold_01/plots/capture_curves.png` | Cumulative-gains curve for every arm — the single best "is this good" visual. |
| `fold_01/plots/confusion_matrix.png`, `roc_curves.png` | Classification-side diagnostics for the deployed arm. |
| Console log / `pipeline_log.json` | Per-arm ranking + classification summary lines, plus timing. |

**A `--final` run (`python run.py train --final`) writes NONE of the
above.** There's no test set in a final fit (it trains on every mature
snapshot to maximize data for the shipped model), so there's nothing to
score it against. `--final` only writes the deployment artifacts
(`model_arm.pkl`, `calibrator.pkl`, `model_bundle.pkl`, `metadata.json`).
**Judge the recipe from the most recent evaluation run, then trust
`--final` to have applied the same recipe to more data.** Don't expect
metrics to appear after a `--final` run — that's expected, not a bug.

---

## Reading `arms_metrics.json`

Each arm's entry has this shape (see `src/evaluation/metrics.py::full_evaluation`):

```
{
  "macro_f1": ...,  "qwk": ...,  "accuracy": ...,           # raw argmax — see the trap below
  "recall_class_0..3": ...,
  "argmax_cal": {...},        # argmax on CALIBRATED probs — isolates calibration's effect
  "cost_rule": {...},         # expected-cost decision, secondary/diagnostic only
  "ranking": {                 # ★ THE HEADLINE — read this first
      "n_ranked": ..., "n_severe": ..., "base_rate": ...,
      "pr_auc": ...,                                          # threshold-free summary
      "at_1_day":   {"k": 5760,   "recall": ..., "precision": ..., "lift": ...},
      "at_1_week":  {"k": 40320,  "recall": ..., "precision": ..., "lift": ...},
      "at_1_month": {"k": 172800, "recall": ..., "precision": ..., "lift": ...},
      "by_current_cat": {
          "current_cat_0": {...ranking block, same shape, this slice only...},
          "current_cat_1": {...},
          "current_cat_2": {...}
      }
  },
  "ranking_single_loan": {...},# same shape, customers holding exactly 1 loan
  "by_current_cat": {...},     # classification-side per-slice (cost_rule based)
  "bootstrap_ci": {...}        # only on the DEPLOYED arm
}
```

### Comparing a loan-grain run against `results_3`…`results_6`

Those runs were scored at **portfolio grain** (one row per customer). A
loan-grain run is not directly comparable on the headline `ranking` block,
for two reasons that have nothing to do with model quality:

1. **The population changed.** Healthy loans belonging to customers with a
   severe loan now enter the queue; under portfolio grain the whole customer
   was carved out as `ALREADY_SEVERE`.
2. **The base rate changed.** A 2-loan customer with one severe loan was 1
   positive in 1 row; as loans it is 1 positive in 2 rows. PR-AUC's no-skill
   floor *is* the base rate, so AP moves with it — a better model can score a
   lower AP. `ranking.base_rate` is reported next to `pr_auc` for exactly
   this reason; read them together.

Use **`ranking_single_loan`** for the comparison. Customers holding one loan
produce bit-identical rows under either grain, so that slice is genuinely
apples-to-apples. Report the full `ranking` block separately, as the new
baseline going forward.

### What `ranking` means, concretely

- **`pr_auc`** — precision-recall AUC of "will this customer go severe," ranked by the model's score. Threshold-free; the single best one-number summary. Compare across arms/runs with this.
- **`at_1_week.recall`** — of everyone who *actually* went severe, what fraction were in the top 40,320 (= 240/hr × 168h) of the ranking? This is literally "if we call the API down this list for a week, what fraction of the real cases do we catch."
- **`at_1_week.lift`** — how many times better than calling customers at random. `lift = precision / base_rate`.
- **`by_current_cat`** — the same three numbers, computed *separately* for customers currently at cat-0 (clean), cat-1, cat-2. **This is where the real signal is** — see the next section.

---

## The trap: aggregate accuracy lies to you here

`WORST_FUTURE_CAT` includes the current month, so `label >= current_cat`
always. For a customer already at cat-2, arithmetic alone makes "will
reach cat-3" a near-certainty if they simply don't pay — no model
intelligence needed. In Run 6, the `current_cat_2` slice hit **F1 = 1.000**
mechanically. That slice is ~10-15% of the population and it inflates
every aggregate number (`macro_f1`, pooled `pr_auc`, pooled recall) toward
"looks great" even if the model has zero insight into who's actually about
to deteriorate.

**The real task — and the number to actually judge the model on — is the
`current_cat_0` slice**: customers who are clean *today*. That's who the
business needs early warning about; customers already at cat-1/cat-2 are
comparatively easy money already flowing through collections. Always read
`ranking.by_current_cat.current_cat_0`, never just the pooled `ranking`
block, when deciding if a model is actually good.

---

## Reference numbers (know what "normal" looks like)

From Run 6 (`results_6/`, single static split, test = Dec 2025, deployed
arm = `multiclass`) and the July-2026 walk-forward check
(`analyze_walk_forward.py`, restricted to folds with realistic training
volume):

| Metric | Pooled | `current_cat_0` slice |
|---|---|---|
| Base rate (severe within horizon) | ~7.7% | **~1.5%** |
| Pooled AP (PR-AUC) | ~0.57–0.58 | ~0.16–0.18 |
| Recall @ 1 week | ~0.50–0.51 | ~0.6–0.8 (small-K noise — see note) |
| Recall @ 1 month | ~0.93 | ~0.95–0.98 |

A `current_cat_0` AP around **0.15–0.20** is the expected, healthy range
given how hard that slice genuinely is (1.5% base rate — an AP of 0.17 on
a 1.5% base rate is roughly an 11× improvement over random, which is what
`lift` will show). Walk-forward confirmed this is stable across four
different test months once the model is trained on realistic data volume
(≥3 snapshots) — see `AGENT_HANDOFF.md` §16 for the full analysis.

**Use these as a sanity range, not a pass/fail gate.** A newly retrained
model landing meaningfully outside this band (say, pooled AP < 0.45 or
`current_cat_0` AP nearly 0) is the signal to investigate, not to ship.

---

## Red flags

| Symptom | Likely cause |
|---|---|
| `current_cat_0` AP near 0, or `pr_auc` undefined/NaN | Check `n_severe` in that slice — if it's tiny, the test snapshot may have too few positives to measure reliably (see the walk-forward note below). Also check the calibrator loaded correctly. |
| Pooled `base_rate` far from ~7-8%, or `current_cat_2` base rate far from ~45% | The label distribution shifted — check the ETL, or that the right snapshot was scored. |
| `bootstrap_ci` interval is very wide relative to the mean | Test set is small, or the metric is noisy (common for `recall_class_3` on a small slice) — don't over-read a single run's point estimate. |
| One arm's ranking AP collapses (< 0.4) while others on the same fold are fine | Check `n_train_snaps` for that fold — models trained on 1-2 snapshots are inherently unstable (see `AGENT_HANDOFF.md` §16); not a sign the architecture is broken. |
| `macro_f1` looks great (>0.9) but `current_cat_0` ranking AP is weak | Expected, not contradictory — see "The trap" above. Don't let a high aggregate F1 talk you out of checking the cat-0 slice. |
| Capture curve (`capture_curves.png`) is close to the diagonal | The model isn't beating random ranking — investigate before shipping; compare against the reference AP range above. |

---

## Reading the plots

- **`capture_curves.png`** — x-axis: hours of API calling (log scale); y-axis: recall of future-severe customers. One line per arm. A good model's curve rises steeply at the left (captures a lot of the real cases early) and flattens out approaching 1.0. Compare arms directly on this plot — the deployed arm's line should be at or near the top.
- **`confusion_matrix.png`** — for the deployed arm, under the cost-rule decision. Expect the bottom-right (predicted severe / actual severe) to be strong and the diagonal generally dominant; off-diagonal mass toward "under-predicting severity" is more concerning than the reverse, given the cost asymmetry.
- **`roc_curves.png`** — one-vs-rest ROC per class on calibrated probabilities. Useful as a secondary check; the capture curve is the metric that actually matches the deliverable, prefer it when the two disagree.

---

## Comparing arms / deciding whether to re-open `DEPLOY_ARM`

Run 6 tested 4 arms (`multiclass`, `binary`, `ordinal`, `per_cat`) and
found them within ~1 point of pooled AP of each other, with `multiclass`
best on both the pooled and `current_cat_0` numbers — full evidence and
reasoning in `AGENT_HANDOFF.md` §15. If re-running this comparison
periodically (recommended every few months, or after a meaningful data/ETL
change):

1. Set `DEPLOY_ARM = "auto"` and keep the full `MODEL_ARMS` list.
2. Run `python run.py train`.
3. Compare `pr_auc` (pooled and `current_cat_0`) across arms in
   `arms_metrics.json` — the log also prints a side-by-side table.
4. If a different arm wins clearly (not just noise — check it wins on
   *both* the pooled and cat-0 numbers, ideally across more than one
   snapshot via walk-forward), update `DEPLOY_ARM` to the explicit winner
   before running `--final`.

## Checking temporal stability (walk-forward)

If a walk-forward run was done (`DEPLOYMENT.md` §2), don't read its raw
per-fold numbers naively — folds trained on very few snapshots are an
artifact of the fold-generation scheme, not a realistic scenario, and will
look unstable even for a good recipe. Use:

```bash
python analyze_walk_forward.py artifacts/<run_dir> --min_train_snaps 3
```

Trust the **restricted** verdict (folds with realistic training volume)
over the naive all-fold average. See `AGENT_HANDOFF.md` §16 for a worked
example of why this matters — an earlier naive read of walk-forward data
appeared to favor a different arm than the one that's actually correct.
