"""
Self-contained monthly monitoring report (one HTML file, charts embedded).

The training server is offline, so nothing is fetched at view time: charts
are matplotlib PNGs inlined as base64, and each is also written to
<run_dir>/charts/ for pasting into slides. Page 1 is for management, page 2
for us; the page break makes "print to PDF" produce the two pages.
"""

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # noqa: E402

import project_config as config  # noqa: E402
from src.monitoring.monitor import PSI_REF_SNAPSHOTS, cutoff_label  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order
SERIES = [BLUE, ORANGE, AQUA]
GRAY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"
CAT_NAMES = ["No delay", "1–60 DPD", "61–155 DPD", "Severe 156+"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def _legend_top(ax, ncol):
    """Legend in a band between the title and the plot, so it never covers data."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=22, color=INK, fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8,
              handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)


def _d(s) -> str:
    s = str(int(s))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def _pct(x, digits=0) -> str:
    return "–" if x is None or x != x else f"{100 * x:.{digits}f}%"


def _num(x) -> str:
    return "–" if x is None or x != x else f"{x:,.0f}"


def _main() -> str:
    return config.MONITOR_MAIN_CUTOFF


def _cap(name: str) -> str:
    label = cutoff_label(name)
    return label[0].upper() + label[1:]


class _Charts:
    def __init__(self, charts_dir: Path):
        self.dir = charts_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def emit(self, fig, name: str, alt: str) -> str:
        fig.tight_layout()
        fig.savefig(self.dir / f"{name}.png", dpi=160, bbox_inches="tight")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


# ── Charts ────────────────────────────────────────────────────────────────────

def _trend(ch, snaps, series, title, name, ci=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = [_d(s) for s in snaps]
    for (label, ys), color in zip(series.items(), SERIES):
        ax.plot(x, ys, color=color, lw=2, marker="o", ms=5, label=label)
        if ci and label in ci:
            lo, hi = ci[label]
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        ax.annotate(_pct(ys[-1]), (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if len(series) > 1:
        _legend_top(ax, len(series))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ch.emit(fig, name, title)


def _early_warning(ch, interim):
    snaps = sorted(interim)
    groups = [(name, _cap(name), c) for name, c in zip(config.MONITOR_CUTOFFS, SERIES)]
    groups.append(("rest", f"Beyond {cutoff_label(_main())}", GRAY))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(snaps))
    width = 0.8 / len(groups)
    for i, (key, label, color) in enumerate(groups):
        ys = [(interim[s]["rest"] if key == "rest" else interim[s]["cutoffs"][key])
              ["deteriorated_rate"] for s in snaps]
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, ys, width * 0.92, color=color, label=label)
    ax.set_xticks(x, [f"{_d(s)}\n{interim[s]['months_elapsed']:.0f} mo" for s in snaps])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Loans already worse than when scored", loc="left", color=INK, fontsize=11)
    _legend_top(ax, len(groups))
    return ch.emit(fig, "early_warning", "Deterioration so far, flagged vs rest")


def _migration(ch, mig):
    groups = [("top", f"{_cap(_main())} of the list"), ("rest", "Rest of the list")]
    groups = [(g, t) for g, t in groups if g in mig]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2, 2.7), squeeze=False)
    n_rank = config.CARVE_CURRENT_CAT_GE
    for ax, (g, title) in zip(axes[0], groups):
        m = np.array([[mig[g].get(r, {}).get(c, 0) for c in range(config.NUM_CLASSES)]
                      for r in range(n_rank)], dtype=float)
        share = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(n_rank):
            for c in range(config.NUM_CLASSES):
                if m[r].sum():
                    ax.text(c, r, _pct(share[r, c]), ha="center", va="center", fontsize=8,
                            color="white" if share[r, c] > 0.55 else INK)
        ax.set_xticks(range(config.NUM_CLASSES), CAT_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n_rank), CAT_NAMES[:n_rank], fontsize=8)
        ax.grid(False)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.set_xlabel("Worst reached in 6 months", fontsize=8)
    axes[0][0].set_ylabel("When scored", fontsize=8)
    return ch.emit(fig, "migration", "Migration, top of list vs rest")


def _calibration(ch, cal):
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(1, len(cal["mean_score"]) + 1)
    ax.bar(x - 0.2, cal["mean_score"], 0.38, color=BLUE, label="Predicted P(severe)")
    ax.bar(x + 0.2, cal["severe_rate"], 0.38, color=ORANGE, label="Actual severe rate")
    ax.set_xticks(x, [f"D{i}" for i in x])
    ax.set_xlabel("List decile (D1 = riskiest)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Predicted vs actual, by list decile", loc="left", color=INK, fontsize=11)
    _legend_top(ax, 2)
    return ch.emit(fig, "calibration", "Calibration by decile")


def _pred_vs_actual(ch, matured):
    snaps = sorted(matured)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(snaps))
    ax.bar(x - 0.2, [matured[s].get("predicted_severe", np.nan) for s in snaps], 0.38,
           color=BLUE, label="Predicted (Σ p3)")
    ax.bar(x + 0.2, [matured[s]["n_severe"] for s in snaps], 0.38, color=ORANGE, label="Actual")
    ax.set_xticks(x, [_d(s) for s in snaps], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_title("Severe loans: predicted vs actual count", loc="left", color=INK, fontsize=11)
    _legend_top(ax, 2)
    return ch.emit(fig, "predicted_vs_actual", "Predicted vs actual severe counts")


def _capture(ch, cc, cuts, n_ranked):
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(cc["frac_called"], cc["recall"], color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":")
    ax.text(0.55, 0.49, "random order", color=MUTED, fontsize=8, rotation=16)
    lines = []
    for name, b in sorted(cuts.items(), key=lambda kv: kv[1]["k"]):
        ax.plot(b["k"] / n_ranked, b["recall"], "o", ms=7, color=BLUE, mec="white", mew=2)
        lines.append(f"{_cap(name)} ({_pct(b['k'] / n_ranked, 1)} of list): {_pct(b['recall'])} caught")
    ax.text(0.98, 0.06, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK, linespacing=1.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of the list, riskiest first")
    ax.set_title("Share of severe loans caught vs how far down the list", loc="left", color=INK,
                 fontsize=11)
    return ch.emit(fig, "capture_curve", "Capture curve")


def _psi(ch, drift):
    rows = [r for r in drift if r["score_psi"] == r["score_psi"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.bar([_d(r["snapshot"]) for r in rows], [r["score_psi"] for r in rows], color=BLUE, width=0.6)
    for y, lab in [(0.1, "watch"), (0.25, "investigate")]:
        ax.axhline(y, color=GRAY, lw=1, ls="--")
        ax.text(len(rows) - 0.5, y, f" {lab}", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_title(f"Score drift (PSI vs previous {PSI_REF_SNAPSHOTS} snapshots)", loc="left", color=INK, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ch.emit(fig, "score_psi", "Score PSI by snapshot")


# ── HTML ──────────────────────────────────────────────────────────────────────

def _tile(value, label, sub="") -> str:
    return (f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div>'
            f'<div class="s">{sub}</div></div>')


def _table(headers, rows) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def _management(ch, m) -> str:
    matured, interim = m["matured"], m["interim"]
    parts = []
    scored = {s: v for s, v in matured.items() if "cutoffs" in v}
    main = _main()
    if scored:
        s = max(scored)
        c = scored[s]
        tiles = []
        for name, b in c["cutoffs"].items():
            lo, hi = b["recall_ci"]
            tiles.append(_tile(_pct(b["recall"]), f"of severe loans were in the {cutoff_label(name)}",
                               f"{b['k']:,} loans · hit rate {_pct(b['precision'])} · "
                               f"95% CI {_pct(lo, 1)}–{_pct(hi, 1)}"))
        mb = c["cutoffs"][main]
        tiles.append(_tile(f"{mb['lift']:.1f}×", f"better than random ({cutoff_label(main)})",
                           f"hit rate {_pct(mb['precision'])} vs {_pct(c['base_rate'], 1)}"))
        tiles.append(_tile(_pct(mb["exposure_caught"]),
                           f"of severe exposure in the {cutoff_label(main)}", "by remaining amount"))
        tiles.append(_tile(_pct(c["base_rate"], 1), "went severe overall",
                           f"{c['n_severe']:,} of {c['n_ranked']:,} loans"))
        parts.append(
            f"<h2>How the {_d(s)} predictions turned out</h2>"
            f'<p class="lead">Of the <b>{c["n_severe"]:,}</b> loans that went severe within six '
            f"months, <b>{_pct(mb['recall'])}</b> were in the {cutoff_label(main)} of the list "
            f"({mb['k']:,} loans) — <b>{mb['lift']:.1f}×</b> the hit rate of a random pick.</p>"
            f'<div class="tiles">{"".join(tiles)}</div>')
        snaps = sorted(scored)
        if len(snaps) > 1:
            series = {_cap(n): [scored[x]["cutoffs"][n]["recall"] for x in snaps]
                      for n in config.MONITOR_CUTOFFS}
            ci = {_cap(n): ([scored[x]["cutoffs"][n]["recall_ci"][0] for x in snaps],
                            [scored[x]["cutoffs"][n]["recall_ci"][1] for x in snaps])
                  for n in config.MONITOR_CUTOFFS}
            parts.append('<div class="row">'
                         + _trend(ch, snaps, series, "Share of severe loans caught, by snapshot",
                                  "trend_recall", ci)
                         + _trend(ch, snaps, {"Severe rate": [scored[x]["base_rate"] for x in snaps]},
                                  "Share of loans that went severe", "trend_base_rate")
                         + "</div>"
                         '<p class="note">Shaded band: 95% interval. Movement inside it is noise. '
                         "Read the fixed-count line next to the severe rate: when more loans go "
                         "severe, the same number of loans can hold a smaller share of them.</p>")
    else:
        parts.append('<div class="empty">No snapshot with predictions has reached its 6-month '
                     "outcome date yet.</div>")

    if interim:
        latest = interim[max(interim)]
        lift = latest["cutoffs"][main]["lift_vs_rest"]
        parts.append(
            "<h2>Early warning — recent predictions, outcome still open</h2>"
            f'<p class="lead">Loans in the {cutoff_label(main)} of the {_d(max(interim))} list '
            f"have already worsened at <b>{lift:.1f}×</b> the rate of the rest of the list.</p>"
            + _early_warning(ch, interim)
            + '<p class="note">"Worse" = a higher delinquency category than when scored. Partial '
            "window: these rates only rise until the 6 months are up.</p>")

    if scored:
        s = max(scored)
        parts.append(f"<h2>Where loans went — {_d(s)}</h2>" + _migration(ch, scored[s]["migration"])
                     + '<p class="note">Each row: loans in that category when scored; cells: '
                     "the worst category they reached within 6 months.</p>")
        rule = scored[s].get("enrichment_rule", {})
        if rule.get("n_flagged"):
            parts.append(
                f"<h2>The enrichment rule — {_d(s)}</h2>"
                f'<p class="lead">The engine sent <b>{rule["n_flagged"]:,}</b> loans to enrichment; '
                f"<b>{_pct(rule['precision'])}</b> of them went severe, covering "
                f"<b>{_pct(rule['recall'], 1)}</b> of all severe loans. Taking the same number of "
                f"loans from the top of the list would have covered "
                f"<b>{_pct(rule['recall_top_same_size'], 1)}</b>.</p>"
                '<p class="note">The rule (remaining &gt; 50M, p3 &gt; 0.5, DPD &gt; 120) can only '
                "reach loans already 121–155 days past due, so its hit rate is mostly loans "
                "crossing into the severe band on schedule — high precision, little early warning."
                "</p>")
    return "".join(parts)


def _analyst(ch, m) -> str:
    matured = {s: v for s, v in m["matured"].items() if "cutoffs" in v}
    parts = []
    if matured:
        snaps = sorted(matured)
        s = snaps[-1]
        c = matured[s]
        if len(snaps) > 1:
            parts.append(_trend(ch, snaps, {
                "All ranked": [matured[x].get("pr_auc") for x in snaps],
                "No delay when scored (cat 0)": [
                    matured[x]["queue_by_cat"].get(0, {}).get("pr_auc", np.nan)
                    for x in snaps]}, "PR-AUC by snapshot", "trend_pr_auc"))
        rows = []
        for cat, sub in sorted(c.get("queue_by_cat", {}).items()):
            cells = [CAT_NAMES[cat], _num(sub["n"]), _num(sub["n_severe"]),
                     _pct(sub["base_rate"], 1), f"{sub['pr_auc']:.3f}"]
            for name in config.MONITOR_CUTOFFS:
                cells += [_pct(sub[f"share_of_{name}"]), _pct(sub[f"recall_{name}"])]
            rows.append(cells)
        heads = ["When scored", "Loans", "Severe", "Severe rate", "PR-AUC within"]
        for name in config.MONITOR_CUTOFFS:
            heads += [f"Share of {cutoff_label(name)}", f"Severe caught, {cutoff_label(name)}"]
        parts.append(f"<h3>By current category — {_d(s)}</h3>" + _table(heads, rows)
                     + '<p class="note">Judge the model on cat 0: for already-delinquent loans the '
                     "outcome is largely mechanical DPD accrual. Share/caught columns are within the "
                     "one pooled list, not re-ranked per category.</p>")
        parts.append('<div class="row">' + _calibration(ch, c["calibration_deciles"])
                     + _pred_vs_actual(ch, matured) + "</div>"
                     + '<p class="note">Predicted counts run below actual: training snapshots are '
                     "depleted of severe loans by upstream deletion (AGENT_HANDOFF §24). The ranking "
                     "survives it; the probability level does not.</p>")
        parts.append(_capture(ch, c["capture_curve"], c["cutoffs"], c["n_ranked"]))

    drift = m["drift"]
    if drift:
        if any(r["score_psi"] == r["score_psi"] for r in drift):
            parts.append(_psi(ch, drift))
        rows = [[_d(r["snapshot"]), _num(r["n_rows"]), _num(r["n_ranked"]),
                 *[_pct(r[f"cat_{k}_share"], 1) for k in range(config.NUM_CLASSES)],
                 _pct(r["score_mean"], 2), _pct(r["score_p99"], 1),
                 "–" if r["score_psi"] != r["score_psi"] else f"{r['score_psi']:.3f}"]
                for r in drift]
        parts.append("<h3>Population and scores</h3>" + _table(
            ["Snapshot", "Rows", "Ranked", *[f"cat {k}" for k in range(config.NUM_CLASSES)],
             "Mean score", "p99 score", "PSI"], rows))

    rule_rows = [[_d(x), _num(v["enrichment_rule"]["n_flagged"]),
                  _pct(v["enrichment_rule"]["precision"]), _pct(v["enrichment_rule"]["recall"], 1),
                  _pct(v["enrichment_rule"]["recall_top_same_size"], 1)]
                 for x, v in sorted(matured.items()) if "enrichment_rule" in v]
    rule_rows += [[f"{_d(x)} ({v['months_elapsed']:.0f} mo, open)", _num(v["enrichment_rule"]["n"]),
                   _pct(v["enrichment_rule"]["severe_so_far_rate"]), "–", "–"]
                  for x, v in sorted(m["interim"].items()) if "enrichment_rule" in v]
    if rule_rows:
        parts.append("<h3>Enrichment rule by snapshot</h3>" + _table(
            ["Snapshot", "Sent to enrichment", "Went severe", "Share of severe covered",
             "Top of list, same size"], rule_rows))

    q = m["quality"]
    unmatched = ", ".join(f"{_d(s)}: {n:,}" for s, n in q["unmatched_by_snapshot"].items()) or "none"
    parts.append("<h3>Data quality</h3><ul>"
                 f"<li>{q['n_prediction_rows']:,} scored rows read.</li>"
                 f"<li>Scored loans missing from {html.escape(config.TRAIN_TABLE)}: {unmatched}.</li>"
                 "</ul>")
    labels = q.get("labels", {})
    hit = {s: v for s, v in labels.items() if v["n_label_lowered"] or v["n_vanished"]}
    if hit:
        parts.append("<h3>Outcome labels changed upstream</h3>" + _table(
            ["Snapshot", "Labels lower than last read", "Loans gone from the feed"],
            [[_d(s), _num(v["n_label_lowered"]), _num(v["n_vanished"])]
             for s, v in sorted(hit.items())])
            + '<p class="note">A rebuilt snapshot has seen more installments, so a label can only '
            "rise. A drop or a vanished loan means installment records were deleted upstream; "
            "the report keeps the highest label it has read for each loan.</p>")
    elif labels:
        parts.append('<p class="note">No outcome label went down and no scored loan left the feed '
                     "since the last run.</p>")
    if q.get("runs"):
        parts.append(_table(
            ["Snapshot", "Engine run used", "Status", "Run date", "Failed loans", "Usable runs"],
            [[_d(r["snapshot"]), html.escape(r["operation_id"][:8]), r["status"],
              html.escape(r["operation_date"][:16]), _num(r["inference_failures"]),
              r["n_usable_runs"]] for r in q["runs"]])
            + '<p class="note">The newest SUCCESS/PARTIAL_SUCCESS run per snapshot is used. '
            "A PARTIAL_SUCCESS run is missing the loans in its failed chunks.</p>")
    return "".join(parts)


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--card:#ffffff;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:24px 16px}
header{border-bottom:1px solid var(--line);margin-bottom:8px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 6px}h3{font-size:15px;margin:22px 0 6px}
.meta,.note{color:var(--muted);font-size:12.5px}.lead{font-size:15px;margin:4px 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .v{font-size:28px;font-weight:600;color:var(--accent)}.tile .l{font-size:13px}
.tile .s{font-size:12px;color:var(--muted)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
img{max-width:100%;height:auto;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--card);display:block;overflow-x:auto}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.empty{padding:16px;border:1px dashed var(--line);border-radius:8px;color:var(--muted)}
.page{break-after:page}
@media print{body{background:#fff}main{max-width:none}}
"""


def build_report(metrics: dict, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ch = _Charts(run_dir / "charts")
    asof = _d(metrics["asof"])
    body = (
        f'<section class="page"><header><h1>Loan early-warning — monthly monitor</h1>'
        f'<div class="meta">As of {asof} · {len(metrics["matured"])} snapshots with final '
        f'outcomes · {len(metrics["interim"])} still open · list ordered by p3, riskiest first'
        f"</div></header>{_management(ch, metrics)}</section>"
        f'<section><h2>Model detail</h2>{_analyst(ch, metrics)}'
        f'<p class="meta">Generated {metrics["generated_at"]}.</p></section>'
    )
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Monthly Model Monitor {asof}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")
    out = run_dir / f"report_{metrics['asof']}.html"
    out.write_text(page, encoding="utf-8")
    return out
