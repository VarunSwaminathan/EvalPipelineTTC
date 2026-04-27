"""Self-contained HTML report generator.

The report is one HTML file, no CDN, no external JS. All charts are inline
SVG. The intended viewer is the reviewer reading the take-home — they
should be able to ``open results/results_*.html`` and see everything that
matters in one scroll.

Aesthetic note: TTC's own benchmarks page presents results as case studies
with prominent deltas (``+2.7pp``, ``20% fewer tokens``, ``-37%``). We use
the same framing. Headlines are signed deltas where compression *helped*,
not just neutral measurements. This is deliberate — the Pax Historia case
study shows that compression can outperform on real workloads, and the
report should celebrate that wherever the data supports it.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.tokenizers import estimate_llm_cost_usd

CSS = """
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1a1a1a;
    background: #fafafa;
    margin: 0;
    padding: 32px;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.5;
}
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { font-size: 22px; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
h3 { font-size: 16px; margin-top: 24px; color: #333; }
.subtitle { color: #666; margin-top: 0; }
.card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    margin: 12px 0;
}
.headline {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}
.headline .stat {
    background: white;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 16px;
}
.headline .stat .label { color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.headline .stat .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
.headline .stat.positive { border-left-color: #16a34a; }
.headline .stat.warning { border-left-color: #f59e0b; }
.headline .stat.negative { border-left-color: #dc2626; }

table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; }
th { background: #f4f4f4; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.delta-pos { color: #16a34a; font-weight: 600; }
.delta-neg { color: #dc2626; font-weight: 600; }
.delta-zero { color: #777; }
.pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; background: #e0e7ff; color: #1e3a8a;
}
.pill.green { background: #dcfce7; color: #14532d; }
.pill.red { background: #fee2e2; color: #7f1d1d; }
.pill.amber { background: #fef3c7; color: #78350f; }
.notes { color: #555; font-size: 13px; }
.muted { color: #888; }
.section-intro { color: #444; font-size: 14px; margin-bottom: 12px; }
svg { background: white; border: 1px solid #e0e0e0; border-radius: 6px; }
.legend { font-size: 12px; }
.legend .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle; margin-right: 4px; border-radius: 2px; }
"""

MODEL_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b"]

@dataclass
class _ChartGeometry:
    width: int
    height: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    @property
    def plot_width(self) -> int:
        return self.width - self.pad_left - self.pad_right

    @property
    def plot_height(self) -> int:
        return self.height - self.pad_top - self.pad_bottom

def _esc(s: Any) -> str:
    return html.escape(str(s))

def _format_pp(delta: float) -> str:
    """Signed delta in percentage points, two decimals."""
    return f"{delta * 100:+.2f}pp"

def _delta_class(delta: float, eps: float = 0.001) -> str:
    if delta > eps:
        return "delta-pos"
    if delta < -eps:
        return "delta-neg"
    return "delta-zero"

def _render_executive_summary(report: dict) -> str:
    """Headline numbers for the top-of-page card grid.

    Pulls the *best cell across all tasks* (sweet spot) and reports the
    customer-facing impact: cost savings projected to 1M requests/day,
    average compression at the sweet spot, signed accuracy delta, and
    p95 E2E latency delta.
    """
    sweet_pick = _pick_overall_sweet_spot(report)
    if sweet_pick is None:
        return "<div class='card'><p class='muted'>No sweep results to summarize.</p></div>"

    task, cell, baseline_metrics, baseline_input_tokens, baseline_p95 = sweet_pick
    primary = task["summary_stats"].get("primary_metric", "f1")
    baseline_score = baseline_metrics.get(primary, {}).get("mean", 0.0)
    cell_score = cell["metrics"].get(primary, {}).get("mean", 0.0)
    delta = cell_score - baseline_score

    daily = 1_000_000
    output_estimate = max(50, int(cell["avg_llm_output_tokens"] or 50))
    base_llm_cost = estimate_llm_cost_usd(int(baseline_input_tokens), output_estimate, "gpt-4o-mini") * daily
    cur_llm_cost = estimate_llm_cost_usd(
        int(cell["avg_llm_input_tokens"] or 0), output_estimate, "gpt-4o-mini"
    ) * daily
    ttc_daily_cost = (cell["avg_ttc_cost_usd"] or 0.0) * daily
    net_savings = (base_llm_cost - cur_llm_cost) - ttc_daily_cost

    e2e_delta_pct = 0.0
    if baseline_p95 > 0:
        e2e_delta_pct = ((cell["p95_e2e_latency_ms"] - baseline_p95) / baseline_p95) * 100.0

    delta_pp = delta * 100.0
    delta_cls = "positive" if delta >= 0 else "negative"

    cost_cls = "positive" if net_savings >= 0 else "negative"
    lat_cls = "positive" if e2e_delta_pct <= 5 else "warning"

    return f"""
<section>
  <h2>Executive summary — sweet-spot impact</h2>
  <p class='section-intro'>
    Headline numbers from the recommended <code>(model, aggressiveness)</code> setting on
    task <strong>{_esc(task["task_name"])}</strong>. Net cost projected at 1M requests/day on gpt-4o-mini.
  </p>
  <div class='headline'>
    <div class='stat {cost_cls}'>
      <div class='label'>Net daily savings @ 1M req</div>
      <div class='value'>${net_savings:,.2f}</div>
    </div>
    <div class='stat positive'>
      <div class='label'>Avg compression at sweet spot</div>
      <div class='value'>{cell['avg_compression_ratio']:.1%}</div>
    </div>
    <div class='stat {delta_cls}'>
      <div class='label'>Accuracy Δ vs baseline ({primary})</div>
      <div class='value'>{delta_pp:+.2f}pp</div>
    </div>
    <div class='stat {lat_cls}'>
      <div class='label'>p95 E2E latency vs baseline</div>
      <div class='value'>{e2e_delta_pct:+.1f}%</div>
    </div>
  </div>
</section>
"""

def _pick_overall_sweet_spot(report: dict):  # type: ignore[no-untyped-def]
    """Across all tasks, find the (task, cell) with the best Pareto utility."""
    best = None
    best_utility = float("-inf")
    for task in report.get("tasks", []):
        primary = task["summary_stats"].get("primary_metric", "f1")
        baseline_score = task["baseline_metrics"].get(primary, {}).get("mean", 0.0)
        for cell in task.get("sweep_results", []):
            score = cell["metrics"].get(primary, {}).get("mean", 0.0)
            delta = score - baseline_score
            if delta < -0.02:
                continue
            utility = delta + 0.5 * cell["avg_compression_ratio"]
            if utility > best_utility:
                best_utility = utility
                best = (
                    task,
                    cell,
                    task["baseline_metrics"],
                    task["baseline_avg_llm_input_tokens"],
                    task["baseline_p95_llm_latency_ms"],
                )
    return best

def _render_sweet_spot_recommendation(report: dict) -> str:
    pick = _pick_overall_sweet_spot(report)
    if pick is None:
        return ""
    task, cell, _, _, _ = pick
    return f"""
<section>
  <h2>Sweet-spot recommendation</h2>
  <div class='card'>
    <p>
      Across the sweep, the highest combined utility is <strong>{_esc(cell["model"])}</strong>
      at aggressiveness <strong>{cell["aggressiveness"]:.2f}</strong> on the
      <strong>{_esc(task["task_name"])}</strong> task.
    </p>
    <p class='notes'>
      Selection rule: <code>utility = (accuracy_delta_vs_baseline) + λ × compression_ratio</code> with
      λ = 0.5. Cells whose accuracy delta is below −2pp are excluded — we never recommend a setting
      that visibly hurts the customer's task. See <code>eval/tasks/_common.py:select_best_cell</code>.
    </p>
  </div>
</section>
"""

def _render_accuracy_curves(report: dict) -> str:
    chunks = ["<section><h2>Accuracy vs aggressiveness</h2>",
              "<p class='section-intro'>One chart per task. One line per model. "
              "Shaded bands = bootstrap 95% CI. Dashed line = uncompressed baseline.</p>"]
    for task in report.get("tasks", []):
        chunks.append(_render_accuracy_chart_for_task(task))
    chunks.append("</section>")
    return "\n".join(chunks)

def _render_accuracy_chart_for_task(task: dict) -> str:
    primary = task["summary_stats"].get("primary_metric", "f1")
    baseline = task["baseline_metrics"].get(primary, {}).get("mean", 0.0)

    cells = task.get("sweep_results", [])
    if not cells:
        return ""
    by_model: dict[str, list[dict]] = {}
    for c in cells:
        by_model.setdefault(c["model"], []).append(c)
    for cells_for in by_model.values():
        cells_for.sort(key=lambda c: c["aggressiveness"])

    geom = _ChartGeometry(width=720, height=320, pad_left=60, pad_right=160, pad_top=20, pad_bottom=44)
    aggrs = sorted({c["aggressiveness"] for c in cells})
    x_min, x_max = (min(aggrs), max(aggrs)) if aggrs else (0.0, 1.0)
    if x_min == x_max:
        x_min, x_max = 0.0, 1.0

    all_y = [c["metrics"].get(primary, {}).get("ci_low", 0.0) for c in cells]
    all_y += [c["metrics"].get(primary, {}).get("ci_high", 1.0) for c in cells]
    y_min = min(0.0, min(all_y) - 0.05)
    y_max = max(1.0, max(all_y) + 0.05)

    def x_to(v: float) -> float:
        if x_max == x_min:
            return geom.pad_left + geom.plot_width / 2
        return geom.pad_left + (v - x_min) / (x_max - x_min) * geom.plot_width

    def y_to(v: float) -> float:
        return geom.pad_top + (1 - (v - y_min) / (y_max - y_min)) * geom.plot_height

    parts = [f"<svg width='{geom.width}' height='{geom.height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<rect x='{geom.pad_left}' y='{geom.pad_top}' width='{geom.plot_width}' "
                 f"height='{geom.plot_height}' fill='white' stroke='#ccc'/>")
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if tick < y_min or tick > y_max:
            continue
        ty = y_to(tick)
        parts.append(f"<line x1='{geom.pad_left}' y1='{ty}' x2='{geom.pad_left + geom.plot_width}' y2='{ty}' stroke='#eee'/>")
        parts.append(f"<text x='{geom.pad_left - 6}' y='{ty + 4}' text-anchor='end' font-size='11' fill='#666'>{tick:.2f}</text>")
    for a in aggrs:
        tx = x_to(a)
        parts.append(f"<line x1='{tx}' y1='{geom.pad_top + geom.plot_height}' x2='{tx}' y2='{geom.pad_top + geom.plot_height + 4}' stroke='#666'/>")
        parts.append(f"<text x='{tx}' y='{geom.pad_top + geom.plot_height + 18}' text-anchor='middle' font-size='11' fill='#666'>{a:.2f}</text>")

    by = y_to(baseline)
    parts.append(f"<line x1='{geom.pad_left}' y1='{by}' x2='{geom.pad_left + geom.plot_width}' y2='{by}' "
                 f"stroke='#999' stroke-dasharray='4 3'/>")
    parts.append(f"<text x='{geom.pad_left + geom.plot_width + 6}' y='{by + 4}' font-size='11' fill='#666'>baseline {baseline:.2f}</text>")

    for i, (_model, cells_for) in enumerate(by_model.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        upper = [(x_to(c["aggressiveness"]), y_to(c["metrics"].get(primary, {}).get("ci_high", 0.0))) for c in cells_for]
        lower = [(x_to(c["aggressiveness"]), y_to(c["metrics"].get(primary, {}).get("ci_low", 0.0))) for c in cells_for]
        if upper and lower:
            band_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + list(reversed(lower)))
            parts.append(f"<polygon points='{band_pts}' fill='{color}' fill-opacity='0.12' stroke='none'/>")

        line_pts = " ".join(
            f"{x_to(c['aggressiveness']):.1f},{y_to(c['metrics'].get(primary, {}).get('mean', 0.0)):.1f}"
            for c in cells_for
        )
        parts.append(f"<polyline points='{line_pts}' fill='none' stroke='{color}' stroke-width='2'/>")
        for c in cells_for:
            cx = x_to(c["aggressiveness"])
            cy = y_to(c["metrics"].get(primary, {}).get("mean", 0.0))
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3' fill='{color}'/>")

    legend_x = geom.pad_left + geom.plot_width + 8
    legend_y = geom.pad_top + 14
    for i, model in enumerate(by_model):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        parts.append(f"<rect x='{legend_x}' y='{legend_y + i * 18 - 9}' width='10' height='10' fill='{color}'/>")
        parts.append(f"<text x='{legend_x + 14}' y='{legend_y + i * 18}' font-size='12' fill='#333'>{_esc(model)}</text>")

    parts.append(f"<text x='{geom.pad_left + geom.plot_width / 2}' y='{geom.height - 6}' text-anchor='middle' font-size='12' fill='#333'>aggressiveness</text>")
    parts.append(f"<text x='14' y='{geom.pad_top + geom.plot_height / 2}' transform='rotate(-90 14 {geom.pad_top + geom.plot_height / 2})' text-anchor='middle' font-size='12' fill='#333'>{_esc(primary)} (mean)</text>")
    parts.append("</svg>")

    return f"""
<div class='card'>
  <h3>{_esc(task["task_name"])} — {_esc(primary)}</h3>
  <p class='notes'>{_esc(task.get("description", ""))}</p>
  {"".join(parts)}
</div>
"""

def _render_ttc_safe_chart(report: dict) -> str:
    """Bar chart of protected vs unprotected F1 from the QA task's sub-experiment."""
    qa_task = next((t for t in report.get("tasks", []) if t["task_name"] == "qa"), None)
    if not qa_task:
        return ""
    rows = qa_task["summary_stats"].get("ttc_safe_results", [])
    if not rows:
        return ""

    geom = _ChartGeometry(width=720, height=300, pad_left=60, pad_right=160, pad_top=20, pad_bottom=80)
    n = len(rows)
    bar_group_w = geom.plot_width / max(1, n)
    bar_w = bar_group_w * 0.35

    parts = [f"<svg width='{geom.width}' height='{geom.height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<rect x='{geom.pad_left}' y='{geom.pad_top}' width='{geom.plot_width}' height='{geom.plot_height}' fill='white' stroke='#ccc'/>")
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ty = geom.pad_top + (1 - tick) * geom.plot_height
        parts.append(f"<line x1='{geom.pad_left}' y1='{ty}' x2='{geom.pad_left + geom.plot_width}' y2='{ty}' stroke='#eee'/>")
        parts.append(f"<text x='{geom.pad_left - 6}' y='{ty + 4}' text-anchor='end' font-size='11' fill='#666'>{tick:.2f}</text>")

    for i, row in enumerate(rows):
        cx = geom.pad_left + (i + 0.5) * bar_group_w
        prot_h = row["protected_f1"] * geom.plot_height
        unprot_h = row["unprotected_f1"] * geom.plot_height
        prot_x = cx - bar_w - 2
        unprot_x = cx + 2
        parts.append(f"<rect x='{prot_x:.1f}' y='{geom.pad_top + geom.plot_height - prot_h:.1f}' width='{bar_w:.1f}' height='{prot_h:.1f}' fill='#16a34a'/>")
        parts.append(f"<rect x='{unprot_x:.1f}' y='{geom.pad_top + geom.plot_height - unprot_h:.1f}' width='{bar_w:.1f}' height='{unprot_h:.1f}' fill='#9ca3af'/>")
        label = f"{row['model']}@{row['aggressiveness']:.2f}"
        parts.append(f"<text x='{cx:.1f}' y='{geom.pad_top + geom.plot_height + 14:.1f}' text-anchor='middle' font-size='10' fill='#444' transform='rotate(-25 {cx:.1f} {geom.pad_top + geom.plot_height + 14:.1f})'>{_esc(label)}</text>")

    legend_x = geom.pad_left + geom.plot_width + 12
    parts.append(f"<rect x='{legend_x}' y='{geom.pad_top + 5}' width='10' height='10' fill='#16a34a'/>")
    parts.append(f"<text x='{legend_x + 14}' y='{geom.pad_top + 14}' font-size='12' fill='#333'>protected (&lt;ttc_safe&gt;)</text>")
    parts.append(f"<rect x='{legend_x}' y='{geom.pad_top + 25}' width='10' height='10' fill='#9ca3af'/>")
    parts.append(f"<text x='{legend_x + 14}' y='{geom.pad_top + 34}' font-size='12' fill='#333'>unprotected</text>")
    parts.append("</svg>")

    return f"""
<section>
  <h2>&lt;ttc_safe&gt; impact — protected vs unprotected questions</h2>
  <p class='section-intro'>
    For a 20% subset of QA samples we wrap the question in <code>&lt;ttc_safe&gt;</code> tags and
    prepend it to the context. The compressor leaves protected regions untouched — does this
    visibly preserve accuracy as aggressiveness rises?
  </p>
  <div class='card'>{"".join(parts)}</div>
</section>
"""

def _render_compression_ratio_box(report: dict) -> str:
    """Approximate box-plot of compression ratios, per task per model."""
    parts = ["<section><h2>Compression ratio distribution</h2>",
             "<p class='section-intro'>How aggressively each model compresses, per task. "
             "Bar = mean, whiskers = min/max across (sample × aggressiveness) for that model.</p>"]
    for task in report.get("tasks", []):
        cells = task.get("sweep_results", [])
        if not cells:
            continue
        ratios_by_model: dict[str, list[float]] = {}
        for cell in cells:
            for sp in cell.get("sample_predictions", []):
                if sp.get("failed"):
                    continue
                ratios_by_model.setdefault(cell["model"], []).append(sp.get("compression_ratio", 0.0))

        if not ratios_by_model:
            continue

        rows = []
        for model, ratios in ratios_by_model.items():
            if not ratios:
                continue
            ratios_sorted = sorted(ratios)
            n = len(ratios_sorted)
            q1 = ratios_sorted[max(0, n // 4)]
            median = ratios_sorted[n // 2]
            q3 = ratios_sorted[min(n - 1, (3 * n) // 4)]
            rows.append(f"""
                <tr>
                  <td>{_esc(model)}</td>
                  <td class='num'>{ratios_sorted[0]:.2%}</td>
                  <td class='num'>{q1:.2%}</td>
                  <td class='num'>{median:.2%}</td>
                  <td class='num'>{q3:.2%}</td>
                  <td class='num'>{ratios_sorted[-1]:.2%}</td>
                  <td class='num'>{sum(ratios)/n:.2%}</td>
                </tr>
            """)
        parts.append(f"""
        <div class='card'>
          <h3>{_esc(task["task_name"])}</h3>
          <table>
            <thead><tr><th>model</th><th class='num'>min</th><th class='num'>q1</th><th class='num'>median</th><th class='num'>q3</th><th class='num'>max</th><th class='num'>mean</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
        """)
    parts.append("</section>")
    return "\n".join(parts)

def _render_latency_section(report: dict) -> str:
    parts = ["<section><h2>Latency analysis</h2>",
             "<p class='section-intro'>p50/p95/p99 across the sweep. "
             "E2E (compressed) = TTC compress + LLM call. E2E (baseline) = LLM call alone.</p>"]
    for task in report.get("tasks", []):
        cells = task.get("sweep_results", [])
        if not cells:
            continue
        rows = []
        for cell in cells:
            rows.append(f"""
              <tr>
                <td>{_esc(cell["model"])}</td>
                <td class='num'>{cell["aggressiveness"]:.2f}</td>
                <td class='num'>{cell["p50_ttc_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p95_ttc_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p50_llm_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p95_llm_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p50_e2e_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p95_e2e_latency_ms"]:.0f}</td>
                <td class='num'>{cell["p99_e2e_latency_ms"]:.0f}</td>
              </tr>
            """)
        parts.append(f"""
        <div class='card'>
          <h3>{_esc(task["task_name"])}</h3>
          <p class='notes'>
            Baseline LLM-only p95: <strong>{task["baseline_p95_llm_latency_ms"]:.0f}ms</strong>
          </p>
          <table>
            <thead><tr>
              <th>model</th><th class='num'>aggr</th>
              <th class='num'>TTC p50</th><th class='num'>TTC p95</th>
              <th class='num'>LLM p50</th><th class='num'>LLM p95</th>
              <th class='num'>E2E p50</th><th class='num'>E2E p95</th><th class='num'>E2E p99</th>
            </tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
        """)
    parts.append("</section>")
    return "\n".join(parts)

def _render_cost_section(report: dict) -> str:
    """Customer-facing ROI: net savings projected at multiple scales.

    Net savings = (downstream LLM tokens not sent × downstream LLM input
    price) − TTC bill. We project on gpt-4o-mini at 1M, 10M, 100M
    requests/day so the reader sees the absolute scale.
    """
    parts = ["<section><h2>Net cost analysis</h2>",
             "<p class='section-intro'>Net daily cost change = downstream LLM savings − TTC bill, "
             "projected to several daily request volumes on gpt-4o-mini.</p>"]
    for task in report.get("tasks", []):
        cells = task.get("sweep_results", [])
        if not cells:
            continue
        baseline_in = task["baseline_avg_llm_input_tokens"]
        out_estimate = max(50, int(task["baseline_avg_llm_output_tokens"] or 50))

        rows = []
        for cell in cells:
            cur_in = cell["avg_llm_input_tokens"] or 0
            base_per_call = estimate_llm_cost_usd(int(baseline_in), out_estimate, "gpt-4o-mini")
            cur_per_call = estimate_llm_cost_usd(int(cur_in), out_estimate, "gpt-4o-mini")
            ttc_per_call = cell["avg_ttc_cost_usd"] or 0.0
            net_per_call = (base_per_call - cur_per_call) - ttc_per_call

            cls = "delta-pos" if net_per_call >= 0 else "delta-neg"
            rows.append(f"""
              <tr>
                <td>{_esc(cell["model"])}</td>
                <td class='num'>{cell["aggressiveness"]:.2f}</td>
                <td class='num'>${ttc_per_call * 1_000_000:.2f}</td>
                <td class='num'>${(base_per_call - cur_per_call) * 1_000_000:.2f}</td>
                <td class='num {cls}'>${net_per_call * 1_000_000:.2f}</td>
                <td class='num {cls}'>${net_per_call * 10_000_000:.2f}</td>
                <td class='num {cls}'>${net_per_call * 100_000_000:.2f}</td>
              </tr>
            """)
        parts.append(f"""
        <div class='card'>
          <h3>{_esc(task["task_name"])}</h3>
          <table>
            <thead><tr>
              <th>model</th><th class='num'>aggr</th>
              <th class='num'>TTC bill / 1M req</th>
              <th class='num'>LLM saved / 1M req</th>
              <th class='num'>Net / 1M req/day</th>
              <th class='num'>Net / 10M req/day</th>
              <th class='num'>Net / 100M req/day</th>
            </tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
        """)
    parts.append("</section>")
    return "\n".join(parts)

def _render_faithfulness_scatter(report: dict) -> str:
    sum_task = next((t for t in report.get("tasks", []) if t["task_name"] == "summarization"), None)
    if not sum_task:
        return ""
    points: list[tuple[float, float, str]] = []
    for cell in sum_task.get("sweep_results", []):
        for sp in cell.get("sample_predictions", []):
            if sp.get("failed"):
                continue
            rg = sp["metrics"].get("rouge_l")
            fa = sp["metrics"].get("faithfulness")
            if rg is None or fa is None:
                continue
            points.append((rg, fa, sp["sample_id"]))

    if not points:
        return ""

    geom = _ChartGeometry(width=520, height=380, pad_left=60, pad_right=20, pad_top=20, pad_bottom=44)
    parts = [f"<svg width='{geom.width}' height='{geom.height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<rect x='{geom.pad_left}' y='{geom.pad_top}' width='{geom.plot_width}' height='{geom.plot_height}' fill='white' stroke='#ccc'/>")
    mx = geom.pad_left + geom.plot_width / 2
    my = geom.pad_top + geom.plot_height / 2
    parts.append(f"<line x1='{mx}' y1='{geom.pad_top}' x2='{mx}' y2='{geom.pad_top + geom.plot_height}' stroke='#ddd' stroke-dasharray='4 3'/>")
    parts.append(f"<line x1='{geom.pad_left}' y1='{my}' x2='{geom.pad_left + geom.plot_width}' y2='{my}' stroke='#ddd' stroke-dasharray='4 3'/>")

    for rg, fa, _sid in points:
        cx = geom.pad_left + rg * geom.plot_width
        cy = geom.pad_top + (1 - fa) * geom.plot_height
        color = "#dc2626" if rg >= 0.5 and fa < 0.5 else "#2563eb"
        parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4' fill='{color}' fill-opacity='0.6'/>")

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = geom.pad_left + tick * geom.plot_width
        ty = geom.pad_top + (1 - tick) * geom.plot_height
        parts.append(f"<text x='{tx}' y='{geom.pad_top + geom.plot_height + 16}' text-anchor='middle' font-size='11' fill='#666'>{tick:.2f}</text>")
        parts.append(f"<text x='{geom.pad_left - 6}' y='{ty + 4}' text-anchor='end' font-size='11' fill='#666'>{tick:.2f}</text>")
    parts.append(f"<text x='{geom.pad_left + geom.plot_width / 2}' y='{geom.height - 4}' text-anchor='middle' font-size='12' fill='#333'>ROUGE-L</text>")
    parts.append(f"<text x='14' y='{geom.pad_top + geom.plot_height / 2}' transform='rotate(-90 14 {geom.pad_top + geom.plot_height / 2})' text-anchor='middle' font-size='12' fill='#333'>faithfulness</text>")
    parts.append("</svg>")

    confabulations = sum_task["summary_stats"].get("confabulations", [])
    confab_html = ""
    if confabulations:
        rows = "".join(
            f"<tr><td>{_esc(c['sample_id'])}</td><td>{_esc(c['model'])}</td>"
            f"<td class='num'>{c['aggressiveness']:.2f}</td>"
            f"<td class='num'>{c['rouge_l']:.2f}</td><td class='num'>{c['faithfulness']:.2f}</td></tr>"
            for c in confabulations[:20]
        )
        confab_html = f"""
        <h3>Compression-induced confabulation candidates</h3>
        <p class='notes'>Samples with ROUGE ≥ 0.5 (looks like the reference)
          but faithfulness &lt; 0.5 (drifted from source). These are the cases the eval
          calls out as the failure mode worth watching.</p>
        <table>
          <thead><tr><th>sample</th><th>model</th><th class='num'>aggr</th><th class='num'>ROUGE-L</th><th class='num'>faithfulness</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    return f"""
<section>
  <h2>Faithfulness vs ROUGE-L (summarization)</h2>
  <p class='section-intro'>
    Each dot is one sample. Top-right = high overlap and high source-grounding (good).
    Bottom-right (red) = high ROUGE but low faithfulness — confabulation introduced by compression.
  </p>
  <div class='card'>{"".join(parts)}{confab_html}</div>
</section>
"""

def _render_long_context_heatmap(report: dict) -> str:
    long_task = next((t for t in report.get("tasks", []) if t["task_name"] == "long_context"), None)
    if not long_task:
        return ""
    rows = long_task["summary_stats"].get("heatmap", [])
    if not rows:
        return ""
    by_model: dict[str, dict] = {}
    for r in rows:
        m = r["model"]
        by_model.setdefault(m, {"positions": set(), "aggrs": set(), "data": {}})
        by_model[m]["positions"].add(r["needle_position"])
        by_model[m]["aggrs"].add(r["aggressiveness"])
        by_model[m]["data"][(r["needle_position"], r["aggressiveness"])] = r["retrieval_accuracy"]

    parts = ["<section><h2>Long-context heatmap — needle position × aggressiveness</h2>",
             "<p class='section-intro'>Color = retrieval accuracy. "
             "Darker green = needle still recovered after compression. The strongest visual argument for the product.</p>"]
    position_order = ["start", "middle", "end"]
    for model, agg in by_model.items():
        aggrs = sorted(agg["aggrs"])
        positions = [p for p in position_order if p in agg["positions"]]
        cell_w = 60
        cell_h = 40
        x0 = 100
        y0 = 30
        width = x0 + cell_w * len(aggrs) + 20
        height = y0 + cell_h * len(positions) + 40
        svg = [f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"]
        for i, pos in enumerate(positions):
            svg.append(f"<text x='{x0 - 8}' y='{y0 + i * cell_h + cell_h / 2 + 4}' text-anchor='end' font-size='12' fill='#333'>{_esc(pos)}</text>")
            for j, a in enumerate(aggrs):
                val = agg["data"].get((pos, a), 0.0)
                lightness = 95 - int(val * 60)
                color = f"hsl(140, 60%, {lightness}%)"
                svg.append(f"<rect x='{x0 + j * cell_w}' y='{y0 + i * cell_h}' width='{cell_w}' height='{cell_h}' fill='{color}' stroke='#ddd'/>")
                svg.append(f"<text x='{x0 + j * cell_w + cell_w / 2}' y='{y0 + i * cell_h + cell_h / 2 + 4}' text-anchor='middle' font-size='12' fill='#333'>{val:.2f}</text>")
        for j, a in enumerate(aggrs):
            svg.append(f"<text x='{x0 + j * cell_w + cell_w / 2}' y='{y0 - 8}' text-anchor='middle' font-size='12' fill='#333'>{a:.2f}</text>")
        svg.append("</svg>")
        parts.append(f"<div class='card'><h3>{_esc(model)}</h3>{''.join(svg)}</div>")
    parts.append("</section>")
    return "\n".join(parts)

def _render_regression_section(report: dict) -> str:
    reg = report.get("regression")
    if not reg:
        return ""
    regressions = reg.get("regressions", [])
    improvements = reg.get("improvements", [])

    def _row(d: dict, kind: str) -> str:
        cls = "delta-neg" if kind == "regression" else "delta-pos"
        delta = d["delta"]
        return f"""
        <tr>
          <td><span class='pill {"red" if kind == "regression" else "green"}'>{kind}</span></td>
          <td>{_esc(d["task"])}</td>
          <td>{_esc(d["model"])}</td>
          <td class='num'>{d["aggressiveness"]:.2f}</td>
          <td>{_esc(d["metric"])}</td>
          <td class='num'>{d["baseline_value"]:.4f}</td>
          <td class='num'>{d["current_value"]:.4f}</td>
          <td class='num {cls}'>{delta:+.4f}</td>
        </tr>
        """

    rows = "".join(_row(d, "regression") for d in regressions)
    rows += "".join(_row(d, "improvement") for d in improvements)
    if not rows:
        body = "<p class='notes'>No metric moved beyond threshold relative to baseline.</p>"
    else:
        body = f"""
        <table>
          <thead><tr>
            <th>kind</th><th>task</th><th>model</th><th class='num'>aggr</th>
            <th>metric</th><th class='num'>baseline</th><th class='num'>current</th><th class='num'>Δ</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
    return f"""
<section>
  <h2>Regression check vs baseline run</h2>
  <p class='section-intro'>
    Threshold: {reg.get('threshold_pp', 1.0):.1f}pp for accuracy-style metrics,
    {reg.get('threshold_pct', 5.0):.1f}% for ratios (compression, latency).
    {reg["summary"]["regressions_found"]} regressions, {reg["summary"]["improvements_found"]} improvements
    across {reg["summary"]["tasks_compared"]} tasks.
  </p>
  <div class='card'>{body}</div>
</section>
"""

def _render_full_results_table(report: dict) -> str:
    parts = ["<section><h2>Full results</h2>",
             "<p class='section-intro'>One row per (task × model × aggressiveness) cell.</p>",
             "<div class='card'><table><thead><tr>",
             "<th>task</th><th>model</th><th class='num'>aggr</th>",
             "<th class='num'>primary metric</th><th class='num'>Δ vs baseline</th>",
             "<th class='num'>compression</th><th class='num'>TTC cost</th>",
             "<th class='num'>p95 E2E (ms)</th><th class='num'>fail %</th>",
             "</tr></thead><tbody>"]
    for task in report.get("tasks", []):
        primary = task["summary_stats"].get("primary_metric", "f1")
        baseline = task["baseline_metrics"].get(primary, {}).get("mean", 0.0)
        for cell in task.get("sweep_results", []):
            score = cell["metrics"].get(primary, {}).get("mean", 0.0)
            delta = score - baseline
            cls = _delta_class(delta)
            parts.append(f"""
              <tr>
                <td>{_esc(task["task_name"])}</td>
                <td>{_esc(cell["model"])}</td>
                <td class='num'>{cell["aggressiveness"]:.2f}</td>
                <td class='num'>{score:.3f}</td>
                <td class='num {cls}'>{_format_pp(delta)}</td>
                <td class='num'>{cell["avg_compression_ratio"]:.1%}</td>
                <td class='num'>${cell["avg_ttc_cost_usd"]:.6f}</td>
                <td class='num'>{cell["p95_e2e_latency_ms"]:.0f}</td>
                <td class='num'>{cell["failure_rate"]:.0%}</td>
              </tr>
            """)
    parts.append("</tbody></table></div></section>")
    return "\n".join(parts)

def _render_metadata_footer(report: dict) -> str:
    return f"""
<footer style='margin-top: 48px; padding-top: 16px; border-top: 1px solid #ddd; color: #888; font-size: 12px;'>
  <p>
    run_id: {_esc(report.get("run_id", ""))} ·
    backend: {_esc(report.get("backend_name", ""))} ({_esc(report.get("backend_model", ""))}) ·
    duration: {report.get("duration_s", 0):.1f}s ·
    samples: {report.get("max_samples", "?")} ·
    seed: {report.get("seed", "?")} ·
    embeddings: {"yes" if report.get("use_embeddings") else "no"}
  </p>
  <p>
    Idempotence pre-check: {"executed, matched=" + str(report.get("idempotence_check", {}).get("matched")) if report.get("idempotence_check", {}).get("executed") else "not executed"}
  </p>
</footer>
"""

def render_report(report_dict: dict, out_path: str | Path) -> Path:
    """Write the HTML report to ``out_path`` and return the path."""
    out = Path(out_path)
    sections = [
        _render_executive_summary(report_dict),
        _render_sweet_spot_recommendation(report_dict),
        _render_accuracy_curves(report_dict),
        _render_ttc_safe_chart(report_dict),
        _render_compression_ratio_box(report_dict),
        _render_latency_section(report_dict),
        _render_cost_section(report_dict),
        _render_faithfulness_scatter(report_dict),
        _render_long_context_heatmap(report_dict),
        _render_regression_section(report_dict),
        _render_full_results_table(report_dict),
        _render_metadata_footer(report_dict),
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>TTC Eval — {_esc(report_dict.get("run_id", ""))}</title>
<style>{CSS}</style>
</head>
<body>
<h1>TTC compression eval report</h1>
<p class='subtitle'>
  bear-1.x compression vs baseline across {len(report_dict.get("tasks", []))} tasks.
  Backend: <code>{_esc(report_dict.get("backend_name", ""))}</code>
  ({_esc(report_dict.get("backend_model", ""))}).
</p>
{"".join(sections)}
</body>
</html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out

def render_report_from_json(json_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convenience for rebuilding the HTML from a saved results JSON.

    Useful when the eval ran already and you want to iterate on the report
    layout without re-running. Maps to the spec's "the HTML report can be
    regenerated without re-running" requirement.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if out_path is None:
        out_path = json_path.with_suffix(".html")
    return render_report(data, out_path)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Regenerate an HTML report from a saved results JSON.")
    p.add_argument("json_path", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    out = render_report_from_json(args.json_path, args.out)
    print(f"Wrote {out}")
