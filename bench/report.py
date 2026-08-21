"""Assemble the graded numbers into one report — bench/results/report.md.

Reads the CSVs the other bench scripts wrote (latency_full.csv,
retrieval_ablation.csv) and renders the percentile table + ablation table +
refusal breakdown as markdown, ready to paste into docs or the README.

Usage:
    uv run --extra dev python -m bench.report
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

RESULTS = Path("bench/results")
BUDGET_MS = 200.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main() -> None:
    lines = ["# Benchmark report", ""]

    latency_csv = RESULTS / "latency_full.csv"
    if latency_csv.exists():
        rows = list(csv.DictReader(latency_csv.open(encoding="utf-8")))
        core = [float(r["t_core_ms"]) for r in rows if r["t_core_ms"] and r["verdict"] != "ERROR"]
        wall = [float(r["wall_ms"]) for r in rows if r["verdict"] != "ERROR"]
        answered = sum(1 for r in rows if r["verdict"] == "ANSWERED")
        refused_out = sum(1 for r in rows if r["verdict"] == "REFUSED" and r["domain"] == "out")
        refused_in = sum(1 for r in rows if r["verdict"] == "REFUSED" and r["domain"] == "in")

        lines += [
            "## Latency (`POST /ask`, full pipeline)",
            "",
            f"{len(core)} successful requests, budget {BUDGET_MS:.0f} ms",
            "",
            "| metric | avg | p50 | p70 | p95 | p99 | p100 |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, vals in [("t_core", core), ("t_e2e", wall)]:
            if vals:
                lines.append(
                    f"| {name} | {statistics.mean(vals):.1f} | {percentile(vals, 50):.1f} | "
                    f"{percentile(vals, 70):.1f} | {percentile(vals, 95):.1f} | "
                    f"{percentile(vals, 99):.1f} | {max(vals):.1f} |"
                )
        over = sum(1 for v in core if v > BUDGET_MS)
        lines += [
            "",
            f"- **Degradation rate:** {over}/{len(core)} ({100 * over / max(len(core), 1):.1f}%) over budget",
            f"- Answered: {answered} · refused out-of-domain (correct): {refused_out} · "
            f"over-refused in-domain: {refused_in}",
            f"- Over-refusal rate: {100 * refused_in / max(refused_in + len(answered), 1):.1f}% of in-domain queries",
            "",
        ]

    ablation_csv = RESULTS / "retrieval_ablation.csv"
    if ablation_csv.exists():
        rows = list(csv.DictReader(ablation_csv.open(encoding="utf-8")))
        best_arm = max(rows, key=lambda r: float(r["ndcg@10"]))
        lines += [
            "## Chunking-strategy ablation",
            "",
            "Recall@10 / nDCG@10 / MRR against the dataset's own `is_selected` labels.",
            "",
            "| arm | recall@10 | nDCG@10 | MRR |",
            "|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: -float(r["ndcg@10"])):
            marker = " **(champion)**" if r["arm"] == best_arm["arm"] else ""
            arm = r["arm"].replace(" <- champion", "")
            lines.append(f"| {arm}{marker} | {r['recall@10']} | {r['ndcg@10']} | {r['mrr']} |")
        lines += ["", f"Champion by nDCG@10: **{best_arm['arm'].replace(' <- champion', '')}**", ""]

    out = RESULTS / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
