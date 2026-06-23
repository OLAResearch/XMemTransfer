"""Comprehensive statistical analysis of Engram transfer experiments.

Generates:
  - analysis-output/analysis-report.md
  - analysis-output/results-tables.md  (LaTeX-ready)
  - Console summary with significance tests
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis-output"


def load_results(tier_dir: str) -> dict[str, list[dict]]:
    """Load results grouped by condition. Returns {condition: [result_dicts]}."""
    base = RESULTS_DIR / tier_dir
    if not base.exists():
        return {}
    grouped = defaultdict(list)
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        results_file = d / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        # Parse condition from dirname: e.g. "transferred_seed42" -> "transferred"
        name = d.name
        parts = name.rsplit("_seed", 1)
        condition = parts[0] if len(parts) == 2 else name
        grouped[condition].append(data)
    return dict(grouped)


def load_test_ppls(tier_dir: str) -> dict[str, list[list[float]]]:
    """Load per-batch test PPLs for bootstrap CI."""
    base = RESULTS_DIR / tier_dir
    if not base.exists():
        return {}
    grouped = defaultdict(list)
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        ppls_file = d / "test_ppls.json"
        if not ppls_file.exists():
            continue
        with open(ppls_file) as f:
            ppls = json.load(f)
        name = d.name
        parts = name.rsplit("_seed", 1)
        condition = parts[0] if len(parts) == 2 else name
        grouped[condition].append(ppls)
    return dict(grouped)


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values)
    means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lo), float(hi)


def paired_t_test(a_ppls: list[list[float]], b_ppls: list[list[float]]) -> dict:
    """Paired t-test on per-seed mean PPLs."""
    a_means = [np.mean(p) for p in a_ppls]
    b_means = [np.mean(p) for p in b_ppls]
    if len(a_means) < 2 or len(b_means) < 2:
        return {"t": float("nan"), "p": float("nan"), "n": 0}
    t_stat, p_val = stats.ttest_rel(a_means, b_means) if len(a_means) == len(b_means) else stats.ttest_ind(a_means, b_means)
    diff = np.array(a_means) - np.array(b_means)
    cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    return {"t": float(t_stat), "p": float(p_val), "cohens_d": cohens_d, "n": len(a_means)}


def analyze_tier(name: str, tier_dir: str, report_lines: list[str], table_lines: list[str]):
    """Analyze a single tier and append to report."""
    results = load_results(tier_dir)
    ppls_data = load_test_ppls(tier_dir)
    if not results:
        report_lines.append(f"\n## {name}\n\nNo results found in `{tier_dir}/`.\n")
        return

    report_lines.append(f"\n## {name}\n")

    # Summary table
    rows = []
    for cond in sorted(results.keys()):
        runs = results[cond]
        ppl_means = [r["test_ppl_mean"] for r in runs]
        gate_means = [r["gate_stats"]["mean"] for r in runs if "gate_stats" in r and r["gate_stats"]]
        n = len(runs)
        ppl_avg = np.mean(ppl_means)
        ppl_std = np.std(ppl_means, ddof=1) if n > 1 else 0
        gate_avg = np.mean(gate_means) if gate_means else float("nan")
        rows.append((cond, n, ppl_avg, ppl_std, gate_avg))

    rows.sort(key=lambda x: x[2])  # Sort by PPL ascending

    report_lines.append("| Condition | Seeds | PPL (mean±std) | Gate mean | Rank |")
    report_lines.append("|-----------|-------|----------------|-----------|------|")
    table_lines.append(f"\n% {name}")
    for rank, (cond, n, ppl_avg, ppl_std, gate_avg) in enumerate(rows, 1):
        gate_str = f"{gate_avg:.3f}" if not np.isnan(gate_avg) else "—"
        report_lines.append(f"| {cond} | {n} | {ppl_avg:.2f}±{ppl_std:.2f} | {gate_str} | {rank} |")
        # LaTeX row
        table_lines.append(f"    {cond.replace('_', ' ')} & {n} & {ppl_avg:.2f}$\\pm${ppl_std:.2f} & {gate_str} \\\\")

    # Statistical tests: transferred vs each other condition
    if "transferred" in ppls_data:
        report_lines.append("\n### Statistical Tests (transferred vs others)\n")
        report_lines.append("| Comparison | t-stat | p-value | Cohen's d | Significant? |")
        report_lines.append("|------------|--------|---------|-----------|--------------|")
        for cond in sorted(ppls_data.keys()):
            if cond == "transferred":
                continue
            test = paired_t_test(ppls_data.get("transferred", []), ppls_data.get(cond, []))
            if np.isnan(test["p"]):
                continue
            sig = "Yes***" if test["p"] < 0.001 else "Yes**" if test["p"] < 0.01 else "Yes*" if test["p"] < 0.05 else "No"
            report_lines.append(
                f"| transferred vs {cond} | {test['t']:.3f} | {test['p']:.4f} | {test['cohens_d']:.3f} | {sig} |"
            )


def analyze_tier4_scaling(report_lines: list[str]):
    """Special analysis for tier4 scaling experiments."""
    results = load_results("tier4")
    if not results:
        return

    report_lines.append("\n## Tier 4 — Scaling Analysis\n")

    # Group by base condition and token budget
    scaling = defaultdict(dict)
    for cond, runs in results.items():
        # Parse: transferred_5M, train_from_scratch_20M, etc.
        parts = cond.rsplit("_", 1)
        if len(parts) == 2 and parts[1].endswith("M"):
            base_cond = parts[0]
            tokens = parts[1]
            ppl_means = [r["test_ppl_mean"] for r in runs]
            scaling[base_cond][tokens] = {
                "mean": np.mean(ppl_means),
                "std": np.std(ppl_means, ddof=1) if len(ppl_means) > 1 else 0,
                "n": len(runs),
                "values": ppl_means,
            }

    token_order = ["5M", "20M", "50M"]
    report_lines.append("| Condition | 5M tokens | 20M tokens | 50M tokens |")
    report_lines.append("|-----------|-----------|------------|------------|")
    for base_cond in sorted(scaling.keys()):
        row = f"| {base_cond} |"
        for tok in token_order:
            d = scaling[base_cond].get(tok)
            if d:
                row += f" {d['mean']:.2f}±{d['std']:.2f} |"
            else:
                row += " — |"
        report_lines.append(row)

    # Flag divergence
    if "transferred" in scaling and "50M" in scaling["transferred"]:
        t50 = scaling["transferred"]["50M"]
        if t50["mean"] > 30:
            report_lines.append(
                f"\n**WARNING**: `transferred_50M` PPL={t50['mean']:.1f} indicates training divergence. "
                f"Frozen memory over-dominates with extended training. "
                f"Consider LR decay, early stopping, or gate regularization."
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    report = ["# Engram Transfer — Statistical Analysis Report\n"]
    tables = ["% LaTeX table rows for paper\n"]

    # Tier 1
    analyze_tier("Tier 1 — Same-Tokenizer Transfer (Pythia-160M → 410M)", "tier1", report, tables)

    # Tier 2
    analyze_tier("Tier 2 — Cross-Tokenizer Transfer (Pythia-160M → TinyLlama-1.1B)", "tier2", report, tables)

    # Tier 3
    analyze_tier("Tier 3 — Full Ablation (Pythia-160M → 410M, 8 conditions)", "tier3", report, tables)

    # Tier 4
    analyze_tier("Tier 4 — Scaling (adaptor training tokens)", "tier4", report, tables)
    analyze_tier4_scaling(report)

    # QWen (if available)
    analyze_tier("QWen 3.5 — Same-Tokenizer Transfer (0.8B → 4B)", "qwen35_tier1", report, tables)
    analyze_tier("QWen 3.5 — Scaling (0.8B → 2B/9B)", "qwen35_scaling", report, tables)

    # Write outputs
    report_text = "\n".join(report)
    tables_text = "\n".join(tables)

    (OUTPUT_DIR / "analysis-report.md").write_text(report_text)
    (OUTPUT_DIR / "results-tables.md").write_text(tables_text)

    print(report_text)
    print(f"\n--- Reports saved to {OUTPUT_DIR}/ ---")


if __name__ == "__main__":
    main()
