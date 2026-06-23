"""Generate all paper figures and tables from experiment results.

Produces:
  - Table 2: Main results (Tier 1 + Tier 2)
  - Table 3: Ablation table (8 conditions)
  - Figure 2: Learning curves (val PPL vs step)
  - Figure 4: Scaling (PPL vs training tokens)
  - Table 4: CKA similarity matrix

Usage:
    uv run python scripts/generate_figures.py --results-root results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

SEEDS = [42, 137, 2024]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    return parser.parse_args()


def load_condition_results(results_dir: Path, condition: str, seeds=SEEDS) -> dict:
    """Load results for a condition across seeds."""
    ppls = []
    all_results = []

    for seed in seeds:
        path = results_dir / f"{condition}_seed{seed}" / "results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            ppls.append(data["test_ppl_mean"])
            all_results.append(data)

    if not ppls:
        return None

    return {
        "condition": condition,
        "mean_ppl": float(np.mean(ppls)),
        "std_ppl": float(np.std(ppls)),
        "ppls": ppls,
        "n_seeds": len(ppls),
        "all_results": all_results,
    }


def generate_table2(results_root: Path, output_dir: Path):
    """Table 2: Main results for Tier 1 + Tier 2."""
    print("\n=== Table 2: Main Results ===")

    rows = []
    for tier, tier_name, model in [
        ("tier1", "Same-tok (Pythia-410M)", "Pythia-410M"),
        ("tier2", "Cross-tok (TinyLlama)", "TinyLlama-1.1B"),
    ]:
        tier_dir = results_root / tier
        if not tier_dir.exists():
            continue

        for condition in ["baseline", "transferred", "random_memory"]:
            data = load_condition_results(tier_dir, condition)
            if data:
                rows.append({
                    "tier": tier_name,
                    "condition": condition,
                    "ppl": f"{data['mean_ppl']:.2f} ± {data['std_ppl']:.2f}",
                    "mean": data["mean_ppl"],
                    "std": data["std_ppl"],
                })

    # Print LaTeX table
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Main results: test perplexity on WikiText-103.}")
    print("\\label{tab:main-results}")
    print("\\begin{tabular}{llc}")
    print("\\toprule")
    print("Setting & Condition & Test PPL ($\\downarrow$) \\\\")
    print("\\midrule")

    current_tier = None
    for row in rows:
        if row["tier"] != current_tier:
            if current_tier is not None:
                print("\\midrule")
            current_tier = row["tier"]
        cond_name = row["condition"].replace("_", " ").title()
        if row["condition"] == "transferred":
            print(f"{row['tier']} & \\textbf{{{cond_name}}} & \\textbf{{{row['ppl']}}} \\\\")
        else:
            print(f"{row['tier']} & {cond_name} & {row['ppl']} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    with open(output_dir / "table2_main_results.json", "w") as f:
        json.dump(rows, f, indent=2)


def generate_table3(results_root: Path, output_dir: Path):
    """Table 3: Ablation results (all 8 conditions)."""
    print("\n=== Table 3: Ablation Results ===")

    tier3_dir = results_root / "tier3"
    if not tier3_dir.exists():
        print("Tier 3 results not found")
        return

    conditions = [
        "baseline", "transferred", "random_memory", "permuted_keys",
        "no_gate", "train_from_scratch", "ffn_only", "affine_stitch",
    ]

    rows = []
    transferred_ppl = None

    for condition in conditions:
        data = load_condition_results(tier3_dir, condition)
        if data:
            if condition == "transferred":
                transferred_ppl = data["mean_ppl"]
            rows.append(data)

    # Print table
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Ablation study (Pythia-160M $\\to$ Pythia-410M, 20M tokens).}")
    print("\\label{tab:ablations}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Condition & Test PPL ($\\downarrow$) & $\\Delta$ vs. Transferred \\\\")
    print("\\midrule")

    for row in rows:
        cond = row["condition"]
        ppl_str = f"{row['mean_ppl']:.2f} ± {row['std_ppl']:.2f}"
        delta = row["mean_ppl"] - transferred_ppl if transferred_ppl else 0

        cond_display = cond.replace("_", " ").title()
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if delta == 0:
            delta_str = "---"

        if cond == "transferred":
            print(f"\\textbf{{{cond_display}}} & \\textbf{{{ppl_str}}} & {delta_str} \\\\")
        else:
            print(f"{cond_display} & {ppl_str} & {delta_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_figure2(results_root: Path, output_dir: Path):
    """Figure 2: Learning curves (val PPL vs step)."""
    print("\n=== Figure 2: Learning Curves ===")

    tier3_dir = results_root / "tier3"
    if not tier3_dir.exists():
        tier3_dir = results_root / "tier1"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = {
        "baseline": "#999999",
        "transferred": "#2196F3",
        "random_memory": "#FF9800",
        "train_from_scratch": "#4CAF50",
        "ffn_only": "#9C27B0",
    }
    labels = {
        "baseline": "Baseline",
        "transferred": "Transferred (ours)",
        "random_memory": "Random memory",
        "train_from_scratch": "Train from scratch",
        "ffn_only": "FFN only",
    }

    for condition in ["baseline", "transferred", "random_memory", "train_from_scratch", "ffn_only"]:
        all_curves = []
        for seed in SEEDS:
            results_path = tier3_dir / f"{condition}_seed{seed}" / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                val_ppls = data.get("val_ppls", [])
                if val_ppls:
                    steps = [v["step"] for v in val_ppls]
                    ppls = [v["val_ppl"] for v in val_ppls]
                    all_curves.append((steps, ppls))

        if not all_curves:
            continue

        # Find common steps
        common_steps = all_curves[0][0]
        ppls_matrix = []
        for steps, ppls in all_curves:
            if steps == common_steps:
                ppls_matrix.append(ppls)

        if ppls_matrix:
            ppls_arr = np.array(ppls_matrix)
            mean_ppls = ppls_arr.mean(axis=0)
            min_ppls = ppls_arr.min(axis=0)
            max_ppls = ppls_arr.max(axis=0)

            color = colors.get(condition, "#000")
            ax.plot(common_steps, mean_ppls, color=color, label=labels.get(condition, condition), linewidth=1.5)
            ax.fill_between(common_steps, min_ppls, max_ppls, alpha=0.15, color=color)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Learning Curves (3-seed min-max bands)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "figure2_learning_curves.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "figure2_learning_curves.png", bbox_inches="tight")
    print(f"Saved to {output_dir / 'figure2_learning_curves.pdf'}")


def generate_figure4(results_root: Path, output_dir: Path):
    """Figure 4: Scaling (PPL vs training tokens)."""
    print("\n=== Figure 4: Scaling ===")

    tier4_dir = results_root / "tier4"
    if not tier4_dir.exists():
        print("Tier 4 results not found")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    budgets = [5, 20, 50]  # in millions
    seeds_t4 = [42, 137]

    for condition, color, label in [
        ("transferred", "#2196F3", "Transferred"),
        ("train_from_scratch", "#FF9800", "Train from scratch"),
    ]:
        means = []
        stds = []
        for budget in budgets:
            ppls = []
            for seed in seeds_t4:
                path = tier4_dir / f"{condition}_{budget}M_seed{seed}" / "results.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    ppls.append(data["test_ppl_mean"])
            if ppls:
                means.append(np.mean(ppls))
                stds.append(np.std(ppls))
            else:
                means.append(None)
                stds.append(None)

        valid = [(b, m, s) for b, m, s in zip(budgets, means, stds) if m is not None]
        if valid:
            b_vals, m_vals, s_vals = zip(*valid)
            ax.errorbar(b_vals, m_vals, yerr=s_vals, marker="o",
                       color=color, label=label, capsize=4, linewidth=1.5)

    ax.set_xlabel("Training Tokens (M)")
    ax.set_ylabel("Test Perplexity")
    ax.set_title("Adaptor Convergence vs. Token Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "figure4_scaling.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "figure4_scaling.png", bbox_inches="tight")
    print(f"Saved to {output_dir / 'figure4_scaling.pdf'}")


def generate_table4(results_root: Path, output_dir: Path):
    """Table 4: CKA similarity matrix."""
    print("\n=== Table 4: CKA Similarity ===")

    cka_path = results_root / "tier4" / "cka" / "cka_results.json"
    if not cka_path.exists():
        print("CKA results not found")
        return

    with open(cka_path) as f:
        data = json.load(f)

    models = data["models"]
    matrix = np.array(data["cka_matrix"])

    # Short names
    short = {
        "EleutherAI/pythia-160m": "Pythia-160M",
        "EleutherAI/pythia-410m": "Pythia-410M",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    }

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Linear CKA similarity at injection layer ($\\ell = N/3$).}")
    print("\\label{tab:cka}")
    cols = "l" + "c" * len(models)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    header = " & ".join([short.get(m, m) for m in models])
    print(f"& {header} \\\\")
    print("\\midrule")
    for i, model in enumerate(models):
        row = [short.get(model, model)]
        for j in range(len(models)):
            row.append(f"{matrix[i,j]:.3f}")
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_table2(results_root, output_dir)
    generate_table3(results_root, output_dir)
    generate_figure2(results_root, output_dir)
    generate_figure4(results_root, output_dir)
    generate_table4(results_root, output_dir)

    print("\nAll figures and tables generated!")


if __name__ == "__main__":
    main()
