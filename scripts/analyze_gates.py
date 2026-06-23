"""Gate activation analysis and visualization.

Produces Figure 3: Gate activation histograms for transferred vs random vs permuted.

Usage:
    uv run python scripts/analyze_gates.py --results-dir results/tier3
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/tier3")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42, help="Which seed to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gate stats for key conditions
    conditions = {
        "transferred": "Transferred (ours)",
        "random_memory": "Random memory",
        "permuted_keys": "Permuted keys",
        "no_gate": "No gate (α=1)",
        "affine_stitch": "Affine stitch",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = {
        "transferred": "#2196F3",
        "random_memory": "#FF9800",
        "permuted_keys": "#4CAF50",
        "no_gate": "#9C27B0",
        "affine_stitch": "#F44336",
    }

    for condition, label in conditions.items():
        stats_path = results_dir / f"{condition}_seed{args.seed}" / "results.json"
        if not stats_path.exists():
            print(f"Skipping {condition} (no results found)")
            continue

        with open(stats_path) as f:
            data = json.load(f)

        gate_stats = data.get("gate_stats", {})
        if not gate_stats:
            continue

        # Create a synthetic histogram from stats
        mean = gate_stats.get("mean", 0.5)
        std = gate_stats.get("std", 0.2)
        q25 = gate_stats.get("q25", 0.3)
        q75 = gate_stats.get("q75", 0.7)

        # Plot as bar showing mean + IQR
        print(f"{condition}: mean={mean:.3f}, std={std:.3f}, IQR=[{q25:.3f}, {q75:.3f}]")

    # If we have the full gate values, plot histograms
    # Otherwise create a summary bar chart
    gate_data = {}
    for condition, label in conditions.items():
        stats_path = results_dir / f"{condition}_seed{args.seed}" / "results.json"
        if stats_path.exists():
            with open(stats_path) as f:
                data = json.load(f)
            gs = data.get("gate_stats", {})
            if gs:
                gate_data[label] = gs

    if gate_data:
        x = np.arange(len(gate_data))
        means = [gs["mean"] for gs in gate_data.values()]
        stds = [gs["std"] for gs in gate_data.values()]
        labels = list(gate_data.keys())

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors.get(k, "#999") for k in conditions if conditions[k] in gate_data],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean gate activation (α)")
        ax.set_title("Gate Activation by Condition")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "figure3_gate_activations.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "figure3_gate_activations.png", bbox_inches="tight")
    print(f"\nSaved to {output_dir / 'figure3_gate_activations.pdf'}")


if __name__ == "__main__":
    main()
