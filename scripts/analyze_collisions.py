"""Cross-tokenizer collision analysis.

Measures key-space agreement between Pythia (GPT-NeoX) and TinyLlama
(SentencePiece) tokenizers using word-boundary canonicalization.

Produces Figure 5: Collision rates per N-gram order.

Usage:
    uv run python scripts/analyze_collisions.py --output-dir results/figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from transformers import AutoTokenizer
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.collision_analyzer import CollisionAnalyzer
from engram.hashing import HashConfig


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
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--n-texts", type=int, default=1000)
    parser.add_argument("--table-size", type=int, default=65536)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    print("Loading tokenizers...")
    tok_pythia = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tok_llama = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load sample texts
    print("Loading WikiText-103 test samples...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 50][:args.n_texts]
    print(f"Analyzing {len(texts)} texts")

    # Measure agreement
    hash_cfg = HashConfig(max_ngram=3, heads_per_order=4, table_size=args.table_size)
    analyzer = CollisionAnalyzer(hash_cfg)

    results = analyzer.measure_agreement(
        texts=texts,
        tokenizer_a=tok_pythia,
        tokenizer_b=tok_llama,
        max_ngram=3,
    )

    print("\nResults:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")

    with open(output_dir / "collision_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    orders = [2, 3]
    means = [results.get(f"order_{n}_mean_agreement", 0) for n in orders]
    stds = [results.get(f"order_{n}_std_agreement", 0) for n in orders]

    bars = ax.bar(
        [f"{n}-gram" for n in orders],
        means,
        yerr=stds,
        capsize=5,
        color=["#2196F3", "#FF9800"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_ylabel("Word N-gram Agreement (Jaccard)")
    ax.set_title("Cross-Tokenizer Key-Space Agreement\n(GPT-NeoX BPE vs Llama SentencePiece)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "figure5_collision_rates.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "figure5_collision_rates.png", bbox_inches="tight")
    print(f"\nSaved to {output_dir / 'figure5_collision_rates.pdf'}")


if __name__ == "__main__":
    main()
