"""Tier 2: Cross-tokenizer transfer (Pythia-160M → TinyLlama-1.1B).

3 conditions (baseline, transferred, random) × 3 seeds = 9 runs.
Uses word-boundary canonicalization.

Usage:
    uv run python scripts/run_tier2.py --source-memory results/source_memory/memory.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 137, 2024]
CONDITIONS = ["baseline", "transferred", "random_memory"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--target-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--output-dir", type=str, default="results/tier2")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    for condition in CONDITIONS:
        for seed in SEEDS:
            subdir = str(output_dir / f"{condition}_seed{seed}")

            results_file = Path(subdir) / "results.json"
            if results_file.exists():
                print(f"Skipping {condition} seed={seed} (already done)")
                continue

            cmd = [
                sys.executable, "scripts/train_adaptor.py",
                "--condition", condition,
                "--target-model", args.target_model,
                "--source-memory", args.source_memory,
                "--memory-config", args.memory_config,
                "--max-tokens", str(args.max_tokens),
                "--seed", str(seed),
                "--output-dir", subdir,
                "--canon-mode", "word_boundary",
            ]

            print(f"\n{'='*60}")
            print(f"Tier 2: {condition} seed={seed}")
            print(f"{'='*60}")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {condition} seed={seed}")
                sys.exit(1)

    print("\nTier 2 complete!")


if __name__ == "__main__":
    main()
