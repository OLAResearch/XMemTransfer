"""Tier 3: Ablations (all 8 conditions on Pythia-160M → Pythia-410M).

8 conditions × 3 seeds = 24 runs.

Usage:
    uv run python scripts/run_tier3.py --source-memory results/source_memory/memory.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 137, 2024]
CONDITIONS = [
    "baseline",
    "transferred",
    "random_memory",
    "permuted_keys",
    "no_gate",
    "train_from_scratch",
    "ffn_only",
    "affine_stitch",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--output-dir", type=str, default="results/tier3")
    # For PBS array jobs: run a specific (condition_idx, seed_idx) pair
    parser.add_argument("--array-index", type=int, default=None,
                        help="PBS array index: 0..23 (condition_idx*3 + seed_idx)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.array_index is not None:
        # Single run from PBS array
        condition_idx = args.array_index // len(SEEDS)
        seed_idx = args.array_index % len(SEEDS)
        conditions = [CONDITIONS[condition_idx]]
        seeds = [SEEDS[seed_idx]]
    else:
        conditions = CONDITIONS
        seeds = SEEDS

    for condition in conditions:
        for seed in seeds:
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
            ]

            print(f"\n{'='*60}")
            print(f"Tier 3: {condition} seed={seed}")
            print(f"{'='*60}")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {condition} seed={seed}")
                if args.array_index is not None:
                    sys.exit(1)

    print("\nTier 3 complete!")


if __name__ == "__main__":
    main()
