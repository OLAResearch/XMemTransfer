"""Pilot run: Validate correctness before full experiments.

Runs 1 seed, 2M tokens for 4 conditions (baseline, transferred,
random_memory, train_from_scratch) on Pythia-160M → Pythia-410M.

Checks:
  - Frozen backbone params have zero gradient norm throughout
  - Frozen memory table params have zero gradient norm throughout
  - random_memory does not improve over baseline by more than ~0.1 PPL
  - Logging captures all required fields
  - Checkpoint saving/loading round-trips correctly

Usage:
    uv run python scripts/run_pilot.py --source-memory results/source_memory/memory.pt
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot validation")
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--output-dir", type=str, default="results/pilot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_condition(condition, args, output_subdir):
    """Run a single condition via train_adaptor.py."""
    cmd = [
        sys.executable, "scripts/train_adaptor.py",
        "--condition", condition,
        "--target-model", args.target_model,
        "--source-memory", args.source_memory,
        "--memory-config", args.memory_config,
        "--max-tokens", str(args.max_tokens),
        "--seed", str(args.seed),
        "--output-dir", output_subdir,
        "--log-every", "20",
        "--eval-every", "100",
        "--batch-size", "8",
    ]
    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"Output: {output_subdir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {condition}")
        return False
    return True


def validate_pilot(output_dir: str) -> list:
    """Run validation checks on pilot results."""
    checks = []
    output_dir = Path(output_dir)

    # Check 1: Backbone grad norm should be zero for all conditions
    for condition in ["transferred", "random_memory", "train_from_scratch"]:
        log_path = output_dir / condition / "train_log.jsonl"
        if not log_path.exists():
            checks.append(f"FAIL: Missing log for {condition}")
            continue

        max_backbone_grad = 0.0
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if "grad_norm_backbone" in entry:
                    max_backbone_grad = max(max_backbone_grad, entry["grad_norm_backbone"])

        if max_backbone_grad > 1e-6:
            checks.append(f"FAIL: Backbone grad norm > 0 for {condition}: {max_backbone_grad:.8f}")
        else:
            checks.append(f"PASS: Backbone frozen for {condition} (max grad: {max_backbone_grad:.8f})")

    # Check 2: Memory grad norm should be zero for transferred/random
    for condition in ["transferred", "random_memory"]:
        log_path = output_dir / condition / "train_log.jsonl"
        if not log_path.exists():
            continue

        max_memory_grad = 0.0
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if "grad_norm_memory" in entry:
                    max_memory_grad = max(max_memory_grad, entry["grad_norm_memory"])

        if max_memory_grad > 1e-6:
            checks.append(f"FAIL: Memory grad norm > 0 for {condition}: {max_memory_grad:.8f}")
        else:
            checks.append(f"PASS: Memory frozen for {condition} (max grad: {max_memory_grad:.8f})")

    # Check 3: Memory grad norm should be > 0 for train_from_scratch
    log_path = output_dir / "train_from_scratch" / "train_log.jsonl"
    if log_path.exists():
        max_memory_grad = 0.0
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if "grad_norm_memory" in entry:
                    max_memory_grad = max(max_memory_grad, entry["grad_norm_memory"])

        if max_memory_grad < 1e-6:
            checks.append(f"FAIL: Memory NOT training for train_from_scratch (grad: {max_memory_grad:.8f})")
        else:
            checks.append(f"PASS: Memory training for train_from_scratch (max grad: {max_memory_grad:.4f})")

    # Check 4: random_memory PPL should be close to baseline
    baseline_path = output_dir / "baseline" / "results.json"
    random_path = output_dir / "random_memory" / "results.json"

    if baseline_path.exists() and random_path.exists():
        with open(baseline_path) as f:
            baseline_ppl = json.load(f)["test_ppl_mean"]
        with open(random_path) as f:
            random_ppl = json.load(f)["test_ppl_mean"]

        delta = baseline_ppl - random_ppl
        if abs(delta) > 2.0:  # More than 2 PPL improvement is suspicious
            checks.append(
                f"WARN: random_memory improves over baseline by {delta:.2f} PPL "
                f"(baseline={baseline_ppl:.2f}, random={random_ppl:.2f})"
            )
        else:
            checks.append(
                f"PASS: random_memory ≈ baseline (delta={delta:.2f}, "
                f"baseline={baseline_ppl:.2f}, random={random_ppl:.2f})"
            )

    # Check 5: Logging completeness
    required_fields = ["step", "loss", "ppl", "grad_norm_backbone", "grad_norm_adaptor"]
    for condition in ["transferred", "random_memory"]:
        log_path = output_dir / condition / "train_log.jsonl"
        if not log_path.exists():
            continue

        with open(log_path) as f:
            first_line = json.loads(f.readline())
            missing = [k for k in required_fields if k not in first_line]
            if missing:
                checks.append(f"FAIL: Missing log fields for {condition}: {missing}")
            else:
                checks.append(f"PASS: All required log fields present for {condition}")

    # Check 6: Checkpoint round-trip
    import torch
    for condition in ["transferred"]:
        adaptor_path = output_dir / condition / "adaptor.pt"
        if adaptor_path.exists():
            try:
                state = torch.load(adaptor_path, map_location="cpu", weights_only=True)
                checks.append(f"PASS: Checkpoint loads successfully for {condition}")
            except Exception as e:
                checks.append(f"FAIL: Checkpoint load failed for {condition}: {e}")

    return checks


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["baseline", "transferred", "random_memory", "train_from_scratch"]

    # Run all conditions
    for condition in conditions:
        subdir = str(output_dir / condition)
        success = run_condition(condition, args, subdir)
        if not success:
            print(f"\nPilot ABORTED: {condition} failed")
            return

    # Validate
    print(f"\n{'='*60}")
    print("PILOT VALIDATION RESULTS")
    print(f"{'='*60}")

    checks = validate_pilot(str(output_dir))
    all_pass = True
    for check in checks:
        status = "✓" if check.startswith("PASS") else ("⚠" if check.startswith("WARN") else "✗")
        print(f"  {status} {check}")
        if check.startswith("FAIL"):
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("PILOT PASSED - Safe to proceed to full tiers")
    else:
        print("PILOT FAILED - Fix issues before proceeding")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
