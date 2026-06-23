"""Tier 4: Scaling analysis + CKA similarity.

- Adaptor convergence vs. training tokens: {5M, 20M, 50M} × 2 seeds = 6 runs
- CKA similarity across Pythia-160M, Pythia-410M, TinyLlama-1.1B

Usage:
    uv run python scripts/run_tier4.py --source-memory results/source_memory/memory.pt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np


SEEDS = [42, 137]
TOKEN_BUDGETS = [5_000_000, 20_000_000, 50_000_000]
CKA_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--output-dir", type=str, default="results/tier4")
    parser.add_argument("--skip-cka", action="store_true")
    return parser.parse_args()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n_samples, d_x) representation matrix
        Y: (n_samples, d_y) representation matrix

    Returns:
        CKA similarity score in [0, 1]
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Use Gram matrices (n×n) so different d_x, d_y are fine
    KX = X @ X.T
    KY = Y @ Y.T

    hsic_xy = np.trace(KX @ KY)
    hsic_xx = np.trace(KX @ KX)
    hsic_yy = np.trace(KY @ KY)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denominator) if denominator > 0 else 0.0


def run_cka_analysis(output_dir: Path, n_samples: int = 10000):
    """Compute CKA similarity between model pairs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cka_dir = output_dir / "cka"
    cka_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Use a shared text for representations
    tokenizer = AutoTokenizer.from_pretrained(CKA_MODELS[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from engram.data import get_dataloader
    loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        seq_len=128,
        batch_size=64,
        max_tokens=n_samples * 128,
        shuffle=False,
    )

    # Collect representations per model at layer N//3
    representations = {}

    for model_name in CKA_MODELS:
        print(f"Extracting representations from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()

        config = model.config
        n_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
        target_layer = n_layers // 3

        reps = []
        hook_output = []

        def get_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_output.append(output[0].detach().float().cpu())
                else:
                    hook_output.append(output.detach().float().cpu())
            return hook_fn

        # Get layer list
        if hasattr(model, "gpt_neox"):
            layers = model.gpt_neox.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            print(f"  Skipping {model_name} (can't find layers)")
            continue

        handle = layers[target_layer].register_forward_hook(get_hook())

        # Re-tokenize for this model's tokenizer if different
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_tokenizer.pad_token is None:
            model_tokenizer.pad_token = model_tokenizer.eos_token

        model_loader = get_dataloader(
            split="test",
            tokenizer=model_tokenizer,
            seq_len=128,
            batch_size=64,
            max_tokens=n_samples * 128,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in model_loader:
                hook_output.clear()
                input_ids = batch["input_ids"].to(device)
                model(input_ids=input_ids)
                if hook_output:
                    # Take mean across sequence dimension
                    rep = hook_output[0].mean(dim=1)  # (B, d_model)
                    reps.append(rep)

        handle.remove()
        del model
        torch.cuda.empty_cache()

        if reps:
            representations[model_name] = torch.cat(reps, dim=0).numpy()
            print(f"  Shape: {representations[model_name].shape}")

    # Compute pairwise CKA
    model_names = list(representations.keys())
    n_models = len(model_names)
    cka_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            # Align sample counts
            n = min(representations[model_names[i]].shape[0],
                    representations[model_names[j]].shape[0])
            cka_matrix[i, j] = linear_cka(
                representations[model_names[i]][:n],
                representations[model_names[j]][:n],
            )

    results = {
        "models": model_names,
        "cka_matrix": cka_matrix.tolist(),
        "n_samples": n,
    }

    with open(cka_dir / "cka_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nCKA Similarity Matrix:")
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            print(f"  {name_i} vs {name_j}: {cka_matrix[i,j]:.4f}")

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Scaling runs
    for budget in TOKEN_BUDGETS:
        for seed in SEEDS:
            subdir = str(output_dir / f"transferred_{budget//1_000_000}M_seed{seed}")

            results_file = Path(subdir) / "results.json"
            if results_file.exists():
                print(f"Skipping {budget//1_000_000}M seed={seed} (already done)")
                continue

            cmd = [
                sys.executable, "scripts/train_adaptor.py",
                "--condition", "transferred",
                "--target-model", args.target_model,
                "--source-memory", args.source_memory,
                "--memory-config", args.memory_config,
                "--max-tokens", str(budget),
                "--seed", str(seed),
                "--output-dir", subdir,
                "--early-stopping-patience", "3",
            ]

            print(f"\n{'='*60}")
            print(f"Tier 4 Scaling: {budget//1_000_000}M tokens, seed={seed}")
            print(f"{'='*60}")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {budget}M seed={seed}")

    # Also run train_from_scratch at same budgets for comparison
    for budget in TOKEN_BUDGETS:
        for seed in SEEDS:
            subdir = str(output_dir / f"train_from_scratch_{budget//1_000_000}M_seed{seed}")

            results_file = Path(subdir) / "results.json"
            if results_file.exists():
                print(f"Skipping train_from_scratch {budget//1_000_000}M seed={seed} (already done)")
                continue

            cmd = [
                sys.executable, "scripts/train_adaptor.py",
                "--condition", "train_from_scratch",
                "--target-model", args.target_model,
                "--source-memory", args.source_memory,
                "--memory-config", args.memory_config,
                "--max-tokens", str(budget),
                "--seed", str(seed),
                "--output-dir", subdir,
                "--early-stopping-patience", "3",
            ]

            print(f"\nTier 4 Scaling: train_from_scratch {budget//1_000_000}M tokens, seed={seed}")
            result = subprocess.run(cmd)

    # CKA analysis
    if not args.skip_cka:
        print(f"\n{'='*60}")
        print("CKA Analysis")
        print(f"{'='*60}")
        run_cka_analysis(output_dir)

    print("\nTier 4 complete!")


if __name__ == "__main__":
    main()
