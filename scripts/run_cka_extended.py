"""Extended CKA analysis: 5×5 matrix including QWen3.5 models.

Extends the original 3-model CKA (Pythia-160M, Pythia-410M, TinyLlama-1.1B)
to include QWen3.5-0.8B (source A2) and QWen3.5-4B (target).

Usage:
    uv run python scripts/run_cka_extended.py
    uv run python scripts/run_cka_extended.py --n-samples 5000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


CKA_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-4B-Base",
]

# Short labels for printing
MODEL_LABELS = {
    "EleutherAI/pythia-160m": "Pythia-160M",
    "EleutherAI/pythia-410m": "Pythia-410M",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "Qwen/Qwen3.5-0.8B-Base": "QWen3.5-0.8B",
    "Qwen/Qwen3.5-4B-Base": "QWen3.5-4B",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/cka_extended")
    parser.add_argument("--n-samples", type=int, default=2560,
                        help="Number of samples for CKA (default: 2560, matching original)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
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

    KX = X @ X.T
    KY = Y @ Y.T

    hsic_xy = np.trace(KX @ KY)
    hsic_xx = np.trace(KX @ KX)
    hsic_yy = np.trace(KY @ KY)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denominator) if denominator > 0 else 0.0


def get_layers(model):
    """Get the transformer layer list from a HF causal LM."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layer list for {type(model)}")


def get_num_layers(config):
    """Get number of hidden layers from model config."""
    if hasattr(config, "text_config"):
        config = config.text_config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine num_layers")


def extract_representations(model_name, n_samples, seq_len, batch_size, device):
    """Extract mean-pooled representations at layer L//3."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram.data import get_dataloader

    print(f"\n{'='*60}")
    print(f"Extracting: {MODEL_LABELS.get(model_name, model_name)}")
    print(f"{'='*60}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    n_layers = get_num_layers(model.config)
    target_layer = n_layers // 3
    print(f"  Layers: {n_layers}, injection point: {target_layer}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        max_tokens=n_samples * seq_len,
        shuffle=False,
    )

    layers = get_layers(model)
    hook_output = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hook_output.append(output[0].detach().float().cpu())
        else:
            hook_output.append(output.detach().float().cpu())

    handle = layers[target_layer].register_forward_hook(hook_fn)

    reps = []
    with torch.no_grad():
        for batch in loader:
            hook_output.clear()
            input_ids = batch["input_ids"].to(device)
            model(input_ids=input_ids)
            if hook_output:
                rep = hook_output[0].mean(dim=1)  # (B, d_model)
                reps.append(rep)

    handle.remove()
    del model
    torch.cuda.empty_cache()

    if reps:
        result = torch.cat(reps, dim=0).numpy()
        print(f"  Shape: {result.shape}")
        return result

    print(f"  WARNING: No representations extracted!")
    return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Extract representations for all models
    representations = {}
    for model_name in CKA_MODELS:
        rep = extract_representations(
            model_name, args.n_samples, args.seq_len, args.batch_size, device,
        )
        if rep is not None:
            representations[model_name] = rep

    # Compute pairwise CKA
    model_names = list(representations.keys())
    n_models = len(model_names)
    cka_matrix = np.zeros((n_models, n_models))

    print(f"\n{'='*60}")
    print(f"Computing CKA ({n_models}×{n_models} matrix)")
    print(f"{'='*60}")

    for i in range(n_models):
        for j in range(n_models):
            n = min(representations[model_names[i]].shape[0],
                    representations[model_names[j]].shape[0])
            cka_matrix[i, j] = linear_cka(
                representations[model_names[i]][:n],
                representations[model_names[j]][:n],
            )

    # Save results
    results = {
        "models": model_names,
        "model_labels": [MODEL_LABELS.get(m, m) for m in model_names],
        "cka_matrix": cka_matrix.tolist(),
        "n_samples": int(min(r.shape[0] for r in representations.values())),
        "seq_len": args.seq_len,
        "layer_rule": "L // 3",
    }

    out_path = output_dir / "cka_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Pretty print
    print(f"\nCKA Similarity Matrix ({results['n_samples']} samples):")
    header = "".ljust(20) + "  ".join(f"{MODEL_LABELS.get(m, m):>14}" for m in model_names)
    print(header)
    for i, name_i in enumerate(model_names):
        row = f"{MODEL_LABELS.get(name_i, name_i):20}"
        row += "  ".join(f"{cka_matrix[i,j]:14.4f}" for j in range(n_models))
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
