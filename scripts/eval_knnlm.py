"""KNN-LM baseline: augment target model with nearest-neighbor datastore.

Builds a FAISS datastore from WikiText-103 using the target model's hidden
states, then evaluates with kNN-interpolated predictions.

Usage:
    uv run python scripts/eval_knnlm.py \
        --target-model EleutherAI/pythia-410m \
        --seed 42 \
        --output-dir results/baselines/knnlm_pythia410m_seed42
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engram.data import get_dataloader
from engram.metrics import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="KNN-LM baseline")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--datastore-tokens", type=int, default=20_000_000,
                        help="Number of tokens for building the datastore")
    parser.add_argument("--k", type=int, default=1024, help="Number of nearest neighbors")
    parser.add_argument("--lmbda", type=float, default=None,
                        help="Interpolation weight (None=tune on val)")
    parser.add_argument("--temperature", type=float, default=10.0,
                        help="Temperature for kNN distribution")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pq-dim", type=int, default=64,
                        help="Product quantization sub-vectors (for FAISS compression)")
    return parser.parse_args()


def build_datastore(model, dataloader, device, dtype, output_dir: Path, hidden_dim: int):
    """Build (key, value) datastore from model's hidden states."""
    import faiss

    keys_path = output_dir / "datastore_keys.npy"
    vals_path = output_dir / "datastore_vals.npy"

    if keys_path.exists() and vals_path.exists():
        print("Loading existing datastore...")
        keys = np.load(str(keys_path))
        vals = np.load(str(vals_path))
        return keys, vals

    print("Building datastore...")
    all_keys = []
    all_vals = []
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Building datastore"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, output_hidden_states=True)
            # Use last hidden state as key (before the LM head)
            hidden = outputs.hidden_states[-1]  # [B, T, D]

            # Keys: hidden states at positions 0..T-2 (predicting next token)
            # Values: target tokens at positions 1..T-1
            batch_keys = hidden[:, :-1, :].reshape(-1, hidden_dim).float().cpu().numpy()
            batch_vals = labels[:, 1:].reshape(-1).cpu().numpy()

            # Filter out padding (-100 labels)
            valid_mask = batch_vals != -100
            batch_keys = batch_keys[valid_mask]
            batch_vals = batch_vals[valid_mask]

            all_keys.append(batch_keys)
            all_vals.append(batch_vals)
            total_tokens += len(batch_vals)

    keys = np.concatenate(all_keys, axis=0).astype(np.float32)
    vals = np.concatenate(all_vals, axis=0).astype(np.int64)
    print(f"Datastore: {len(keys):,} entries, {keys.nbytes / 1e9:.2f} GB")

    np.save(str(keys_path), keys)
    np.save(str(vals_path), vals)
    return keys, vals


def build_faiss_index(keys: np.ndarray, pq_dim: int, output_dir: Path):
    """Build a FAISS index with IVF + PQ for efficient search."""
    import faiss

    index_path = output_dir / "faiss_index.bin"
    if index_path.exists():
        print("Loading existing FAISS index...")
        return faiss.read_index(str(index_path))

    d = keys.shape[1]
    n = keys.shape[0]
    nlist = min(4096, max(64, int(np.sqrt(n))))

    print(f"Building FAISS index: n={n:,}, d={d}, nlist={nlist}, pq_dim={pq_dim}")

    # Use IVF + PQ for memory efficiency
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_dim, 8)  # 8 bits per sub-vector

    # Train on a subset if too large
    train_size = min(n, 500_000)
    if train_size < n:
        train_idx = np.random.choice(n, train_size, replace=False)
        train_data = keys[train_idx]
    else:
        train_data = keys

    print("Training index...")
    index.train(train_data)
    print("Adding vectors...")
    index.add(keys)

    index.nprobe = 32
    faiss.write_index(index, str(index_path))
    print(f"Index saved. Memory: {index.sa_code_size() * n / 1e9:.2f} GB (compressed)")
    return index


def knn_eval(model, index, vals, dataloader, device, dtype, hidden_dim,
             k, lmbda, temperature, vocab_size):
    """Evaluate with kNN-interpolated predictions."""
    model.eval()
    all_ppls = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="KNN-LM eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, output_hidden_states=True)
            logits = outputs.logits  # [B, T, V]
            hidden = outputs.hidden_states[-1]  # [B, T, D]

            B, T, V = logits.shape

            # Model probability (shifted by 1)
            model_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
            target = labels[:, 1:]  # [B, T-1]

            # Get hidden states for query (positions predicting next token)
            queries = hidden[:, :-1, :].reshape(-1, hidden_dim).float().cpu().numpy()
            target_flat = target.reshape(-1)

            # kNN search
            distances, indices = index.search(queries, k)  # [B*(T-1), k]
            knn_targets = vals[indices]  # [B*(T-1), k] — neighbor token IDs

            # Compute kNN distribution
            # Convert L2 distances to probabilities via softmax with temperature
            knn_dists = torch.tensor(-distances / temperature, dtype=torch.float32)  # [N, k]
            knn_weights = torch.softmax(knn_dists, dim=-1)  # [N, k]
            knn_targets_t = torch.tensor(knn_targets, dtype=torch.long)  # [N, k]

            # Scatter into vocabulary-sized distribution
            N = queries.shape[0]
            knn_probs = torch.zeros(N, vocab_size, dtype=torch.float32)
            knn_probs.scatter_add_(1, knn_targets_t, knn_weights)
            knn_log_probs = torch.log(knn_probs + 1e-10)  # [N, V]

            # Interpolate: p = λ * p_knn + (1-λ) * p_model
            model_lp = model_log_probs.reshape(N, V).float().cpu()
            interp_log_probs = torch.log(
                lmbda * torch.exp(knn_log_probs) + (1 - lmbda) * torch.exp(model_lp)
            )

            # Compute per-token NLL
            target_cpu = target_flat.cpu()
            valid_mask = target_cpu != -100
            nll = -interp_log_probs[torch.arange(N), target_cpu.clamp(min=0)]
            nll = nll[valid_mask]

            # Per-batch PPL
            batch_ppl = float(torch.exp(nll.mean()).item())
            all_ppls.append(batch_ppl)

    return all_ppls


def tune_lambda(model, index, vals, val_loader, device, dtype, hidden_dim,
                k, temperature, vocab_size):
    """Tune interpolation weight λ on validation set."""
    print("\nTuning λ on validation set...")
    best_ppl = float("inf")
    best_lmbda = 0.0

    for lmbda in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        ppls = knn_eval(model, index, vals, val_loader, device, dtype,
                        hidden_dim, k, lmbda, temperature, vocab_size)
        mean_ppl = sum(ppls) / len(ppls)
        print(f"  λ={lmbda:.2f}: val PPL = {mean_ppl:.2f}")
        if mean_ppl < best_ppl:
            best_ppl = mean_ppl
            best_lmbda = lmbda

    print(f"Best λ = {best_lmbda:.2f} (val PPL = {best_ppl:.2f})")
    return best_lmbda


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        _cfg = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
        model_dtype = getattr(_cfg, "torch_dtype", None)
        dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    else:
        dtype = torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size

    # Data loaders
    ds_loader = get_dataloader(
        split="train", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=args.datastore_tokens,
        shuffle=False, seed=args.seed,
    )
    val_loader = get_dataloader(
        split="validation", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=2_000_000, shuffle=False,
    )
    test_loader = get_dataloader(
        split="test", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=None, shuffle=False,
    )

    start_time = time.time()

    # Step 1: Build datastore
    keys, vals = build_datastore(model, ds_loader, device, dtype, output_dir, hidden_dim)

    # Step 2: Build FAISS index
    index = build_faiss_index(keys, args.pq_dim, output_dir)

    # Step 3: Tune λ on validation set
    if args.lmbda is None:
        lmbda = tune_lambda(model, index, vals, val_loader, device, dtype,
                            hidden_dim, args.k, args.temperature, vocab_size)
    else:
        lmbda = args.lmbda

    # Step 4: Evaluate on test set
    print(f"\nEvaluating on test set with λ={lmbda:.2f}...")
    test_ppls = knn_eval(model, index, vals, test_loader, device, dtype,
                         hidden_dim, args.k, lmbda, args.temperature, vocab_size)

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
    print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

    results = {
        "condition": "knnlm",
        "seed": args.seed,
        "target_model": args.target_model,
        "datastore_tokens": args.datastore_tokens,
        "datastore_entries": len(vals),
        "k": args.k,
        "lambda": lmbda,
        "temperature": args.temperature,
        "trainable_params": 0,
        "test_ppl_mean": mean_ppl,
        "test_ppl_ci_lower": ci_lower,
        "test_ppl_ci_upper": ci_upper,
        "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
        "n_test_batches": len(test_ppls),
        "elapsed_hours": (time.time() - start_time) / 3600,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "test_ppls.json", "w") as f:
        json.dump(test_ppls, f)

    # Also evaluate without kNN (λ=0) as sanity check
    print("\nSanity check: λ=0 (pure model)...")
    pure_ppls = knn_eval(model, index, vals, test_loader, device, dtype,
                         hidden_dim, args.k, 0.0, args.temperature, vocab_size)
    pure_mean, _, _ = bootstrap_ci(pure_ppls)
    print(f"Pure model PPL: {pure_mean:.2f}")
    results["pure_model_ppl"] = pure_mean
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
