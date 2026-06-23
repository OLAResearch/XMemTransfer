"""Cross-LoRA baseline: transfer LoRA weights from source to target model.

Inspired by Cross-LoRA (Xia et al., 2025) and LoRA-X (Zhang et al., 2025).
1. Train LoRA on source model (Pythia-160M)
2. Project LoRA weights via SVD alignment of base weight matrices
3. Initialize target LoRA with projected weights
4. Fine-tune on target model

Usage:
    uv run python scripts/train_crosslora.py \
        --source-model EleutherAI/pythia-160m \
        --target-model EleutherAI/pythia-410m \
        --source-lora-rank 8 \
        --target-lora-rank 7 \
        --max-tokens 20000000 \
        --seed 42 \
        --output-dir results/baselines/crosslora_pythia410m_seed42
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engram.data import get_dataloader
from engram.metrics import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-LoRA baseline")
    parser.add_argument("--source-model", type=str, required=True)
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--source-lora-rank", type=int, default=8)
    parser.add_argument("--target-lora-rank", type=int, default=7)
    parser.add_argument("--source-tokens", type=int, default=20_000_000,
                        help="Tokens for training source LoRA")
    parser.add_argument("--target-tokens", type=int, default=20_000_000,
                        help="Tokens for fine-tuning target LoRA")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_lora_target_modules(model_name: str) -> list[str]:
    name_lower = model_name.lower()
    if "pythia" in name_lower:
        return ["query_key_value", "dense"]
    elif "tinyllama" in name_lower or "llama" in name_lower:
        return ["q_proj", "v_proj"]
    elif "qwen" in name_lower:
        return ["q_proj", "v_proj"]
    else:
        return ["q_proj", "v_proj"]


def train_source_lora(args, device, dtype) -> Path:
    """Phase 1: Train LoRA on source model."""
    source_lora_dir = Path(args.output_dir) / "source_lora"
    if (source_lora_dir / "adapter_config.json").exists():
        print("Source LoRA already trained, skipping...")
        return source_lora_dir

    print(f"\n=== Phase 1: Training LoRA on {args.source_model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    target_modules = get_lora_target_modules(args.source_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.source_lora_rank,
        lora_alpha=2 * args.source_lora_rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_loader = get_dataloader(
        split="train", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=args.source_tokens,
        shuffle=True, seed=args.seed,
    )

    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.source_tokens // tokens_per_step

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    step = 0
    epoch = 0
    model.train()
    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  Source step {step}/{total_steps} | Loss {loss.item():.4f}")

    model.save_pretrained(source_lora_dir)
    del model
    torch.cuda.empty_cache()
    return source_lora_dir


def project_lora_weights(source_lora_dir: Path, source_model_name: str,
                         target_model_name: str, source_rank: int,
                         target_rank: int, device, dtype) -> dict:
    """Phase 2: Project source LoRA weights to target space via SVD alignment.

    For each LoRA module, project using the SVD of the base weight matrices:
    - Compute SVD of source base weight: W_s = U_s S_s V_s^T
    - Compute SVD of target base weight: W_t = U_t S_t V_t^T
    - Project: A_t = A_s @ V_s[:, :r_s]^T @ V_t[:, :r_t]
    -          B_t = U_t[:, :r_t]^T @ U_s[:, :r_s] @ B_s
    """
    print("\n=== Phase 2: Projecting LoRA weights via SVD alignment ===")

    # Load source LoRA state dict (PEFT saves as safetensors by default)
    import safetensors.torch
    safetensors_path = source_lora_dir / "adapter_model.safetensors"
    bin_path = source_lora_dir / "adapter_model.bin"
    if safetensors_path.exists():
        source_state = safetensors.torch.load_file(str(safetensors_path))
    elif bin_path.exists():
        source_state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {source_lora_dir}")

    # Load base models for SVD
    source_base = AutoModelForCausalLM.from_pretrained(
        source_model_name, torch_dtype=torch.float32, trust_remote_code=True)
    target_base = AutoModelForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.float32, trust_remote_code=True)

    projected_state = {}

    # Find matching layers by name pattern
    source_modules = get_lora_target_modules(source_model_name)
    target_modules = get_lora_target_modules(target_model_name)

    n_source_layers = source_base.config.num_hidden_layers
    n_target_layers = target_base.config.num_hidden_layers

    # Map source layers to target layers (proportional mapping)
    for tgt_layer_idx in range(n_target_layers):
        src_layer_idx = int(tgt_layer_idx * n_source_layers / n_target_layers)

        for src_mod, tgt_mod in zip(source_modules, target_modules):
            # Get source LoRA A and B
            src_A_key = None
            src_B_key = None
            for key in source_state:
                if f"layers.{src_layer_idx}" in key and src_mod in key:
                    if "lora_A" in key:
                        src_A_key = key
                    elif "lora_B" in key:
                        src_B_key = key

            if src_A_key is None or src_B_key is None:
                continue

            A_s = source_state[src_A_key].float()  # [r_s, d_in_s]
            B_s = source_state[src_B_key].float()  # [d_out_s, r_s]

            # Get base weights for SVD
            src_weight = None
            tgt_weight = None
            for name, param in source_base.named_parameters():
                if f"layers.{src_layer_idx}" in name and src_mod in name and "weight" in name:
                    if "lora" not in name:
                        src_weight = param.data.float()
                        break
            for name, param in target_base.named_parameters():
                if f"layers.{tgt_layer_idx}" in name and tgt_mod in name and "weight" in name:
                    if "lora" not in name:
                        tgt_weight = param.data.float()
                        break

            if src_weight is None or tgt_weight is None:
                continue

            # SVD alignment
            r_s = min(source_rank, min(src_weight.shape))
            r_t = min(target_rank, min(tgt_weight.shape))

            U_s, S_s, Vh_s = torch.linalg.svd(src_weight, full_matrices=False)
            U_t, S_t, Vh_t = torch.linalg.svd(tgt_weight, full_matrices=False)

            # Project A: A_t = A_s @ V_s[:r_s]^T @ V_t[:r_t]  (input space alignment)
            V_s = Vh_s[:r_s, :]  # [r_s, d_in_s]
            V_t = Vh_t[:r_t, :]  # [r_t, d_in_t]

            # A_s: [r_s_lora, d_in_s], project to [r_t_lora, d_in_t]
            # Simplified: just truncate/pad the projected delta
            delta_s = B_s @ A_s  # [d_out_s, d_in_s] — full LoRA delta
            # Project: delta_t = U_t[:, :r_proj]^T @ U_s[:, :r_proj] @ delta_s @ V_s[:r_proj]^T @ V_t[:r_proj]
            r_proj = min(r_s, r_t, delta_s.shape[0], delta_s.shape[1])

            # Use truncated SVD bases for projection
            U_s_r = U_s[:, :r_proj]  # [d_out_s, r_proj]
            U_t_r = U_t[:, :r_proj]  # [d_out_t, r_proj]
            V_s_r = Vh_s[:r_proj, :]  # [r_proj, d_in_s]
            V_t_r = Vh_t[:r_proj, :]  # [r_proj, d_in_t]

            # Projected delta in reduced space
            delta_reduced = U_s_r.T @ delta_s @ V_s_r.T  # [r_proj, r_proj]

            # Reconstruct in target space with target rank
            A_t = torch.zeros(target_rank, tgt_weight.shape[1])
            B_t = torch.zeros(tgt_weight.shape[0], target_rank)

            r_use = min(target_rank, r_proj)
            # Initialize A_t and B_t from the projected delta
            A_t[:r_use, :] = V_t_r[:r_use, :]
            B_t[:, :r_use] = U_t_r[:, :r_use] @ delta_reduced[:r_use, :r_use]

            # Store with target layer naming
            tgt_A_key = src_A_key.replace(
                f"layers.{src_layer_idx}", f"layers.{tgt_layer_idx}"
            ).replace(src_mod, tgt_mod)
            tgt_B_key = src_B_key.replace(
                f"layers.{src_layer_idx}", f"layers.{tgt_layer_idx}"
            ).replace(src_mod, tgt_mod)

            projected_state[tgt_A_key] = A_t.to(dtype)
            projected_state[tgt_B_key] = B_t.to(dtype)

    del source_base, target_base
    torch.cuda.empty_cache()
    print(f"Projected {len(projected_state)} LoRA weight tensors")
    return projected_state


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

    # Phase 1: Train source LoRA
    source_lora_dir = train_source_lora(args, device, dtype)

    # Phase 2: Project LoRA weights
    projected_state = project_lora_weights(
        source_lora_dir, args.source_model, args.target_model,
        args.source_lora_rank, args.target_lora_rank, device, dtype,
    )

    # Phase 3: Fine-tune projected LoRA on target model
    print(f"\n=== Phase 3: Fine-tuning projected LoRA on {args.target_model} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    target_modules = get_lora_target_modules(args.target_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.target_lora_rank,
        lora_alpha=2 * args.target_lora_rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Initialize with projected weights (where available)
    current_state = model.state_dict()
    loaded = 0
    for key, val in projected_state.items():
        matching_keys = [k for k in current_state if key.split(".")[-2:] == k.split(".")[-2:]
                         and k.split("layers.")[1].split(".")[0] == key.split("layers.")[1].split(".")[0]]
        for mk in matching_keys:
            if current_state[mk].shape == val.shape:
                current_state[mk] = val.to(current_state[mk].device)
                loaded += 1
    if loaded > 0:
        model.load_state_dict(current_state, strict=False)
        print(f"Loaded {loaded} projected LoRA weights")
    else:
        print("WARNING: Could not load projected weights, using random init")

    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training (same as LoRA baseline)
    train_loader = get_dataloader(
        split="train", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=args.target_tokens,
        shuffle=True, seed=args.seed,
    )
    val_loader = get_dataloader(
        split="validation", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=2_000_000, shuffle=False,
    )
    test_loader = get_dataloader(
        split="test", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=None, shuffle=False,
    )

    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.target_tokens // tokens_per_step

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    log_file = open(output_dir / "train_log.jsonl", "w")
    step = 0
    epoch = 0
    start_time = time.time()
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience

    print(f"Training Cross-LoRA for {total_steps} steps...")
    model.train()

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                log_entry = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.item()),
                    "ppl": float(torch.exp(loss).item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_s": time.time() - start_time,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                print(f"  Step {step}/{total_steps} | Loss {loss.item():.4f} | PPL {torch.exp(loss).item():.2f}")

            if step % args.eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_out = model(input_ids=val_ids, labels=val_labels)
                        val_losses.append(val_out.loss.item())

                val_ppl = float(torch.exp(torch.tensor(sum(val_losses) / len(val_losses))).item())
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step
                    patience_counter = 0
                    model.save_pretrained(output_dir / "best_lora")
                    print(f"  >> Val PPL: {val_ppl:.2f} (new best)")
                else:
                    patience_counter += 1
                    print(f"  >> Val PPL: {val_ppl:.2f} (patience {patience_counter}/{patience})")
                log_file.write(json.dumps({"step": step, "val_ppl": val_ppl, "type": "eval"}) + "\n")
                log_file.flush()
                model.train()
                if patience > 0 and patience_counter >= patience:
                    break
        if patience > 0 and patience_counter >= patience:
            break
    log_file.close()

    # Test evaluation
    if patience > 0 and (output_dir / "best_lora" / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(
            args.target_model, torch_dtype=dtype, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(base, str(output_dir / "best_lora"))

    print("\nRunning test evaluation...")
    model.eval()
    test_ppls = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            ppl = float(torch.exp(outputs.loss).item())
            test_ppls.append(ppl)

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
    print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

    results = {
        "condition": "crosslora",
        "seed": args.seed,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "source_lora_rank": args.source_lora_rank,
        "target_lora_rank": args.target_lora_rank,
        "trainable_params": trainable_params,
        "test_ppl_mean": mean_ppl,
        "test_ppl_ci_lower": ci_lower,
        "test_ppl_ci_upper": ci_upper,
        "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
        "n_test_batches": len(test_ppls),
        "best_step": best_step if patience > 0 else step,
        "elapsed_hours": (time.time() - start_time) / 3600,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "test_ppls.json", "w") as f:
        json.dump(test_ppls, f)

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
