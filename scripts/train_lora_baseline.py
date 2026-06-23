"""LoRA baseline: iso-parameter fine-tuning of the target model.

Applies LoRA to the target model with the same parameter budget as
the Engram adaptor, trained on the same data (WikiText-103).

Usage:
    uv run python scripts/train_lora_baseline.py \
        --target-model EleutherAI/pythia-410m \
        --lora-rank 7 \
        --max-tokens 20000000 \
        --seed 42 \
        --output-dir results/baselines/lora_pythia410m_seed42
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engram.data import get_dataloader
from engram.metrics import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA baseline")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: 2 * rank)")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
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
    """Return the correct LoRA target module names for each model family."""
    name_lower = model_name.lower()
    if "pythia" in name_lower:
        # GPT-NeoX: fused QKV + output projection
        return ["query_key_value", "dense"]
    elif "tinyllama" in name_lower or "llama" in name_lower:
        # LLaMA family: separate Q, V projections
        return ["q_proj", "v_proj"]
    elif "qwen" in name_lower:
        return ["q_proj", "v_proj"]
    else:
        # Fallback: try common names
        return ["q_proj", "v_proj"]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect dtype
    if torch.cuda.is_available():
        _cfg = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
        model_dtype = getattr(_cfg, "torch_dtype", None)
        dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    else:
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}, LoRA rank: {args.lora_rank}")
    torch.manual_seed(args.seed)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    # Apply LoRA
    target_modules = get_lora_target_modules(args.target_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    # Data
    print("Loading WikiText-103...")
    train_loader = get_dataloader(
        split="train", tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=args.batch_size, max_tokens=args.max_tokens,
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
    total_steps = args.max_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training
    log_file = open(output_dir / "train_log.jsonl", "w")
    step = 0
    epoch = 0
    start_time = time.time()
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience

    print(f"\nTraining LoRA for {total_steps} steps...")
    model.train()

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                log_entry = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.item()),
                    "ppl": float(torch.exp(loss).item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_s": elapsed,
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

                mean_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = float(torch.exp(torch.tensor(mean_val_loss)).item())

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
                    print(f"\nEarly stopping at step {step}")
                    break
        if patience > 0 and patience_counter >= patience:
            break

    log_file.close()

    # Restore best
    best_dir = output_dir / "best_lora"
    if patience > 0 and best_dir.exists():
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            args.target_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base_model, str(best_dir))
        print(f"Restored best LoRA from step {best_step}")

    # Test evaluation
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
        "condition": "lora",
        "seed": args.seed,
        "target_model": args.target_model,
        "lora_rank": args.lora_rank,
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
