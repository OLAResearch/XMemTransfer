"""Knowledge Distillation baseline: transfer knowledge via logit matching.

The source model (teacher) generates soft targets for the target model (student)
on WikiText-103. Only same-tokenizer transfer supported (Pythia → Pythia).

Usage:
    uv run python scripts/train_kd_baseline.py \
        --teacher-model EleutherAI/pythia-160m \
        --student-model EleutherAI/pythia-410m \
        --max-tokens 20000000 \
        --seed 42 \
        --output-dir results/baselines/kd_pythia410m_seed42
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engram.data import get_dataloader
from engram.metrics import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation baseline")
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5,
                        help="Weight for KD loss (1-alpha for CE loss)")
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
    # LoRA for student to keep iso-parameter
    parser.add_argument("--lora-rank", type=int, default=7,
                        help="LoRA rank for student (iso-param with Engram adaptor)")
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        _cfg = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
        model_dtype = getattr(_cfg, "torch_dtype", None)
        dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    else:
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Teacher: {args.teacher_model}, Student: {args.student_model}")
    torch.manual_seed(args.seed)

    # Load teacher (frozen)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student with LoRA (iso-parameter)
    from peft import LoraConfig, get_peft_model, TaskType
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    target_modules = get_lora_target_modules(args.student_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable params: {trainable_params:,}")

    # Use teacher's tokenizer (same tokenizer assumed)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify same vocab size
    teacher_vocab = teacher.config.vocab_size
    student_vocab = student.config.vocab_size
    if teacher_vocab != student_vocab:
        print(f"WARNING: vocab mismatch (teacher={teacher_vocab}, student={student_vocab}). "
              f"KD will use min vocab size.")
    min_vocab = min(teacher_vocab, student_vocab)

    # Data
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
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    T = args.kd_temperature
    alpha = args.kd_alpha

    log_file = open(output_dir / "train_log.jsonl", "w")
    step = 0
    epoch = 0
    start_time = time.time()
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience

    print(f"\nTraining KD for {total_steps} steps (T={T}, α={alpha})...")

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits[:, :, :min_vocab].float()

            # Student forward
            student.train()
            student_outputs = student(input_ids=input_ids, labels=labels)
            student_logits = student_outputs.logits[:, :, :min_vocab].float()
            ce_loss = student_outputs.loss

            # KD loss (KL divergence on soft targets)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

            # Combined loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                log_entry = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.item()),
                    "ce_loss": float(ce_loss.item()),
                    "kd_loss": float(kd_loss.item()),
                    "ppl": float(torch.exp(ce_loss).item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_s": elapsed,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                print(f"  Step {step}/{total_steps} | CE {ce_loss.item():.4f} | "
                      f"KD {kd_loss.item():.4f} | PPL {torch.exp(ce_loss).item():.2f}")

            if step % args.eval_every == 0:
                student.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_out = student(input_ids=val_ids, labels=val_labels)
                        val_losses.append(val_out.loss.item())

                val_ppl = float(torch.exp(torch.tensor(sum(val_losses) / len(val_losses))).item())
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step
                    patience_counter = 0
                    student.save_pretrained(output_dir / "best_lora")
                    print(f"  >> Val PPL: {val_ppl:.2f} (new best)")
                else:
                    patience_counter += 1
                    print(f"  >> Val PPL: {val_ppl:.2f} (patience {patience_counter}/{patience})")

                log_file.write(json.dumps({"step": step, "val_ppl": val_ppl, "type": "eval"}) + "\n")
                log_file.flush()

                if patience > 0 and patience_counter >= patience:
                    break
        if patience > 0 and patience_counter >= patience:
            break

    log_file.close()

    # Delete teacher to free memory
    del teacher
    torch.cuda.empty_cache()

    # Restore best
    best_dir = output_dir / "best_lora"
    if patience > 0 and best_dir.exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.student_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        student = PeftModel.from_pretrained(base, str(best_dir))

    # Test evaluation
    print("\nRunning test evaluation...")
    student.eval()
    test_ppls = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = student(input_ids=input_ids, labels=labels)
            ppl = float(torch.exp(outputs.loss).item())
            test_ppls.append(ppl)

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
    print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

    results = {
        "condition": "kd",
        "seed": args.seed,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "kd_temperature": T,
        "kd_alpha": alpha,
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
