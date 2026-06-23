"""Out-of-domain evaluation: test transferred memory on datasets beyond WikiText-103.

Evaluates baseline (no memory) vs transferred memory on diverse downstream tasks:
  - LAMBADA (long-range narrative cloze)
  - WikiText-2 (Wikipedia, domain-adjacent)
  - PTB (Penn Treebank, news/WSJ)
  - C4 (web crawl, 2M token sample)
  - PG-19 (Project Gutenberg books, literary fiction)
  - OpenWebText (Reddit-filtered web text)
  - arXiv (scientific papers)

Usage:
    uv run python scripts/eval_ood.py \
        --target-model EleutherAI/pythia-410m \
        --adaptor-dir results/tier1/transferred_seed42 \
        --source-memory results/source_memory/memory.pt \
        --memory-config results/source_memory/memory_config.json \
        --output-dir results/ood_expanded/pythia410m \
        --datasets lambada wikitext2 ptb c4 pg19 openwebtext arxiv
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import build_adaptor
from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import build_canonicalizer, WordBoundaryCanonicalizer
from engram.hashing import HashConfig, WordNgramHasher
from engram.metrics import compute_perplexity, compute_batch_perplexities, bootstrap_ci


# ── OOD Dataset Loading ──────────────────────────────────────────────


class GenericLMDataset(Dataset):
    """Generic LM dataset: tokenize and chunk into fixed-length sequences."""

    def __init__(self, token_ids: list[int], seq_len: int = 512):
        self.seq_len = seq_len
        n_seqs = len(token_ids) // seq_len
        self.tokens = torch.tensor(
            token_ids[: n_seqs * seq_len], dtype=torch.long
        ).view(n_seqs, seq_len)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def load_lambada(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load LAMBADA test set."""
    from datasets import load_dataset

    ds = load_dataset("lambada", split="test")
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_wikitext2(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load WikiText-2 test set (different domain split from WikiText-103)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_c4(tokenizer, seq_len: int = 512, max_tokens: int = 5_000_000):
    """Load C4 validation sample (web text)."""
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/c4", "en", split="validation", streaming=True,
    )
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_ptb(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load Penn Treebank test set (news/WSJ text)."""
    from datasets import load_dataset

    ds = load_dataset("ptb_text_only", "penn_treebank", split="test",
                      trust_remote_code=True)
    all_ids = []
    for ex in ds:
        text = ex["sentence"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_pg19(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load PG-19 test set (Project Gutenberg literary fiction)."""
    from datasets import load_dataset

    ds = load_dataset("deepmind/pg19", split="test", streaming=True,
                      trust_remote_code=True)
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_openwebtext(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load OpenWebText sample (Reddit-filtered web text)."""
    from datasets import load_dataset

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                      trust_remote_code=True)
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


def load_arxiv(tokenizer, seq_len: int = 512, max_tokens: int = 2_000_000):
    """Load arXiv papers test set (scientific text)."""
    from datasets import load_dataset

    ds = load_dataset("ccdv/arxiv-summarization", split="test",
                      trust_remote_code=True)
    all_ids = []
    for ex in ds:
        text = ex.get("article", ex.get("text", ""))
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
    return GenericLMDataset(all_ids, seq_len)


DATASET_LOADERS = {
    "lambada": load_lambada,
    "wikitext2": load_wikitext2,
    "ptb": load_ptb,
    "c4": load_c4,
    "pg19": load_pg19,
    "openwebtext": load_openwebtext,
    "arxiv": load_arxiv,
}


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_condition(
    model_name: str,
    memory: EngramMemory | None,
    condition: str,
    adaptor_path: str | None,
    dataset: Dataset,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 16,
    canon_mode: str = "word_boundary",
) -> dict:
    """Evaluate a single condition on a dataset."""

    wrapper = BackboneWrapper(
        model_name=model_name,
        memory=memory,
        condition=condition,
        device=device,
        dtype=dtype,
    )

    # Load adaptor weights if provided
    if adaptor_path and Path(adaptor_path).exists() and wrapper.adaptor is not None:
        state_dict = torch.load(adaptor_path, map_location="cpu", weights_only=True)
        wrapper.adaptor.load_state_dict(state_dict)
        wrapper.adaptor.to(device)

    # Set up canonicalization
    tokenizer = wrapper.tokenizer
    set_canon_fn = None

    if memory is not None:
        with open(args.memory_config) as f:
            mem_cfg_dict = json.load(f)

        hash_cfg = HashConfig(
            max_ngram=mem_cfg_dict["max_ngram"],
            heads_per_order=mem_cfg_dict["heads_per_order"],
            table_size=mem_cfg_dict["table_size"],
            seed=mem_cfg_dict.get("hash_seed", mem_cfg_dict.get("seed", 0)),
        )

        if canon_mode == "word_boundary":
            wb_canon = WordBoundaryCanonicalizer(tokenizer, max_ngram=hash_cfg.max_ngram)
            wb_hasher = WordNgramHasher(hash_cfg)

            def set_canon_fn(input_ids):
                word_ngrams = wb_canon.compute_word_ngrams(input_ids)
                hash_indices = wb_hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
                wrapper.set_hash_indices(hash_indices)
        else:
            # Vocab mode: use VocabCanonicalizer with ID map
            canonicalizer = build_canonicalizer(tokenizer, mode="vocab", max_ngram=hash_cfg.max_ngram)
            canon_id_map = canonicalizer.build_id_map(tokenizer).to(device)

            def set_canon_fn(input_ids):
                canon_ids = canon_id_map[input_ids]
                wrapper.set_canon_ids(canon_ids)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True,
    )

    ppls = compute_batch_perplexities(
        wrapper, dataloader, device, set_canon_fn=set_canon_fn,
    )

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(ppls)

    return {
        "mean_ppl": mean_ppl,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(ppls)),
        "n_batches": len(ppls),
    }


# ── Main ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-domain evaluation")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--adaptor-dir", type=str, required=True,
                        help="Directory containing adaptor.pt (or adaptor_best.pt)")
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--datasets", nargs="+", default=["lambada", "wikitext2", "c4"],
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--canon-mode", type=str, default="word_boundary",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    global args
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            dtype = torch.bfloat16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    results_file = output_dir / "ood_results.json"
    if results_file.exists():
        print(f"Results already exist at {results_file}, skipping.")
        return

    print(f"Target model: {args.target_model}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Datasets: {args.datasets}")

    # Load memory
    with open(args.memory_config) as f:
        mem_cfg_dict = json.load(f)

    mem_cfg = MemoryConfig(
        max_ngram=mem_cfg_dict["max_ngram"],
        heads_per_order=mem_cfg_dict["heads_per_order"],
        table_size=mem_cfg_dict["table_size"],
        d_head=mem_cfg_dict["d_head"],
        hash_seed=mem_cfg_dict["hash_seed"],
    )

    # Load tokenizer for dataset loading
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    # Find adaptor checkpoint
    adaptor_dir = Path(args.adaptor_dir)
    adaptor_path = adaptor_dir / "adaptor_best.pt"
    if not adaptor_path.exists():
        adaptor_path = adaptor_dir / "adaptor.pt"
    if not adaptor_path.exists():
        raise FileNotFoundError(f"No adaptor checkpoint found in {adaptor_dir}")

    print(f"Adaptor: {adaptor_path}")

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Evaluating on: {ds_name}")
        print(f"{'='*60}")

        try:
            # Load dataset
            print(f"Loading {ds_name}...")
            dataset = DATASET_LOADERS[ds_name](tokenizer, seq_len=args.seq_len)
            print(f"  {len(dataset)} sequences ({len(dataset) * args.seq_len:,} tokens)")

            ds_results = {}

            # Baseline (no memory)
            print(f"\n  [1/2] Baseline (no memory)...")
            t0 = time.time()
            baseline_res = evaluate_condition(
                model_name=args.target_model,
                memory=None,
                condition="baseline",
                adaptor_path=None,
                dataset=dataset,
                device=device,
                dtype=dtype,
                batch_size=args.batch_size,
                canon_mode=args.canon_mode,
            )
            baseline_res["elapsed_s"] = time.time() - t0
            ds_results["baseline"] = baseline_res
            print(f"    PPL: {baseline_res['mean_ppl']:.2f} [{baseline_res['ci_lower']:.2f}, {baseline_res['ci_upper']:.2f}]")

            # Clear GPU memory before loading next condition
            torch.cuda.empty_cache()

            # Transferred memory
            print(f"\n  [2/2] Transferred memory...")
            memory = EngramMemory(mem_cfg)
            memory.load_state_dict(
                torch.load(args.source_memory, map_location="cpu", weights_only=True)
            )
            for p in memory.parameters():
                p.requires_grad = False

            t0 = time.time()
            transferred_res = evaluate_condition(
                model_name=args.target_model,
                memory=memory,
                condition="transferred",
                adaptor_path=str(adaptor_path),
                dataset=dataset,
                device=device,
                dtype=dtype,
                batch_size=args.batch_size,
                canon_mode=args.canon_mode,
            )
            transferred_res["elapsed_s"] = time.time() - t0
            ds_results["transferred"] = transferred_res
            print(f"    PPL: {transferred_res['mean_ppl']:.2f} [{transferred_res['ci_lower']:.2f}, {transferred_res['ci_upper']:.2f}]")

            # Compute delta
            delta_ppl = transferred_res["mean_ppl"] - baseline_res["mean_ppl"]
            rel_change = delta_ppl / baseline_res["mean_ppl"] * 100
            ds_results["delta_ppl"] = delta_ppl
            ds_results["relative_change_pct"] = rel_change

            print(f"\n  Delta: {delta_ppl:+.2f} PPL ({rel_change:+.1f}% relative)")

            all_results[ds_name] = ds_results

            # Free GPU
            torch.cuda.empty_cache()
            del memory

        except Exception as e:
            print(f"\n  !! ERROR on {ds_name}: {e}")
            print(f"  Skipping {ds_name}, continuing with remaining datasets...")
            torch.cuda.empty_cache()
            continue

    # Save results
    final = {
        "target_model": args.target_model,
        "adaptor_dir": args.adaptor_dir,
        "seed": args.seed,
        "canon_mode": args.canon_mode,
        "datasets": all_results,
    }
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print("  OOD Evaluation Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Baseline':>10} {'Transferred':>12} {'Delta':>8} {'Rel%':>8}")
    print("-" * 52)
    for ds_name, res in all_results.items():
        bl = res["baseline"]["mean_ppl"]
        tr = res["transferred"]["mean_ppl"]
        d = res["delta_ppl"]
        r = res["relative_change_pct"]
        print(f"{ds_name:<12} {bl:>10.2f} {tr:>12.2f} {d:>+8.2f} {r:>+7.1f}%")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
