"""Preprocess a text corpus into MLPMemory-style sliding windows.

This mirrors the official MLPMemory preprocessing pattern:
  1. tokenize with the target model tokenizer
  2. build overlapping fixed-length windows
  3. mask duplicate-label positions with `padding_index`
  4. save the processed DatasetDict to disk

For Wikipedia-2021 transfer experiments, run this once with the target
tokenizer (for example, Llama-2-7b) and then point training/eval scripts to
the resulting local dataset directory.
"""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess corpus for Engram/MLPMemory experiments")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="HF dataset repo, local save_to_disk directory, or local json/jsonl file")
    parser.add_argument("--dataset-config-name", type=str, default=None,
                        help="Optional HF dataset config name")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Text column to tokenize")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Tokenizer/model path used to preprocess the corpus")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--padding-index", type=int, default=-100)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.002,
                        help="If no test split exists, carve out this fraction from train")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_raw_datasets(dataset_name: str, dataset_config_name: str | None):
    dataset_path = Path(dataset_name)

    if dataset_path.exists():
        if dataset_path.is_dir():
            try:
                return load_from_disk(str(dataset_path))
            except Exception:
                pass

        suffixes = dataset_path.suffixes
        if suffixes and suffixes[-1] in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(dataset_path))

    return load_dataset(dataset_name, dataset_config_name)


def normalize_dataset_splits(raw_datasets, test_size: float, seed: int) -> DatasetDict:
    if isinstance(raw_datasets, DatasetDict):
        dataset_dict = DatasetDict(raw_datasets)
    else:
        dataset_dict = DatasetDict({"train": raw_datasets})

    if "train" not in dataset_dict:
        first_split = next(iter(dataset_dict.keys()))
        dataset_dict = DatasetDict({"train": dataset_dict[first_split]})

    if "test" not in dataset_dict:
        for candidate in ("validation", "valid", "dev"):
            if candidate in dataset_dict:
                dataset_dict = DatasetDict({
                    "train": dataset_dict["train"],
                    "test": dataset_dict[candidate],
                })
                break
        else:
            split_dataset = dataset_dict["train"].train_test_split(
                test_size=test_size,
                seed=seed,
                shuffle=True,
            )
            dataset_dict = DatasetDict({
                "train": split_dataset["train"],
                "test": split_dataset["test"],
            })

    return dataset_dict


def tokenize_and_group_text(
    raw_datasets: DatasetDict,
    tokenizer,
    text_column: str,
    block_size: int,
    stride: int,
    padding_index: int,
    num_proc: int,
) -> DatasetDict:
    if text_column not in raw_datasets["train"].column_names:
        raise KeyError(
            f"Text column {text_column!r} not found. "
            f"Available columns: {raw_datasets['train'].column_names}"
        )

    def tokenize_function(examples):
        return tokenizer(examples[text_column], add_special_tokens=False)

    def group_texts(examples):
        concatenated_examples = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        input_ids = []
        labels = []
        attention_mask = []

        total_chunks = len(range(0, total_length, stride))
        for offset in tqdm(range(0, total_length, stride), total=total_chunks):
            begin_loc = max(offset + stride - block_size, 0)
            end_loc = min(offset + stride, total_length)
            target_length = end_loc - offset

            cur_input_ids = concatenated_examples["input_ids"][begin_loc:end_loc]
            cur_labels = list(cur_input_ids)
            cur_labels[:-target_length] = [padding_index] * (len(cur_labels) - target_length)

            if len(cur_input_ids) < block_size:
                pad_token_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                )
                padding_size = block_size - len(cur_input_ids)
                cur_input_ids += [pad_token_id] * padding_size
                cur_labels += [padding_index] * padding_size

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
            attention_mask.append([1] * len(cur_labels))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=True,
        desc="Tokenizing dataset",
    )

    return tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc=f"Grouping texts into block_size={block_size}, stride={stride}",
    )


def add_dstore_ranges(lm_datasets: DatasetDict, padding_index: int) -> dict:
    dstore_summary = {}

    for split_name, split_dataset in lm_datasets.items():
        dataset_cnt = []
        dstore_size = 0

        for chunk in split_dataset["labels"]:
            cur_len = len([token for token in chunk[1:] if token != padding_index])
            dataset_cnt.append(cur_len)
            dstore_size += cur_len

        dstore_range = []
        idx = 0
        for cnt in dataset_cnt:
            dstore_range.append((idx, idx + cnt))
            idx += cnt

        lm_datasets[split_name] = split_dataset.add_column("dstore_range", dstore_range)
        dstore_summary[split_name] = {
            "dstore_size": dstore_size,
            "dataset_cnt_len": len(dataset_cnt),
        }

    return dstore_summary


def save_preprocessing_config(output_dir: Path, args) -> None:
    config = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "text_column": args.text_column,
        "tokenizer_path": args.tokenizer_path,
        "block_size": args.block_size,
        "stride": args.stride,
        "padding_index": args.padding_index,
        "num_proc": args.num_proc,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    with open(output_dir / "preprocessing_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()

    raw_datasets = load_raw_datasets(args.dataset_name, args.dataset_config_name)
    raw_datasets = normalize_dataset_splits(raw_datasets, test_size=args.test_size, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_datasets = tokenize_and_group_text(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        text_column=args.text_column,
        block_size=args.block_size,
        stride=args.stride,
        padding_index=args.padding_index,
        num_proc=args.num_proc,
    )
    dstore_summary = add_dstore_ranges(lm_datasets, padding_index=args.padding_index)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lm_datasets.save_to_disk(str(output_dir))
    save_preprocessing_config(output_dir, args)

    with open(output_dir / "dstore_summary.json", "w") as f:
        json.dump(dstore_summary, f, indent=2)

    print(f"Saved preprocessed dataset to: {output_dir}")
    print(f"Tokenizer alignment: {args.tokenizer_path}")
    for split_name, summary in dstore_summary.items():
        print(
            f"  {split_name}: dstore_size={summary['dstore_size']:,}, "
            f"windows={summary['dataset_cnt_len']:,}"
        )


if __name__ == "__main__":
    main()
