# XMemTransfer Code Release

Implementation accompanying the XMemTransfer project on cross-model transfer of frozen Engram memory.

Project links:

- Website: [https://olaresearch.org/XMemTransfer](https://olaresearch.github.io/XMemTransfer)
- Code: [https://github.com/OLAResearch](https://github.com/OLAResearch)
- Models: [https://huggingface.co/OLAResearchX](https://huggingface.co/OLAResearchX)

## Abstract

We study cross-model transfer of a fixed-capacity conditional-memory module (*Engram*) between heterogeneous LLM backbones. The framework decomposes transfer into two phases: (1) key-space unification via a tokenizer-agnostic canonicalization, and (2) value-space alignment via a lightweight LoRA-scale adaptor with a learned gate. Across four model families (Pythia, TinyLlama, QWen3.5, Phi-4-mini), a 3×3 cross-architecture transfer matrix shows positive perplexity gains in every cell (up to −15.7%), and downstream evaluation on six knowledge-intensive tasks shows selective improvements on factual retrieval (BoolQ +14.6% on QWen-2B with domain-matched memory).

## Repository layout

```
.
├── engram/          Core library (memory, adaptor, canonicalizer, hashing, gating)
├── scripts/         Runnable entry points (Phase 1/2 training, evaluation, analysis)
├── configs/         YAML configs for each experiment
├── tests/           Unit tests (canonicalization, freezing, data splits, integration)
├── run/             Turnkey shell wrappers, one per paper section
├── pyproject.toml   Python project definition (uv-managed)
└── uv.lock          Fully-pinned lockfile for reproducibility
```

## Installation

Prerequisites: Python ≥ 3.11, a CUDA-capable GPU, and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
bash run/00_install.sh
```

This runs `uv sync`, which resolves and installs all dependencies from `uv.lock` into `.venv/`, then smoke-tests the imports.

## Quickstart

A 30-minute pilot run validates the pipeline end-to-end on Pythia-160M → Pythia-410M:

```bash
bash run/01_pilot.sh
```

The pilot checks: frozen-backbone gradient norms are zero, random-memory control does not improve over baseline, checkpoints round-trip correctly.

## Reproducing paper results

Experiments are grouped by paper section under `run/`. Each script is idempotent (skips completed runs).

| Paper section | Command |
|---|---|
| §5.1.1 Tier 1 same-tokenizer | `bash run/10_tier1_same_tokenizer.sh` |
| §5.1.2 Tier 2 cross-tokenizer | `bash run/11_tier2_cross_tokenizer.sh` |
| §5.1.3 Tier 3 ablations | `bash run/12_tier3_ablations.sh` |
| §5.1.4 Tier 4 scaling | `bash run/13_tier4_scaling.sh` |
| §5.1.6 Extended CKA | `bash run/14_cka_extended.sh` |
| §5.2 Downstream (FW matched) | `bash run/20_downstream_fw_matched.sh` |
| §5.2 Domain alignment | `bash run/21_domain_alignment.sh` |
| §5.2 Corpus control | `bash run/22_corpus_control.sh` |
| Iso-parameter baselines | `bash run/30_baselines.sh` |
| Figures & analysis | `bash run/40_analysis.sh` |

The Phase 1 source memories (one per source-model + corpus combination) must be trained first:

```bash
bash run/02_source_memories.sh
```

See `run/README.md` for per-script runtime estimates and dependencies.

## Expected outputs

Each run produces a self-contained results directory with:

- `results.json` — perplexity / accuracy deltas, seed, wall-clock, config snapshot
- `adaptor.pt` or `adaptor_best.pt` — adaptor checkpoint (≈ 1–2 M parameters)
- `memory.pt` + `memory_config.json` — for source-memory runs
- `training_log.jsonl` — per-step loss, gradient norms, gate activations

Aggregation and figure generation are handled by `run/40_analysis.sh`.

## Hardware notes

- **Pythia-160M / TinyLlama runs**: fit on a single 24 GB GPU (RTX 4090, A6000).
- **QWen3.5-4B source training**: 80 GB recommended; gradient-checkpointing enabled.
- **QWen3.5-9B source training (200M tokens, 8-bit optimizer)**: ≈ 96 h on H100 80 GB.
- **Downstream evaluation**: CPU-bound tokenization dominates; any GPU works.
- All model weights download from the HuggingFace Hub on first use.

## License

Apache 2.0 — see `LICENSE`.
