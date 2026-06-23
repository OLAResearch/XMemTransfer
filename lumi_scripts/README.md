# LUMI Scripts

These scripts follow the official LUMI AI Guide pattern:

- run from `/project` or `/scratch`, not `$HOME`
- load `lumi-aif-singularity-bindings`
- use a LUMI AI Factory PyTorch container
- layer a small Python environment on top of the container and optionally pack it as `sqsh`

References:

- Quickstart: <https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart>
- Environment setup: <https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment>

## Files

- `create_overlay_venv.sh`
  Creates a virtualenv on top of the container with `--system-site-packages`
  and packs it into `.sqsh` if `mksquashfs` is available.
- `run_mistral_minimal_openqa.slurm`
  Single-GPU smoke/minimal pipeline for:
  1. source memory training on `Wikipedia-2021`
  2. transferred adaptor fitting on `Wikipedia-2021`
  3. OpenQA eval on `nq webqa triviaqa truthfulqa hotpotqa`
- `train_wiki2021_source.slurm`
  Generic source-memory trainer for `Wikipedia-2021`, parameterized by
  `MODEL`, `MAX_TOKENS`, `TABLE_SIZE`, and early-stopping settings.
- `train_wiki2021_transfer.slurm`
  Generic transferred-adaptor trainer for `Wikipedia-2021`, parameterized by
  `TARGET_MODEL`, `SOURCE_OUT`, `MAX_TOKENS`, and early-stopping settings.
- `eval_transfer_openqa.slurm`
  Generic OpenQA evaluator for a source-memory/adaptor pair.
- `submit_mistral_from_llama2_wiki2021_table_scale.sh`
  Submits the `Llama-2-7B -> Mistral-7B` table-size scaling experiment on
  `Wikipedia-2021` for `2x/4x/6x` Engram capacity.
- `submit_mistral_from_llama2_wiki2021_token_scale.sh`
  Submits the `Llama-2-7B -> Mistral-7B` token-scaling experiment at fixed
  Engram size, with `source/adaptor = 15M/25M` and `20M/30M`, and evaluates
  only the `transferred` condition on OpenQA.

## Typical usage

```bash
cd /scratch/<project>/<user>/llm-memory-transfer

bash lumi_scripts/create_overlay_venv.sh

# Edit the --account line first.
sbatch lumi_scripts/run_mistral_minimal_openqa.slurm
```

## Notes

- The provided Slurm script is a minimal validation run, not the full Table 1 reproduction.
- If `mksquashfs` is unavailable, the Slurm script can also bind the unpacked `.lumi-venv/` directory directly.
- By default it evaluates all examples for `nq webqa triviaqa truthfulqa hotpotqa`.
- Set `OPENQA_MAX_EXAMPLES` only when you explicitly want a smoke test cap.
- If the source memory or adaptor artifacts already exist, the Slurm script reuses them and reruns the evaluation stage.
