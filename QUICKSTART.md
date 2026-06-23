# Quickstart (5 minutes to first run)

## 1. Install uv (skip if you already have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install dependencies

```bash
bash run/00_install.sh
```

Resolves `uv.lock` into `.venv/` and smoke-tests that `engram.*` imports. Takes under 5 minutes.

## 3. Run the pilot (30 min)

```bash
bash run/01_pilot.sh
```

Trains a source memory on Pythia-160M (2M tokens), then runs four conditions (`baseline`, `transferred`, `random_memory`, `train_from_scratch`) on Pythia-410M with a single seed. Outputs land in `results/pilot/` and `results/pilot_source/`.

## 4. Inspect the pilot output

```bash
for f in results/pilot/*/results.json; do
    echo "=== $f ==="
    cat "$f" | python -m json.tool | head -20
done
```

Success criteria printed by `scripts/run_pilot.py`:
- backbone gradient norms == 0 (frozen backbone enforced)
- memory-table gradients == 0 in Phase 2 (memory frozen after source training)
- `random_memory` perplexity ≈ `baseline` (no transfer signal from random keys)

## 5. Next steps

See `README.md` and `run/README.md` for full-paper reproduction.

For a single paper-section reproduction, pick one of:

```bash
bash run/02_source_memories.sh        # Phase 1 (days on H100)
bash run/14_cka_extended.sh           # §5.1.6 CKA matrix (~1 h)
bash run/10_tier1_same_tokenizer.sh   # §5.1.1 Pythia 160M→410M (~8 h)
```

All runners are idempotent — re-running resumes from the last completed checkpoint.
