"""Perplexity computation and bootstrap confidence intervals."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity from logits and labels.

    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target token IDs
        ignore_index: Token ID to ignore in loss computation

    Returns:
        Perplexity (exp of mean cross-entropy loss)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return float(torch.exp(loss).item())


def compute_batch_perplexities(
    model,
    dataloader,
    device: torch.device,
    set_canon_fn=None,
    max_batches: Optional[int] = None,
) -> List[float]:
    """Compute per-batch perplexities on a dataset.

    Args:
        model: BackboneWrapper or similar
        dataloader: DataLoader yielding dicts with input_ids and labels
        device: Computation device
        set_canon_fn: Optional function to set canon_ids before each batch
        max_batches: Maximum number of batches to evaluate

    Returns:
        List of per-batch perplexity values
    """
    model.eval()
    perplexities = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if set_canon_fn is not None:
                set_canon_fn(input_ids)

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            ppl = compute_perplexity(logits, labels)
            perplexities.append(ppl)

    return perplexities


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values
        n_resamples: Number of bootstrap resamples
        ci: Confidence level

    Returns:
        (mean, lower_bound, upper_bound)
    """
    arr = np.array(values)
    rng = np.random.RandomState(seed)
    n = len(arr)

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(arr))

    return mean, lower, upper


def paired_bootstrap_delta(
    values_a: List[float],
    values_b: List[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute paired bootstrap CI for the difference (A - B).

    Uses paired resampling (same batch indices) for tighter estimates.

    Args:
        values_a: Per-batch values for condition A
        values_b: Per-batch values for condition B
        n_resamples: Number of bootstrap resamples
        ci: Confidence level

    Returns:
        Dict with delta_mean, ci_lower, ci_upper
    """
    a = np.array(values_a)
    b = np.array(values_b)
    assert len(a) == len(b), "Paired bootstrap requires equal-length arrays"

    deltas = a - b
    rng = np.random.RandomState(seed)
    n = len(deltas)

    boot_deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[i] = np.mean(deltas[idx])

    alpha = 1 - ci
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(boot_deltas, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_deltas, 100 * (1 - alpha / 2))),
    }
