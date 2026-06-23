"""Gate activation recording and statistics for analysis."""

from typing import Optional

import torch
import numpy as np


class GateAnalyzer:
    """Records and analyzes gate activation values across training/eval."""

    def __init__(self):
        self._recordings: list[torch.Tensor] = []

    def record(self, gate_values: Optional[torch.Tensor]) -> None:
        """Record a batch of gate values."""
        if gate_values is not None:
            self._recordings.append(gate_values.cpu().float())

    def clear(self) -> None:
        self._recordings = []

    def get_all(self) -> torch.Tensor:
        """Return all recorded gate values as a flat tensor."""
        if not self._recordings:
            return torch.tensor([])
        return torch.cat([g.flatten() for g in self._recordings])

    def compute_stats(self) -> dict:
        """Compute summary statistics over all recorded gate values."""
        all_gates = self.get_all()
        if all_gates.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}

        arr = all_gates.numpy()
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "frac_below_01": float(np.mean(arr < 0.1)),
            "frac_above_09": float(np.mean(arr > 0.9)),
        }
