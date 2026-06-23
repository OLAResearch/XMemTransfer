"""Cross-tokenizer collision measurement.

Measures how often two different tokenizers produce the same hash indices
for the same input text, quantifying canonicalization effectiveness.
"""

from typing import List, Tuple

import torch
import numpy as np

from .canonicalization import WordBoundaryCanonicalizer
from .hashing import HashConfig, WordNgramHasher


class CollisionAnalyzer:
    """Measure hash collision and agreement rates across tokenizer pairs."""

    def __init__(self, hash_cfg: HashConfig):
        self.hash_cfg = hash_cfg
        self.hasher = WordNgramHasher(hash_cfg)

    def measure_agreement(
        self,
        texts: List[str],
        tokenizer_a,
        tokenizer_b,
        max_ngram: int = 3,
    ) -> dict:
        """Measure key-space agreement between two tokenizers.

        For each input text, tokenize with both tokenizers, compute
        word-level N-gram hashes, and measure agreement rate.

        Args:
            texts: Input texts to analyze
            tokenizer_a: Source tokenizer (HF)
            tokenizer_b: Target tokenizer (HF)
            max_ngram: N-gram order

        Returns:
            Dict with per-order agreement rates and overall stats
        """
        canon_a = WordBoundaryCanonicalizer(tokenizer_a, max_ngram)
        canon_b = WordBoundaryCanonicalizer(tokenizer_b, max_ngram)

        per_order_matches = {n: [] for n in range(2, max_ngram + 1)}

        for text in texts:
            # Tokenize
            ids_a = tokenizer_a(text, return_tensors="pt")["input_ids"]
            ids_b = tokenizer_b(text, return_tensors="pt")["input_ids"]

            # Get word N-grams
            ngrams_a = canon_a.compute_word_ngrams(ids_a)
            ngrams_b = canon_b.compute_word_ngrams(ids_b)

            # Extract completed word sequences for comparison
            words_a = self._extract_word_sequence(canon_a, ids_a)
            words_b = self._extract_word_sequence(canon_b, ids_b)

            # Compare word sequences
            for n in range(2, max_ngram + 1):
                ngrams_set_a = set()
                ngrams_set_b = set()

                for i in range(n, len(words_a) + 1):
                    ngrams_set_a.add(tuple(words_a[i - n : i]))
                for i in range(n, len(words_b) + 1):
                    ngrams_set_b.add(tuple(words_b[i - n : i]))

                if ngrams_set_a or ngrams_set_b:
                    intersection = ngrams_set_a & ngrams_set_b
                    union = ngrams_set_a | ngrams_set_b
                    agreement = len(intersection) / len(union) if union else 1.0
                    per_order_matches[n].append(agreement)

        results = {}
        for n in range(2, max_ngram + 1):
            matches = per_order_matches[n]
            results[f"order_{n}_mean_agreement"] = float(np.mean(matches)) if matches else 0.0
            results[f"order_{n}_std_agreement"] = float(np.std(matches)) if matches else 0.0

        all_matches = []
        for v in per_order_matches.values():
            all_matches.extend(v)
        results["overall_mean_agreement"] = float(np.mean(all_matches)) if all_matches else 0.0

        return results

    def _extract_word_sequence(
        self, canon: WordBoundaryCanonicalizer, input_ids: torch.LongTensor
    ) -> List[str]:
        """Extract the sequence of canonical words from tokenized input."""
        B, T = input_ids.shape
        words = []
        current_parts = []

        for t in range(T):
            tid = input_ids[0, t].item()
            tok_str = canon.id_to_surface.get(tid, "")

            if canon.is_word_boundary(tok_str) and current_parts:
                from .canonicalization import normalize_token
                word = normalize_token("".join(current_parts))
                if word:
                    words.append(word)
                current_parts = []

            from .canonicalization import _strip_prefix
            surface = _strip_prefix(tok_str)
            if surface:
                current_parts.append(surface)

        # Flush remaining
        if current_parts:
            from .canonicalization import normalize_token
            word = normalize_token("".join(current_parts))
            if word:
                words.append(word)

        return words
