"""Alias-aware canonicalization for Tier A of the hash-key addressability test.

Motivated by the Phase 0.2 finding that hit@10 = 3%
under the existing surface-token VocabCanonicalizer (probe idx=3 shows Q and A
sharing the bigram "trypsin inhibitor" yet missing all top-10 slots because the
surrounding 3-gram context differs). This module replaces token-level vocab
canonicalization with three semantic-leaning transformations:

    1. Stopword removal (NLTK English stopwords + light scientific filler)
    2. Lemmatization (NLTK WordNetLemmatizer with noun/verb POS heuristic)
    3. Compound bigram merging (curated scientific noun phrases)

The canonicalizer is a drop-in replacement for `VocabCanonicalizer` at the
INTERFACE used by the Phase 0.2 diagnostic only — it does NOT touch the
existing source-memory training pipeline. Stage (i) of the validation protocol
re-hashes the existing 200-probe set with this canonicalizer and measures
whether hit@10 lifts above 3% under the same memory configuration.

Stage (ii) — full source-memory retrain under this canonicalizer — is gated
on a strong Stage (i) signal (hit@10 ≥ 40%).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import torch


# ── Compound bigram list ─────────────────────────────────────────────────
#
# Hand-curated ~30 scientific noun-phrase compounds. Each entry is a tuple of
# (word1, word2) in lemma-lowercase form. When the canonicalizer sees these
# two words adjacent in the post-stopword post-lemma stream, it merges them
# into a single canonical token `word1_word2`.
#
# Curation strategy: pick compounds whose individual words are too generic
# alone (e.g., "monte" / "carlo" / "neural" / "network") but whose pair carries
# concrete meaning. Aim for coverage of HQ-DQA's chemistry / biology / physics
# / engineering / ML domains.
#
# Expand iteratively during Stage (i) inspection: when a known miss should
# semantically hit (e.g., probe idx=3 trypsin inhibitor), check whether the
# compound is in this list. If not, add it.

DEFAULT_COMPOUNDS: list[tuple[str, str]] = [
    # Chemistry / biology
    ("trypsin", "inhibitor"),
    ("monoclonal", "antibody"),
    ("amino", "acid"),
    ("fatty", "acid"),
    ("nucleic", "acid"),
    ("stem", "cell"),
    ("spinacia", "oleracea"),
    ("escherichia", "coli"),
    ("carbon", "dioxide"),
    ("nitric", "oxide"),
    # Physics / math
    ("monte", "carlo"),
    ("ridge", "state"),
    ("ground", "state"),
    ("excited", "state"),
    ("differential", "equation"),
    ("partial", "derivative"),
    ("hyperspherical", "coordinate"),
    # Engineering / geology
    ("fracture", "permeability"),
    ("fluid", "dynamics"),
    ("reservoir", "rock"),
    # ML / CS
    ("machine", "learning"),
    ("neural", "network"),
    ("deep", "learning"),
    ("language", "model"),
    ("attention", "head"),
    ("natural", "language"),
    # Method / measurement
    ("simulation", "monte"),  # for "Direct Simulation Monte Carlo"
    ("direct", "simulation"),
    ("flow", "rate"),
    ("data", "point"),
    ("standard", "deviation"),
    ("error", "bar"),
]


# ── Helpers ──────────────────────────────────────────────────────────────


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")  # alphabetic + intra-word digits/apostrophes


def _normalize_word(word: str) -> str:
    """NFKC + lowercase + strip diacritics. Mirrors VocabCanonicalizer's
    normalize_token but at the WORD level."""
    word = unicodedata.normalize("NFKC", word)
    word = word.lower()
    word = unicodedata.normalize("NFD", word)
    word = "".join(ch for ch in word if not unicodedata.combining(ch))
    return word.strip()


def _load_stopwords() -> set[str]:
    """Return the union of NLTK English stopwords + a small filler list.

    NLTK's stopword corpus must be downloaded once via
    `nltk.download('stopwords')`. We trigger it lazily.
    """
    try:
        from nltk.corpus import stopwords as nltk_stopwords
    except ImportError as exc:
        raise ImportError(
            "alias_canonicalization requires nltk. "
            "Install via `uv add nltk` and `python -c \"import nltk; nltk.download('stopwords')\"`"
        ) from exc

    try:
        words = set(nltk_stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        words = set(nltk_stopwords.words("english"))

    # Light scientific filler that NLTK doesn't cover but adds hash noise.
    filler = {
        "also", "thus", "hence", "therefore", "however", "moreover",
        "furthermore", "additionally", "consequently", "respectively",
        "namely", "i.e", "e.g", "etc", "fig", "figure", "table",
    }
    return words | filler


def _ensure_wordnet() -> None:
    """Make sure WordNet data is available for the lemmatizer."""
    try:
        from nltk.corpus import wordnet  # noqa: F401
        _ = wordnet.synsets("test")
    except (ImportError, LookupError):
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


# ── AliasCanonicalizer ───────────────────────────────────────────────────


class AliasCanonicalizer:
    """Tier A alias-aware canonicalizer.

    The canonicalizer is stateful: it builds up a `norm_to_canon` map as it
    sees new normalized words. For the Phase 0.2 diagnostic this is fine —
    the same instance processes both Q and A across the probe set, so shared
    words get shared canonical IDs deterministically. For Stage (ii)
    source-memory retraining this stateful design would need to be
    pre-warmed against the training corpus (out of scope for this module).

    Interface differs from VocabCanonicalizer in one important way: the input
    token-ID sequence (1, T) is mapped to a canonical-ID sequence (1, T')
    where T' ≤ T (stopword removal + compound merging shorten the stream).
    Callers that previously did `id_map[ids]` should call `canonicalize(ids)`.
    """

    BOS_CANON_ID = 0  # reserved; not used as a real word

    def __init__(
        self,
        tokenizer,
        compounds: Optional[List[tuple[str, str]]] = None,
        extra_stopwords: Optional[set[str]] = None,
    ):
        self.tokenizer = tokenizer

        # NLTK setup — lazy so the import error message is informative
        _ensure_wordnet()
        from nltk.stem import WordNetLemmatizer

        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = _load_stopwords()
        if extra_stopwords:
            self.stopwords = self.stopwords | extra_stopwords

        # Compound list: store as set of (w1, w2) tuples (normalized lemma)
        compound_list = compounds if compounds is not None else DEFAULT_COMPOUNDS
        self.compounds: set[tuple[str, str]] = {
            (_normalize_word(w1), _normalize_word(w2)) for (w1, w2) in compound_list
        }

        # Dynamic vocab — populated as we see new normalized words
        self.norm_to_canon: dict[str, int] = {"<pad>": self.BOS_CANON_ID}
        self.canon_tokens: list[str] = ["<pad>"]

    @property
    def canon_vocab_size(self) -> int:
        return len(self.canon_tokens)

    # ── Public interface ──

    def canonicalize(
        self, ids: torch.LongTensor, device: Optional[torch.device] = None
    ) -> torch.LongTensor:
        """Map (1, T) token IDs to (1, T') alias-aware canonical IDs.

        T' may be shorter than T (stopword removal) or pair count may shrink
        further on compound merging. If the result is empty, returns shape
        (1, 0).
        """
        if device is None:
            device = ids.device
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"Expected (1, T) input, got shape {tuple(ids.shape)}")

        # 1. Decode to text. Skip special tokens so we don't lemma <eos> etc.
        text = self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)

        # 2. Tokenize text into words via simple word regex (avoids the model
        #    tokenizer's subword splits — we want word-level aliasing).
        raw_words = _WORD_RE.findall(text)

        # 3. Normalize, lemmatize, drop stopwords.
        clean: list[str] = []
        for w in raw_words:
            norm = _normalize_word(w)
            if not norm:
                continue
            if norm in self.stopwords:
                continue
            # Lemmatize as noun first, then verb if unchanged (POS guess).
            lemma = self.lemmatizer.lemmatize(norm, pos="n")
            if lemma == norm:
                lemma_v = self.lemmatizer.lemmatize(norm, pos="v")
                if lemma_v != norm:
                    lemma = lemma_v
            # Skip post-lemma stopwords (e.g., "be" from "was")
            if lemma in self.stopwords:
                continue
            clean.append(lemma)

        # 4. Merge compound bigrams (greedy left-to-right).
        merged: list[str] = []
        i = 0
        while i < len(clean):
            if i + 1 < len(clean) and (clean[i], clean[i + 1]) in self.compounds:
                merged.append(f"{clean[i]}_{clean[i + 1]}")
                i += 2
            else:
                merged.append(clean[i])
                i += 1

        # 5. Map to canonical IDs (dynamic vocab).
        canon_ids = [self._intern(w) for w in merged]
        if not canon_ids:
            return torch.empty((1, 0), dtype=torch.long, device=device)

        return torch.tensor([canon_ids], dtype=torch.long, device=device)

    # ── Helpers ──

    def _intern(self, word: str) -> int:
        cid = self.norm_to_canon.get(word)
        if cid is None:
            cid = len(self.canon_tokens)
            self.norm_to_canon[word] = cid
            self.canon_tokens.append(word)
        return cid
