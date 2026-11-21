"""
language.py — Speech Community Container, Typological Profile, and Inventory Statistics.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple, Sequence
import numpy as np

from phonetics import readout_batch, _ATTRACTOR_SYMBOLS, _V_IDX, _C_IDX
from word import Word

_NONPULMONIC_CHARS = ("ʼ", "ɓ", "ɗ", "ᶑ", "ʄ", "ɠ", "ʛ", "ʘ", "ǀ", "ǃ", "ǂ", "ǁ")
_NONPULM_MASK = np.array([any(ch in sym for ch in _NONPULMONIC_CHARS) for sym in _ATTRACTOR_SYMBOLS], dtype=bool)
_TONE_STRIP = str.maketrans("", "", "ː˥˦˧˨˩")


def inventory_counts(words: Sequence) -> np.ndarray:
    gs = [p.g for w in words for p in w.phonemes]
    if not gs:
        return np.zeros(len(_ATTRACTOR_SYMBOLS), dtype=np.float32)
    fs = [float(getattr(w, "usage_frequency", 1.0)) for w in words for _ in w.phonemes]
    return np.bincount(
        readout_batch(np.asarray(gs, dtype=np.float32)),
        weights=np.asarray(fs, dtype=np.float32),
        minlength=len(_ATTRACTOR_SYMBOLS),
    ).astype(np.float32)


def inventory_size(words: Sequence, min_share: float = 0.01) -> dict[str, int]:
    n = inventory_counts(words)
    tot = float(n.sum())
    if tot <= 0.0:
        return {"vowels": 0, "consonants": 0, "nonpulmonic": 0, "total": 0}
    keep = (n / tot) >= min_share
    v = int(np.sum(keep[_V_IDX]))
    c = int(np.sum(keep[_C_IDX]))
    non_p = int(np.sum(keep & _NONPULM_MASK))
    return {"vowels": v, "consonants": c, "nonpulmonic": non_p, "total": v + c}


def cluster_contrastive_phonemes(words: Sequence, min_share: float = 0.01) -> tuple[list[str], list[str]]:
    v_usage: dict[str, float] = Counter()
    c_usage: dict[str, float] = Counter()

    for w in words:
        u = float(getattr(w, "usage_frequency", 1.0))
        for p in w.phonemes:
            clean = p.ipa.translate(_TONE_STRIP)
            if p.is_vowel:
                v_usage[clean] += u
            else:
                c_usage[clean] += u

    v_tot = sum(v_usage.values()) or 1.0
    c_tot = sum(c_usage.values()) or 1.0
    vowels = [s for s, u in v_usage.most_common() if (u / v_tot) >= min_share]
    consonants = [s for s, u in c_usage.most_common() if (u / c_tot) >= min_share]
    return vowels, consonants


class LanguageProfile(NamedTuple):
    name: str
    head_directionality: float = 0.50
    synthesis_index: float = 0.50
    syllable_complexity: float = 0.50
    max_transition_cost: float = 3.00
    acoustic_roughness: float = 0.50
    alignment: str = "nominative_accusative"
    metrical_bias: float = 0.50

    @property
    def dominant_word_order(self) -> str:
        h = self.head_directionality
        p_sov = math.exp(-6.0 * (h - 0.15) ** 2)
        p_svo = math.exp(-6.0 * (h - 0.50) ** 2)
        p_vso = math.exp(-6.0 * (h - 0.85) ** 2)
        return max((("SOV", p_sov), ("SVO", p_svo), ("VSO", p_vso)), key=lambda x: x[1])[0]


class Language:
    def __init__(
        self,
        name: str,
        profile: LanguageProfile | None = None,
        population: float = 0.50,
    ):
        self.name = name
        self.profile = profile or LanguageProfile(name=f"{name} Profile")
        self.population = float(population)
        self.words: dict[int, Word] = {}
        self.cultural_attention: list[float] = [1.0] * 7
        self._cached_grammar_paradigm = None

    def add_word(self, word: Word) -> Word:
        self.words[word.id] = word
        return word

    def extract_phonemic_inventory(self, paradigm=None) -> tuple[list[str], list[str]]:
        cohort: list[Word] = list(self.words.values())
        if paradigm:
            grammatical_boosts = (
                (paradigm.spatial_operators, 18.0),
                (paradigm.temporal_operators, 16.0),
                (paradigm.pronouns.values(), 15.0),
                (paradigm.digits.values(), 12.0),
                (paradigm.articles.values(), 14.0),
                (paradigm.number_operators.values(), 14.0),
            )
            for ops, boost in grammatical_boosts:
                for op in ops:
                    if op:
                        w_op = op.copy()
                        w_op.usage_frequency = boost
                        cohort.append(w_op)

        return cluster_contrastive_phonemes(cohort)