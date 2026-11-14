"""
language.py — Lightweight Speech Community Container and Typological Profile.
Integrates Active Grammatical Morphemes into the Contrastive Phonemic Inventory.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import Sequence, NamedTuple
from collections import Counter
import numpy as np

from phonetics import cluster_contrastive_phonemes
from word import Word


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
        return max([("SOV", p_sov), ("SVO", p_svo), ("VSO", p_vso)], key=lambda x: x[1])[0]


class Language:
    """
    Lightweight continuous dialect realization container.
    """
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
        """
        Extracts contrastive phonemic inventory across the communicative stream.
        Tally includes both content roots AND active inflectional affixes, case markers,
        person clitics, and numerals according to their functional communicative load.
        """
        combined_cohort: list[Word] = list(self.words.values())

        if paradigm:
            # Inject active grammatical operators weighted by their high sentence frequency
            grammatical_boosts = [
                (paradigm.spatial_operators, 18.0),
                (paradigm.temporal_operators, 16.0),
                (list(paradigm.pronouns.values()), 15.0),
                (list(paradigm.digits.values()), 12.0),
                (list(paradigm.articles.values()), 14.0),
                (list(paradigm.number_operators.values()), 14.0),
            ]
            for op_list, boost in grammatical_boosts:
                for op in op_list:
                    if op:
                        w_op = op.copy()
                        w_op.usage_frequency = boost
                        combined_cohort.append(w_op)

        return cluster_contrastive_phonemes(combined_cohort)