"""
language.py — Diachronic Speech Community Engine, Replicator Dynamics,
and Societal Stratum Integration.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from typing import Sequence, NamedTuple
import numpy as np

from phonetics import (
    Phoneme,
    word_energy,
    repair_phonemes,
    apply_prosody,
    trajectory_distance,
    relax_gesture_gradient,
    apply_syncope_and_clustering,
    cluster_contrastive_phonemes,
    resolve_tract_vector,
)
from semantics import SemanticVector, DerivationRecord, DerivationType
from word import Word, Morpheme, LexicalCategory


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

    def get_lexicon_sample(self, n: int = 32) -> list[Word]:
        words = list(self.words.values())
        if len(words) <= n:
            return words
        return random.sample(words, n)

    def extract_vocalic_neighbor_pool(self, max_samples: int = 24) -> list[np.ndarray]:
        pool = []
        for w in list(self.words.values())[-40:]:
            for p in w.phonemes:
                if p.is_vowel:
                    pool.append(p.g)
                    if len(pool) >= max_samples:
                        return pool
        return pool

    def compete_lexicon(self, cohort_sample: Sequence[Word] | None = None) -> list[int]:
        """
        Replicator Dynamics (Survival of the Fittest Words):
        Fitter words (lower Joules, higher modulation) actively devour usage from
        sluggish, unoptimized words competing for the same semantic space.
        Returns IDs of purged/extinct words.
        """
        words = list(self.words.values())
        if len(words) <= 12:
            return []

        purged_ids: list[int] = []
        sample = list(cohort_sample or words[:30])

        # Evaluate fitness: Fitness(w) = 10.0 / E_word
        fitness_map = {}
        for w in words:
            cost = w.energetic_cost(profile=self.profile, cohort_sample=sample)
            fitness_map[w.id] = max(0.1, 10.0 / max(2.0, cost))

        # Pairwise semantic competition
        for i in range(len(words)):
            w1 = words[i]
            if w1.id in purged_ids:
                continue

            for j in range(i + 1, min(i + 8, len(words))):
                w2 = words[j]
                if w2.id in purged_ids or w1.category != w2.category:
                    continue

                d_sem = w1.vector.distance_to(w2.vector)
                if d_sem < 0.22:  # Direct semantic collision
                    f1 = fitness_map[w1.id]
                    f2 = fitness_map[w2.id]
                    
                    # Fitter word absorbs usage from less efficient word
                    transfer = 0.25 * math.tanh(f1 - f2) * w2.usage_frequency
                    w1.usage_frequency += transfer
                    w2.usage_frequency -= transfer

                    # Less fit word starves and dies
                    if w2.usage_frequency < 0.15 and not w2.is_proto_root:
                        purged_ids.append(w2.id)

        for pid in set(purged_ids):
            if pid in self.words:
                del self.words[pid]

        return list(set(purged_ids))

    def apply_superstrate_stratum(self, conqueror_words: Sequence[Word], imperial_share: float) -> None:
        """
        Applies sociolinguistic stratum shift under imperial conquest:
        High-prestige domains (potency, law, social hierarchy) are replaced by the conqueror's
        vocabulary, while domestic substrate roots (water, mother, eye) are retained.
        """
        if imperial_share < 0.40 or not conqueror_words:
            return

        for donor in conqueror_words:
            # Check if donor word represents a superstrate domain (potency or sociality > 0.60)
            if donor.vector.potency > 0.58 or donor.vector.sociality > 0.58:
                # Find matching native target
                target = min(self.words.values(), key=lambda w: w.vector.distance_to(donor.vector), default=None)
                if target and target.id != donor.id:
                    # Imperial word displaces the native root
                    loan = donor.copy()
                    loan.id = target.id
                    loan.usage_frequency = target.usage_frequency * (1.0 + imperial_share)
                    self.words[target.id] = loan

    def compound_concepts(
        self,
        w1: Word,
        w2: Word,
        blended_vec: SemanticVector,
        new_id: int,
        epoch: int,
        alt: float,
        hum: float,
        temp: float,
        t_drift: float,
        rng: random.Random,
    ) -> Word | None:
        valence_intensity = abs(blended_vec.valence - 0.50) + (blended_vec.potency * 0.4)

        # Head-Dependent Metrical Truncation: extract prominent syllables, avoiding 18-phoneme runaways
        p1 = list(w1.phonemes)
        p2 = list(w2.phonemes)

        if len(p1) > 4:
            syls1 = w1.syllables
            p1 = syls1[0].to_phonemes() if syls1 else p1[:3]
        if len(p2) > 4:
            syls2 = w2.syllables
            p2 = (syls2[-1].to_phonemes() if len(syls2) > 1 else syls2[0].to_phonemes()) if syls2 else p2[-3:]

        raw_p = p1 + p2
        bias = np.zeros(8, dtype=np.float32)
        vowel_pool = self.extract_vocalic_neighbor_pool()

        est_cost = len(raw_p) * 2.2
        mutated_p = []

        for i, p_seg in enumerate(raw_p):
            adj = raw_p[i - 1] if i > 0 else (raw_p[i + 1] if i + 1 < len(raw_p) else None)
            mutated_seg = relax_gesture_gradient(
                p=p_seg,
                adjacent_p=adj,
                neighbor_g_pool=vowel_pool,
                valence_intensity=valence_intensity,
                bias=bias,
                alt=alt,
                hum=hum,
                temp=temp,
                t_drift=t_drift,
                rng=rng,
                word_cost=est_cost,
            )
            mutated_p.append(mutated_seg)

        comp_usage = rng.uniform(0.8, 1.6)
        eroded_p = apply_syncope_and_clustering(
            mutated_p,
            usage_frequency=comp_usage,
            alt=alt,
            hum=hum,
            temp=temp,
            profile=self.profile,
            word_cost=est_cost,
        )

        if not eroded_p or not any(p.is_vowel for p in eroded_p):
            return None

        if len(eroded_p) >= 2:
            ratio = len(p1) / max(1, len(raw_p))
            split = max(1, min(len(eroded_p) - 1, int(round(len(eroded_p) * ratio))))
            m1 = Morpheme(eroded_p[:split], root_family_id=w1.derivation.root_family_id)
            m2 = Morpheme(eroded_p[split:], root_family_id=w2.derivation.root_family_id)
            morphemes = [m1, m2]
        else:
            morphemes = [Morpheme(eroded_p, root_family_id=w1.derivation.root_family_id)]

        comp_word = Word(
            word_id=new_id,
            vector=blended_vec,
            phonemes=eroded_p,
            morphemes=morphemes,
            derivation=DerivationRecord(
                root_family_id=w1.derivation.root_family_id,
                derivation_type=DerivationType.COMPOUND,
                local_parent_ids=(w1.id, w2.id),
            ),
            usage_frequency=comp_usage,
            category=LexicalCategory.CONTENT_OPEN,
            generation_born=max(0, epoch - rng.randint(0, min(epoch, 500))),
        )
        return comp_word

    def extract_phonemic_inventory(self) -> tuple[list[str], list[str]]:
        return cluster_contrastive_phonemes(list(self.words.values()))

    def fork(self, daughter_name: str, initial_head_directionality: float | None = None) -> Language:
        h = initial_head_directionality if initial_head_directionality is not None else self.profile.head_directionality
        prof = LanguageProfile(
            name=f"{daughter_name} Profile",
            head_directionality=h,
            synthesis_index=self.profile.synthesis_index,
            syllable_complexity=self.profile.syllable_complexity,
            max_transition_cost=self.profile.max_transition_cost,
            alignment=self.profile.alignment,
            metrical_bias=self.profile.metrical_bias,
        )
        daughter = Language(name=daughter_name, profile=prof, population=self.population)
        for w in self.words.values():
            daughter.add_word(w.copy())
        return daughter