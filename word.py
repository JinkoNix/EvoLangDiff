"""
word.py — Compact Lexical Container wrapping (L, 9) Continuous Articulatory Tensors.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Sequence
import numpy as np

from phonetics import (
    Phoneme,
    Syllable,
    word_energy,
    render_word_trajectory,
    continuous_sonority,
    syllabify,
    assign_stress,
)
from semantics import SemanticVector, DerivationRecord, DerivationType, ClineStage


class LexicalCategory(Enum):
    CONTENT_OPEN = "content_open"
    FUNCTIONAL_CLOSED = "functional"


class Morpheme:
    __slots__ = ('phonemes', 'root_family_id', 'is_grammatical', 'stage')

    def __init__(
        self,
        phonemes: Sequence[Phoneme],
        root_family_id: int,
        is_grammatical: bool = False,
        stage: ClineStage = ClineStage.FREE_ROOT,
    ):
        self.phonemes = list(phonemes)
        self.root_family_id = int(root_family_id)
        self.is_grammatical = bool(is_grammatical)
        self.stage = stage

    @property
    def ipa(self) -> str:
        return "".join(p.ipa for p in self.phonemes)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phonemes)

    def erode(
        self,
        reduction_rate: float = 0.80,
        usage_frequency: float = 1.0,
        is_long_word: bool = False,
        syllable_complexity: float = 0.50,
    ) -> Morpheme:
        if self.stage == ClineStage.ZERO or (not self.is_grammatical and not is_long_word):
            return self

        p_list = list(self.phonemes)
        if len(p_list) <= 1:
            return self

        freq_multiplier = math.log(usage_frequency + 1.2) * (1.20 if self.is_grammatical else 0.70)
        p_drop = 1.0 - math.exp(-0.25 * reduction_rate * freq_multiplier)
        if random.random() > p_drop:
            return self

        v_indices = [idx for idx, seg in enumerate(p_list) if seg.is_vowel]
        candidates = [idx for idx in range(len(p_list)) if not (len(v_indices) == 1 and idx == v_indices[0])]
        if not candidates:
            return self

        least_eff = min(
            candidates,
            key=lambda idx: p_list[idx].energy.audibility_payoff / (
                p_list[idx].energy.motor_effort + p_list[idx].energy.air_consumption + 1e-4
            )
        )
        p_list.pop(least_eff)
        return Morpheme(phonemes=p_list, root_family_id=self.root_family_id, is_grammatical=self.is_grammatical, stage=self.stage)

    def copy(self) -> Morpheme:
        return Morpheme(
            phonemes=[p.drift() for p in self.phonemes],
            root_family_id=self.root_family_id,
            is_grammatical=self.is_grammatical,
            stage=self.stage,
        )

    def __repr__(self) -> str:
        cat = "GRAM" if self.is_grammatical else "LEX"
        return f"Morpheme(/{self.ipa}/, fam={self.root_family_id}, {cat})"


class Word:
    __slots__ = (
        'id', 'vector', '_phonemes', '_syllables', '_morphemes',
        'derivation', 'usage_frequency', 'category', 'generation_born',
        'epochs_idle', 'senses', '_form', '_plain_form',
        '_cached_cost', '_cost_env_key'
    )

    _ID_COUNTER = 1000
    _EPHEMERAL_COUNTER = -1

    def __init__(
        self,
        vector: SemanticVector,
        phonemes: Sequence[Phoneme] | None = None,
        syllables: Sequence[Syllable] | None = None,
        morphemes: Sequence[Morpheme] | None = None,
        derivation: DerivationRecord | None = None,
        word_id: int | None = None,
        usage_frequency: float = 1.0,
        category: LexicalCategory = LexicalCategory.CONTENT_OPEN,
        generation_born: int = 0,
        senses: Sequence[SemanticVector] | tuple[SemanticVector, ...] | None = None,
        is_ephemeral: bool = False,
        **kwargs,
    ):
        if word_id is not None:
            self.id = word_id
        elif is_ephemeral:
            Word._EPHEMERAL_COUNTER -= 1
            self.id = Word._EPHEMERAL_COUNTER
        else:
            Word._ID_COUNTER += 1
            self.id = Word._ID_COUNTER

        self.vector = vector
        self._syllables = list(syllables) if syllables else []
        self._morphemes = list(morphemes) if morphemes else []
        self._form = None
        self._plain_form = None
        self._cached_cost = None
        self._cost_env_key = None

        if syllables:
            flat: list[Phoneme] = []
            for s in self._syllables:
                flat.extend(s.to_phonemes())
            self._phonemes = flat
        elif phonemes is not None:
            self._phonemes = list(phonemes)
        elif self._morphemes:
            flat = []
            for m in self._morphemes:
                flat.extend(m.phonemes)
            self._phonemes = flat
        else:
            self._phonemes = []

        self.derivation = derivation or DerivationRecord(root_family_id=self.id)
        self.usage_frequency = max(0.01, float(usage_frequency))
        self.category = category
        self.generation_born = int(generation_born)
        self.epochs_idle = 0

        self.senses = senses if isinstance(senses, tuple) else (tuple(senses) if senses else (self.vector,))

        if not self._morphemes and self._phonemes:
            self._morphemes = [
                Morpheme(
                    phonemes=list(self._phonemes),
                    root_family_id=self.derivation.root_family_id,
                    is_grammatical=(self.category == LexicalCategory.FUNCTIONAL_CLOSED),
                )
            ]

    def _invalidate_cache(self) -> None:
        self._form = None
        self._plain_form = None
        self._cached_cost = None
        self._cost_env_key = None

    @property
    def phonemes(self) -> list[Phoneme]:
        return self._phonemes

    @phonemes.setter
    def phonemes(self, new_phonemes: Sequence[Phoneme]) -> None:
        self._phonemes = list(new_phonemes)
        self._syllables = []
        self._invalidate_cache()

        n_clean = len(self._phonemes)
        if len(self._morphemes) <= 1:
            if not self._morphemes:
                self._morphemes = [Morpheme(self._phonemes, self.derivation.root_family_id)]
            else:
                self._morphemes[0].phonemes = list(self._phonemes)
        else:
            total_old = sum(len(m.phonemes) for m in self._morphemes) or 1
            cursor = 0
            for m_idx, m in enumerate(self._morphemes):
                if m_idx == len(self._morphemes) - 1:
                    m.phonemes = list(self._phonemes[cursor:])
                else:
                    m_len = max(0, min(n_clean - cursor, int(round(n_clean * (len(m.phonemes) / total_old)))))
                    m.phonemes = list(self._phonemes[cursor:cursor + m_len])
                    cursor += m_len
            self._morphemes = [m for m in self._morphemes if m.phonemes] or [
                Morpheme(self._phonemes, self.derivation.root_family_id)
            ]

    @property
    def morphemes(self) -> list[Morpheme]:
        return self._morphemes

    @morphemes.setter
    def morphemes(self, new_morphemes: Sequence[Morpheme]) -> None:
        self._morphemes = list(new_morphemes)
        flat = []
        for m in self._morphemes:
            flat.extend(m.phonemes)
        self._phonemes = flat
        self._syllables = []
        self._invalidate_cache()

    @property
    def tensor(self) -> np.ndarray:
        if not self._phonemes:
            return np.zeros((0, 9), dtype=np.float32)
        return np.stack([p.vector for p in self._phonemes], axis=0)

    @property
    def form(self) -> str:
        if self._form is None:
            if self._phonemes:
                self._form, self._plain_form = render_word_trajectory(self.tensor)
            else:
                self._form, self._plain_form = "", ""
        return self._form

    @property
    def plain_form(self) -> str:
        if self._plain_form is None:
            _ = self.form
        return self._plain_form

    @property
    def syllables(self) -> list[Syllable]:
        if not self._syllables and self._phonemes:
            self._syllables = syllabify(self._phonemes)
        return self._syllables

    @syllables.setter
    def syllables(self, new_syllables: Sequence[Syllable]) -> None:
        self._syllables = list(new_syllables)
        if self._syllables:
            flat: list[Phoneme] = []
            for s in self._syllables:
                flat.extend(s.to_phonemes())
            self._phonemes = flat
        self._invalidate_cache()

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self._phonemes)

    @property
    def is_proto_root(self) -> bool:
        return self.derivation.derivation_type == DerivationType.ROOT and self.generation_born == 0

    @property
    def is_ephemeral(self) -> bool:
        return self.id < 0

    def energetic_cost(
        self,
        profile=None,
        env=None,
        cultural_attention: Sequence[float] | None = None,
        cohort_sample: Sequence[Word] | None = None,
    ) -> float:
        env_key = (
            getattr(profile, "synthesis_index", 0.50),
            getattr(env, "altitude", 0.0),
            getattr(env, "temperature", 0.50),
            getattr(env, "humidity", 0.50),
            len(cohort_sample) if cohort_sample else 0
        )
        if self._cached_cost is None or self._cost_env_key != env_key:
            self._cached_cost = word_energy(
                self,
                profile=profile,
                env=env,
                cultural_attention=cultural_attention,
                cohort_sample=cohort_sample,
            )
            self._cost_env_key = env_key
        return self._cached_cost

    def record_usage(self, boost: float = 1.0) -> None:
        self.usage_frequency += max(0.0, float(boost))
        self.epochs_idle = 0

    def is_obsolete(self, max_idle_epochs: int = 25, cultural_inertia: float = 1.0) -> bool:
        if self.category == LexicalCategory.FUNCTIONAL_CLOSED or self.is_proto_root:
            return False
        effective_max = max(10, int(round(max_idle_epochs * max(0.40, cultural_inertia))))
        # Cost-driven obsolescence: words with excessive energy dissipation fail to survive
        energy_wear = (self._cached_cost or 2.0) / 4.0
        return (self.epochs_idle >= effective_max) and (self.usage_frequency < (0.15 * energy_wear))

    def copy(self) -> Word:
        w = Word(
            word_id=self.id,
            vector=self.vector,
            phonemes=[p.drift() for p in self._phonemes],
            syllables=[s for s in self._syllables],
            morphemes=[m.copy() for m in self._morphemes],
            derivation=self.derivation,
            usage_frequency=self.usage_frequency,
            category=self.category,
            generation_born=self.generation_born,
            senses=self.senses,
        )
        w._form = self._form
        w._plain_form = self._plain_form
        w._cached_cost = self._cached_cost
        w._cost_env_key = self._cost_env_key
        w.epochs_idle = self.epochs_idle
        return w

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        if self.id < 0 or other.id < 0:
            return self is other
        return self.id == other.id

    def __len__(self) -> int:
        return len(self._phonemes)

    def __getitem__(self, idx: int) -> Phoneme:
        return self._phonemes[idx]

    def __iter__(self):
        return iter(self._phonemes)

    def __repr__(self) -> str:
        cat = "FUNC" if self.category == LexicalCategory.FUNCTIONAL_CLOSED else "CONTENT"
        return f"Word(#{self.id}, /{self.form}/, U={self.usage_frequency:.2f}, {cat})"