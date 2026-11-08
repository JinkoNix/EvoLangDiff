"""
semantics.py — Continuous 7D Cognitive Space, Typed Derivation Trees, and Biological Prototype Anchors.
Optimized for vectorized SIMD operations and continuous parameter spaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import numpy as np


class ClineStage(Enum):
    FREE_ROOT = "Free Root"
    PERIPHRASIS = "Periphrastic Auxiliary"
    CLITIC = "Clitic"
    BOUND_AFFIX = "Bound Affix"
    FUSED_INTERNAL = "Fused / Suprasegmental"
    ZERO = "Zero / Lost"


class DerivationType(Enum):
    ROOT = "root"
    COMPOUND = "compound"
    AFFIXATION = "affixation"
    LOANWORD = "loanword"
    SEMANTIC_SPLIT = "split"


@dataclass
class LoanProvenance:
    source_language: str
    donor_word_id: int
    donor_form_at_borrowing: str
    donor_vector_at_borrowing: SemanticVector
    generation_borrowed: int = 0


@dataclass
class DerivationRecord:
    root_family_id: int
    derivation_type: DerivationType = DerivationType.ROOT
    local_parent_ids: tuple[int, ...] = ()
    provenance: LoanProvenance | None = None


_CLINE_DEPTHS: dict[ClineStage, float] = {
    ClineStage.FREE_ROOT: 0.00,
    ClineStage.PERIPHRASIS: 0.20,
    ClineStage.CLITIC: 0.45,
    ClineStage.BOUND_AFFIX: 0.75,
    ClineStage.FUSED_INTERNAL: 0.92,
    ClineStage.ZERO: 1.00,
}

_BASE_CLINE_DECAY = np.array([2.40, 2.80, 0.00, 1.80, 0.00, 0.00, 0.00], dtype=np.float32)


class SemanticVector:
    """A continuous coordinate point in 7D cognitive semantic space [0.0, 1.0]^7."""

    __slots__ = ('_coords', '_data')

    DIMENSION_NAMES = (
        "concreteness",
        "animacy",
        "valence",
        "potency",
        "dynamism",
        "sociality",
        "extension",
    )

    def __init__(
        self,
        concreteness: float,
        animacy: float,
        valence: float,
        potency: float,
        dynamism: float,
        sociality: float,
        extension: float,
        _data: np.ndarray | None = None,
    ):
        if _data is not None:
            self._data = np.clip(_data.astype(np.float32), 0.0, 1.0)
            self._coords = tuple(float(x) for x in self._data)
        else:
            c = (
                max(0.0, min(1.0, float(concreteness))),
                max(0.0, min(1.0, float(animacy))),
                max(0.0, min(1.0, float(valence))),
                max(0.0, min(1.0, float(potency))),
                max(0.0, min(1.0, float(dynamism))),
                max(0.0, min(1.0, float(sociality))),
                max(0.0, min(1.0, float(extension))),
            )
            self._coords = c
            self._data = np.array(c, dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> SemanticVector:
        clamped = np.clip(arr.astype(np.float32), 0.0, 1.0)
        vec = cls.__new__(cls)
        vec._coords = tuple(float(x) for x in clamped)
        vec._data = clamped
        return vec

    @property
    def coords(self) -> tuple[float, float, float, float, float, float, float]:
        return self._coords

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def concreteness(self) -> float: return self._coords[0]
    @property
    def animacy(self) -> float: return self._coords[1]
    @property
    def valence(self) -> float: return self._coords[2]
    @property
    def potency(self) -> float: return self._coords[3]
    @property
    def dynamism(self) -> float: return self._coords[4]
    @property
    def sociality(self) -> float: return self._coords[5]
    @property
    def extension(self) -> float: return self._coords[6]

    @property
    def grammaticality_index(self) -> float:
        lexical_mass = (self._coords[0] + 0.60 * self._coords[1]) / 1.60
        return max(0.0, min(1.0, 1.0 - lexical_mass))

    @property
    def is_grammatical(self) -> bool:
        # Logistic distribution replacing threshold cliff
        p_gram = 1.0 / (1.0 + math.exp(-12.0 * (self.grammaticality_index - 0.70)))
        return p_gram >= 0.50

    def distance_to(self, other: SemanticVector, weights: Sequence[float] | None = None) -> float:
        diff_sq = (self._data - other._data) ** 2
        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            return float(np.sqrt(np.sum(w * diff_sq)))
        return float(np.sqrt(np.sum(diff_sq)))

    def weighted_distance_to(self, other: SemanticVector, weights: Sequence[float]) -> float:
        w = np.asarray(weights, dtype=np.float32)
        diff_sq = (self._data - other._data) ** 2
        return float(np.sqrt(np.sum(w * diff_sq)))

    def drift(self, delta: Sequence[float]) -> SemanticVector:
        d = np.asarray(delta, dtype=np.float32)
        return SemanticVector.from_array(self._data + d)

    def blend(self, other: SemanticVector, weight: float = 0.5) -> SemanticVector:
        w = float(np.clip(weight, 0.0, 1.0))
        blended = w * self._data + (1.0 - w) * other._data
        return SemanticVector.from_array(blended)

    def generalize_for_cline(self, stage: ClineStage, cultural_attention: Sequence[float] | None = None) -> SemanticVector:
        t = _CLINE_DEPTHS.get(stage, 0.0)
        if t <= 0.0:
            return self

        rates = _BASE_CLINE_DECAY.copy()
        if cultural_attention is not None:
            attn = np.asarray(cultural_attention, dtype=np.float32)
            rates[0] /= max(0.5, float(attn[0]))
            rates[1] /= max(0.5, float(attn[1]))

        decayed = self._data * np.exp(-rates * t)
        return SemanticVector.from_array(decayed)

    def __iter__(self):
        return iter(self._coords)

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.2f}" for v in self._coords)
        return f"SemanticVector({vals})"


class SemanticSpace:
    """Cognitive Prototype Anchors (Rosch & Lakoff Prototype Theory)."""
    WATER = SemanticVector(1.0, 0.0, 0.7, 0.5, 0.5, 0.1, 0.4)
    FIRE  = SemanticVector(0.9, 0.1, 0.4, 0.8, 0.9, 0.2, 0.3)
    SUN   = SemanticVector(0.9, 0.1, 0.8, 1.0, 0.6, 0.1, 0.9)
    MOON  = SemanticVector(0.9, 0.0, 0.6, 0.7, 0.3, 0.1, 0.8)
    EARTH = SemanticVector(1.0, 0.0, 0.6, 0.8, 0.1, 0.1, 0.8)
    STONE = SemanticVector(1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1)
    TREE  = SemanticVector(1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4)

    MOTHER = SemanticVector(0.9, 1.0, 0.85, 0.40, 0.40, 0.95, 0.20)
    FATHER = SemanticVector(0.9, 1.0, 0.75, 0.75, 0.60, 0.85, 0.30)
    PERSON = SemanticVector(0.9, 1.0, 0.60, 0.50, 0.60, 0.70, 0.30)
    BEAST  = SemanticVector(0.9, 0.9, 0.50, 0.70, 0.80, 0.20, 0.40)
    SELF   = SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.30, 0.10)
    OTHER  = SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.80, 0.10)

    HEAD  = SemanticVector(0.9, 0.8, 0.6, 0.7, 0.4, 0.3, 0.2)
    BODY  = SemanticVector(1.0, 0.9, 0.6, 0.6, 0.5, 0.4, 0.5)
    EYE   = SemanticVector(0.9, 0.8, 0.6, 0.4, 0.5, 0.3, 0.3)
    HAND  = SemanticVector(1.0, 0.8, 0.6, 0.6, 0.6, 0.4, 0.2)
    HEART = SemanticVector(0.7, 0.9, 0.7, 0.6, 0.4, 0.6, 0.2)
    BLOOD = SemanticVector(0.9, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2)

    MAKE  = SemanticVector(0.5, 0.8, 0.6, 0.6, 0.8, 0.4, 0.3)
    MOVE  = SemanticVector(0.6, 0.7, 0.5, 0.4, 0.8, 0.2, 0.5)
    SEE   = SemanticVector(0.4, 0.8, 0.6, 0.4, 0.5, 0.3, 0.4)
    HEAR  = SemanticVector(0.4, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4)
    SPEAK = SemanticVector(0.4, 0.9, 0.6, 0.4, 0.7, 1.0, 0.3)
    EAT   = SemanticVector(0.8, 0.8, 0.8, 0.5, 0.6, 0.3, 0.2)
    DIE   = SemanticVector(0.5, 0.8, 0.1, 0.8, 0.4, 0.3, 0.6)

    GOOD  = SemanticVector(0.2, 0.3, 1.0, 0.6, 0.3, 0.7, 0.5)
    BAD   = SemanticVector(0.2, 0.3, 0.0, 0.6, 0.4, 0.3, 0.5)
    HUGE  = SemanticVector(0.4, 0.1, 0.6, 0.95, 0.3, 0.1, 0.85)
    MINI  = SemanticVector(0.4, 0.1, 0.5, 0.10, 0.3, 0.1, 0.10)
    THIS  = SemanticVector(0.3, 0.0, 0.5, 0.3, 0.0, 0.0, 0.1)
    THAT  = SemanticVector(0.3, 0.0, 0.5, 0.3, 0.0, 0.0, 0.6)
    IN    = SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.1)
    ABOVE = SemanticVector(0.2, 0.0, 0.5, 0.6, 0.0, 0.0, 0.5)
    NOT   = SemanticVector(0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.1)
    ONE   = SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.1)
    MANY  = SemanticVector(0.1, 0.0, 0.5, 0.6, 0.0, 0.2, 0.6)

    BIOLOGICAL_COGNITIVE_SEEDS = [
        WATER, FIRE, SUN, MOON, EARTH, STONE, TREE,
        MOTHER, FATHER, PERSON, BEAST, SELF, OTHER,
        HEAD, BODY, EYE, HAND, HEART, BLOOD,
        MAKE, MOVE, SEE, HEAR, SPEAK, EAT, DIE,
        GOOD, BAD, HUGE, MINI, THIS, THAT, IN, ABOVE, NOT, ONE, MANY
    ]