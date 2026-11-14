"""
translator.py — Balanced Continuous Part-of-Speech Simplex, Dense Prototype Anchors,
and Clean Semantic Glossing (No spurious polar adjective spam).
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import numpy as np

from semantics import SemanticVector, DerivationType
from word import LexicalCategory

if TYPE_CHECKING:
    from word import Word
    from language import Language


_NOUN_ANCHORS: list[tuple[tuple[float, ...], str]] = [
    # Hydrology & Water
    ((1.0, 0.0, 0.7, 0.4, 0.4, 0.1, 0.3), "water"),
    ((1.0, 0.0, 0.7, 0.5, 0.8, 0.1, 0.8), "river"),
    ((1.0, 0.0, 0.7, 0.8, 0.6, 0.1, 1.0), "ocean"),
    ((0.8, 0.0, 0.6, 0.4, 0.9, 0.1, 0.6), "rain"),
    ((0.9, 0.0, 0.5, 0.2, 0.2, 0.1, 0.1), "dew"),
    ((0.9, 0.0, 0.4, 0.6, 0.2, 0.1, 0.4), "ice"),
    ((0.8, 0.0, 0.6, 0.3, 0.5, 0.1, 0.5), "spring"),
    ((0.9, 0.0, 0.5, 0.4, 0.3, 0.1, 0.6), "lake"),
    ((0.9, 0.0, 0.4, 0.7, 0.8, 0.1, 0.7), "torrent"),

    # Fire, Light & Sky
    ((0.9, 0.1, 0.4, 0.7, 0.9, 0.1, 0.3), "fire"),
    ((0.9, 0.0, 0.3, 0.4, 0.6, 0.1, 0.4), "smoke"),
    ((0.9, 0.0, 0.3, 0.2, 0.1, 0.1, 0.2), "ash"),
    ((0.8, 0.0, 0.9, 1.0, 0.6, 0.1, 0.9), "sun"),
    ((0.8, 0.0, 0.6, 0.6, 0.3, 0.1, 0.7), "moon"),
    ((0.6, 0.0, 0.7, 0.3, 0.2, 0.1, 0.4), "star"),
    ((0.5, 0.0, 0.8, 0.4, 0.1, 0.1, 1.0), "sky"),
    ((0.7, 0.0, 0.5, 0.3, 0.5, 0.1, 0.6), "cloud"),
    ((0.6, 0.0, 0.6, 0.5, 0.8, 0.1, 0.8), "wind"),
    ((0.6, 0.0, 0.3, 1.0, 1.0, 0.1, 0.7), "thunder"),
    ((0.8, 0.0, 0.8, 0.6, 0.7, 0.1, 0.3), "flame"),

    # Earth, Stone & Topography
    ((1.0, 0.0, 0.6, 0.7, 0.1, 0.1, 0.7), "earth"),
    ((1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1), "stone"),
    ((1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.9), "mountain"),
    ((1.0, 0.0, 0.4, 0.8, 0.0, 0.0, 0.4), "cliff"),
    ((1.0, 0.0, 0.5, 0.2, 0.1, 0.0, 0.6), "sand"),
    ((1.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.3), "mud"),
    ((1.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.2), "iron"),
    ((0.9, 0.0, 0.9, 0.6, 0.0, 0.1, 0.2), "gold"),
    ((1.0, 0.0, 0.5, 0.8, 0.1, 0.0, 0.7), "ridge"),
    ((0.9, 0.0, 0.3, 0.6, 0.2, 0.0, 0.5), "ravine"),
    ((0.8, 0.0, 0.5, 0.4, 0.1, 0.0, 0.4), "cavern"),
    ((0.9, 0.0, 0.6, 0.4, 0.1, 0.1, 0.6), "vale"),

    # Vegetation & Flora
    ((1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4), "tree"),
    ((1.0, 0.4, 0.6, 0.7, 0.2, 0.1, 0.9), "forest"),
    ((0.9, 0.4, 0.8, 0.2, 0.2, 0.2, 0.2), "flower"),
    ((1.0, 0.3, 0.6, 0.4, 0.1, 0.1, 0.2), "wood"),
    ((0.9, 0.3, 0.7, 0.3, 0.3, 0.2, 0.2), "leaf"),
    ((0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.1), "fruit"),
    ((0.9, 0.3, 0.6, 0.3, 0.1, 0.1, 0.1), "seed"),
    ((0.9, 0.4, 0.6, 0.4, 0.2, 0.1, 0.5), "grove"),
    ((0.8, 0.4, 0.5, 0.3, 0.2, 0.1, 0.3), "reed"),
    ((0.9, 0.3, 0.6, 0.5, 0.1, 0.1, 0.2), "root"),

    # Kinship & Social
    ((0.9, 1.0, 0.85, 0.40, 0.40, 0.95, 0.20), "mother"),
    ((0.9, 1.0, 0.75, 0.75, 0.60, 0.85, 0.30), "father"),
    ((0.9, 1.0, 0.85, 0.20, 0.50, 0.90, 0.10), "child"),
    ((0.9, 1.0, 0.70, 0.70, 0.70, 0.80, 0.30), "man"),
    ((0.9, 1.0, 0.80, 0.40, 0.50, 0.85, 0.20), "woman"),
    ((0.9, 1.0, 0.60, 0.50, 0.50, 0.70, 0.30), "person"),
    ((0.8, 1.0, 0.80, 0.40, 0.40, 0.90, 0.40), "friend"),
    ((0.8, 1.0, 0.20, 0.80, 0.70, 0.10, 0.40), "enemy"),
    ((0.8, 1.0, 0.60, 0.80, 0.50, 0.85, 0.60), "leader"),
    ((0.8, 1.0, 0.70, 0.50, 0.50, 0.90, 0.70), "clan"),
    ((0.8, 1.0, 0.65, 0.45, 0.40, 0.85, 0.40), "kinsman"),
    ((0.8, 1.0, 0.70, 0.60, 0.30, 0.80, 0.30), "elder"),
    ((0.8, 0.9, 0.60, 0.60, 0.60, 0.80, 0.30), "craftsman"),
    ((0.8, 0.9, 0.50, 0.70, 0.70, 0.50, 0.40), "hunter"),
    ((0.8, 0.9, 0.40, 0.85, 0.80, 0.60, 0.40), "warrior"),

    # Fauna & Animals
    ((0.9, 0.9, 0.50, 0.60, 0.70, 0.20, 0.40), "animal"),
    ((0.9, 0.9, 0.30, 0.90, 0.90, 0.10, 0.50), "wolf"),
    ((0.9, 0.9, 0.70, 0.30, 0.80, 0.40, 0.30), "deer"),
    ((0.8, 0.8, 0.60, 0.30, 0.70, 0.30, 0.20), "bird"),
    ((0.8, 0.8, 0.40, 0.80, 0.90, 0.10, 0.40), "eagle"),
    ((0.9, 0.8, 0.60, 0.40, 0.70, 0.10, 0.30), "fish"),
    ((0.9, 0.8, 0.20, 0.60, 0.60, 0.10, 0.30), "snake"),
    ((0.9, 0.8, 0.60, 0.70, 0.80, 0.30, 0.50), "steed"),
    ((0.8, 0.8, 0.60, 0.30, 0.70, 0.40, 0.20), "prey"),

    # Anatomy
    ((0.9, 0.8, 0.60, 0.70, 0.40, 0.30, 0.20), "head"),
    ((1.0, 0.9, 0.60, 0.60, 0.50, 0.40, 0.50), "body"),
    ((0.9, 0.8, 0.60, 0.40, 0.50, 0.30, 0.20), "eye"),
    ((0.9, 0.8, 0.60, 0.40, 0.40, 0.40, 0.20), "ear"),
    ((0.9, 0.8, 0.60, 0.30, 0.30, 0.10, 0.10), "nose"),
    ((0.9, 0.8, 0.60, 0.50, 0.60, 0.60, 0.20), "mouth"),
    ((1.0, 0.8, 0.60, 0.60, 0.60, 0.40, 0.20), "hand"),
    ((1.0, 0.8, 0.60, 0.60, 0.70, 0.20, 0.30), "foot"),
    ((0.7, 0.9, 0.70, 0.60, 0.40, 0.60, 0.20), "heart"),
    ((0.9, 0.6, 0.40, 0.70, 0.50, 0.30, 0.20), "blood"),
    ((1.0, 0.5, 0.50, 0.80, 0.10, 0.10, 0.20), "bone"),
    ((0.9, 0.5, 0.50, 0.40, 0.20, 0.20, 0.40), "skin"),

    # Material Culture & Tools
    ((1.0, 0.0, 0.30, 0.90, 0.80, 0.10, 0.30), "sword"),
    ((1.0, 0.0, 0.50, 0.50, 0.50, 0.10, 0.20), "blade"),
    ((1.0, 0.0, 0.40, 0.80, 0.80, 0.10, 0.60), "spear"),
    ((1.0, 0.0, 0.50, 0.70, 0.70, 0.10, 0.50), "bow"),
    ((1.0, 0.0, 0.60, 0.80, 0.30, 0.20, 0.40), "shield"),
    ((1.0, 0.2, 0.80, 0.70, 0.10, 0.80, 0.60), "hearth"),
    ((1.0, 0.2, 0.70, 0.60, 0.20, 0.90, 0.80), "haven"),
    ((0.9, 0.1, 0.60, 0.50, 0.40, 0.30, 0.30), "vessel"),
    ((0.9, 0.2, 0.60, 0.50, 0.20, 0.70, 0.50), "shelter"),
    ((0.8, 0.1, 0.50, 0.40, 0.50, 0.20, 0.70), "trail"),
    ((0.8, 0.1, 0.50, 0.70, 0.10, 0.40, 0.80), "boundary"),

    # Cultural & Cognitive Abstractions
    ((0.2, 0.9, 0.20, 0.10, 0.40, 0.70, 0.60), "ghost"),
    ((0.2, 0.9, 0.70, 0.20, 0.40, 0.80, 0.60), "ancestor"),
    ((0.5, 0.8, 0.10, 0.40, 0.40, 0.30, 0.50), "death"),
    ((0.4, 0.9, 0.60, 0.40, 0.40, 0.80, 0.30), "voice"),
    ((0.4, 0.9, 0.70, 0.50, 0.40, 0.90, 0.30), "song"),
    ((0.5, 0.7, 0.50, 0.40, 0.40, 0.30, 0.50), "journey"),
    ((0.3, 0.8, 0.70, 0.60, 0.30, 0.80, 0.40), "covenant"),
    ((0.3, 0.8, 0.50, 0.80, 0.40, 0.75, 0.60), "dominion"),
]

_VERB_ANCHORS: list[tuple[tuple[float, ...], str]] = [
    ((0.5, 0.8, 0.60, 0.60, 0.80, 0.40, 0.30), "make"),
    ((0.6, 0.7, 0.50, 0.40, 0.70, 0.20, 0.50), "walk"),
    ((0.6, 0.7, 0.50, 0.80, 1.00, 0.20, 0.70), "run"),
    ((0.5, 0.7, 0.60, 0.70, 0.90, 0.10, 0.80), "fly"),
    ((0.5, 0.7, 0.60, 0.50, 0.70, 0.10, 0.50), "swim"),
    ((0.4, 0.8, 0.60, 0.40, 0.50, 0.30, 0.40), "see"),
    ((0.4, 0.8, 0.60, 0.40, 0.40, 0.40, 0.40), "hear"),
    ((0.4, 0.9, 0.60, 0.40, 0.70, 1.00, 0.30), "speak"),
    ((0.4, 0.9, 0.70, 0.50, 0.70, 0.90, 0.30), "sing"),
    ((0.8, 0.8, 0.80, 0.50, 0.60, 0.30, 0.20), "eat"),
    ((0.7, 0.9, 0.75, 0.40, 0.70, 0.35, 0.10), "devour"),
    ((0.8, 0.5, 0.75, 0.35, 0.60, 0.60, 0.10), "feast"),
    ((0.7, 0.8, 0.85, 0.30, 0.50, 0.80, 0.20), "feed"),
    ((0.8, 0.8, 0.70, 0.40, 0.50, 0.20, 0.20), "drink"),
    ((0.4, 0.7, 0.60, 0.10, 0.10, 0.30, 0.20), "sleep"),
    ((0.5, 0.8, 0.10, 0.40, 0.40, 0.30, 0.50), "die"),
    ((0.6, 0.8, 0.10, 0.95, 0.95, 0.10, 0.40), "strike"),
    ((0.6, 0.8, 0.30, 0.70, 0.80, 0.40, 0.40), "hunt"),
    ((0.4, 0.8, 0.85, 0.30, 0.20, 0.95, 0.30), "love"),
    ((0.5, 0.8, 0.70, 0.40, 0.60, 0.80, 0.40), "give"),
    ((0.7, 0.8, 0.80, 0.70, 0.70, 0.60, 0.50), "nourish"),
    ((0.6, 0.8, 0.85, 0.60, 0.60, 0.80, 0.40), "heal"),
    ((0.6, 0.8, 0.50, 0.80, 0.80, 0.85, 0.60), "lead"),
    ((0.8, 0.0, 0.80, 0.90, 0.70, 0.10, 0.80), "shine"),
    ((0.9, 0.0, 0.30, 0.90, 0.80, 0.10, 0.40), "cut"),
    ((0.9, 0.1, 0.40, 0.80, 0.90, 0.10, 0.40), "burn"),
]

_ADJ_ANCHORS: list[tuple[tuple[float, ...], str]] = [
    ((0.2, 0.3, 0.90, 0.60, 0.30, 0.70, 0.50), "good"),
    ((0.2, 0.3, 0.10, 0.60, 0.40, 0.30, 0.50), "bad"),
    ((0.4, 0.1, 0.60, 0.90, 0.30, 0.10, 0.90), "big"),
    ((0.4, 0.1, 0.50, 0.10, 0.30, 0.10, 0.10), "small"),
    ((0.3, 0.1, 0.60, 0.80, 0.10, 0.10, 0.80), "long"),
    ((0.3, 0.1, 0.50, 0.20, 0.10, 0.10, 0.20), "short"),
    ((0.6, 0.1, 0.50, 0.90, 0.80, 0.10, 0.20), "hard"),
    ((0.5, 0.1, 0.60, 0.20, 0.20, 0.30, 0.30), "soft"),
    ((0.6, 0.0, 0.70, 0.80, 0.60, 0.10, 0.30), "hot"),
    ((0.7, 0.0, 0.40, 0.60, 0.20, 0.10, 0.30), "cold"),
    ((0.3, 0.0, 0.90, 0.60, 0.40, 0.10, 0.80), "bright"),
    ((0.3, 0.0, 0.20, 0.60, 0.20, 0.10, 0.70), "dark"),
    ((0.4, 0.1, 0.85, 0.40, 0.30, 0.85, 0.50), "holy"),
    ((0.4, 0.1, 0.15, 0.80, 0.40, 0.15, 0.50), "cursed"),
    ((0.5, 0.3, 0.50, 0.80, 0.90, 0.20, 0.50), "swift"),
    ((0.7, 0.1, 0.60, 0.60, 0.10, 0.30, 0.70), "ancient"),
    ((0.4, 0.0, 0.80, 0.90, 0.50, 0.20, 0.80), "mighty"),
    ((0.3, 0.2, 0.70, 0.30, 0.20, 0.40, 0.30), "gentle"),
    ((0.8, 0.0, 0.20, 0.90, 0.70, 0.10, 0.40), "fierce"),
    ((0.5, 0.0, 0.50, 0.50, 0.20, 0.10, 0.90), "deep"),
]

_PARTICLE_ANCHORS: list[tuple[tuple[float, ...], str]] = [
    ((0.3, 0.0, 0.50, 0.30, 0.00, 0.00, 0.10), "this"),
    ((0.3, 0.0, 0.50, 0.30, 0.00, 0.00, 0.60), "that"),
    ((0.1, 0.0, 0.50, 0.20, 0.00, 0.00, 0.10), "in"),
    ((0.2, 0.0, 0.50, 0.60, 0.00, 0.00, 0.50), "on"),
    ((0.0, 0.0, 0.30, 0.70, 0.00, 0.00, 0.10), "not"),
    ((0.1, 0.0, 0.50, 0.20, 0.00, 0.00, 0.10), "one"),
    ((0.1, 0.0, 0.50, 0.60, 0.00, 0.20, 0.60), "many"),
    ((0.1, 0.0, 0.60, 0.70, 0.00, 0.30, 0.90), "all"),
    ((0.2, 0.0, 0.50, 0.30, 0.00, 0.10, 0.30), "some"),
    ((0.3, 0.4, 0.60, 0.40, 0.20, 0.80, 0.20), "with"),
    ((0.2, 0.0, 0.50, 0.40, 0.60, 0.20, 0.70), "from"),
    ((0.2, 0.0, 0.50, 0.40, 0.60, 0.20, 0.30), "to"),
    ((0.1, 0.0, 0.50, 0.30, 0.10, 0.50, 0.40), "and"),
    ((0.1, 0.0, 0.40, 0.60, 0.20, 0.30, 0.30), "but"),
    ((0.1, 0.0, 0.50, 0.50, 0.30, 0.20, 0.50), "if"),
    ((0.8, 1.0, 0.60, 0.50, 0.50, 0.30, 0.10), "I"),
    ((0.8, 1.0, 0.60, 0.50, 0.50, 0.80, 0.10), "you"),
]

_MAT_NOUN = np.array([pt for pt, _ in _NOUN_ANCHORS], dtype=np.float32)
_WORDS_NOUN = [word for _, word in _NOUN_ANCHORS]

_MAT_VERB = np.array([pt for pt, _ in _VERB_ANCHORS], dtype=np.float32)
_WORDS_VERB = [word for _, word in _VERB_ANCHORS]

_MAT_ADJ = np.array([pt for pt, _ in _ADJ_ANCHORS], dtype=np.float32)
_WORDS_ADJ = [word for _, word in _ADJ_ANCHORS]

_MAT_PARTICLE = np.array([pt for pt, _ in _PARTICLE_ANCHORS], dtype=np.float32)
_WORDS_PARTICLE = [word for _, word in _PARTICLE_ANCHORS]


class SemanticTranslator:
    @classmethod
    def get_pos(cls, vector: SemanticVector) -> str:
        c, a, v, p, d, s, e = vector.coords

        score_noun = 2.20 * c + 1.60 * a - 0.80 * d
        score_verb = 2.40 * d + 1.10 * p - 0.70 * c
        qual_salience = max(abs(v - 0.50), abs(p - 0.50), abs(e - 0.50))
        score_adj = 2.10 * qual_salience + 0.60 * (1.0 - d) + 0.40 * (1.0 - a) - 0.80 * c
        score_particle = 3.20 * vector.grammaticality_index

        scores = [
            (score_verb, "Verb"),
            (score_noun, "Noun"),
            (score_adj, "Adjective"),
            (score_particle, "Particle"),
        ]
        return max(scores, key=lambda item: item[0])[1]

    @classmethod
    def translate_vector_compositional(cls, vector: SemanticVector, target_pos: str | None = None) -> str:
        """
        Translates a 7D semantic vector directly to its closest prototype anchor.
        Only attaches an adjective if the vector exhibits an extreme, defining outlier
        deviation (|delta| >= 0.40), completely avoiding spurious adjective spam.
        """
        pos = target_pos or cls.get_pos(vector)

        if pos == "Verb":
            mat, word_list = _MAT_VERB, _WORDS_VERB
        elif pos == "Adjective":
            mat, word_list = _MAT_ADJ, _WORDS_ADJ
        elif pos == "Particle":
            mat, word_list = _MAT_PARTICLE, _WORDS_PARTICLE
        else:
            mat, word_list = _MAT_NOUN, _WORDS_NOUN

        diffs = mat - vector.data
        dists_sq = np.sum(diffs ** 2, axis=1)
        best_idx = int(np.argmin(dists_sq))
        base_gloss = word_list[best_idx]
        min_dist = math.sqrt(float(dists_sq[best_idx]))

        # Normal conceptual radius: clean direct prototype noun/verb
        if min_dist < 0.32 or pos != "Noun":
            return base_gloss

        # Only extreme, defining outliers (|mag| >= 0.40) trigger descriptors
        anchor_coords = mat[best_idx]
        delta = vector.data - anchor_coords
        max_dim = int(np.argmax(np.abs(delta)))
        mag = delta[max_dim]

        qualifier = ""
        if abs(mag) >= 0.40:
            if max_dim == 2:
                qualifier = "sacred" if mag > 0 else "dark"
            elif max_dim == 3:
                qualifier = "mighty" if mag > 0 else "slender"
            elif max_dim == 4:
                qualifier = "surging" if mag > 0 else "slumbering"
            elif max_dim == 6:
                qualifier = "boundless" if mag > 0 else "minute"

        if qualifier and qualifier != base_gloss:
            return f"{qualifier} {base_gloss}"
        return base_gloss

    @classmethod
    def translate_word(cls, word: Word, lang: Language | None = None) -> str:
        if word.category == LexicalCategory.FUNCTIONAL_CLOSED:
            word_pos = "Particle"
        else:
            word_pos = cls.get_pos(word.vector)

        base_gloss = cls.translate_vector_compositional(word.vector, target_pos=word_pos)
        deriv = word.derivation

        # Transparent morphological compound glossing
        if deriv.derivation_type == DerivationType.COMPOUND and len(deriv.local_parent_ids) >= 2 and lang:
            parent_glosses = []
            for pid in deriv.local_parent_ids:
                p_word = lang.words.get(pid)
                if p_word and p_word.id != word.id:
                    p_pos = "Particle" if p_word.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(p_word.vector)
                    parent_glosses.append(cls.translate_vector_compositional(p_word.vector, target_pos=p_pos))

            if len(parent_glosses) >= 2 and parent_glosses[0] != parent_glosses[1]:
                compound_label = f"{parent_glosses[0]}+{parent_glosses[1]}"
                return f"{compound_label} ({base_gloss})" if base_gloss not in parent_glosses else compound_label

        elif deriv.derivation_type == DerivationType.AFFIXATION and len(deriv.local_parent_ids) >= 2 and lang:
            stem_w = lang.words.get(deriv.local_parent_ids[0])
            affix_w = lang.words.get(deriv.local_parent_ids[1])
            if stem_w and affix_w:
                s_pos = "Particle" if stem_w.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(stem_w.vector)
                a_pos = "Particle" if affix_w.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(affix_w.vector)
                stem_g = cls.translate_vector_compositional(stem_w.vector, target_pos=s_pos)
                affix_g = cls.translate_vector_compositional(affix_w.vector, target_pos=a_pos)
                return f"{stem_g}+{affix_g} ({base_gloss})"

        return base_gloss

    @classmethod
    def translate(cls, target: SemanticVector | Word, lang: Language | None = None) -> str:
        if hasattr(target, "derivation") and hasattr(target, "vector"):
            return cls.translate_word(target, lang)
        elif hasattr(target, "coords"):
            return cls.translate_vector_compositional(target)
        return str(target)