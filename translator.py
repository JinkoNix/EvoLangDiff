"""
translator.py — Vectorized Semantic Decoder and Compositional Gloss Translator.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import numpy as np

from semantics import SemanticVector, DerivationType

if TYPE_CHECKING:
    from word import Word
    from language import Language


_NOUN_ANCHORS: list[tuple[tuple[float, ...], str]] = [
    ((1.0, 0.0, 0.7, 0.4, 0.4, 0.1, 0.3), "water"),
    ((1.0, 0.0, 0.7, 0.5, 0.8, 0.1, 0.8), "river"),
    ((1.0, 0.0, 0.7, 0.8, 0.6, 0.1, 1.0), "ocean"),
    ((0.8, 0.0, 0.6, 0.4, 0.9, 0.1, 0.6), "rain"),
    ((0.9, 0.0, 0.5, 0.2, 0.2, 0.1, 0.1), "dew"),
    ((0.9, 0.0, 0.4, 0.6, 0.2, 0.1, 0.4), "ice"),
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
    ((1.0, 0.0, 0.6, 0.7, 0.1, 0.1, 0.7), "earth"),
    ((1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1), "stone"),
    ((1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.9), "mountain"),
    ((1.0, 0.0, 0.4, 0.8, 0.0, 0.0, 0.4), "cliff"),
    ((1.0, 0.0, 0.5, 0.2, 0.1, 0.0, 0.6), "sand"),
    ((1.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.3), "mud"),
    ((1.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.2), "iron"),
    ((0.9, 0.0, 0.9, 0.6, 0.0, 0.1, 0.2), "gold"),
    ((1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4), "tree"),
    ((1.0, 0.4, 0.6, 0.7, 0.2, 0.1, 0.9), "forest"),
    ((0.9, 0.4, 0.8, 0.2, 0.2, 0.2, 0.2), "flower"),
    ((1.0, 0.3, 0.6, 0.4, 0.1, 0.1, 0.2), "wood"),
    ((0.9, 0.3, 0.7, 0.3, 0.3, 0.2, 0.2), "leaf"),
    ((0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.1), "fruit"),
    ((0.9, 0.3, 0.6, 0.3, 0.1, 0.1, 0.1), "seed"),
    ((0.9, 1.0, 0.85, 0.40, 0.40, 0.95, 0.20), "mother"),
    ((0.9, 1.0, 0.75, 0.75, 0.60, 0.85, 0.30), "father"),
    ((0.9, 1.0, 0.85, 0.20, 0.50, 0.90, 0.10), "child"),
    ((0.9, 1.0, 0.70, 0.70, 0.70, 0.80, 0.30), "man"),
    ((0.9, 1.0, 0.80, 0.40, 0.50, 0.85, 0.20), "woman"),
    ((0.9, 1.0, 0.60, 0.50, 0.50, 0.70, 0.30), "person"),
    ((0.8, 1.0, 0.80, 0.40, 0.40, 0.90, 0.40), "friend"),
    ((0.8, 1.0, 0.20, 0.80, 0.70, 0.10, 0.40), "enemy"),
    ((0.8, 1.0, 0.60, 0.80, 0.50, 0.85, 0.60), "leader"),
    ((0.8, 1.0, 0.70, 0.50, 0.50, 0.90, 0.70), "tribe"),
    ((0.9, 0.9, 0.50, 0.60, 0.70, 0.20, 0.40), "animal"),
    ((0.9, 0.9, 0.30, 0.90, 0.90, 0.10, 0.50), "wolf"),
    ((0.9, 0.9, 0.70, 0.30, 0.80, 0.40, 0.30), "deer"),
    ((0.8, 0.8, 0.60, 0.30, 0.70, 0.30, 0.20), "bird"),
    ((0.8, 0.8, 0.40, 0.80, 0.90, 0.10, 0.40), "eagle"),
    ((0.9, 0.8, 0.60, 0.40, 0.70, 0.10, 0.30), "fish"),
    ((0.9, 0.8, 0.20, 0.60, 0.60, 0.10, 0.30), "snake"),
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
    ((1.0, 0.0, 0.30, 0.90, 0.80, 0.10, 0.30), "sword"),
    ((1.0, 0.0, 0.50, 0.50, 0.50, 0.10, 0.20), "knife"),
    ((1.0, 0.0, 0.40, 0.80, 0.80, 0.10, 0.60), "spear"),
    ((1.0, 0.0, 0.50, 0.70, 0.70, 0.10, 0.50), "bow"),
    ((1.0, 0.0, 0.60, 0.80, 0.30, 0.20, 0.40), "shield"),
    ((1.0, 0.2, 0.80, 0.70, 0.10, 0.80, 0.60), "house"),
    ((1.0, 0.2, 0.70, 0.60, 0.20, 0.90, 0.80), "village"),
    ((0.2, 0.9, 0.20, 0.10, 0.40, 0.70, 0.60), "ghost"),
    ((0.2, 0.9, 0.70, 0.20, 0.40, 0.80, 0.60), "ancestor"),
    ((0.5, 0.8, 0.10, 0.40, 0.40, 0.30, 0.50), "death"),
    ((0.4, 0.9, 0.60, 0.40, 0.40, 0.80, 0.30), "voice"),
    ((0.4, 0.9, 0.70, 0.50, 0.40, 0.90, 0.30), "song"),
    ((0.5, 0.7, 0.50, 0.40, 0.40, 0.30, 0.50), "journey"),
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
    ((0.6, 0.8, 0.10, 0.95, 0.95, 0.10, 0.40), "kill"),
    ((0.7, 0.8, 0.20, 0.85, 0.90, 0.20, 0.30), "strike"),
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
        score_verb = 1.70 * d + 0.50 * p - 0.40 * c
        score_noun = 1.40 * c + 1.20 * a - 0.50 * d
        qual_salience = max(abs(v - 0.50), abs(p - 0.50), abs(e - 0.50))
        score_adj = 1.60 * qual_salience + 0.30 * (1.0 - a) - 0.40 * d
        score_particle = 1.80 * vector.grammaticality_index

        scores = [
            (score_verb, "Verb"),
            (score_noun, "Noun"),
            (score_adj, "Adjective"),
            (score_particle, "Particle"),
        ]
        return max(scores, key=lambda item: item[0])[1]

    @classmethod
    def translate_vector_constrained(cls, vector: SemanticVector, target_pos: str | None = None) -> str:
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
        best_dist = math.sqrt(float(dists_sq[best_idx]))
        base_word = word_list[best_idx]

        if pos == "Particle":
            return base_word

        anchor_coords = mat[best_idx]
        delta_v = vector.valence - anchor_coords[2]
        delta_p = vector.potency - anchor_coords[3]
        delta_e = vector.extension - anchor_coords[6]
        delta_d = vector.dynamism - anchor_coords[4]

        # Continuous qualification via sigmoid confidence
        p_qualify = 1.0 / (1.0 + math.exp(-12.0 * (best_dist - 0.28)))
        if p_qualify >= 0.50:
            if pos == "Verb":
                if delta_v >= 0.30: return f"sacredly {base_word}"
                if delta_v <= -0.30: return f"darkly {base_word}"
                if delta_p >= 0.30: return f"fiercely {base_word}"
                if delta_d >= 0.30: return f"swiftly {base_word}"
                if delta_e >= 0.30: return f"widely {base_word}"
            elif pos == "Noun":
                if delta_v >= 0.30: return f"holy {base_word}"
                if delta_v <= -0.30: return f"dark {base_word}"
                if delta_p >= 0.30: return f"mighty {base_word}"
                if delta_e >= 0.30: return f"great {base_word}"

        return base_word

    @classmethod
    def translate_word(cls, word: Word, lang: Language | None = None) -> str:
        from word import LexicalCategory
        if word.category == LexicalCategory.FUNCTIONAL_CLOSED:
            word_pos = "Particle"
        else:
            word_pos = cls.get_pos(word.vector)

        base_gloss = cls.translate_vector_constrained(word.vector, target_pos=word_pos)
        deriv = word.derivation

        if deriv.derivation_type == DerivationType.COMPOUND and len(deriv.local_parent_ids) >= 2 and lang:
            parent_glosses = []
            for pid in deriv.local_parent_ids:
                p_word = lang.words.get(pid)
                if p_word and p_word.id != word.id:
                    p_pos = "Particle" if p_word.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(p_word.vector)
                    parent_glosses.append(cls.translate_vector_constrained(p_word.vector, target_pos=p_pos))

            if len(parent_glosses) >= 2 and parent_glosses[0] != parent_glosses[1]:
                compound_label = f"{parent_glosses[0]}-{parent_glosses[1]}"
                return f"{compound_label} ({base_gloss})" if base_gloss not in parent_glosses else compound_label

        elif deriv.derivation_type == DerivationType.AFFIXATION and len(deriv.local_parent_ids) >= 2 and lang:
            stem_w = lang.words.get(deriv.local_parent_ids[0])
            affix_w = lang.words.get(deriv.local_parent_ids[1])
            if stem_w and affix_w:
                s_pos = "Particle" if stem_w.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(stem_w.vector)
                a_pos = "Particle" if affix_w.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(affix_w.vector)
                stem_g = cls.translate_vector_constrained(stem_w.vector, target_pos=s_pos)
                affix_g = cls.translate_vector_constrained(affix_w.vector, target_pos=a_pos)
                return f"{stem_g}+{affix_g} ({base_gloss})"

        return base_gloss

    @classmethod
    def translate(cls, target: SemanticVector | Word, lang: Language | None = None) -> str:
        if hasattr(target, "derivation") and hasattr(target, "vector"):
            return cls.translate_word(target, lang)
        elif hasattr(target, "coords"):
            return cls.translate_vector_constrained(target)
        return str(target)

    @classmethod
    def identify_metaphor_pathway(cls, primary: SemanticVector, extended: SemanticVector) -> str:
        diffs = [extended.coords[i] - primary.coords[i] for i in range(7)]
        relational_gains = [(diffs[i], SemanticVector.DIMENSION_NAMES[i]) for i in range(2, 7)]
        relational_gains.sort(key=lambda item: item[0], reverse=True)
        max_gain, target_dim = relational_gains[0]

        if diffs[0] < -0.08 or diffs[1] < -0.08:
            if target_dim == "sociality" and max_gain > 0.05:
                return "Physical → Social / Hierarchical"
            elif target_dim == "extension" and max_gain > 0.05:
                return "Spatial → Temporal / Extensional"
            elif target_dim == "dynamism" and max_gain > 0.05:
                return "Stative → Process / Cognitive"
            elif target_dim in ("potency", "valence") and max_gain > 0.05:
                return "Physical → Evaluative / Intensive"
            return "Concrete → Abstract Projection"

        return "Semantic Drift"

    @classmethod
    def translate_polysemy(cls, word: Word, lang: Language | None = None) -> list[tuple[str, str, SemanticVector]]:
        from word import LexicalCategory
        if not word.senses or not isinstance(word.senses[0], SemanticVector):
            return [(cls.translate_word(word, lang), "Primary Literal", word.vector)]

        primary_vec = word.senses[0]
        primary_gloss = cls.translate_word(word, lang)
        results = [(primary_gloss, "Primary Literal", primary_vec)]
        seen_meanings = {primary_gloss}

        for ext_vec in word.senses[1:]:
            if not isinstance(ext_vec, SemanticVector):
                continue
            ext_pos = "Particle" if word.category == LexicalCategory.FUNCTIONAL_CLOSED else cls.get_pos(ext_vec)
            ext_gloss = cls.translate_vector_constrained(ext_vec, target_pos=ext_pos)
            if ext_gloss not in seen_meanings:
                pathway = cls.identify_metaphor_pathway(primary_vec, ext_vec)
                results.append((ext_gloss, pathway, ext_vec))
                seen_meanings.add(ext_gloss)

        return results