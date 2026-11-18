"""
lexicon.py — Productive Derivational Morphology, Energy-Based Root Coinage,
and Demographically Driven Semantic Expansion.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence
import numpy as np

from phonetics import (
    Phoneme,
    repair_phonemes,
    render_word_trajectory,
    inventory_counts,
    sample_segment,
    sound_change_step,
    _ATTRACTOR_MATRIX,
    _ATTRACTOR_SYMBOLS,
)
from word import Word, Morpheme, ClineStage, LexicalCategory
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType


def _np_rng(rng) -> np.random.Generator:
    """Accepts a numpy Generator, a stdlib Random, or None."""
    if rng is None:
        return np.random.default_rng()
    if hasattr(rng, "integers"):
        return rng
    if hasattr(rng, "getrandbits"):
        return np.random.default_rng(rng.getrandbits(32))
    return np.random.default_rng()


class LexiconGenerator:
    TOKEN_JITTER = 0.015

    @classmethod
    def _realize(cls, symbol_index: int, rng: np.random.Generator, vowel: bool) -> Phoneme:
        g = _ATTRACTOR_MATRIX[symbol_index].astype(np.float32).copy()
        g = np.clip(g + rng.normal(0.0, cls.TOKEN_JITTER, size=9).astype(np.float32), 0.0, 1.0)
        sym = _ATTRACTOR_SYMBOLS[symbol_index]
        if vowel:
            tau = 1.35 if rng.random() < 0.20 else 1.00
        elif sym in {"ʘ", "ǀ", "ǃ", "ǂ", "ǁ"}:
            tau = 0.85
        else:
            tau = 1.00
        return Phoneme(g, duration=tau)

    @classmethod
    def synthesize_constriction_gesture(cls, env=None, counts: np.ndarray | None = None, rng=None) -> Phoneme:
        rng = _np_rng(rng)
        pop = float(getattr(env, "population", 0.5)) if env else 0.5
        return cls._realize(sample_segment(False, counts, env, pop, rng), rng, vowel=False)

    @classmethod
    def synthesize_resonant_gesture(cls, env=None, counts: np.ndarray | None = None, rng=None) -> Phoneme:
        rng = _np_rng(rng)
        pop = float(getattr(env, "population", 0.5)) if env else 0.5
        return cls._realize(sample_segment(True, counts, env, pop, rng), rng, vowel=True)

    @classmethod
    def generate_gestural_root(
        cls,
        profile=None,
        env=None,
        existing_words: Sequence[Word] | None = None,
        rng=None,
        existing_counts: np.ndarray | None = None,
        existing_forms: set[str] | None = None,
    ) -> list[Phoneme]:
        rng = _np_rng(rng)
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50

        counts = existing_counts if (existing_counts is not None and float(existing_counts.sum()) > 0.0) else None
        if counts is None and existing_words:
            c = inventory_counts(existing_words)
            counts = c if float(c.sum()) > 0.0 else None

        forms_set = existing_forms if existing_forms is not None else (
            {w.plain_form for w in existing_words} if existing_words else set()
        )

        fallback = None
        for _ in range(40):
            gestures = [
                cls.synthesize_constriction_gesture(env, counts, rng),
                cls.synthesize_resonant_gesture(env, counts, rng),
            ]
            if rng.random() < (s_comp * 0.70):
                gestures.append(cls.synthesize_constriction_gesture(env, counts, rng))

            repaired = repair_phonemes(list(gestures), profile=profile)
            if not repaired or not any(p.is_vowel for p in repaired):
                continue
            fallback = repaired
            if forms_set:
                plain = render_word_trajectory(np.stack([p.vector for p in repaired], axis=0))[1]
                if plain in forms_set:
                    continue
            return repaired

        if fallback is not None:
            return fallback
        return [cls.synthesize_constriction_gesture(env, counts, rng), cls.synthesize_resonant_gesture(env, counts, rng)]

    @classmethod
    def derive_productive_concept(
        cls,
        base_word: Word,
        derivation_kind: str,
        lang_lexicon: Sequence[Word],
        profile=None,
        env=None,
    ) -> Word | None:
        anchor_map = {
            "agent": SemanticSpace.PERSON,
            "instrument": SemanticSpace.STONE,
            "locative": SemanticSpace.EARTH,
            "diminutive": SemanticSpace.MINI,
            "augmentative": SemanticSpace.HUGE,
        }
        target_anchor = anchor_map.get(derivation_kind, SemanticSpace.PERSON)

        cand_modifiers = [w for w in lang_lexicon if w.id != base_word.id]
        if not cand_modifiers:
            return None
        mod_word = min(cand_modifiers, key=lambda w: w.vector.distance_to(target_anchor))

        if derivation_kind == "agent":
            new_data = base_word.vector.data * 0.55 + mod_word.vector.data * 0.45
            new_data[1] = max(new_data[1], 0.85)
        elif derivation_kind == "instrument":
            new_data = base_word.vector.data * 0.50 + mod_word.vector.data * 0.50
            new_data[0] = max(new_data[0], 0.80)
            new_data[1] = min(new_data[1], 0.20)
        elif derivation_kind == "locative":
            new_data = base_word.vector.data * 0.40 + mod_word.vector.data * 0.60
            new_data[6] = max(new_data[6], 0.70)
        elif derivation_kind == "diminutive":
            new_data = base_word.vector.data.copy()
            new_data[3] = max(0.05, new_data[3] - 0.35)
            new_data[6] = max(0.05, new_data[6] - 0.40)
        else:
            new_data = base_word.vector.data.copy()
            new_data[3] = min(0.95, new_data[3] + 0.35)
            new_data[6] = min(0.95, new_data[6] + 0.35)

        new_vec = SemanticVector.from_array(new_data)

        affix_p = mod_word.phonemes[:1] if len(mod_word.phonemes) > 1 else mod_word.phonemes
        h = getattr(profile, "head_directionality", 0.50)
        raw_p = (base_word.phonemes + affix_p) if h >= 0.50 else (affix_p + base_word.phonemes)
        repaired_p = repair_phonemes(list(raw_p), profile=profile)
        if not repaired_p or not any(p.is_vowel for p in repaired_p):
            return None

        m_base = Morpheme(base_word.phonemes, root_family_id=base_word.derivation.root_family_id)
        m_aff = Morpheme(affix_p, root_family_id=mod_word.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)

        return Word(
            vector=new_vec,
            phonemes=repaired_p,
            morphemes=[m_base, m_aff] if h >= 0.50 else [m_aff, m_base],
            derivation=DerivationRecord(
                root_family_id=base_word.derivation.root_family_id,
                derivation_type=DerivationType.AFFIXATION,
                local_parent_ids=(base_word.id, mod_word.id)
            ),
            category=LexicalCategory.CONTENT_OPEN,
            usage_frequency=1.2,
        )

    @classmethod
    def populate_proto_lexicon(cls, population: float = 0.50, rng=None, **kwargs) -> list[Word]:
        rng = _np_rng(rng if rng is not None else kwargs.get("seed"))
        words: list[Word] = []
        seed_names = [
            "water", "fire", "sun", "moon", "earth", "stone", "tree", "mother", "father",
            "person", "beast", "self", "other", "head", "body", "eye", "hand", "heart",
            "blood", "make", "move", "see", "hear", "speak", "eat", "die", "good", "bad",
            "big", "small", "this", "that", "in", "above", "not", "one", "many"
        ]
        env = SimpleNamespace(population=population)

        for i, seed_vec in enumerate(SemanticSpace.BIOLOGICAL_COGNITIVE_SEEDS):
            phonemes = cls.generate_gestural_root(profile=None, env=env, existing_words=words, rng=rng)
            valence_intensity = abs(seed_vec.valence - 0.50) + seed_vec.potency * 0.40
            scaled_phonemes = [
                Phoneme(p.g, duration=round(p.duration * (0.85 + 0.30 * valence_intensity), 2))
                for p in phonemes
            ]
            cat = LexicalCategory.FUNCTIONAL_CLOSED if seed_vec.is_grammatical else LexicalCategory.CONTENT_OPEN

            w = Word(
                vector=seed_vec,
                phonemes=scaled_phonemes,
                derivation=DerivationRecord(root_family_id=len(words) + 1),
                usage_frequency=1.5,
                category=cat,
                generation_born=0,
            )
            w.senses = (seed_names[i] if i < len(seed_names) else f"concept_{i}",)
            words.append(w)

        # Light self-organization from zero initial inventory without over-merging
        for _ in range(15):
            sound_change_step(words, rng=rng, T=0.05, env=env, p_merge=0.20, p_cond=0.20)
        for w in words:
            w._invalidate_cache()

        return words