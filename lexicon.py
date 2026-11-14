"""
lexicon.py — Productive Derivational Morphology, Biomechanical Root Synthesis,
and Demographically Driven Semantic Expansion.
Velaric double-closures (clicks) restricted to small, low-noise foraging communities.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from typing import Sequence
import numpy as np

from phonetics import Phoneme, repair_phonemes, apply_prosody
from word import Word, Morpheme, ClineStage, LexicalCategory
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType


class LexiconGenerator:
    @classmethod
    def synthesize_constriction_gesture(cls, env=None) -> Phoneme:
        """
        Samples continuous 9D tract space.
        Clicks are strictly limited: they require an isolated foraging band
        in open, silent savannah with low population, and incur a heavy motor penalty.
        """
        alt = getattr(env, "altitude", 0.0) if env else 0.0
        hum = getattr(env, "humidity", 0.5) if env else 0.5
        veg = getattr(env, "vegetation", 0.3) if env else 0.3
        pop = getattr(env, "population", 0.5) if env else 0.5
        rho = math.exp(-0.85 * max(0.0, alt))

        g = np.random.uniform(0.0, 1.0, size=9).astype(np.float32)
        g[1] = 0.0  # Lip protrusion default
        g[6] = 0.0  # Velum closed default
        g[8] = 0.50 # Resting neutral larynx

        # Clicks are an extraordinary, rare innovation:
        # Require low population (foragers), flat open terrain, low vegetation, and low noise
        open_plain_suitability = max(0.0, 1.0 - veg * 3.0) * max(0.0, 1.0 - alt * 2.5) * max(0.0, 1.0 - pop * 1.5)
        # Global base rate is very small (< 2.5% even in optimal plains, 0% elsewhere)
        is_click_roll = (random.random() < open_plain_suitability * 0.025)

        if is_click_roll:
            if random.random() < 0.50:
                g[0] = 0.00
                g[2] = 1.00
            else:
                g[0] = 1.00
                g[2] = 0.00
                g[3] = float(np.random.choice([0.05, 0.25, 0.50]))

            # Posterior velar closure (g4=0, g5>=0.75)
            g[4] = 0.00
            g[5] = 0.75
            g[7] = 0.40
            return Phoneme(g, duration=0.85)

        # Standard Single-Constriction Plosive / Fricative
        organ_choice = int(np.random.choice([0, 2, 4]))
        g[organ_choice] = np.random.uniform(0.0, 0.08)

        if organ_choice == 2:
            g[3] = float(np.random.choice([0.25, 0.85]))
        elif organ_choice == 4:
            g[5] = float(np.random.choice([0.15, 0.75]))

        g[7] = float(np.random.choice([0.40, 0.60, 0.02]))

        # High-altitude hypoxia promotes larynx elevation (ejectives) to conserve pulmonary air
        if rho < 0.65 and g[organ_choice] < 0.08:
            if random.random() < (1.0 - rho) * 0.80:
                g[7] = 0.02
                g[8] = 0.85

        # Arid heat promotes pharyngeal constriction and tongue backing
        if hum < 0.32:
            g[5] = max(g[5], 0.80)
            if random.random() < (0.38 - hum) * 0.75:
                g[8] = 0.95

        return Phoneme(g, duration=1.0)

    @classmethod
    def synthesize_resonant_gesture(cls, env=None) -> Phoneme:
        hum = getattr(env, "humidity", 0.5) if env else 0.5
        drying = max(0.0, 0.45 - hum)

        # Disperse vowels across the full F1/F2 space: high-front, low-open, high-back
        corner = random.choices(["front_high", "open_low", "back_high", "mid"], weights=[0.28, 0.30, 0.25, 0.17])[0]

        if corner == "front_high":
            target_height = float(np.clip(np.random.normal(0.28, 0.08), 0.20, 0.40))
            target_backness = float(np.clip(np.random.normal(0.12, 0.06), 0.05, 0.25))
            lip_p = 0.0
        elif corner == "back_high":
            target_height = float(np.clip(np.random.normal(0.30, 0.08), 0.20, 0.45))
            target_backness = float(np.clip(np.random.normal(0.82, 0.08), 0.70, 0.95))
            lip_p = 0.85
        elif corner == "open_low":
            target_height = float(np.clip(np.random.normal(0.90 - 0.10 * drying, 0.08), 0.75, 0.98))
            target_backness = float(np.clip(np.random.normal(0.45, 0.20), 0.20, 0.80))
            lip_p = 0.0
        else:
            target_height = float(np.clip(np.random.normal(0.55, 0.10), 0.45, 0.65))
            target_backness = float(np.clip(np.random.normal(0.50, 0.15), 0.30, 0.70))
            lip_p = 0.40 if target_backness > 0.50 else 0.0

        lip_d = (1.0 - lip_p) * 1.00 + lip_p * 0.35
        g = np.array([lip_d, lip_p, 1.00, 0.00, target_height, target_backness, 0.00, 0.40, 0.50], dtype=np.float32)
        tau = 1.35 if random.random() < 0.20 else 1.00
        return Phoneme(g, duration=tau)

    @classmethod
    def generate_gestural_root(cls, profile=None, env=None, existing_words: Sequence[Word] | None = None) -> list[Phoneme]:
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50

        for _ in range(40):
            gestures = [cls.synthesize_constriction_gesture(env), cls.synthesize_resonant_gesture(env)]
            if random.random() < (s_comp * 0.70):
                gestures.append(cls.synthesize_constriction_gesture(env))

            repaired = repair_phonemes(gestures, profile=profile)
            if not repaired or not any(p.is_vowel for p in repaired):
                continue
            if existing_words:
                cand_ipa = "".join(p.ipa for p in repaired)
                if any(w.plain_form == cand_ipa for w in existing_words):
                    continue
            return repaired
        return repaired

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
        repaired_p = repair_phonemes(raw_p, profile=profile)
        if not repaired_p or not any(p.is_vowel for p in repaired_p):
            return None

        m_base = Morpheme(base_word.phonemes, root_family_id=base_word.derivation.root_family_id)
        m_aff = Morpheme(affix_p, root_family_id=mod_word.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)

        derived_word = Word(
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
        return derived_word

    @classmethod
    def populate_proto_lexicon(cls, population: float = 0.50, **kwargs) -> list[Word]:
        words: list[Word] = []
        seed_names = [
            "water", "fire", "sun", "moon", "earth", "stone", "tree", "mother", "father",
            "person", "beast", "self", "other", "head", "body", "eye", "hand", "heart",
            "blood", "make", "move", "see", "hear", "speak", "eat", "die", "good", "bad",
            "big", "small", "this", "that", "in", "above", "not", "one", "many"
        ]

        for i, seed_vec in enumerate(SemanticSpace.BIOLOGICAL_COGNITIVE_SEEDS):
            phonemes = cls.generate_gestural_root(profile=None, env=None, existing_words=words)
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

        return words