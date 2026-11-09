"""
lexicon.py — Continuous Thermodynamic Neologisms, Compression, and Information Dynamics.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from typing import Sequence
import numpy as np

from phonetics import (
    Phoneme,
    repair_phonemes,
    apply_prosody,
    sensory_distance,
    continuous_sonority,
)
from word import Word, Morpheme, ClineStage, LexicalCategory
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType
from gpu_accelerator import GPUSemanticEngine


def _extract_metrical_head(word: Word, mode: str = "head") -> list[Phoneme]:
    syls = word.syllables
    if not syls or len(syls) <= 2:
        return list(word.phonemes)

    stressed_idx = next((i for i, s in enumerate(syls) if s.stressed), 0)
    if mode == "tail":
        extracted_syls = syls[-2:] if len(syls) >= 2 else syls[-1:]
    elif mode == "head":
        extracted_syls = syls[:2]
    else:
        if stressed_idx + 1 < len(syls):
            extracted_syls = syls[stressed_idx:stressed_idx + 2]
        elif stressed_idx > 0:
            extracted_syls = syls[stressed_idx - 1:stressed_idx + 1]
        else:
            extracted_syls = syls[:2]

    extracted: list[Phoneme] = []
    for s in extracted_syls:
        extracted.extend(s.to_phonemes())
    return extracted if extracted else list(word.phonemes)


class LexiconGenerator:
    """
    Synthesizes gestures via continuous Gaussian Mixture sampling across anatomical attractors
    conditioned on thermodynamic barometric pressure.
    """

    @classmethod
    def synthesize_constriction_gesture(cls, env=None) -> Phoneme:
        alt = getattr(env, "altitude", 0.0) if env else 0.0
        hum = getattr(env, "humidity", 0.5) if env else 0.5
        rho_air = math.exp(-0.90 * max(0.0, alt))

        # Base neutral tract: tip relaxed on floor (g2=1.0)
        g = np.array([1.0, 0.0, 1.0, 0.25, 0.80, 0.50, 0.0, 0.40], dtype=np.float32)
        organ_roll = random.random()

        if organ_roll < 0.32:  # Labials (p, b, m, f, v)
            manner = random.random()
            if manner < 0.55:
                g[0], g[7] = 0.00, 0.40 if random.random() < 0.50 else 0.60  # p, b
            elif manner < 0.80:
                g[0], g[6], g[7] = 0.00, 0.60, 0.40                          # m
            else:
                g[0], g[7] = 0.08, 0.65 if random.random() < 0.50 else 0.40  # f, v

        elif organ_roll < 0.72:  # Coronals: explicitly synthesizes t, d, s, z, l, r, n
            manner = random.random()
            g[3] = 0.25
            if manner < 0.40:
                g[2], g[7] = 0.00, 0.60 if random.random() < 0.55 else 0.40  # t, d (stops)
            elif manner < 0.65:
                g[2], g[7] = 0.06, 0.65 if random.random() < 0.60 else 0.40  # s, z (sibilants)
            elif manner < 0.80:
                g[2], g[7] = 0.28, 0.40                                       # l (lateral liquid)
            elif manner < 0.90:
                g[2], g[7] = 0.14, 0.40                                       # r (rhotic tap)
            else:
                g[2], g[6], g[7] = 0.00, 0.60, 0.40                          # n (nasal)

        else:  # Dorsals (k, g, x, ŋ)
            manner = random.random()
            g[5] = 0.75
            if manner < 0.65:
                g[4], g[7] = 0.00, 0.60 if random.random() < 0.55 else 0.40  # k, g
            elif manner < 0.85:
                g[4], g[7] = 0.10, 0.65 if random.random() < 0.50 else 0.40  # x, ɣ
            else:
                g[4], g[6], g[7] = 0.00, 0.60, 0.40                          # ŋ

        # Alpine ejective emergence
        if alt > 0.45 and hum < 0.35 and (g[0] == 0.0 or g[2] == 0.0 or g[4] == 0.0):
            if random.random() < 0.35:
                g[7] = 0.02

        # Lowland humid implosive emergence
        if alt < 0.25 and hum > 0.60 and (g[0] == 0.0 or g[2] == 0.0):
            if random.random() < 0.30:
                g[7] = 0.15

        return Phoneme(g, duration=1.0)

    @classmethod
    def synthesize_resonant_gesture(cls, env=None) -> Phoneme:
        hum = getattr(env, "humidity", 0.5) if env else 0.5
        drying = max(0.0, 0.45 - hum)

        # Explore vowel space smoothly across height and backness
        target_height = float(np.clip(np.random.normal(0.60 - 0.15 * drying, 0.18), 0.25, 0.95))
        target_backness = float(np.clip(np.random.normal(0.50, 0.25), 0.08, 0.90))

        # Physiological rounding coupling
        p_round = 1.0 / (1.0 + math.exp(-8.0 * (target_backness - 0.55)))
        lip_p = float(np.clip(np.random.normal(p_round, 0.10), 0.0, 1.0))
        lip_d = (1.0 - lip_p) * 1.00 + lip_p * 0.35

        if target_height > 0.75:
            lip_p = min(lip_p, 1.15 - target_height)
            lip_d = max(lip_d, 0.50)

        # Vowels have tongue tip relaxed on the floor (g2 = 1.0)
        g = np.array([lip_d, lip_p, 1.00, 0.00, target_height, target_backness, 0.00, 0.40], dtype=np.float32)
        tau = 1.40 if random.random() < 0.20 else 1.00
        return Phoneme(g, duration=tau)

    @classmethod
    def sample_phonotactic_template(cls, profile=None) -> list[str]:
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
        p_coda = 1.0 / (1.0 + math.exp(-7.0 * (s_comp - 0.40)))
        p_cc_onset = 1.0 / (1.0 + math.exp(-8.0 * (s_comp - 0.60)))
        p_poly = 0.20 + 0.45 * (1.0 - s_comp)

        def make_syl() -> str:
            on = "CC" if random.random() < p_cc_onset else "C"
            cod = "C" if random.random() < p_coda else ""
            return f"{on}V{cod}"

        syls = [make_syl()]
        if random.random() < p_poly:
            syls.append(make_syl())
        return syls

    @classmethod
    def generate_gestural_root(cls, profile=None, env=None, existing_words: Sequence[Word] | None = None) -> list[Phoneme]:
        templates = cls.sample_phonotactic_template(profile)
        for _ in range(40):
            gestures = []
            for pat in templates:
                for ch in pat:
                    if ch == "C":
                        gestures.append(cls.synthesize_constriction_gesture(env))
                    else:
                        gestures.append(cls.synthesize_resonant_gesture(env))
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

    @classmethod
    def expand_vocabulary(
        cls,
        lang,
        growth_rate: float = 0.03,
        d_sem_min: float = 0.06,
        current_generation: int = 0,
        profile=None,
    ) -> list[Word]:
        living = list(lang.words.values())
        if not living:
            return []

        prof = profile or getattr(lang, "profile", None)
        env = getattr(lang, "environment", None)
        pop = getattr(lang, "population", 0.50)
        s_idx = getattr(prof, "synthesis_index", 0.50) if prof else 0.50
        s_comp = getattr(prof, "syllable_complexity", 0.50) if prof else 0.50
        dir_bias = getattr(prof, "head_directionality", 0.50) if prof else 0.50
        attn = lang.cultural_attention

        k_cap = round(2500.0 * (1.0 + math.tanh(pop * 1.5)))
        n_current = len(living)
        if n_current >= k_cap:
            return []

        damping = max(0.0, 1.0 - (n_current / k_cap))
        attempts = max(2, min(30, int(round(n_current * growth_rate * damping))))

        density = min(1.0, n_current / 800.0)
        mean_len = sum(len(w.phonemes) for w in living) / max(1, n_current)
        len_pen = math.tanh(max(0.0, mean_len - 5.0))

        e_root = 1.20 * (1.0 - density) - 0.40 * len_pen
        e_comp = 1.40 * (1.0 - s_idx) - 0.80 * len_pen
        e_affx = 1.50 * s_idx + 0.60 * len_pen
        pots = np.array([e_root, e_comp, e_affx], dtype=np.float32)
        exp_p = np.exp(pots - np.max(pots))
        strat_probs = exp_p / np.sum(exp_p)

        max_family_id = max((w.derivation.root_family_id for w in living), default=0)
        new_born: list[Word] = []
        candidate_plans = []

        for _ in range(attempts):
            strat = random.choices(["root", "compound", "affix"], weights=strat_probs, k=1)[0]
            if strat == "root" or len(living) < 10:
                coords = [float(np.clip(random.gauss(0.5, 0.22), 0.02, 0.98)) for _ in range(7)]
                candidate_plans.append({"type": "root", "vec": SemanticVector(*coords), "parents": ()})
            elif strat == "compound":
                w1, w2 = random.sample(living, 2)
                candidate_plans.append({"type": "compound", "vec": w1.vector.blend(w2.vector, 0.55), "parents": (w1, w2)})
            else:
                base, host = random.sample(living, 2)
                candidate_plans.append({"type": "affix", "vec": base.vector.blend(host.vector, 0.70), "parents": (base, host)})

        if not candidate_plans:
            return []

        existing_coords = [w.vector.coords for w in living]
        cand_coords = [p["vec"].coords for p in candidate_plans]
        valid_mask = GPUSemanticEngine.filter_candidate_density_gpu(
            candidates_coords=cand_coords,
            existing_coords=existing_coords,
            weights=attn,
            d_min=d_sem_min,
        )

        for plan, is_valid in zip(candidate_plans, valid_mask):
            if not is_valid:
                continue
            ptype, cand_vec = plan["type"], plan["vec"]

            if ptype == "root":
                p_root = cls.generate_gestural_root(profile=prof, env=env, existing_words=living)
                if not p_root:
                    continue
                max_family_id += 1
                word = Word(
                    vector=cand_vec,
                    phonemes=p_root,
                    derivation=DerivationRecord(root_family_id=max_family_id, derivation_type=DerivationType.ROOT),
                    usage_frequency=0.90,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )
            elif ptype == "compound":
                w1, w2 = plan["parents"]
                h1 = repair_phonemes(_extract_metrical_head(w1, "head"), prof)
                h2 = repair_phonemes(_extract_metrical_head(w2, "tail"), prof)
                raw = h1 + h2
                cleaned = repair_phonemes(raw, prof)
                split = max(1, min(len(cleaned) - 1, len(h1)))
                m1 = Morpheme(cleaned[:split], root_family_id=w1.derivation.root_family_id)
                m2 = Morpheme(cleaned[split:], root_family_id=w2.derivation.root_family_id)
                word = Word(
                    vector=cand_vec,
                    phonemes=cleaned,
                    morphemes=[m1, m2],
                    derivation=DerivationRecord(
                        root_family_id=w1.derivation.root_family_id,
                        derivation_type=DerivationType.COMPOUND,
                        local_parent_ids=(w1.id, w2.id),
                    ),
                    usage_frequency=1.10,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )
            else:
                base, host = plan["parents"]
                p_base = repair_phonemes(_extract_metrical_head(base, "head"), prof)
                p_aff = repair_phonemes(host.phonemes[:2], prof)
                raw = p_base + p_aff
                cleaned = repair_phonemes(raw, prof)
                split = max(1, min(len(cleaned) - 1, len(p_base)))
                m_base = Morpheme(cleaned[:split], root_family_id=base.derivation.root_family_id)
                m_aff = Morpheme(cleaned[split:], root_family_id=host.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)
                word = Word(
                    vector=cand_vec,
                    phonemes=cleaned,
                    morphemes=[m_base, m_aff],
                    derivation=DerivationRecord(
                        root_family_id=base.derivation.root_family_id,
                        derivation_type=DerivationType.AFFIXATION,
                        local_parent_ids=(base.id, host.id),
                    ),
                    usage_frequency=1.00,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )

            apply_prosody(word, directional_bias=dir_bias, syllable_complexity=s_comp)
            if not any(w.plain_form == word.plain_form for w in living):
                lang.add_word(word)
                living.append(word)
                new_born.append(word)

        return new_born