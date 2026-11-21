"""
lexicon.py — Gesture Sampling, Root Coinage, and Derivational Morphology.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence
import numpy as np
import math

from acoustics import AcousticEngine
from language import inventory_counts
from phonetics import (
    render_word_trajectory,
    segment_effort,
    sensory_distance,
    _ATTRACTOR_MATRIX,
    _ATTRACTOR_SYMBOLS,
    _C_IDX,
    _V_IDX,
)
from semantics import DerivationRecord, DerivationType, SemanticSpace, SemanticVector
from word import ClineStage, LexicalCategory, Morpheme, Phoneme, Word, repair_phonemes

_ATTRACTOR_CLARITY: np.ndarray | None = None


def _attractor_clarities() -> np.ndarray:
    global _ATTRACTOR_CLARITY
    if _ATTRACTOR_CLARITY is None:
        ones = np.ones((len(_ATTRACTOR_MATRIX), 1), dtype=np.float32)
        vecs = np.hstack([_ATTRACTOR_MATRIX, ones])
        _ATTRACTOR_CLARITY = np.array(
            [AcousticEngine.get_auditory_features([v]).clarity_score for v in vecs],
            dtype=np.float64,
        )
    return _ATTRACTOR_CLARITY


def _np_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if hasattr(rng, "integers"):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if hasattr(rng, "getrandbits"):
        return np.random.default_rng(rng.getrandbits(32))
    return np.random.default_rng()


def sample_segment(
    vowel: bool,
    counts: np.ndarray | None = None,
    env=None,
    rng=None,
    T: float = 0.09,
    alpha: float = 1.0,
) -> int:
    rng = _np_rng(rng)
    idx = _V_IDX if vowel else _C_IDX
    c = (
        np.asarray(counts, dtype=np.float64)[idx]
        if (counts is not None and len(counts) >= len(_ATTRACTOR_SYMBOLS))
        else np.zeros(len(idx), dtype=np.float64)
    )
    c_sum = float(c.sum())
    cand_mat = _ATTRACTOR_MATRIX[idx]
    efforts = segment_effort(cand_mat, env=env)
    clarities = _attractor_clarities()[idx]

    if c_sum > 0.0 and rng.random() >= (alpha / (c_sum + alpha)):
        # Reuse existing phonemes weighted by usage and clarity-adjusted effort
        score = efforts - 0.35 * clarities
        w = c * np.exp(-(score - score.min()) / 0.35)
        w /= w.sum()
        return int(idx[int(rng.choice(len(idx), p=w))])

    active = c > 0.0
    repulsion = np.zeros(len(idx), dtype=np.float64)
    if np.any(active):
        d_mat = sensory_distance(cand_mat[:, None, :], cand_mat[active][None, :, :])
        repulsion = np.sum(np.exp(-(d_mat ** 2) / (2.0 * 0.55 ** 2)), axis=1)

    cost = efforts - 0.38 * clarities + 0.55 * repulsion
    w = np.exp(-(cost - cost.min()) / T)
    w /= w.sum()
    return int(idx[int(rng.choice(len(idx), p=w))])


class LexiconGenerator:
    TOKEN_JITTER = 0.006

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
        return cls._realize(sample_segment(False, counts=counts, env=env, rng=rng), rng, vowel=False)

    @classmethod
    def synthesize_resonant_gesture(cls, env=None, counts: np.ndarray | None = None, rng=None) -> Phoneme:
        rng = _np_rng(rng)
        return cls._realize(sample_segment(True, counts=counts, env=env, rng=rng), rng, vowel=True)

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
        from phonetics import readout_batch, trajectory_distance, word_energy

        rng = _np_rng(rng)
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50

        counts = existing_counts if (existing_counts is not None and float(existing_counts.sum()) > 0.0) else None
        if counts is None and existing_words:
            c = inventory_counts(existing_words)
            counts = c if float(c.sum()) > 0.0 else None

        active_set = set(np.flatnonzero(counts > 0).tolist()) if counts is not None else set()
        forms_set = existing_forms if existing_forms is not None else (
            {w.plain_form for w in existing_words} if existing_words else set()
        )

        def pick_fresh(is_v: bool) -> Phoneme:
            return cls._realize(sample_segment(is_v, counts=None, env=env, rng=rng, T=0.07), rng, vowel=is_v)

        def pick_inv(is_v: bool) -> Phoneme:
            return cls.synthesize_resonant_gesture(env, counts, rng) if is_v else cls.synthesize_constriction_gesture(env, counts, rng)

        ref_pool = list(existing_words[-28:]) if existing_words else []
        ref_data = [
            (w.tensor[:, :9], AcousticEngine.get_auditory_features(w.tensor))
            for w in ref_pool if w.phonemes
        ]

        candidates: list[list[Phoneme]] = []
        for k in range(18):
            draw_v = lambda: cls._realize(sample_segment(True, counts=counts, env=env, rng=rng, T=0.08, alpha=1.4), rng, vowel=True)
            draw_c = lambda: cls._realize(sample_segment(False, counts=counts, env=env, rng=rng, T=0.08, alpha=1.4), rng, vowel=False)

            roll = rng.random()
            if roll < 0.14:
                v1 = draw_v()
                raw = [v1] if (len(ref_data) < 4 and rng.random() < 0.50) else [v1, v1.drift()]
            elif roll < 0.46:
                raw = [draw_c(), draw_v()]
            elif roll < (0.46 + 0.20 * s_comp):
                # CCV onset cluster (e.g. pra, ska, tra)
                raw = [draw_c(), draw_c(), draw_v()]
            elif roll < (0.68 + 0.15 * s_comp):
                raw = [draw_c(), draw_v(), draw_c()]
            else:
                c1, v1 = draw_c(), draw_v()
                if rng.random() < 0.50:
                    raw = [c1, v1, c1.drift(), v1.drift()]
                else:
                    raw = [c1, v1, draw_c(), draw_v()]

            repaired = repair_phonemes(raw, profile=profile)
            if repaired and any(p.is_vowel for p in repaired):
                candidates.append(repaired)

        if not candidates:
            return [cls.synthesize_constriction_gesture(env, counts, rng), cls.synthesize_resonant_gesture(env, counts, rng)]

        scored: list[tuple[float, list[Phoneme]]] = []
        for cand in candidates:
            cand_tensor = np.stack([p.vector for p in cand], axis=0)
            e_cost = word_energy(cand_tensor, profile=profile, env=env)
            cand_af = AcousticEngine.get_auditory_features(cand_tensor)

            # Inventory economy: reusing established phonemes avoids acquisition cost
            novel_count = sum(1 for sym_i in readout_batch(cand_tensor[:, :9]) if int(sym_i) not in active_set) if active_set else 0
            inv_pen = 1.15 * novel_count

            # Fierce auditory + perceptual separation against ALL existing words
            conf_pen = 0.0
            for ref_g, ref_af in ref_data:
                d_aud = AcousticEngine.auditory_distance(cand_af, ref_af)
                d_traj = trajectory_distance(cand_tensor[:, :9], ref_g)
                conf_pen += 5.0 * math.exp(-((min(d_aud, d_traj * 1.3) / 1.15) ** 2))
                if d_aud < 1.15:
                    conf_pen += 25.0 * (1.15 - d_aud)
                if d_traj < 0.70:
                    conf_pen += 20.0 * (0.70 - d_traj)

            plain = render_word_trajectory(cand_tensor)[1]
            dup_pen = 30.0 if plain in forms_set else 0.0
            scored.append((e_cost + inv_pen + conf_pen + dup_pen, cand))

        scored.sort(key=lambda item: item[0])
        # Pick from top 3 via low-temperature softmax so different hearths don't pick identical first words
        top_k = scored[:min(3, len(scored))]
        vals = np.array([s for s, _ in top_k], dtype=np.float64)
        probs = np.exp(-(vals - vals.min()) / 0.18)
        probs /= probs.sum()
        return top_k[int(rng.choice(len(top_k), p=probs))][1]

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

        first_v = next((idx for idx, p in enumerate(mod_word.phonemes) if p.is_vowel), None)
        affix_p = mod_word.phonemes[: first_v + 1] if first_v is not None else mod_word.phonemes[:1]

        h = getattr(profile, "head_directionality", 0.50)
        raw_p = (base_word.phonemes + affix_p) if h >= 0.50 else (affix_p + base_word.phonemes)
        repaired_p = repair_phonemes(raw_p, profile=profile)
        if not repaired_p or not any(p.is_vowel for p in repaired_p):
            return None

        base_af = AcousticEngine.get_auditory_features(base_word.tensor)
        new_tensor = np.stack([p.vector for p in repaired_p], axis=0)
        new_af = AcousticEngine.get_auditory_features(new_tensor)
        if AcousticEngine.auditory_distance(new_af, base_af) < 0.55 or new_af.clarity_score < base_af.clarity_score * 0.85:
            return None

        m_base = Morpheme(base_word.phonemes, root_family_id=base_word.derivation.root_family_id)
        m_aff = Morpheme(
            affix_p,
            root_family_id=mod_word.derivation.root_family_id,
            is_grammatical=True,
            stage=ClineStage.BOUND_AFFIX,
        )

        return Word(
            vector=new_vec,
            phonemes=repaired_p,
            morphemes=[m_base, m_aff] if h >= 0.50 else [m_aff, m_base],
            derivation=DerivationRecord(
                root_family_id=base_word.derivation.root_family_id,
                derivation_type=DerivationType.AFFIXATION,
                local_parent_ids=(base_word.id, mod_word.id),
            ),
            category=LexicalCategory.CONTENT_OPEN,
            usage_frequency=1.2,
        )

    @classmethod
    def populate_proto_lexicon(cls, population: float = 0.50, rng=None, env=None, **kwargs) -> list[Word]:
        rng = _np_rng(rng if rng is not None else kwargs.get("seed"))
        words: list[Word] = []
        env = env or SimpleNamespace(population=population)

        seeds = list(SemanticSpace.BIOLOGICAL_COGNITIVE_SEEDS)
        order = rng.permutation(len(seeds))

        for idx in order:
            seed_vec = seeds[int(idx)]
            phonemes = cls.generate_gestural_root(profile=None, env=env, existing_words=words, rng=rng)
            valence_intensity = abs(seed_vec.valence - 0.50) + seed_vec.potency * 0.40
            scaled_phonemes = [
                Phoneme(p.g, duration=round(p.duration * (0.85 + 0.30 * valence_intensity), 2))
                for p in phonemes
            ]
            cat = LexicalCategory.FUNCTIONAL_CLOSED if seed_vec.is_grammatical else LexicalCategory.CONTENT_OPEN

            words.append(
                Word(
                    vector=seed_vec,
                    phonemes=scaled_phonemes,
                    derivation=DerivationRecord(root_family_id=len(words) + 1),
                    usage_frequency=1.5,
                    category=cat,
                    generation_born=0,
                )
            )

        return words