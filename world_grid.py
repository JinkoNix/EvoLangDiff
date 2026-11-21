"""
world_grid.py — Continuous 2D Tensor Geography, Planetary Climate Thermodynamics,
Reaction-Diffusion Hegemony, GPU Word-Presence Diffusion, and Sound Change.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
from scipy.ndimage import distance_transform_edt as _edt

from gpu_accelerator import GPUSemanticEngine
from language import inventory_counts
from phonetics import (
    REST_POSTURE,
    ambient_noise,
    readout_batch,
    render_word_trajectory,
    sensory_distance,
    trajectory_distance,
    word_energy,
    _ATTRACTOR_MATRIX,
    _ATTRACTOR_SYMBOLS,
    _C_IDX,
    _V_IDX,
    _VOWEL_SET,
)
from semantics import DerivationRecord, SemanticVector
from word import LexicalCategory, Phoneme, Word


# -----------------------------------------------------------------------------
# 1. SPATIAL OPERATORS & SOUND CHANGE STEP
# -----------------------------------------------------------------------------

_CARDINAL_SLICES = [
    (slice(None, -2), slice(1, -1)),
    (slice(2, None), slice(1, -1)),
    (slice(1, -1), slice(None, -2)),
    (slice(1, -1), slice(2, None)),
]

_DIAGONAL_SLICES = [
    (slice(None, -2), slice(None, -2)),
    (slice(None, -2), slice(2, None)),
    (slice(2, None), slice(None, -2)),
    (slice(2, None), slice(2, None)),
]


def diffuse_masked_2d(f: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if f.ndim == 3:
        m = m[..., np.newaxis]
        D = D[..., np.newaxis]
        fp = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode='edge')
        Dp = np.pad(D, ((1, 1), (1, 1), (0, 0)), mode='edge')
        mp = np.pad(m, ((1, 1), (1, 1), (0, 0)), mode='edge')
    else:
        fp = np.pad(f, 1, mode='edge')
        Dp = np.pad(D, 1, mode='edge')
        mp = np.pad(m, 1, mode='edge')

    cardinal_flux = sum(0.5 * (Dp[s] + D) * mp[s] * (fp[s] - f) for s in _CARDINAL_SLICES)
    diagonal_flux = sum(0.5 * (Dp[s] + D) * mp[s] * (fp[s] - f) for s in _DIAGONAL_SLICES)
    return ((2.0 / 3.0) * cardinal_flux + (1.0 / 6.0) * diagonal_flux) * m


def laplacian_isotropic_2d(f: np.ndarray) -> np.ndarray:
    fp = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode='edge') if f.ndim == 3 else np.pad(f, 1, mode='edge')
    cardinal = sum(fp[s] for s in _CARDINAL_SLICES)
    diagonal = sum(fp[s] for s in _DIAGONAL_SLICES)
    return (2.0 / 3.0) * cardinal + (1.0 / 6.0) * diagonal - (10.0 / 3.0) * f


def conservative_advection_divergence_2d(
    p: np.ndarray, vx: np.ndarray, vy: np.ndarray, mobility: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    m = mask.astype(np.float32)
    face_mx = 0.5 * (mobility[:, :-1] + mobility[:, 1:]) * m[:, :-1] * m[:, 1:]
    face_vx = 0.5 * (vx[:, :-1] + vx[:, 1:])
    fx = face_mx * np.where(face_vx >= 0.0, p[:, :-1], p[:, 1:]) * face_vx

    face_my = 0.5 * (mobility[:-1, :] + mobility[1:, :]) * m[:-1, :] * m[1:, :]
    face_vy = 0.5 * (vy[:-1, :] + vy[1:, :])
    fy = face_my * np.where(face_vy >= 0.0, p[:-1, :], p[1:, :]) * face_vy

    div = np.zeros_like(p, dtype=np.float32)
    div[:, :-1] -= fx
    div[:, 1:] += fx
    div[:-1, :] -= fy
    div[1:, :] += fy
    return div


def conservative_flux_divergence_2d(
    p: np.ndarray, potential: np.ndarray, mobility: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(p, dtype=bool)
    m = mask.astype(np.float32)

    face_mx = 0.5 * (mobility[:, :-1] + mobility[:, 1:]) * m[:, :-1] * m[:, 1:]
    grad_fx = potential[:, 1:] - potential[:, :-1]
    fx = face_mx * np.where(grad_fx >= 0.0, p[:, :-1], p[:, 1:]) * grad_fx

    face_my = 0.5 * (mobility[:-1, :] + mobility[1:, :]) * m[:-1, :] * m[1:, :]
    grad_fy = potential[1:, :] - potential[:-1, :]
    fy = face_my * np.where(grad_fy >= 0.0, p[:-1, :], p[1:, :]) * grad_fy

    div = np.zeros_like(p, dtype=np.float32)
    div[:, :-1] -= fx
    div[:, 1:] += fx
    div[:-1, :] -= fy
    div[1:, :] += fy
    return div


def upwind_shadow(b: np.ndarray, wind: np.ndarray, K: int = 8, lam: float = 0.75) -> np.ndarray:
    H, W = b.shape
    P = K + 1
    bp = np.pad(b, P, mode='edge')
    out = np.zeros_like(b, dtype=np.float32)
    for k in range(1, K + 1):
        dr = int(round(-wind[1] * k))
        dc = int(round(-wind[0] * k))
        out += (lam ** k) * bp[P + dr : P + dr + H, P + dc : P + dc + W]
    return out * (1.0 - lam)


def chaikin_smooth(points: list[list[float]], iterations: int = 2) -> list[list[float]]:
    if len(points) < 3:
        return points
    pts = points
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            new_pts.append([0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1], p0[2]])
            new_pts.append([0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1], p1[2]])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def _as_np_rng(rng) -> np.random.Generator:
    if hasattr(rng, "integers"):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if hasattr(rng, "getrandbits"):
        return np.random.default_rng(rng.getrandbits(32))
    return np.random.default_rng()


def sound_change_step(
    words: list,
    rng,
    T: float = 0.05,
    p_merge: float = 0.12,
    p_cond: float = 0.42,
    p_erode: float = 0.08,
    env=None,
    profile=None,
    lam_conf: float = 1.0,
    n_words_eval: int = 48,
    n_ref: int = 24,
    conf_scale: float = 28.0,
    shift_sigma: float = 0.020,
    ctx_sigma: float = 0.55,
    **kwargs,
) -> dict:
    """Metropolis-Hastings diachronic step: mergers, conditioned drift, unconditioned drift, and word erosion."""
    rng = _as_np_rng(rng)
    env_fn = env if callable(env) else (lambda _w: env)

    # Erosion mode: drop a weak segment from a multi-segment word (CVCV -> CVC / CCV / CV)
    erodible = [wi for wi, w in enumerate(words) if len(w.phonemes) >= 3]
    if erodible and rng.random() < p_erode:
        wi = int(rng.choice(erodible))
        w = words[wi]
        phs = list(w.phonemes)
        # Prefer dropping unstressed/final/medial weak segments while keeping >= 1 vowel
        cands = [j for j in range(1, len(phs)) if any(p.is_vowel for k, p in enumerate(phs) if k != j)]
        if cands:
            drop_j = int(rng.choice(cands))
            ref_idx = rng.choice(len(words), size=min(n_ref, len(words)), replace=False)
            ref_words = [words[i] for i in ref_idx if words[i] is not w and len(words[i].phonemes) > 0]
            env_w = env_fn(w)
            sigma = 0.38 + 0.45 * ambient_noise(env_w)
            inv_2sig2 = 1.0 / (2.0 * sigma * sigma)
            w_freq = float(getattr(w, "usage_frequency", 1.0))

            def erode_energy() -> float:
                e_t = w_freq * word_energy(w, profile=profile, env=env_w)
                if not ref_words:
                    return e_t
                ga = w.tensor[:, :9]
                conf = 0.0
                for rw in ref_words:
                    d_sem = float(w.vector.distance_to(rw.vector)) if hasattr(w, "vector") and hasattr(rw, "vector") else 1.0
                    s_ij = 1.0 - math.exp(-((d_sem / 0.35) ** 2))
                    if s_ij > 1e-3:
                        dw = trajectory_distance(ga, rw.tensor[:, :9])
                        conf += s_ij * float(getattr(rw, "usage_frequency", 1.0)) * math.exp(-(dw * dw) * inv_2sig2)
                return e_t + lam_conf * conf_scale * w_freq * (conf / max(1, len(ref_words)))

            e0 = erode_energy()
            w.phonemes = phs[:drop_j] + phs[drop_j + 1 :]
            e1 = erode_energy()
            dE = e1 - e0
            accept = dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-6))
            if not accept:
                w.phonemes = phs
            return {"accepted": bool(accept), "kind": "erode", "symbol": phs[drop_j].ipa, "target": None, "n_tokens": 1, "n_words": 1, "dE": float(dE)}

    refs, gs, fs = [], [], []
    for wi, w in enumerate(words):
        fw = float(getattr(w, "usage_frequency", 1.0))
        for pi, p in enumerate(w.phonemes):
            refs.append((wi, pi))
            gs.append(p.g)
            fs.append(fw)
    if not gs:
        return {"accepted": False, "kind": "none"}

    G = np.array(gs, dtype=np.float32)
    F = np.array(fs, dtype=np.float32)
    sym = readout_batch(G)
    counts = np.bincount(sym, weights=F, minlength=len(_ATTRACTOR_SYMBOLS)).astype(np.float32)
    active_syms = np.flatnonzero(counts > 0)

    s = int(rng.choice(active_syms)) if (rng.random() < 0.50 and len(active_syms) > 0) else int(sym[int(rng.integers(len(G)))])
    members = np.flatnonzero(sym == s)
    env_ref = env_fn(words[refs[int(members[0])][0]])
    sigma = 0.38 + 0.45 * ambient_noise(env_ref)

    s_is_vowel = _ATTRACTOR_SYMBOLS[s] in _VOWEL_SET
    pool = _V_IDX if s_is_vowel else _C_IDX
    cand_active = pool[(pool != s) & (counts[pool] > 0)]

    roll = rng.random()
    target = None
    if roll < p_merge and len(cand_active) > 0:
        d = sensory_distance(_ATTRACTOR_MATRIX[s][None, :], _ATTRACTOR_MATRIX[cand_active])
        order = np.argsort(d)[:4]
        pr = np.exp(-d[order])
        pr /= pr.sum()
        t = int(cand_active[order[int(rng.choice(len(order), p=pr))]])
        G_new = np.clip(G[members] + (_ATTRACTOR_MATRIX[t] - _ATTRACTOR_MATRIX[s])[None, :], 0.0, 1.0).astype(np.float32)
        kind, target = "merge", _ATTRACTOR_SYMBOLS[t]

    elif roll < (p_merge + p_cond) and len(members) >= 2:
        k0 = int(members[int(rng.integers(len(members)))])
        wi0, pi0 = refs[k0]
        ph0 = words[wi0].phonemes
        prev0 = ph0[pi0 - 1].g if pi0 > 0 else REST_POSTURE
        next0 = ph0[pi0 + 1].g if (pi0 + 1) < len(ph0) else REST_POSTURE

        prev_mat = np.empty((len(members), 9), dtype=np.float32)
        next_mat = np.empty((len(members), 9), dtype=np.float32)
        coart = np.zeros((len(members), 9), dtype=np.float32)
        for m_i, k in enumerate(members):
            wi, pi = refs[k]
            phs = words[wi].phonemes
            pg = phs[pi - 1].g if pi > 0 else REST_POSTURE
            ng = phs[pi + 1].g if (pi + 1) < len(phs) else REST_POSTURE
            prev_mat[m_i], next_mat[m_i] = pg, ng
            coart[m_i] = 0.022 * ((pg - G[k]) + (ng - G[k]))

        w_ctx = np.exp(-(
            (sensory_distance(prev_mat, prev0[None, :]) ** 2 + sensory_distance(next_mat, next0[None, :]) ** 2)
            / (2.0 * ctx_sigma ** 2)
        ))
        keep = w_ctx > 0.18
        if not np.any(keep):
            keep[0] = True
        members = members[keep]
        noise = shift_sigma * rng.normal(0.0, 1.0, size=(len(members), 9)).astype(np.float32)
        G_new = np.clip(G[members] + w_ctx[keep][:, None] * (coart[keep] + noise), 0.0, 1.0).astype(np.float32)
        kind = "conditioned"

    else:
        shift = (shift_sigma * rng.normal(0.0, 1.0, size=9)).astype(np.float32)[None, :]
        G_new = np.clip(G[members] + shift, 0.0, 1.0).astype(np.float32)
        kind = "drift"

    aff = sorted({refs[k][0] for k in members})
    sample = [aff[i] for i in rng.choice(len(aff), size=min(n_words_eval, len(aff)), replace=False)]
    scale = len(aff) / len(sample)
    ref_idx = rng.choice(len(words), size=min(n_ref, len(words)), replace=False)
    sample_words = [words[wi] for wi in sample]
    ref_words = [words[i] for i in ref_idx]
    n_ref_total = max(1, len(ref_words))
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    if all(hasattr(w, "vector") and hasattr(w.vector, "data") for w in sample_words + ref_words):
        S_v = np.stack([w.vector.data for w in sample_words], axis=0)
        R_v = np.stack([w.vector.data for w in ref_words], axis=0)
        d_sem = np.sqrt(np.sum((S_v[:, None, :] - R_v[None, :, :]) ** 2, axis=-1))
    else:
        d_sem = np.ones((len(sample_words), len(ref_words)), dtype=np.float32)

    s_ij = 1.0 - np.exp(-((d_sem / 0.35) ** 2))
    ref_freqs = np.array([float(getattr(o, "usage_frequency", 1.0)) for o in ref_words], dtype=np.float32)
    sample_freqs = np.array([float(getattr(w, "usage_frequency", 1.0)) for w in sample_words], dtype=np.float32)
    pair_weights = np.where(s_ij < 1e-3, 0.0, s_ij) * ref_freqs[None, :]
    for a_i, wa in enumerate(sample_words):
        for b_j, wb in enumerate(ref_words):
            if wa is wb:
                pair_weights[a_i, b_j] = 0.0

    s_by_len: dict[int, list[int]] = {}
    for a_i, wa in enumerate(sample_words):
        if wa.phonemes:
            s_by_len.setdefault(len(wa.phonemes), []).append(a_i)
    r_by_len: dict[int, list[int]] = {}
    for b_j, wb in enumerate(ref_words):
        if wb.phonemes:
            r_by_len.setdefault(len(wb.phonemes), []).append(b_j)

    def local_energy() -> tuple[float, float]:
        e_tok = sum(
            float(sample_freqs[a_i]) * word_energy(w_, profile=profile, env=env_fn(w_))
            for a_i, w_ in enumerate(sample_words)
        )
        r_tensors = {
            L2: np.stack([np.stack([p.g for p in ref_words[j].phonemes], axis=0) for j in r_idx], axis=0)
            for L2, r_idx in r_by_len.items()
        }
        conf_per_s = np.zeros(len(sample_words), dtype=np.float64)
        for L1, s_idx in s_by_len.items():
            A1 = np.stack([np.stack([p.g for p in sample_words[i].phonemes], axis=0) for i in s_idx], axis=0)
            for L2, r_idx in r_by_len.items():
                A2 = r_tensors[L2]
                if L1 == L2:
                    S1, S2 = A1[:, None, :, :], A2[None, :, :, :]
                else:
                    max_L = max(L1, L2)
                    i1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
                    i2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
                    S1, S2 = A1[:, i1, :][:, None, :, :], A2[:, i2, :][None, :, :, :]
                D = np.mean(sensory_distance(S1, S2), axis=-1)
                s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
                s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
                is_tonal = ((1.0 / (1.0 + math.exp(6.0 * (s_comp - 0.40)))) * 0.6 + (1.0 / (1.0 + math.exp(6.0 * (s_idx - 0.40)))) * 0.4) >= 0.55
                if is_tonal:
                    # Compare glottal/laryngeal pitch trajectories (g7) across words
                    P1 = S1[..., 7]
                    P2 = S2[..., 7]
                    D = D + 1.40 * np.mean(np.abs(P1 - P2), axis=-1)
                conf_per_s[s_idx] += np.sum(pair_weights[np.ix_(s_idx, r_idx)] * np.exp(-(D * D) * inv_2sig2), axis=1)
        e_conf = float(lam_conf * conf_scale * np.sum(sample_freqs * (conf_per_s / n_ref_total)))
        return e_tok, e_conf

    tok0, conf0 = local_energy()

    undo = []
    for row, k in enumerate(members):
        wi, pi = refs[k]
        old = words[wi].phonemes[pi]
        undo.append((wi, pi, old))
        words[wi].phonemes[pi] = Phoneme(G_new[row], duration=old.duration)
    for wi in aff:
        words[wi].phonemes = words[wi].phonemes

    tok1, conf1 = local_energy()
    dE = scale * ((tok1 - tok0) + (conf1 - conf0))
    accept = dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-6))

    if not accept:
        for wi, pi, old in undo:
            words[wi].phonemes[pi] = old
        for wi in aff:
            words[wi].phonemes = words[wi].phonemes

    return {
        "accepted": bool(accept),
        "kind": kind,
        "symbol": _ATTRACTOR_SYMBOLS[s],
        "target": target,
        "n_tokens": int(len(members)),
        "n_words": len(aff),
        "dE": float(dE),
        "d_token": float(scale * (tok1 - tok0)),
        "d_conf": float(scale * (conf1 - conf0)),
    }


# -----------------------------------------------------------------------------
# 2. DATA CONTAINERS & WORLD GRID
# -----------------------------------------------------------------------------

@dataclass
class Lake:
    x: float
    y: float
    radius: float
    discharge: float

    def to_dict(self) -> dict:
        return {
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "radius": round(float(self.radius), 2),
            "discharge": round(float(self.discharge), 2),
        }


@dataclass
class Landmark:
    id: int
    name_endonym: str
    gloss: str
    feature_type: str
    x: float
    y: float
    source_culture_id: int
    epoch_named: int
    semantic_vector: list[float]
    infrastructure_level: float = 1.0
    is_named: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name_endonym,
            "gloss": self.gloss,
            "feature_type": self.feature_type,
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "source_culture_id": self.source_culture_id,
            "epoch_named": self.epoch_named,
            "semantic_vector": [round(float(v), 2) for v in self.semantic_vector],
            "infrastructure": round(float(self.infrastructure_level), 1),
        }


@dataclass
class Polity:
    id: int
    name: str
    literal_meaning: str
    x: int
    y: int
    prestige: float
    color: list[int]
    age: int
    dist_cells: np.ndarray
    semantic_vector: list[float]
    government_type: str = "Nomadic Clan"
    centralization: float = 0.35
    infrastructure_bias: float = 0.30
    sacred_focus: float = 0.45
    tolerance: float = 0.60
    is_standardized: bool = False
    standard_epoch: int = 0
    standard_tier: int = 0
    canonical_roots: dict[int, list[Phoneme]] = field(default_factory=dict)
    classical_roots: dict[int, list[Phoneme]] = field(default_factory=dict)
    lexicon_phonemes: dict[int, list[Phoneme]] = field(default_factory=dict)
    is_decapitated: bool = False
    last_secession_epoch: int = 0
    _cached_inv_counts: np.ndarray | None = field(default=None, repr=False)
    _cached_plain_forms: set[str] | None = field(default=None, repr=False)


class WorldGrid:
    MAX_POLITIES = 40

    def __init__(self, size: int = 64, seed: int = 42, proto_roots: Sequence[Word] | None = None):
        self.H = size
        self.W = size
        self.epoch = 0
        self.rng = np.random.default_rng(seed)

        self.wind = np.array([0.80, -0.25], dtype=np.float32)
        self.wind /= np.linalg.norm(self.wind)

        y_coords, x_coords = np.meshgrid(
            np.linspace(0.0, 1.0, self.H, dtype=np.float32),
            np.linspace(0.0, 1.0, self.W, dtype=np.float32),
            indexing='ij',
        )
        self.X_grid = x_coords
        self.Y_grid = y_coords

        self.elev = np.zeros((self.H, self.W), dtype=np.float32)
        self.slope = np.zeros((self.H, self.W), dtype=np.float32)
        self.roughness = np.zeros((self.H, self.W), dtype=np.float32)
        self.temp = np.zeros((self.H, self.W), dtype=np.float32)
        self.hum = np.zeros((self.H, self.W), dtype=np.float32)
        self.veg = np.zeros((self.H, self.W), dtype=np.float32)
        self.fauna = np.zeros((self.H, self.W), dtype=np.float32)
        self.soil_health = np.ones((self.H, self.W), dtype=np.float32)

        self.base_capacity = np.zeros((self.H, self.W), dtype=np.float32)
        self.capacity = np.zeros((self.H, self.W), dtype=np.float32)
        self.resource_value = np.zeros((self.H, self.W), dtype=np.float32)

        self.discharge = np.zeros((self.H, self.W), dtype=np.float32)
        self.river_paths: list[list[list[float]]] = []
        self.lakes: list[Lake] = []

        self.pop = np.zeros((self.H, self.W), dtype=np.float32)
        self.migration_vx = np.zeros((self.H, self.W), dtype=np.float32)
        self.migration_vy = np.zeros((self.H, self.W), dtype=np.float32)
        self.infrastructure = np.zeros((self.H, self.W), dtype=np.float32)
        self.prestige = np.zeros((self.H, self.W), dtype=np.float32)

        self.attention = np.full((self.H, self.W, 7), 1.0, dtype=np.float32)
        self.typology = np.full((self.H, self.W, 3), 0.50, dtype=np.float32)
        self.permeability = np.ones((self.H, self.W), dtype=np.float32)
        self.border_barrier = np.zeros((self.H, self.W), dtype=np.float32)
        self.danger_zones = np.zeros((self.H, self.W), dtype=np.float32)

        self.phonetic_bias = np.zeros((self.H, self.W, 9), dtype=np.float32)
        self.phonetic_drift_field = np.zeros((self.H, self.W, 9), dtype=np.float32)

        self.hegemony = np.zeros((self.H, self.W, 16), dtype=np.float32)
        self.polities: list[Polity] = []
        self.landmarks: dict[int, Landmark] = {}
        self.next_landmark_id = 1
        self.next_polity_id = 0
        self.new_city_events: list[dict] = []
        self.events: list[str] = []
        self._probe_cache: dict[tuple[int, int, int], dict] = {}
        self._probe_words_cache: dict[tuple[int, int, int], dict[int, Word]] = {}
        self._phono_cursor = 0
        self.phonology_every = 4
        self.phonology_steps = 2
        self.phonology_cohort = 240

        self._grid_y, self._grid_x = np.indices((self.H, self.W), dtype=np.float32)

        self.lexical_manifold: list[Word] = [w.copy() for w in proto_roots] if proto_roots else []
        self._hearth_word_indices: dict[int, list[int]] = {}
        self._wp_buf = GPUSemanticEngine.alloc_presence_buffer(max(256, len(self.lexical_manifold)), self.H, self.W)
        self._wp_cpu_cache: np.ndarray | None = None
        self._wp_dirty = True

        self._elev_version = 1
        self._terrain_cache: dict[str, Any] = {}

        self._init_geography()
        self._init_hydrology()
        self._init_demographics(use_shared_proto=bool(proto_roots))
        self._init_word_presence()

    @property
    def word_presence(self) -> np.ndarray:
        n = max(1, len(self.lexical_manifold))
        if self._wp_dirty or self._wp_cpu_cache is None or self._wp_cpu_cache.shape[0] < n:
            buf = self._wp_buf
            self._wp_cpu_cache = buf[:n].detach().cpu().numpy() if hasattr(buf, "detach") else buf[:n].copy()
            self._wp_dirty = False
        return self._wp_cpu_cache

    @word_presence.setter
    def word_presence(self, arr: np.ndarray) -> None:
        self._wp_buf = GPUSemanticEngine.ensure_presence_capacity(self._wp_buf, arr.shape[0])
        for i in range(arr.shape[0]):
            GPUSemanticEngine.set_presence_slice(self._wp_buf, i, arr[i])
        self._wp_dirty = True

    def get_presence_column(self, x: int, y: int) -> np.ndarray:
        return GPUSemanticEngine.get_presence_column(self._wp_buf, len(self.lexical_manifold), x, y)

    def _init_geography(self) -> None:
        x, y = self.X_grid, self.Y_grid
        w1 = np.sin(x * 3.8 + y * 2.2) * np.cos(y * 3.4 - x * 1.8)
        w2 = 0.50 * np.sin(x * 7.6 - y * 4.8 + 1.2) * np.cos(y * 6.8 + x * 3.6)
        w3 = 0.25 * np.cos(x * 15.2 + y * 13.6 - 0.8)
        base_relief = 0.55 * (w1 + w2 + w3)

        ridge_center = 0.50 + 0.16 * np.sin(y * 4.2)
        ridge_dist_sq = ((x - ridge_center) * 2.4) ** 2 + ((y - 0.50) * 1.4) ** 2
        mountains = 0.98 * np.exp(-ridge_dist_sq / 0.040)

        dx = x - 0.50
        dy = y - 0.50
        radial_mask = 1.0 - (dx ** 2 + dy ** 2) * 2.5
        island_mass = 0.42 * np.exp(-(((x - 0.80) * 2.6) ** 2 + ((y - 0.36) * 2.6) ** 2) / 0.02)

        raw_elev = (base_relief + mountains + island_mass) * radial_mask - 0.06
        self.elev = np.clip(raw_elev, -1.0, 1.25).astype(np.float32)

        grad_ey, grad_ex = np.gradient(self.elev)
        self.slope = np.sqrt(grad_ey ** 2 + grad_ex ** 2).astype(np.float32)
        self.roughness = np.clip(self.slope / 0.15, 0.0, 1.0).astype(np.float32)
        self._elev_version += 1

    def _init_hydrology(self) -> None:
        grad_y, grad_x = np.gradient(self.elev)
        slope_lift = grad_x * self.wind[0] + grad_y * self.wind[1]
        orographic_rain = np.where(slope_lift > 0, np.minimum(0.40, slope_lift * 4.2), 0.0)

        self.discharge.fill(0.0)
        river_paths: list[list[list[float]]] = []
        lakes: list[Lake] = []

        spring_mask = (self.elev > 0.38) & (orographic_rain > 0.06)
        sy_arr, sx_arr = np.nonzero(spring_mask)
        if len(sy_arr) > 0:
            order = np.argsort(-((self.elev * orographic_rain)[sy_arr, sx_arr]))
            sy_arr, sx_arr = sy_arr[order], sx_arr[order]

        chosen_springs: list[tuple[float, float]] = []
        for sy, sx in zip(sy_arr, sx_arr):
            if not any(math.hypot(sx - cx, sy - cy) < 6.0 for cx, cy in chosen_springs):
                chosen_springs.append((float(sx), float(sy)))
                if len(chosen_springs) >= 9:
                    break

        for start_x, start_y in chosen_springs:
            pos = np.array([start_x, start_y], dtype=np.float32)
            vel = np.zeros(2, dtype=np.float32)
            acc_discharge = 0.5
            raw_path: list[list[float]] = [[float(pos[0]), float(pos[1]), float(acc_discharge)]]
            visited = set()

            for _ in range(120):
                ix = int(np.clip(round(pos[0]), 0, self.W - 1))
                iy = int(np.clip(round(pos[1]), 0, self.H - 1))

                acc_discharge += 0.10
                self.discharge[iy, ix] += acc_discharge
                visited.add((ix, iy))

                if self.elev[iy, ix] <= 0.0:
                    break

                gx, gy = grad_x[iy, ix], grad_y[iy, ix]
                slope_vec = np.array([-gx, -gy], dtype=np.float32)
                norm_s = float(np.linalg.norm(slope_vec))

                if norm_s < 0.015:
                    if not any(math.hypot(pos[0] - lk.x, pos[1] - lk.y) < 3.0 for lk in lakes):
                        lakes.append(Lake(x=float(pos[0]), y=float(pos[1]), radius=0.8 + 0.3 * math.sqrt(acc_discharge), discharge=acc_discharge))

                    best_target = None
                    min_e = 999.0
                    for radius in range(2, 6):
                        for dy_s in range(-radius, radius + 1):
                            for dx_s in range(-radius, radius + 1):
                                nx = int(np.clip(ix + dx_s, 0, self.W - 1))
                                ny = int(np.clip(iy + dy_s, 0, self.H - 1))
                                if (nx, ny) not in visited and self.elev[ny, nx] < min_e:
                                    min_e = self.elev[ny, nx]
                                    best_target = np.array([dx_s, dy_s], dtype=np.float32)
                        if best_target is not None and min_e < self.elev[iy, ix]:
                            break

                    if best_target is not None:
                        slope_vec = best_target / (np.linalg.norm(best_target) + 1e-4)
                    else:
                        break

                vel = 0.70 * vel + 0.30 * (slope_vec / (np.linalg.norm(slope_vec) + 1e-4))
                pos += vel * 0.80
                if pos[0] < 0.5 or pos[0] >= self.W - 0.5 or pos[1] < 0.5 or pos[1] >= self.H - 0.5:
                    break
                raw_path.append([float(pos[0]), float(pos[1]), float(acc_discharge)])

            if len(raw_path) >= 6:
                river_paths.append(chaikin_smooth(raw_path, iterations=2))

        self.river_paths = river_paths
        self.lakes = lakes
        self._elev_version += 1
        self._update_climate(epoch=0)

    def _ensure_terrain_cache(self) -> dict[str, Any]:
        if self._terrain_cache.get("ver") != self._elev_version:
            grad_y, grad_x = np.gradient(self.elev)
            slope_lift = grad_x * self.wind[0] + grad_y * self.wind[1]
            oro = np.where(slope_lift > 0, np.minimum(0.45, slope_lift * 5.0), 0.0).astype(np.float32)
            high_barrier = np.maximum(0.0, self.elev - 0.20)
            rainshadow = (upwind_shadow(high_barrier, self.wind, K=8, lam=0.75) * 0.70).astype(np.float32)
            fluvial = (6.0 * (np.clip(self.discharge / 2.0, 0.0, 3.0) ** 1.6)).astype(np.float32)
            coastal = np.where((self.elev > 0.0) & (self.elev < 0.10), 3.0, 0.0).astype(np.float32)
            land_m = (self.elev > 0.0).astype(np.float32)
            c_edge = np.clip(
                np.sqrt(np.gradient(land_m, axis=0) ** 2 + np.gradient(land_m, axis=1) ** 2) * 3.0,
                0.0, 1.0,
            ).astype(np.float32)
            self._terrain_cache = {
                "ver": self._elev_version,
                "oro": oro,
                "rainshadow": rainshadow,
                "fluvial": fluvial,
                "coastal": coastal,
                "coast_edge": c_edge,
                "slope_exp": np.exp(-6.0 * self.slope).astype(np.float32),
                "mobility": (0.065 * np.exp(-5.0 * self.slope) * land_m).astype(np.float32),
                "river_conduit": np.clip(self.discharge / 3.0, 0.0, 2.5).astype(np.float32),
                "river_junction": (1.2 * np.log1p(self.discharge)).astype(np.float32),
                "ambient_moisture": (0.35 + 0.22 * np.sin(self.X_grid * 3.2 + self.Y_grid * 2.6)).astype(np.float32),
            }
        return self._terrain_cache

    def _update_climate(self, epoch: int) -> None:
        tc = self._ensure_terrain_cache()
        season = math.sin(2.0 * math.pi * (epoch % 12) / 12.0)

        insolation = np.clip(np.sin(np.pi * np.clip(self.Y_grid + season * 0.08, 0.0, 1.0)), 0.0, 1.0)
        base_thermal = 0.06 + 0.88 * (insolation ** 1.35)
        sea_thermal = np.clip(0.05 + 0.72 * (insolation ** 1.2), 0.02, 0.82)
        land_temp = np.clip(base_thermal - 1.25 * np.maximum(0.0, self.elev), 0.02, 1.0)
        self.temp = np.where(self.elev <= 0.0, sea_thermal, land_temp)

        raw_hum = np.where(self.elev <= 0.0, 0.88, tc["ambient_moisture"] + tc["oro"] - tc["rainshadow"])
        self.hum = np.clip(raw_hum, 0.05, 0.98)

        thermal_gate = np.tanh(4.5 * np.maximum(0.0, self.temp - 0.18))
        pet = 0.20 + 0.90 * self.temp
        water_availability = (self.hum + np.clip(self.discharge / 2.5, 0.0, 2.5) * 0.35) / np.maximum(0.10, pet)
        climax_veg = 0.95 * thermal_gate * (water_availability / (0.85 + water_availability))

        timber_depletion = 0.010 * np.clip(self.pop / 3.0, 0.0, 1.0) * (self.attention[..., 0] / 1.5)
        self.soil_health = np.clip(self.soil_health + (0.006 - timber_depletion), 0.35, 1.0)

        emergent_veg = np.where(self.elev <= 0.0, 0.0, climax_veg * self.soil_health)
        self.veg = np.clip(emergent_veg * np.exp(-0.20 * (self.pop ** 0.8)), 0.005, 0.98)

        thermal_fauna_peak = np.exp(-((self.temp - 0.55) ** 2) / 0.12)
        raw_fauna = 2.5 * (self.veg ** 0.8) * thermal_fauna_peak * (1.0 + 0.5 * np.tanh(self.discharge))
        human_hunting = np.exp(-0.35 * self.pop * (1.0 + self.attention[..., 1] * 0.4))
        self.fauna = np.where(self.elev <= 0.0, 0.0, np.clip(raw_fauna * human_hunting, 0.01, 3.5))

        base_food = self.veg * 2.0 + self.fauna * 1.5 + tc["fluvial"] + tc["coastal"]
        self.base_capacity = np.where(self.elev <= 0.0, 0.0, np.clip(0.12 + 0.65 * (base_food ** 1.35), 0.05, 25.0))
        self.capacity = np.clip(self.base_capacity * (1.0 + 0.60 * np.log1p(self.infrastructure * 0.40)), 0.05, 45.0)
        self.resource_value = np.where(self.elev <= 0.0, 0.0, base_food * (1.0 + self.elev * 0.2) * (1.0 + 0.3 * np.log1p(self.infrastructure)))

    def assign_polity_chroma(self, polity_id: int) -> list[int]:
        hue = (0.08 + polity_id * 0.618033988749895) % 1.0
        r_f, g_f, b_f = colorsys.hls_to_rgb(hue, 0.54, 0.85)
        return [int(r_f * 255), int(g_f * 255), int(b_f * 255)]

    def evaluate_site_semantic_vector(self, x: float, y: float) -> list[float]:
        ix = int(np.clip(round(x), 0, self.W - 1))
        iy = int(np.clip(round(y), 0, self.H - 1))
        disch, elev, veg, pop, infra = self.discharge[iy, ix], self.elev[iy, ix], self.veg[iy, ix], self.pop[iy, ix], self.infrastructure[iy, ix]
        return [
            float(np.clip(0.20 + 0.60 * max(0.0, elev) + 0.40 * (1.0 - veg), 0.0, 1.0)),
            float(np.clip(0.15 + 0.70 * veg + 0.30 * math.tanh(pop * 0.4), 0.0, 1.0)),
            float(np.clip(0.30 + 0.50 * veg - 0.25 * max(0.0, elev), 0.0, 1.0)),
            float(np.clip(0.20 + 0.60 * math.tanh(infra * 0.25) + 0.35 * max(0.0, elev), 0.0, 1.0)),
            float(np.clip(0.15 + 0.65 * math.tanh(disch * 0.35) + 0.30 * math.tanh(pop * 0.25), 0.0, 1.0)),
            float(np.clip(0.10 + 0.80 * math.tanh(pop * 0.35) + 0.20 * math.tanh(infra * 0.2), 0.0, 1.0)),
            float(np.clip(0.20 + 0.60 * (1.0 - self.permeability[iy, ix]), 0.0, 1.0)),
        ]

    def _spawn_clan(self, sx: int, sy: int, footprint: np.ndarray, name_hint: str = "") -> None:
        nid = self.next_polity_id
        self.next_polity_id += 1
        if nid >= self.hegemony.shape[-1]:
            self.hegemony = np.pad(self.hegemony, ((0, 0), (0, 0), (0, 8)), mode='constant')

        nearest_p = min(self.polities, key=lambda pol: math.hypot(pol.x - sx, pol.y - sy)) if self.polities else None
        dist_to_nearest = math.hypot(nearest_p.x - sx, nearest_p.y - sy) if nearest_p else 999.0

        inherited_phon: dict[int, list[Phoneme]] = {}
        if nearest_p and dist_to_nearest < 14.0 and nearest_p.lexicon_phonemes:
            inherited_phon = {wid: [ph.drift() for ph in plist] for wid, plist in nearest_p.lexicon_phonemes.items()}
            clan_type = "Branch Tribe"
        else:
            from lexicon import LexiconGenerator
            iso_rng = np.random.default_rng(int(self.rng.integers(100000)) + nid * 331)
            iso_env = SimpleNamespace(
                altitude=float(self.elev[sy, sx]),
                humidity=float(self.hum[sy, sx]),
                temperature=float(self.temp[sy, sx]),
                vegetation=float(self.veg[sy, sx]),
                fauna=float(self.fauna[sy, sx]),
                roughness=float(self.roughness[sy, sx]),
                population=float(self.pop[sy, sx]),
            )
            iso_words = LexiconGenerator.populate_proto_lexicon(
                population=float(self.pop[sy, sx]),
                env=iso_env,
                rng=iso_rng,
            )

            for w_idx, w in enumerate(iso_words):
                w.id = base_id + w_idx
                w.derivation.root_family_id = w.id
                self.lexical_manifold.append(w)
                inherited_phon[w.id] = list(w.phonemes)
                new_idx = len(self.lexical_manifold) - 1
                self._wp_buf = GPUSemanticEngine.ensure_presence_capacity(self._wp_buf, new_idx + 1)
                GPUSemanticEngine.set_presence_slice(self._wp_buf, new_idx, iso_fp)
                self._wp_dirty = True
            clan_type = "Indigenous Isolate"

        dist = np.hypot(self._grid_x - sx, self._grid_y - sy)
        self.polities.append(Polity(
            id=nid,
            name=name_hint or f"{clan_type} #{nid}",
            literal_meaning="",
            x=int(sx),
            y=int(sy),
            prestige=1.2,
            color=self.assign_polity_chroma(nid),
            age=self.epoch,
            dist_cells=dist,
            semantic_vector=self.evaluate_site_semantic_vector(float(sx), float(sy)),
            government_type="Nomadic Clan",
            centralization=0.30,
            infrastructure_bias=0.25,
            sacred_focus=0.45,
            tolerance=0.65,
            lexicon_phonemes=inherited_phon,
        ))
        self.hegemony[..., nid] = np.where(footprint, 5.0 * np.exp(-dist ** 2 / 6.0), 0.0)
        self.events.append(f"Epoch {self.epoch}: Autonomous settlers founded {clan_type} #{nid} at ({sx}, {sy}).")

    def _seed_unclaimed_clans(self, k_max: int = 4, r_min: float = 6.0) -> None:
        land = self.elev > 0.0
        claimed = self.hegemony.sum(axis=-1) > 0.25
        free = land & (self.pop > 0.40)

        for _ in range(k_max):
            if len(self.polities) >= self.MAX_POLITIES:
                break
            d = _edt(~claimed) if claimed.any() else np.full(claimed.shape, 1e3)
            score = np.where(free & ~claimed & (d >= r_min), self.pop * d, -1.0)
            sy, sx = np.unravel_index(np.argmax(score), score.shape)
            if score[sy, sx] <= 0.0:
                break
            self._spawn_clan(int(sx), int(sy), land & ~claimed)
            claimed |= np.hypot(self._grid_x - sx, self._grid_y - sy) < r_min

    def _init_demographics(self, use_shared_proto: bool = False) -> None:
        from lexicon import LexiconGenerator

        fertility = np.where(self.elev > 0.10, self.capacity * self.veg, -1.0)
        sorted_indices = np.argsort(fertility.ravel())[::-1]

        hearth_nodes: list[tuple[int, int]] = []
        for idx in sorted_indices:
            cy, cx = np.unravel_index(idx, fertility.shape)
            if not any(math.hypot(cx - hx, cy - hy) < 12.0 for hy, hx in hearth_nodes):
                hearth_nodes.append((int(cy), int(cx)))
                if len(hearth_nodes) >= 6:
                    break

        self.polities = []
        self._hearth_word_indices = {}
        if not use_shared_proto:
            self.lexical_manifold = []

        for pid, (cy, cx) in enumerate(hearth_nodes):
            dy, dx = self.Y_grid - self.Y_grid[cy, cx], self.X_grid - self.X_grid[cy, cx]
            self.pop += (0.90 * np.exp(-(dx ** 2 + dy ** 2) / 0.003)).astype(np.float32)

            d2_h = (self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2
            self.infrastructure = np.maximum(self.infrastructure, (0.90 * np.exp(-d2_h / 1.8)).astype(np.float32))

            is_highland = self.elev[cy, cx] > 0.40
            is_river = self.discharge[cy, cx] > 1.2
            self.typology[cy, cx, 0] = 0.18 if is_highland else (0.80 if self.elev[cy, cx] < 0.08 else 0.50)
            self.typology[cy, cx, 1] = 0.70 if is_highland else (0.35 if is_river else 0.50)
            self.typology[cy, cx, 2] = 0.75 if is_highland else (0.30 if self.elev[cy, cx] < 0.08 else 0.50)

            hearth_env = SimpleNamespace(
                altitude=float(self.elev[cy, cx]),
                humidity=float(self.hum[cy, cx]),
                temperature=float(self.temp[cy, cx]),
                vegetation=float(self.veg[cy, cx]),
                fauna=float(self.fauna[cy, cx]),
                roughness=float(self.roughness[cy, cx]),
                population=float(self.pop[cy, cx]),
            )

            if use_shared_proto:
                lex_phon = {w.id: [p.drift() for p in w.phonemes] for w in self.lexical_manifold}
            else:
                hearth_rng = np.random.default_rng(int(self.rng.integers(100000)) + pid * 997)
                family_words = LexiconGenerator.populate_proto_lexicon(
                    population=float(self.pop[cy, cx]),
                    env=hearth_env,
                    rng=hearth_rng,
                )
                base_id = (pid + 1) * 10000
                h_indices = []
                for w_idx, w in enumerate(family_words):
                    w.id = base_id + w_idx
                    w.derivation.root_family_id = w.id
                    self.lexical_manifold.append(w)
                    h_indices.append(len(self.lexical_manifold) - 1)
                self._hearth_word_indices[pid] = h_indices
                lex_phon = {w.id: list(w.phonemes) for w in family_words}

            env_label = "Valley Clan" if is_river else ("Upland Band" if is_highland else "Coastal Tribe")
            self.polities.append(Polity(
                id=pid,
                name=f"Hearth #{pid} ({env_label})",
                literal_meaning=f"Original {env_label}",
                x=int(cx),
                y=int(cy),
                prestige=1.4,
                color=self.assign_polity_chroma(pid),
                age=0,
                dist_cells=np.sqrt((self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2),
                semantic_vector=self.evaluate_site_semantic_vector(float(cx), float(cy)),
                government_type="Nomadic Clan",
                centralization=0.30,
                infrastructure_bias=0.25,
                sacred_focus=0.45,
                tolerance=0.65,
                lexicon_phonemes=lex_phon,
            ))
            self.hegemony[cy, cx, pid] = 4.5

        self.next_polity_id = len(hearth_nodes)
        self._discover_and_update_landmarks()

    def _init_word_presence(self) -> None:
        if not self.polities or not self.lexical_manifold:
            return
        n_total = len(self.lexical_manifold)
        self._wp_buf = GPUSemanticEngine.ensure_presence_capacity(self._wp_buf, n_total + 64)

        if self._hearth_word_indices:
            for p in self.polities:
                d2 = (self._grid_x - p.x) ** 2 + (self._grid_y - p.y) ** 2
                local_fp = (2.2 * np.exp(-d2 / 12.0)).astype(np.float32)
                for w_idx in self._hearth_word_indices.get(p.id, []):
                    GPUSemanticEngine.set_presence_slice(self._wp_buf, w_idx, local_fp)
        else:
            for p in self.polities:
                d2 = (self._grid_x - p.x) ** 2 + (self._grid_y - p.y) ** 2
                GPUSemanticEngine.add_to_presence_slice(self._wp_buf, n_total, np.exp(-d2 / 20.0).astype(np.float32), scale=1.5)

        self._wp_dirty = True

    def _name_landmark_from_manifold(self, lm: Landmark, owner_id: int) -> None:
        if lm.is_named or not self.lexical_manifold:
            return
        from translator import SemanticTranslator

        ix = int(np.clip(round(lm.x), 0, self.W - 1))
        iy = int(np.clip(round(lm.y), 0, self.H - 1))
        target_vec = SemanticVector(*lm.semantic_vector)

        n = len(self.lexical_manifold)
        pres = self.get_presence_column(ix, iy)
        cand_idx = np.where(pres > 0.03)[0]
        cands = [self.lexical_manifold[int(i)] for i in cand_idx] if len(cand_idx) > 0 else self.lexical_manifold[:min(37, n)]

        best_w = min(cands, key=lambda w: w.vector.distance_to(target_vec))
        dom_pol = next((p for p in self.polities if p.id == owner_id), None)
        ph_list = dom_pol.lexicon_phonemes.get(best_w.id, best_w.phonemes) if dom_pol else best_w.phonemes

        phi = self.phonetic_bias[iy, ix] + self.phonetic_drift_field[iy, ix]
        phi_v = phi.copy()
        phi_v[7], phi_v[8] = 0.0, 0.0
        phi_c = phi.copy()
        phi_c[4] *= 0.25
        local_ph = [Phoneme(np.clip(p.g + (phi_v if p.is_vowel else phi), 0.0, 1.0), duration=p.duration) for p in ph_list]
        form, _ = render_word_trajectory(np.stack([p.vector for p in local_ph], axis=0), directional_bias=float(self.typology[iy, ix, 0]))
        gloss = SemanticTranslator.translate_vector_compositional(best_w.vector)
        lm.name_endonym = form.replace("ˈ", "").capitalize()
        lm.gloss = f"{gloss} ({lm.feature_type})"
        lm.is_named = True

    def _discover_and_update_landmarks(self) -> None:
        tot_heg = np.sum(self.hegemony, axis=-1)
        dom_owners = np.where(tot_heg > 0.25, np.argmax(self.hegemony, axis=-1), -1)

        py_arr, px_arr = np.nonzero((self.infrastructure >= 3.0) & (self.pop >= 1.4) & (self.elev > 0.0))
        for py, px in zip(py_arr, px_arr):
            existing_city = next((lm for lm in self.landmarks.values() if lm.feature_type == "city" and math.hypot(lm.x - px, lm.y - py) < 3.2), None)
            owner_id = int(dom_owners[py, px])
            if existing_city is None:
                sem_vec = self.evaluate_site_semantic_vector(float(px), float(py))
                new_lm = Landmark(
                    id=self.next_landmark_id, name_endonym="", gloss="", feature_type="city",
                    x=float(px), y=float(py), source_culture_id=owner_id, epoch_named=self.epoch,
                    semantic_vector=sem_vec, infrastructure_level=float(self.infrastructure[py, px]),
                )
                self._name_landmark_from_manifold(new_lm, owner_id)
                self.landmarks[self.next_landmark_id] = new_lm
                self.next_landmark_id += 1
                self.new_city_events.append({"x": int(px), "y": int(py), "name": new_lm.name_endonym or "Settlement", "color": [255, 215, 0]})
            else:
                existing_city.infrastructure_level = float(self.infrastructure[py, px])
                if not existing_city.is_named:
                    self._name_landmark_from_manifold(existing_city, owner_id)

        pad_elev = np.pad(self.elev, 1, mode='edge')
        is_m_peak = (
            (self.elev >= pad_elev[:-2, 1:-1]) & (self.elev >= pad_elev[2:, 1:-1]) &
            (self.elev >= pad_elev[1:-1, :-2]) & (self.elev >= pad_elev[1:-1, 2:]) &
            (self.elev > 0.52)
        )
        my_arr, mx_arr = np.nonzero(is_m_peak)
        for py, px in zip(my_arr, mx_arr):
            if not any(lm.feature_type == "mountain" and math.hypot(lm.x - px, lm.y - py) < 4.5 for lm in self.landmarks.values()):
                cid = int(dom_owners[py, px])
                new_m = Landmark(
                    id=self.next_landmark_id, name_endonym="", gloss="", feature_type="mountain",
                    x=float(px), y=float(py), source_culture_id=cid, epoch_named=self.epoch,
                    semantic_vector=self.evaluate_site_semantic_vector(float(px), float(py)),
                )
                self._name_landmark_from_manifold(new_m, cid)
                self.landmarks[self.next_landmark_id] = new_m
                self.next_landmark_id += 1

    def _step_reaction_diffusion_hegemony(self, dt: float) -> None:
        land_mask = self.elev > 0.0
        res_pull = 1.0 + 0.85 * (self.resource_value / (1.0 + self.resource_value))
        river_conduit = self._ensure_terrain_cache()["river_conduit"]

        active_polities: list[Polity] = []
        for p in self.polities:
            pid, px, py = p.id, p.x, p.y
            u = self.hegemony[..., pid]
            tot_u = float(np.sum(u))

            rival_at_cap = float(np.sum(self.hegemony[py, px]) - u[py, px])
            u_cap = float(u[py, px])
            if rival_at_cap > u_cap:
                p.is_decapitated = True
            elif u_cap > 1.5 * rival_at_cap:
                p.is_decapitated = False

            if (tot_u < 0.80 or (p.is_decapitated and tot_u < 2.5)) and self.epoch > p.age + 15:
                self.hegemony[..., pid] = 0.0
                self.events.append(f"Epoch {self.epoch}: Capital of {p.name or ('#' + str(pid))} fell. Dynasty dissolved.")
                continue

            active_polities.append(p)
            p_prest = float(self.prestige[py, px])
            p.prestige = p_prest

            institutional_mass = p_prest * (1.0 + math.log1p(float(self.infrastructure[py, px])))
            if not p.is_standardized and institutional_mass > 12.0 and (self.epoch - p.age) > 12:
                p.is_standardized = True
                p.standard_tier = 1
                p.standard_epoch = self.epoch
                p.canonical_roots = {
                    w.id: [phon.drift() for phon in p.lexicon_phonemes.get(w.id, w.phonemes)]
                    for w in self.lexical_manifold[:min(80, len(self.lexical_manifold))]
                }
                self.events.append(f"Epoch {self.epoch}: {p.name or ('#' + str(pid))} codified its Regional Standard Register.")
            elif p.is_standardized and p.standard_tier == 1 and institutional_mass > 24.0:
                p.standard_tier = 2
                if not p.classical_roots:
                    p.classical_roots = {wid: [ph.drift() for ph in plist] for wid, plist in p.canonical_roots.items()}
                self.events.append(f"Epoch {self.epoch}: {p.name or ('#' + str(pid))} rose to Imperial Lingua Franca status.")
            elif p.is_standardized and (self.epoch - p.standard_epoch) >= 40 and p.canonical_roots:
                diffs = [
                    trajectory_distance(np.stack([ph.g for ph in p.lexicon_phonemes[wid]], axis=0), np.stack([ph.g for ph in canon_ph], axis=0))
                    for wid, canon_ph in list(p.canonical_roots.items())[:25]
                    if p.lexicon_phonemes.get(wid)
                ]
                if diffs and float(np.mean(diffs)) > 0.30:
                    if not p.classical_roots:
                        p.classical_roots = {wid: [ph.drift() for ph in plist] for wid, plist in p.canonical_roots.items()}
                    p.canonical_roots = {
                        w.id: [phon.drift() for phon in p.lexicon_phonemes.get(w.id, w.phonemes)]
                        for w in self.lexical_manifold[:min(80, len(self.lexical_manifold))]
                    }
                    p.standard_epoch = self.epoch
                    self.events.append(f"Epoch {self.epoch}: {p.name or ('#' + str(pid))} re-codified its Neo-Vernacular Standard.")

        self.polities = active_polities
        if not self.polities:
            self._seed_unclaimed_clans()
            return

        U_stack = np.stack([self.hegemony[..., p.id] for p in self.polities], axis=-1)
        combat_reach_stack = np.zeros_like(U_stack)
        inv_infra = 1.0 / (1.0 + 0.30 * np.log1p(self.infrastructure))
        base_cost = 1.0 + 1.8 * (1.0 - self.permeability)
        for idx, p in enumerate(self.polities):
            cap_E, cap_S = float(self.attention[p.y, p.x, 6]), float(self.attention[p.y, p.x, 5])
            combat_reach_stack[..., idx] = np.exp(-(p.dist_cells * base_cost * inv_infra) / (6.0 + 6.0 * cap_E + 4.0 * cap_S))

        terrain_defense = 1.0 + 0.8 * self.roughness + 0.4 * self.veg
        prest_arr = np.array([p.prestige for p in self.polities], dtype=np.float32)
        adv = 1.0 / (1.0 + np.exp(-(prest_arr[np.newaxis, :] - prest_arr[:, np.newaxis]) / 1.2))
        np.fill_diagonal(adv, 0.0)

        A = U_stack * combat_reach_stack
        Abar = np.maximum(0.0, A + 0.125 * laplacian_isotropic_2d(A))

        rate = 0.08 / terrain_defense[..., np.newaxis]
        out_loss = rate * U_stack * np.einsum('hwi,ji->hwj', Abar, adv)
        scale = np.minimum(1.0, U_stack / (dt * out_loss + 1e-6))
        loss = scale * out_loss
        gain = rate * Abar * np.einsum('hwj,ji->hwi', U_stack * scale, adv)

        combat_dev = np.sum(loss, axis=-1)
        self.pop = np.maximum(0.0, self.pop - np.clip(0.16 * np.tanh(combat_dev * 0.4) * self.pop, 0.0, self.pop * 0.5) * dt)
        self.infrastructure = np.maximum(0.0, self.infrastructure - np.clip(0.20 * np.tanh(combat_dev * 0.5) * self.infrastructure, 0.0, self.infrastructure) * dt)

        unclaimed_mask = (U_stack.sum(axis=-1) < 1.0) & land_mask
        colonization = np.where(
            unclaimed_mask[..., np.newaxis],
            0.040 * Abar * np.clip(self.pop[..., np.newaxis] / 1.5, 0.1, 1.5) * res_pull[..., np.newaxis],
            0.0,
        )
        diff_flux_stack = diffuse_masked_2d(U_stack, 0.045 * res_pull * (self.permeability + 0.4 * river_conduit), land_mask)

        for idx, p in enumerate(self.polities):
            pid, px, py = p.id, p.x, p.y
            u = U_stack[..., idx]
            tot_u = float(np.sum(u))

            source = np.zeros((self.H, self.W), dtype=np.float32)
            if not p.is_decapitated:
                source[py, px] = max(0.5, p.prestige * float(np.tanh(self.pop[py, px] * 0.40)) * (1.2 + p.centralization * 0.6))

            reach_field = combat_reach_stack[..., idx]
            logistical_cap = max(14.0, p.prestige * 14.0 * float(np.tanh(self.pop[py, px] * 0.40)) * (1.0 + p.infrastructure_bias * 0.5))
            overstretch_decay = 0.035 * (max(0.0, (tot_u - logistical_cap) / (logistical_cap + 0.1)) ** 1.3)

            k_carrying = np.clip(1.5 + 0.5 * p.prestige + 0.8 * self.pop, 1.5, 20.0)
            growth = 0.10 * u * (1.0 - (u / k_carrying))
            decay = (0.008 + 0.015 * (1.0 - reach_field) + overstretch_decay) * u

            new_u = np.where(
                land_mask,
                np.clip(u + (diff_flux_stack[..., idx] + source + growth + gain[..., idx] + colonization[..., idx] - decay - loss[..., idx]) * dt, 0.0, 30.0),
                0.0,
            )
            self.hegemony[..., pid] = np.where(new_u < 0.01, 0.0, new_u)

        if self.epoch % 6 == 0 and self.epoch > 12 and len(self.polities) < self.MAX_POLITIES:
            tot_h = np.sum(self.hegemony, axis=-1)
            dom_owner = np.where(tot_h > 0.25, np.argmax(self.hegemony, axis=-1), -1)

            for idx, p in enumerate(list(self.polities)):
                pid, px, py = p.id, p.x, p.y
                if float(np.sum(self.hegemony[..., pid])) < 5.0 or (self.epoch - p.last_secession_epoch) < 14:
                    continue

                delta_A = np.sum(np.abs(self.attention - self.attention[py, px]), axis=-1)
                reach = combat_reach_stack[..., idx]
                owned = (dom_owner == pid) & (self.pop > 0.35) & (reach < 0.42) & land_mask
                local_autonomy = self.pop * (0.6 + 0.4 * np.log1p(self.infrastructure)) * (1.0 + p.dist_cells / 14.0)
                rebellion_pressure = np.where(owned, (local_autonomy / (reach + 0.20)) * (0.35 + delta_A), 0.0)

                if np.max(rebellion_pressure) > 1.20:
                    sy, sx = np.unravel_index(np.argmax(rebellion_pressure), rebellion_pressure.shape)
                    new_id = self.next_polity_id
                    self.next_polity_id += 1
                    if new_id >= self.hegemony.shape[-1]:
                        self.hegemony = np.pad(self.hegemony, ((0, 0), (0, 0), (0, 8)), mode='constant')

                    infra_val = float(self.infrastructure[sy, sx])
                    sacred_val = float(self.attention[sy, sx, 2] + self.attention[sy, sx, 3]) / 2.0
                    gov_label = "Domain" if infra_val > 3.0 else ("Order" if sacred_val > 1.3 else "Clan")

                    new_polity = Polity(
                        id=new_id, name="", literal_meaning="", x=int(sx), y=int(sy),
                        prestige=float(self.prestige[sy, sx]) * 1.5, color=self.assign_polity_chroma(new_id),
                        age=self.epoch, dist_cells=np.sqrt((self._grid_x - sx) ** 2 + (self._grid_y - sy) ** 2),
                        semantic_vector=self.evaluate_site_semantic_vector(float(sx), float(sy)),
                        government_type=gov_label,
                        centralization=float(np.clip(0.40 + 0.10 * infra_val, 0.2, 0.9)),
                        infrastructure_bias=float(np.clip(infra_val / 8.0, 0.2, 0.9)),
                        sacred_focus=float(np.clip(sacred_val / 2.5, 0.1, 0.9)),
                        tolerance=float(np.clip(1.0 - infra_val / 10.0, 0.2, 0.8)),
                        lexicon_phonemes={wid: [ph.drift() for ph in plist] for wid, plist in p.lexicon_phonemes.items()},
                        classical_roots={wid: [ph.drift() for ph in plist] for wid, plist in (p.classical_roots or p.canonical_roots).items()},
                    )
                    self.polities.append(new_polity)
                    p.last_secession_epoch = self.epoch

                    rebel_basin = np.exp(-((self._grid_x - sx) ** 2 + (self._grid_y - sy) ** 2) / 4.0)
                    self.hegemony[..., new_id] = np.where(owned, 6.0 * rebel_basin, 0.0)
                    self.hegemony[..., pid] = np.where(self.hegemony[..., new_id] > 1.0, self.hegemony[..., pid] * 0.15, self.hegemony[..., pid])

                    border_mask = (rebel_basin > 0.2) & (rebel_basin < 0.8)
                    self.border_barrier = np.maximum(self.border_barrier, 0.95 * border_mask)
                    self.danger_zones = np.maximum(self.danger_zones, 1.0 * border_mask)
                    self.events.append(f"Epoch {self.epoch}: Frontier broke away as {gov_label} #{new_id} at ({sx}, {sy}).")
                    break

        if self.epoch % 4 == 0:
            self._seed_unclaimed_clans()

    def _step_lexical_presence_and_drift(self, dt: float) -> None:
        n_words = len(self.lexical_manifold)
        land_mask = self.elev > 0.0

        rho = np.exp(-0.85 * np.maximum(0.0, self.elev))
        target_b8 = np.clip(0.40 * (1.0 - rho) - 0.50 * self.hum * np.maximum(0.0, 0.30 - self.elev), -0.30, 0.30)
        target_b7 = np.clip(-0.50 * (1.0 - rho) * (1.0 - self.hum), -0.60, 0.0)

        self.phonetic_bias[..., 8] += 0.05 * (target_b8 - self.phonetic_bias[..., 8]) * dt
        self.phonetic_bias[..., 7] += 0.05 * (target_b7 - self.phonetic_bias[..., 7]) * dt

        coh = self.attention[..., 2] - 1.0
        force = self.attention[..., 3] - 1.0
        self.phonetic_drift_field[..., 4] += 0.025 * (np.clip(0.08 * coh - 0.06 * force, -0.18, 0.18) - self.phonetic_drift_field[..., 4]) * dt
        self.phonetic_drift_field[..., 6] += 0.025 * (np.clip(0.06 * coh, -0.12, 0.12) - self.phonetic_drift_field[..., 6]) * dt
        self.phonetic_drift_field[..., 8] += 0.025 * (np.clip(0.10 * force, -0.20, 0.20) - self.phonetic_drift_field[..., 8]) * dt

        id_to_idx = {w.id: idx for idx, w in enumerate(self.lexical_manifold)}
        for p in self.polities:
            if p.is_standardized and p.id < self.hegemony.shape[-1]:
                h_share = self.hegemony[..., p.id]
                decay_dist = 1.0 / (1.0 + p.dist_cells * 0.08)
                standard_pull = 0.03 * p.centralization * h_share * decay_dist * self.permeability
                self.phonetic_drift_field *= np.clip(1.0 - standard_pull * dt, 0.0, 1.0)[..., np.newaxis]

                if self.epoch % 4 == 0 and p.canonical_roots:
                    boost = (0.02 * h_share * p.centralization * self.permeability).astype(np.float32)
                    for root_id in p.canonical_roots.keys():
                        w_idx = id_to_idx.get(root_id)
                        if w_idx is not None:
                            GPUSemanticEngine.add_presence_point_masked(self._wp_buf, w_idx, boost)
                    self._wp_dirty = True

        self.phonetic_drift_field = np.clip(
            self.phonetic_drift_field + diffuse_masked_2d(self.phonetic_drift_field, 0.015 * self.permeability, land_mask) * dt,
            -0.45, 0.45,
        )

        if n_words > 0 and self.epoch % 3 == 0:
            GPUSemanticEngine.step_word_presence(self._wp_buf, n_words, 0.035 * self.permeability, land_mask, self.pop, dt_scale=2.0)
            self._wp_dirty = True

        from lexicon import LexiconGenerator
        from language import LanguageProfile

        total_global_pop = float(np.sum(self.pop[land_mask]))
        global_lexical_target = int(350 + 2500 * math.tanh(total_global_pop * 0.005) + 35 * max(1, len(self.polities)))
        settled_y, settled_x = np.nonzero((self.pop > 0.25) & land_mask)

        if len(settled_y) > 0 and n_words < global_lexical_target and self.epoch % 2 == 0:
            growth_pressure = (global_lexical_target - n_words) / float(global_lexical_target)
            batch_target = max(1, min(14, int(math.ceil(growth_pressure * 16.0))))
            pop_weights = self.pop[settled_y, settled_x].astype(np.float64)
            sample_indices = self.rng.choice(len(settled_y), size=min(batch_target, len(settled_y)), replace=False, p=pop_weights / pop_weights.sum())
            max_fam = max((w.derivation.root_family_id for w in self.lexical_manifold), default=0)

            for s_idx in sample_indices:
                sy, sx = int(settled_y[s_idx]), int(settled_x[s_idx])
                cell_pop = float(self.pop[sy, sx])
                env_site = SimpleNamespace(
                    altitude=float(self.elev[sy, sx]), humidity=float(self.hum[sy, sx]),
                    temperature=float(self.temp[sy, sx]), vegetation=float(self.veg[sy, sx]),
                    fauna=float(self.fauna[sy, sx]), roughness=float(self.roughness[sy, sx]),
                    population=cell_pop,
                )
                prof_site = LanguageProfile(
                    name=f"Site_{sx}_{sy}",
                    head_directionality=float(self.typology[sy, sx, 0]),
                    synthesis_index=float(self.typology[sy, sx, 1]),
                    syllable_complexity=float(self.typology[sy, sx, 2]),
                )

                local_scores = self.get_presence_column(sx, sy)
                raw_act = np.where(local_scores > 0.04)[0]
                if len(raw_act) > 120:
                    raw_act = raw_act[np.argsort(-local_scores[raw_act])[:120]]
                active_local = [self.lexical_manifold[int(idx)] for idx in raw_act]

                h_cell = self.hegemony[sy, sx]
                dom_pol = next((pol for pol in self.polities if pol.id == int(np.argmax(h_cell))), None) if float(h_cell.max()) >= 0.25 else None

                ext_counts, ext_forms = None, None
                local_pool = active_local if len(active_local) >= 6 else self.lexical_manifold[-60:]
                if dom_pol is not None:
                    if dom_pol._cached_inv_counts is None or dom_pol._cached_plain_forms is None:
                        overlaid = self._overlay_view(local_pool, sx, sy)
                        dom_pol._cached_inv_counts = inventory_counts(overlaid)
                        dom_pol._cached_plain_forms = {w.plain_form for w in overlaid}
                    ext_counts = dom_pol._cached_inv_counts
                    ext_forms = dom_pol._cached_plain_forms
                else:
                    local_pool = self._overlay_view(local_pool, sx, sy)

                new_word = None
                coin_roll = self.rng.random()
                max_fam += 1

                if coin_roll < 0.65 and len(active_local) >= 2:
                    content_cands = [w for w in active_local if w.category == LexicalCategory.CONTENT_OPEN]
                    if content_cands:
                        base_candidate = content_cands[int(self.rng.integers(len(content_cands)))]
                        deriv_mode = str(self.rng.choice(["agent", "instrument", "locative", "diminutive", "augmentative"]))
                        new_word = LexiconGenerator.derive_productive_concept(base_candidate, deriv_mode, active_local, profile=prof_site, env=env_site)
                elif coin_roll < 0.85 and len(active_local) >= 1:
                    if self.elev[sy, sx] < 0.08:
                        base_pt = np.array([0.85, 0.45, 0.65, 0.55, 0.75, 0.25, 0.85], dtype=np.float32)
                    elif self.elev[sy, sx] > 0.45:
                        base_pt = np.array([0.92, 0.15, 0.45, 0.82, 0.20, 0.15, 0.78], dtype=np.float32)
                    elif self.discharge[sy, sx] > 1.2:
                        base_pt = np.array([0.88, 0.48, 0.72, 0.52, 0.80, 0.30, 0.78], dtype=np.float32)
                    else:
                        base_pt = np.array([0.65, 0.55, 0.55, 0.50, 0.50, 0.55, 0.45], dtype=np.float32)

                    jittered = np.clip(base_pt + self.rng.normal(0.0, 0.20, size=7), 0.05, 0.95)
                    target_vec = SemanticVector(*jittered)

                    # Semantic saturation: skip if an existing local word already occupies this exact meaning
                    if not any(target_vec.distance_to(e.vector) < 0.22 for e in active_local):
                        new_phonemes = LexiconGenerator.generate_gestural_root(
                            profile=prof_site, env=env_site, existing_words=local_pool, rng=self.rng, existing_counts=ext_counts, existing_forms=ext_forms
                        )
                        if new_phonemes and any(p.is_vowel for p in new_phonemes):
                            new_word = Word(
                                vector=target_vec, phonemes=new_phonemes,
                                derivation=DerivationRecord(root_family_id=max_fam),
                                category=LexicalCategory.CONTENT_OPEN, generation_born=self.epoch,
                            )
                else:
                    new_phonemes = LexiconGenerator.generate_gestural_root(
                        profile=prof_site, env=env_site, existing_words=local_pool, rng=self.rng, existing_counts=ext_counts, existing_forms=ext_forms
                    )
                    if new_phonemes and any(p.is_vowel for p in new_phonemes):
                        coords = [float(np.clip(self.rng.normal(0.5, 0.22), 0.05, 0.95)) for _ in range(7)]
                        new_word = Word(
                            vector=SemanticVector(*coords), phonemes=new_phonemes,
                            derivation=DerivationRecord(root_family_id=max_fam),
                            category=LexicalCategory.CONTENT_OPEN, generation_born=self.epoch,
                        )

                if new_word is None:
                    continue

                cand_tensor = new_word.tensor[:, :9]
                cand_plain = new_word.plain_form
                cand_parents = new_word.derivation.local_parent_ids
                clash_pool = active_local + self.lexical_manifold[-40:]
                if any(
                    (cand_parents and e.derivation.local_parent_ids == cand_parents)
                    or (
                        new_word.vector.distance_to(e.vector) < 0.15
                        and len(e.phonemes) > 0
                        and (e.plain_form == cand_plain or trajectory_distance(cand_tensor, e.tensor[:, :9]) < 0.28)
                    )
                    for e in clash_pool
                ):
                    continue

                self.lexical_manifold.append(new_word)
                new_idx = len(self.lexical_manifold) - 1
                if dom_pol is not None and dom_pol._cached_plain_forms is not None:
                    dom_pol._cached_plain_forms.add(new_word.plain_form)

                self._wp_buf = GPUSemanticEngine.ensure_presence_capacity(self._wp_buf, new_idx + 1)
                d2 = (self._grid_x - sx) ** 2 + (self._grid_y - sy) ** 2
                footprint = (float(np.clip(2.5 * math.log1p(cell_pop * 1.5), 1.2, 4.0)) * np.exp(-d2 / 16.0)).astype(np.float32)
                GPUSemanticEngine.set_presence_slice(self._wp_buf, new_idx, footprint)
                self._wp_dirty = True

    def _overlay_view(self, words: Sequence[Word], x: int, y: int) -> list:
        h = self.hegemony[y, x]
        if float(h.max()) < 0.25:
            return list(words)
        poly = next((p for p in self.polities if p.id == int(np.argmax(h))), None)
        if poly is None or not poly.lexicon_phonemes:
            return list(words)
        out = []
        for w in words:
            ph = poly.lexicon_phonemes.get(w.id)
            if ph is None:
                out.append(w)
            else:
                _, plain = render_word_trajectory(np.stack([p.vector for p in ph], axis=0))
                out.append(SimpleNamespace(id=w.id, plain_form=plain, usage_frequency=w.usage_frequency, phonemes=ph))
        return out

    def _step_phonology(self) -> None:
        if not self.polities or not self.lexical_manifold:
            return

        top_pol = max(self.polities, key=lambda pol: pol.prestige)
        n_pols = min(len(self.polities), max(1, (len(self.polities) + 3) // 4))
        for _ in range(n_pols):
            p = self.polities[self._phono_cursor % len(self.polities)]
            self._phono_cursor += 1

            pres = self.get_presence_column(p.x, p.y)
            valid = np.flatnonzero(pres > 0.02)
            if len(valid) < 12:
                continue
            if len(valid) > self.phonology_cohort:
                ranked = valid[np.argsort(-pres[valid])]
                core = ranked[:160]
                tail = self.rng.choice(ranked[160:], size=self.phonology_cohort - 160, replace=False)
                top = np.concatenate([core, tail])
            else:
                top = valid

            env = SimpleNamespace(
                altitude=float(self.elev[p.y, p.x]), humidity=float(self.hum[p.y, p.x]),
                temperature=float(self.temp[p.y, p.x]), vegetation=float(self.veg[p.y, p.x]),
                fauna=float(self.fauna[p.y, p.x]), roughness=float(self.roughness[p.y, p.x]),
                population=float(self.pop[p.y, p.x]),
            )
            prof = SimpleNamespace(
                synthesis_index=float(self.typology[p.y, p.x, 1]),
                syllable_complexity=float(self.typology[p.y, p.x, 2]),
            )

            bases = [self.lexical_manifold[int(i)] for i in top]
            proxies = [
                Word(
                    word_id=b.id, vector=b.vector,
                    phonemes=list(p.lexicon_phonemes.get(b.id, b.phonemes)),
                    derivation=b.derivation, category=b.category,
                    usage_frequency=float(0.5 + 3.0 * pres[int(i)]),
                    generation_born=b.generation_born,
                )
                for b, i in zip(bases, top)
            ]
            last_res = {"accepted": False, "kind": "none", "symbol": ""}
            for _ in range(self.phonology_steps):
                last_res = sound_change_step(proxies, self.rng, T=0.05, env=env, profile=prof)

            for b, px in zip(bases, proxies):
                p.lexicon_phonemes[b.id] = list(px.phonemes)
                if last_res.get("accepted"):
                    is_owner = (b.id >= 10000 and (b.id // 10000 - 1) == p.id) or (b.id < 10000 and p is top_pol)
                    if is_owner:
                        b.record_evolution(self.epoch, px.phonemes, event=f"{last_res['kind'].capitalize()} ({last_res['symbol']})")
            p._cached_inv_counts = None
            p._cached_plain_forms = None

    def step(self, dt: float = 1.0) -> None:
        self.epoch += 1
        self._probe_cache.clear()
        self._probe_words_cache.clear()
        self._update_climate(self.epoch)

        tc = self._ensure_terrain_cache()
        land_mask = self.elev > 0.0
        mobility = tc["mobility"]

        is_shallow = (self.elev <= 0.0) & (self.elev > -0.16)
        eff_mobility = mobility + np.where(is_shallow, 0.02 * np.tanh(self.infrastructure * 0.15), 0.0) + 0.025 * tc["river_conduit"]

        agglomeration_potential = np.tanh(self.prestige * 0.25) * 1.8 + np.log1p(self.infrastructure * 0.30) * 0.8 + tc["river_junction"]
        urban_flux = conservative_flux_divergence_2d(self.pop, agglomeration_potential, eff_mobility * 0.50, mask=land_mask)

        C_matter, A_bio, S_coord = self.attention[..., 0], self.attention[..., 1], self.attention[..., 5]
        yield_agri = (2.2 * self.veg + 3.5 * np.tanh(self.discharge)) * (0.6 + 0.4 * C_matter / 1.5) * (0.6 + 0.4 * S_coord / 1.5) * np.maximum(0.2, self.temp)
        yield_pastoral = (1.8 * self.fauna + 1.0 * self.veg) * (0.6 + 0.4 * A_bio / 1.5)
        exposure_deficit = np.maximum(0.0, 0.20 - self.temp) * 3.5 * np.maximum(0.0, 1.0 - C_matter / 1.2)

        caloric_yield = np.maximum(0.10, yield_agri + yield_pastoral - exposure_deficit) * (1.0 + 0.40 * np.log1p(self.infrastructure))
        caloric_demand = self.pop * 1.05
        net_calories = caloric_yield - caloric_demand

        agri_unlock = 1.0 / (1.0 + np.exp(-4.5 * (self.infrastructure - 0.78)))
        forage_cap = 0.50 + np.maximum(0.0, self.capacity - 0.50) * agri_unlock
        K_eff = np.maximum(0.05, np.minimum(self.capacity, forage_cap))

        deficit = np.maximum(0.0, -net_calories) / np.maximum(0.1, caloric_demand)
        starvation = 0.15 * self.pop * (deficit ** 1.3)
        pop_growth = np.where(net_calories >= 0, 0.08 * self.pop * (1.0 - self.pop / K_eff), 0.0)

        tot_heg = np.sum(self.hegemony, axis=-1)
        threat_gy, threat_gx = np.gradient(tot_heg)
        pasture_gy, pasture_gx = np.gradient(self.veg * 0.6 + self.fauna * 0.8 + np.tanh(self.discharge) * 0.6)

        is_threatened = (np.sqrt(threat_gx ** 2 + threat_gy ** 2) > 0.3) & (self.pop < 1.2)
        self.migration_vx = np.where(is_threatened, -threat_gx * 0.55, pasture_gx * 0.35)
        self.migration_vy = np.where(is_threatened, -threat_gy * 0.55, pasture_gy * 0.35)

        refugee_flux = conservative_advection_divergence_2d(
            self.pop, self.migration_vx, self.migration_vy, eff_mobility * np.clip(self.attention[..., 6] / 1.5, 0.5, 2.0), mask=land_mask
        )
        diff_flux = diffuse_masked_2d(self.pop, eff_mobility, land_mask)
        self.pop = np.where(land_mask, np.clip(self.pop + (diff_flux + pop_growth - starvation + urban_flux + refugee_flux) * dt, 0.0, 25.0), 0.0)

        labor_surplus = np.clip(net_calories / np.maximum(0.1, caloric_demand), 0.0, 2.0)
        infra_growth = 0.045 * self.pop * (0.2 + 0.8 * labor_surplus) * (C_matter / 1.2) * (S_coord / 1.2)
        infra_decay = 0.007 * self.infrastructure + 0.0010 * (self.infrastructure ** 2)
        self.infrastructure = np.where(land_mask, np.clip(self.infrastructure + (infra_growth - infra_decay) * dt, 0.0, 50.0), 0.0)
        self.prestige = np.where(land_mask, self.pop * (1.0 + 0.40 * np.log1p(self.infrastructure)), 0.0)

        self._step_reaction_diffusion_hegemony(dt)
        self._step_lexical_presence_and_drift(dt)
        if self.phonology_every > 0 and self.epoch % self.phonology_every == 0:
            self._step_phonology()
        if self.epoch % 6 == 0:
            self._discover_and_update_landmarks()

        real_coast = tc["coast_edge"]
        target_A = np.zeros_like(self.attention)
        target_A[..., 0] = 0.20 + 0.90 * np.maximum(0.0, self.elev) + 0.50 * (1.0 - self.veg)
        target_A[..., 1] = 0.20 + 1.20 * self.fauna * 0.6 + 0.60 * self.veg
        target_A[..., 2] = 0.25 + 0.85 * (1.0 - np.abs(self.hum - 0.55)) * (1.0 - np.abs(self.temp - 0.55))
        target_A[..., 3] = 0.20 + 0.80 * np.maximum(0.0, self.elev) + 0.80 * np.tanh(self.infrastructure * 0.20)
        target_A[..., 4] = 0.20 + 0.70 * np.tanh(self.pop * 0.35) + 0.50 * np.tanh(self.infrastructure * 0.25) + 0.40 * np.tanh(self.discharge * 0.40) + 0.40 * real_coast
        target_A[..., 5] = 0.15 + 1.20 * np.tanh(self.pop * 0.35) + 0.40 * np.tanh(self.infrastructure * 0.20)
        target_A[..., 6] = 0.20 + 1.10 * real_coast + 0.60 * np.tanh(self.infrastructure * 0.25)

        target_A = 7.0 * (target_A / np.maximum(0.1, np.sum(target_A, axis=-1, keepdims=True)))
        d_lap_A = diffuse_masked_2d(self.attention, self.permeability * (0.010 + 0.025 * np.tanh(self.prestige * 0.20)), land_mask)
        self.attention = np.clip(self.attention + (0.06 * (target_A - self.attention) + d_lap_A) * dt, 0.02, 7.0)
        self.attention = 7.0 * (self.attention / np.maximum(0.1, np.sum(self.attention, axis=-1, keepdims=True)))

        target_sigma = 0.15 + 0.70 / (1.0 + np.exp(-3.5 * (np.maximum(0.0, self.elev) * 1.5 - self.hum * 1.2)))
        target_s = 0.15 + 0.70 / (1.0 + np.exp(-2.5 * (np.tanh(self.pop * 0.5) - np.maximum(0.0, self.elev) * 0.4)))
        target_h = 0.50 + 0.25 * np.tanh((self.attention[..., 6] - self.attention[..., 5]) * 0.4)

        tot_heg = np.sum(self.hegemony, axis=-1, keepdims=True)
        advect_h = np.zeros((self.H, self.W), dtype=np.float32)
        advect_s = np.zeros((self.H, self.W), dtype=np.float32)
        for p in self.polities:
            if p.id >= self.hegemony.shape[-1]:
                continue
            share = self.hegemony[..., p.id] / np.maximum(0.05, tot_heg[..., 0])
            cap_t = self.typology[p.y, p.x]
            advect_h += 0.006 * share * (cap_t[0] - self.typology[..., 0])
            advect_s += 0.006 * share * (cap_t[1] - self.typology[..., 1])

        jespersen = -0.020 * (self.typology[..., 1] ** 2) - 0.025 * np.tanh(self.pop * 0.35)
        urban_simp = -0.030 * np.tanh(self.pop * 0.40)

        target_T = np.stack([target_h + advect_h, target_s + advect_s + jespersen, target_sigma + urban_simp], axis=-1)
        self.typology = np.clip(self.typology + (0.04 * (target_T - self.typology) + diffuse_masked_2d(self.typology, 0.02 * self.permeability, land_mask)) * dt, 0.05, 0.95)
        self.phonetic_bias = np.clip(self.phonetic_bias + diffuse_masked_2d(self.phonetic_bias, 0.015 * self.permeability, land_mask) * dt, -0.45, 0.45)

        self.border_barrier *= 0.97
        self.danger_zones *= 0.90

        grad_hx, grad_hy = np.gradient(self.typology[..., 0])
        grad_sx, grad_sy = np.gradient(self.typology[..., 1])
        ling_grad = np.sqrt(grad_hx ** 2 + grad_hy ** 2 + grad_sx ** 2 + grad_sy ** 2)
        self.permeability = np.clip((tc["slope_exp"] / (1.0 + np.exp(16.0 * (ling_grad - 0.22)))) * (1.0 - self.border_barrier), 0.01, 1.0)

    def get_cell_state(self, x: int, y: int) -> dict:
        r, c = int(np.clip(y, 0, self.H - 1)), int(np.clip(x, 0, self.W - 1))
        h_vals = self.hegemony[r, c]
        tot_h = float(np.sum(h_vals))
        max_h = float(np.max(h_vals))

        dom_id = int(np.argmax(h_vals)) if max_h >= 0.25 else -1
        dom_polity = next((p for p in self.polities if p.id == dom_id), None)
        is_danger = float(self.danger_zones[r, c]) > 0.35

        infra, cell_p = float(self.infrastructure[r, c]), float(self.pop[r, c])
        if self.elev[r, c] <= 0.0:
            settlement_tier = "Open Ocean"
        elif infra >= 10.0 and cell_p >= 3.5:
            settlement_tier = "Regional Metropolis"
        elif infra >= 4.0 and cell_p >= 1.8:
            settlement_tier = "Walled Town / Port"
        elif infra >= 1.5 and cell_p >= 0.7:
            settlement_tier = "Agrarian Village"
        elif cell_p >= 0.15:
            settlement_tier = "Nomadic Camp / Station"
        elif infra >= 1.5:
            settlement_tier = "Abandoned Ruins"
        else:
            settlement_tier = "Wilderness"

        c_matter, a_bio, s_coord = self.attention[r, c, 0], self.attention[r, c, 1], self.attention[r, c, 5]
        yield_agri = (2.2 * self.veg[r, c] + 3.5 * math.tanh(self.discharge[r, c])) * (0.6 + 0.4 * c_matter / 1.5) * (0.6 + 0.4 * s_coord / 1.5) * max(0.2, self.temp[r, c])
        yield_pastoral = (1.8 * self.fauna[r, c] + 1.0 * self.veg[r, c]) * (0.6 + 0.4 * a_bio / 1.5)
        exp_def = max(0.0, 0.20 - self.temp[r, c]) * 3.5 * max(0.0, 1.0 - c_matter / 1.2)
        c_yield = max(0.10, yield_agri + yield_pastoral - exp_def) * (1.0 + 0.40 * math.log1p(infra))
        net_cal = c_yield - cell_p * 1.05

        mvx, mvy = float(self.migration_vx[r, c]), float(self.migration_vy[r, c])
        spd = math.hypot(mvx, mvy)
        if cell_p < 0.05 or self.elev[r, c] <= 0.0:
            migration_str = "None (Uninhabited)"
        elif spd < 0.06:
            migration_str = "Stationary Settled Population"
        else:
            angle = math.atan2(-mvy, mvx) * 180.0 / math.pi
            if -22.5 <= angle < 22.5: heading = "E"
            elif 22.5 <= angle < 67.5: heading = "NE"
            elif 67.5 <= angle < 112.5: heading = "N"
            elif 112.5 <= angle < 157.5: heading = "NW"
            elif angle >= 157.5 or angle < -157.5: heading = "W"
            elif -157.5 <= angle < -112.5: heading = "SW"
            elif -112.5 <= angle < -67.5: heading = "S"
            else: heading = "SE"
            motive = "Fleeing Conflict" if is_danger else "Pastoral Grazing"
            migration_str = f"Migrating {heading} ({round(spd * 4.2, 1)} km/yr) — {motive}"

        return {
            "x": c, "y": r,
            "elevation": round(float(self.elev[r, c]), 3),
            "temperature": round(float(self.temp[r, c]), 3),
            "humidity": round(float(self.hum[r, c]), 3),
            "vegetation": round(float(self.veg[r, c]), 3),
            "fauna": round(float(self.fauna[r, c]), 3),
            "soil_health": round(float(self.soil_health[r, c]), 2),
            "population": round(cell_p, 3),
            "carrying_capacity": round(float(self.capacity[r, c]), 3),
            "infrastructure": round(infra, 2),
            "settlement_tier": settlement_tier,
            "migration_activity": migration_str,
            "net_calories": round(float(net_cal), 2),
            "prestige": round(float(self.prestige[r, c]), 2),
            "attention": [round(float(v), 3) for v in self.attention[r, c]],
            "head_directionality": round(float(self.typology[r, c, 0]), 3),
            "synthesis_index": round(float(self.typology[r, c, 1]), 3),
            "syllable_complexity": round(float(self.typology[r, c, 2]), 3),
            "phonetic_bias": [round(float(v), 3) for v in self.phonetic_bias[r, c]],
            "phonetic_drift": [round(float(v), 3) for v in self.phonetic_drift_field[r, c]],
            "permeability": round(float(self.permeability[r, c]), 3),
            "is_danger_zone": is_danger,
            "polity": {
                "dominant_id": dom_id,
                "name": dom_polity.name if dom_polity else "Unclaimed Territory",
                "literal_meaning": dom_polity.literal_meaning if dom_polity else "Wilderness",
                "government_type": dom_polity.government_type if dom_polity else "None",
                "is_standardized": dom_polity.is_standardized if dom_polity else False,
                "standard_epoch": dom_polity.standard_epoch if dom_polity else 0,
                "control_percentage": round(float(h_vals[dom_id] / max(0.01, tot_h)) * 100.0, 1) if dom_id >= 0 else 0.0,
                "is_capital": any(p.x == c and p.y == r for p in self.polities),
                "color": dom_polity.color if dom_polity else [80, 80, 80],
            },
        }