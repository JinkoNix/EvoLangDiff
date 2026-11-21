"""
phonetics.py — Pure Articulatory Energetics, Perceptual Metric, and Passive IPA Readout.
Evaluates (L, 10) gestural tensors strictly on:
  1. Biomechanical production effort (muscular work, airflow/hypoxia, transition inertia)
  2. Acoustic transmission clarity (via AcousticEngine)
IPA attractors are used strictly as a passive transcription lens.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from acoustics import AcousticEngine

# Neutral resting vocal tract posture & physical stiffness/inertia tensors
REST_POSTURE = np.array([0.90, 0.00, 0.90, 0.25, 0.70, 0.45, 0.00, 0.42, 0.50], dtype=np.float32)
STIFFNESS = np.array([0.12, 0.03, 0.12, 0.00, 0.26, 0.48, 0.10, 0.65, 2.20], dtype=np.float32)
INERTIA = np.array([0.65, 0.15, 0.65, 0.40, 0.65, 1.10, 0.45, 0.70, 1.20], dtype=np.float32)

_CHAO_BARS = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
_VOWEL_SET = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o', 'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ'
}
_VOICELESS_STOPS = {'p', 't̪', 't', 'ʈ', 'c', 'k', 'q', 'ʔ'}


# -----------------------------------------------------------------------------
# 1. ENERGETICS & PERCEPTUAL METRICS
# -----------------------------------------------------------------------------

def oral_aperture(g: np.ndarray) -> np.ndarray:
    phar = np.clip(1.0 - np.maximum(0.0, g[..., 5] - 0.85) / 0.15, 0.0, 1.0)
    return np.minimum(np.minimum(g[..., 0], g[..., 2]), np.minimum(g[..., 4], phar))


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    """Passive classification for IPA/syllable transcription only."""
    g = np.asarray(g, dtype=np.float32)
    open_tract = (g[..., 0] > 0.25) & (g[..., 2] > 0.55) & (g[..., 4] > 0.20) & (g[..., 5] < 0.91)
    g8 = g[..., 8] if g.shape[-1] > 8 else 0.50
    modal = (np.abs(g[..., 7] - 0.40) < 0.18) & (np.abs(g8 - 0.50) < 0.22)
    res = open_tract & modal
    return bool(res) if np.ndim(res) == 0 else res


def sensory_distance(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Perceptual distance combining closure-weighted tract space and acoustic formants."""
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    mid = 0.50 * (a1 + a2)

    w = np.ones_like(mid)
    w[..., 0] = 1.0 + 12.0 * np.exp(-(mid[..., 0] / 0.14) ** 2)
    w[..., 2] = 1.0 + 14.0 * np.exp(-(mid[..., 2] / 0.14) ** 2)
    w[..., 4] = 1.5 + 12.0 * np.exp(-(mid[..., 4] / 0.14) ** 2)
    w[..., 6] = 1.0 + 8.0 * np.exp(-(mid[..., 6] / 0.12) ** 2)
    w[..., 7] = 1.5 + 10.0 * np.exp(-((mid[..., 7] - 0.50) / 0.15) ** 2)

    d_tract_sq = np.sum(w * ((a1 - a2) ** 2), axis=-1)

    # Normalized (F1, F2) formant distance for open voiced gestures
    f1_1 = (a1[..., 4] ** 1.35) * (0.75 + 0.25 * a1[..., 0])
    f1_2 = (a2[..., 4] ** 1.35) * (0.75 + 0.25 * a2[..., 0])
    f2_1 = 0.50 + (1.0 - 0.50 * f1_1) * ((1.0 - a1[..., 5]) * (1.0 - 0.60 * a1[..., 1]) - 0.50)
    f2_2 = 0.50 + (1.0 - 0.50 * f1_2) * ((1.0 - a2[..., 5]) * (1.0 - 0.60 * a2[..., 1]) - 0.50)

    v1 = np.clip(oral_aperture(a1) * 4.0, 0.0, 1.0) * np.exp(-((a1[..., 7] - 0.40) / 0.20) ** 2)
    v2 = np.clip(oral_aperture(a2) * 4.0, 0.0, 1.0) * np.exp(-((a2[..., 7] - 0.40) / 0.20) ** 2)
    d_formant_sq = 2.2 * v1 * v2 * (((f1_1 - f1_2) ** 2) + ((f2_1 - f2_2) ** 2))

    return np.sqrt(np.maximum(0.0, d_tract_sq + d_formant_sq))


def trajectory_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    if a1.ndim == 1:
        a1 = a1[None, :]
    if a2.ndim == 1:
        a2 = a2[None, :]
    L1, L2 = len(a1), len(a2)
    if L1 == 0 or L2 == 0:
        return 1.0
    if L1 == L2:
        return float(np.mean(sensory_distance(a1, a2)))
    max_L = max(L1, L2)
    idx1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
    idx2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
    return float(np.mean(sensory_distance(a1[idx1], a2[idx2])))


def segment_effort(G: np.ndarray, taus: np.ndarray | float = 1.0, env=None) -> np.ndarray:
    """Biomechanical effort (>= 0): posture, extreme-place strain, double-closure clicks, and airflow."""
    G = np.asarray(G, dtype=np.float32)[..., :9]
    alt = float(getattr(env, "altitude", 0.0)) if env else 0.0
    veg = float(getattr(env, "vegetation", 0.30)) if env else 0.30
    rough = float(getattr(env, "roughness", 0.10)) if env else 0.10
    rho = math.exp(-0.95 * max(0.0, alt))

    apical_strain = 1.65 * np.maximum(0.0, 1.0 - G[..., 2]) * ((G[..., 3] - 0.25) ** 2)
    phar_strain = 0.85 * np.maximum(0.0, (G[..., 5] - 0.85) / 0.15)
    palatal_strain = 3.20 * np.maximum(0.0, 0.28 - G[..., 4]) * np.maximum(0.0, 0.28 - G[..., 5])
    nonback_round = 0.38 * G[..., 1] * np.maximum(0.0, 0.72 - G[..., 5])
    labiodental_strain = 0.30 * np.exp(-(((G[..., 0] - 0.07) / 0.035) ** 2)) * (1.0 - G[..., 1])

    a_oral = oral_aperture(G)
    affricate_prec = 0.40 * np.exp(-(((a_oral - 0.04) / 0.018) ** 2)) * (1.0 - G[..., 6])

    ant_seal = np.maximum(
        np.exp(-np.minimum(G[..., 0], G[..., 2]) / 0.06),
        np.exp(-(((G[..., 2] - 0.28) / 0.05) ** 2)),
    )
    double_closure = 0.95 * np.exp(-G[..., 4] / 0.06) * ant_seal
    click_masking = double_closure * (0.40 + 1.50 * veg + 1.50 * rough)

    e_muscle = (
        np.sum(STIFFNESS * ((G - REST_POSTURE) ** 2), axis=-1)
        + apical_strain + phar_strain + palatal_strain + nonback_round + labiodental_strain + affricate_prec + double_closure + click_masking
    )

    g6, g7, g8 = G[..., 6], G[..., 7], G[..., 8]
    is_glottalic = np.exp(-((g7 - 0.02) / 0.10) ** 2) * (1.0 - np.exp(-((g8 - 0.50) / 0.20) ** 2))
    turbulence = np.exp(-(((a_oral - 0.06) / 0.045) ** 2))
    pulmonic_flow = (0.06 + 0.18 * turbulence + 0.05 * g6) * (0.25 + 1.15 * np.maximum(0.0, g7 - 0.25))
    e_air = (1.0 - 0.85 * is_glottalic) * (pulmonic_flow / (rho ** 2.0))

    voicing_drag = 0.24 * np.exp(-((g7 - 0.40) / 0.14) ** 2) * np.exp(-15.0 * a_oral) * ((1.0 - g6) ** 2)
    return (0.06 + e_muscle + e_air + voicing_drag) * taus


def transition_cost(p1, p2, tau1: float = 1.0, tau2: float = 1.0, **kwargs) -> float:
    """Kinetic inertia + same-class OCP friction between consecutive gestures."""
    g1 = p1.g if hasattr(p1, "g") else np.asarray(p1, dtype=np.float32)[:9]
    g2 = p2.g if hasattr(p2, "g") else np.asarray(p2, dtype=np.float32)[:9]
    t1 = p1.duration if hasattr(p1, "duration") else tau1
    t2 = p2.duration if hasattr(p2, "duration") else tau2
    min_tau = max(0.10, min(float(t1), float(t2)))

    v1 = 1.0 if bool(is_vowel_gesture(g1)) else 0.0
    v2 = 1.0 if bool(is_vowel_gesture(g2)) else 0.0
    same_class = v1 * v2 + (1.0 - v1) * (1.0 - v2)

    kin = float(np.sum(INERTIA * ((g1 - g2) ** 2)) / (min_tau ** 0.85))
    ocp = 1.60 * math.exp(-3.5 * float(sensory_distance(g1, g2))) * same_class
    return round(kin + ocp, 3)


def ambient_noise(env=None) -> float:
    if env is None:
        return 0.10
    veg = float(getattr(env, "vegetation", 0.30))
    rough = float(getattr(env, "roughness", 0.10))
    return float(np.clip(0.10 + 0.25 * veg + 0.25 * rough, 0.05, 0.65))


def word_energy(word_or_tensor, profile=None, env=None, base_f0: float = 120.0, **kwargs) -> float:
    """
    Physical + communicative energy (>= 0 J):
      Production effort (posture, airflow, transitions, length)
      + Hyper-articulation deficit against ambient noise (via AcousticEngine)
    """
    if isinstance(word_or_tensor, np.ndarray):
        tensor = np.asarray(word_or_tensor, dtype=np.float32)
        usage_freq = 1.0
    elif hasattr(word_or_tensor, "tensor"):
        tensor = word_or_tensor.tensor
        usage_freq = float(getattr(word_or_tensor, "usage_frequency", 1.0))
    else:
        ph = list(word_or_tensor)
        if not ph:
            return 0.0
        tensor = np.stack([p.vector for p in ph], axis=0)
        usage_freq = 1.0

    n = len(tensor)
    if n == 0:
        return 0.0

    G = tensor[:, :9]
    taus = tensor[:, 9] if tensor.shape[1] > 9 else np.ones(n, dtype=np.float32)

    e_seg = float(np.sum(segment_effort(G, taus=taus, env=env)))
    e_trans = 0.0
    if n > 1:
        min_t = np.maximum(0.10, np.minimum(taus[:-1], taus[1:]))
        e_trans = 0.25 * float(np.sum(np.sum(INERTIA * ((G[:-1] - G[1:]) ** 2), axis=-1) / (min_t ** 0.85)))

    s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
    e_length = 0.10 * (n ** 1.45) * (1.0 - 0.45 * s_idx) * (0.60 + 0.40 * math.log1p(usage_freq))

    af = AcousticEngine.get_auditory_features(list(tensor), base_f0=base_f0)
    amb = ambient_noise(env)
    clarity_target = 6.5 + 3.0 * amb
    clarity_deficit = (0.45 + 0.35 * amb) * max(0.0, clarity_target - af.clarity_score)

    return round(float(e_seg + e_trans + e_length + clarity_deficit), 3)


# -----------------------------------------------------------------------------
# 2. PASSIVE IPA OBSERVER CODEBOOK & TONE TRANSCRIPTION
# -----------------------------------------------------------------------------

_CANONICAL_ATTRACTORS: list[tuple[str, list[float]]] = [
    ("i",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.08, 0.0, 0.40, 0.50]),
    ("y",  [0.35, 1.0, 1.00, 0.00, 0.25, 0.08, 0.0, 0.40, 0.50]),
    ("ɨ",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.50, 0.0, 0.40, 0.50]),
    ("ʉ",  [0.35, 1.0, 1.00, 0.00, 0.25, 0.50, 0.0, 0.40, 0.50]),
    ("ɯ",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.85, 0.0, 0.40, 0.50]),
    ("u",  [0.35, 1.0, 1.00, 0.00, 0.25, 0.85, 0.0, 0.40, 0.50]),
    ("ɪ",  [1.00, 0.0, 1.00, 0.00, 0.38, 0.15, 0.0, 0.40, 0.50]),
    ("ʏ",  [0.40, 0.8, 1.00, 0.00, 0.38, 0.15, 0.0, 0.40, 0.50]),
    ("ʊ",  [0.40, 0.8, 1.00, 0.00, 0.38, 0.75, 0.0, 0.40, 0.50]),
    ("e",  [1.00, 0.0, 1.00, 0.00, 0.50, 0.10, 0.0, 0.40, 0.50]),
    ("ø",  [0.40, 0.8, 1.00, 0.00, 0.50, 0.10, 0.0, 0.40, 0.50]),
    ("ɘ",  [1.00, 0.0, 1.00, 0.00, 0.50, 0.50, 0.0, 0.40, 0.50]),
    ("ɵ",  [0.40, 0.8, 1.00, 0.00, 0.50, 0.50, 0.0, 0.40, 0.50]),
    ("ɤ",  [1.00, 0.0, 1.00, 0.00, 0.50, 0.85, 0.0, 0.40, 0.50]),
    ("o",  [0.35, 0.9, 1.00, 0.00, 0.50, 0.85, 0.0, 0.40, 0.50]),
    ("ə",  [1.00, 0.0, 1.00, 0.00, 0.60, 0.50, 0.0, 0.40, 0.50]),
    ("ɛ",  [1.00, 0.0, 1.00, 0.00, 0.72, 0.15, 0.0, 0.40, 0.50]),
    ("œ",  [0.45, 0.7, 1.00, 0.00, 0.72, 0.15, 0.0, 0.40, 0.50]),
    ("ɜ",  [1.00, 0.0, 1.00, 0.00, 0.72, 0.50, 0.0, 0.40, 0.50]),
    ("ɞ",  [0.45, 0.7, 1.00, 0.00, 0.72, 0.50, 0.0, 0.40, 0.50]),
    ("ʌ",  [1.00, 0.0, 1.00, 0.00, 0.72, 0.85, 0.0, 0.40, 0.50]),
    ("ɔ",  [0.40, 0.85, 1.00, 0.00, 0.72, 0.85, 0.0, 0.40, 0.50]),
    ("æ",  [1.00, 0.0, 1.00, 0.00, 0.85, 0.15, 0.0, 0.40, 0.50]),
    ("ɐ",  [1.00, 0.0, 1.00, 0.00, 0.85, 0.50, 0.0, 0.40, 0.50]),
    ("a",  [1.00, 0.0, 1.00, 0.00, 0.95, 0.35, 0.0, 0.40, 0.50]),
    ("ɶ",  [0.45, 0.7, 1.00, 0.00, 0.95, 0.15, 0.0, 0.40, 0.50]),
    ("ɑ",  [1.00, 0.0, 1.00, 0.00, 0.95, 0.85, 0.0, 0.40, 0.50]),
    ("ɒ",  [0.45, 0.7, 1.00, 0.00, 0.95, 0.85, 0.0, 0.40, 0.50]),

    ("p",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("b",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("t̪",  [1.00, 0.0, 0.00, 0.05, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("d̪",  [1.00, 0.0, 0.00, 0.05, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("t",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("d",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("ʈ",  [1.00, 0.0, 0.00, 0.90, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("ɖ",  [1.00, 0.0, 0.00, 0.90, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("c",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.0, 0.60, 0.50]),
    ("ɟ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.0, 0.40, 0.50]),
    ("k",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.60, 0.50]),
    ("ɡ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.40, 0.50]),
    ("q",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.88, 0.0, 0.60, 0.50]),
    ("ɢ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.88, 0.0, 0.40, 0.50]),
    ("ʔ",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.02, 0.50]),

    ("t͡s", [1.00, 0.0, 0.04, 0.25, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("d͡z", [1.00, 0.0, 0.04, 0.25, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("t͡ʃ", [1.00, 0.3, 0.04, 0.50, 0.60, 0.20, 0.0, 0.60, 0.50]),
    ("d͡ʒ", [1.00, 0.3, 0.04, 0.50, 0.60, 0.20, 0.0, 0.40, 0.50]),
    ("ʈ͡ʂ", [1.00, 0.0, 0.04, 0.90, 0.60, 0.40, 0.0, 0.60, 0.50]),
    ("k͡x", [1.00, 0.0, 1.00, 0.00, 0.05, 0.75, 0.0, 0.60, 0.50]),

    ("ɸ",  [0.05, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.65, 0.50]),
    ("β",  [0.05, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("f",  [0.08, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.65, 0.50]),
    ("v",  [0.08, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("θ",  [1.00, 0.0, 0.08, 0.05, 0.80, 0.50, 0.0, 0.65, 0.50]),
    ("ð",  [1.00, 0.0, 0.08, 0.05, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("s",  [1.00, 0.0, 0.06, 0.25, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("z",  [1.00, 0.0, 0.06, 0.25, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("ʃ",  [1.00, 0.4, 0.06, 0.50, 0.60, 0.20, 0.0, 0.60, 0.50]),
    ("ʒ",  [1.00, 0.4, 0.06, 0.50, 0.60, 0.20, 0.0, 0.40, 0.50]),
    ("ʂ",  [1.00, 0.0, 0.06, 0.90, 0.60, 0.40, 0.0, 0.65, 0.50]),
    ("ʐ",  [1.00, 0.0, 0.06, 0.90, 0.60, 0.40, 0.0, 0.40, 0.50]),
    ("ç",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.08, 0.0, 0.65, 0.50]),
    ("ʝ",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.08, 0.0, 0.40, 0.50]),
    ("x",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.75, 0.0, 0.65, 0.50]),
    ("ɣ",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.75, 0.0, 0.40, 0.50]),
    ("χ",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.88, 0.0, 0.65, 0.50]),
    ("ʁ",  [1.00, 0.0, 1.00, 0.00, 0.09, 0.88, 0.0, 0.40, 0.50]),
    ("ħ",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.98, 0.0, 0.85, 0.50]),
    ("ʕ",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.98, 0.0, 0.45, 0.50]),
    ("h",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.90, 0.50]),
    ("ɦ",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.55, 0.50]),
    ("ɬ",  [1.00, 0.0, 0.12, 0.25, 0.70, 0.30, 0.0, 0.65, 0.50]),
    ("ɮ",  [1.00, 0.0, 0.12, 0.25, 0.70, 0.30, 0.0, 0.40, 0.50]),

    ("m",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.60, 0.40, 0.50]),
    ("m̥",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.60, 0.60, 0.50]),
    ("ɱ",  [0.08, 0.0, 1.00, 0.00, 0.80, 0.50, 0.60, 0.40, 0.50]),
    ("n",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.60, 0.40, 0.50]),
    ("n̥",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.60, 0.60, 0.50]),
    ("ɳ",  [1.00, 0.0, 0.00, 0.90, 0.80, 0.50, 0.60, 0.40, 0.50]),
    ("ɲ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.60, 0.40, 0.50]),
    ("ŋ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.60, 0.40, 0.50]),
    ("ŋ̊",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.60, 0.60, 0.50]),
    ("ɴ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.88, 0.60, 0.40, 0.50]),

    ("l",  [1.00, 0.0, 0.28, 0.25, 0.60, 0.30, 0.00, 0.40, 0.50]),
    ("l̥",  [1.00, 0.0, 0.28, 0.25, 0.60, 0.30, 0.00, 0.60, 0.50]),
    ("ɭ",  [1.00, 0.0, 0.28, 0.90, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("ʎ",  [1.00, 0.0, 1.00, 0.00, 0.06, 0.08, 0.00, 0.40, 0.50]),
    ("ʟ",  [1.00, 0.0, 1.00, 0.00, 0.06, 0.75, 0.00, 0.40, 0.50]),
    ("ɾ",  [1.00, 0.0, 0.14, 0.25, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("r",  [1.00, 0.0, 0.06, 0.25, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("r̥",  [1.00, 0.0, 0.06, 0.25, 0.60, 0.40, 0.00, 0.60, 0.50]),
    ("ɽ",  [1.00, 0.0, 0.14, 0.90, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("ʀ",  [1.00, 0.0, 1.00, 0.00, 0.05, 0.88, 0.00, 0.40, 0.50]),
    ("ɹ",  [1.00, 0.0, 0.22, 0.42, 0.52, 0.42, 0.00, 0.40, 0.50]),
    ("j",  [1.00, 0.0, 1.00, 0.00, 0.15, 0.08, 0.00, 0.40, 0.50]),
    ("ɰ",  [1.00, 0.0, 1.00, 0.00, 0.15, 0.85, 0.00, 0.40, 0.50]),
    ("w",  [0.15, 1.0, 1.00, 0.00, 0.15, 0.85, 0.00, 0.40, 0.50]),
    ("ʋ",  [0.20, 0.0, 1.00, 0.00, 0.80, 0.50, 0.00, 0.40, 0.50]),

    ("pʼ", [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("tʼ", [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("ʈʼ", [1.00, 0.0, 0.00, 0.90, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("cʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.0, 0.02, 0.85]),
    ("kʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.02, 0.85]),
    ("qʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.88, 0.0, 0.02, 0.85]),
    ("sʼ", [1.00, 0.0, 0.05, 0.25, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("t͡sʼ", [1.00, 0.0, 0.04, 0.25, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("t͡ʃʼ", [1.00, 0.3, 0.04, 0.50, 0.60, 0.20, 0.0, 0.02, 0.85]),
    ("xʼ", [1.00, 0.0, 1.00, 0.00, 0.09, 0.75, 0.0, 0.02, 0.85]),

    ("ɓ",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40, 0.15]),
    ("ɗ",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.40, 0.15]),
    ("ᶑ",  [1.00, 0.0, 0.00, 0.90, 0.80, 0.50, 0.0, 0.40, 0.15]),
    ("ʄ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.0, 0.40, 0.15]),
    ("ɠ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.40, 0.15]),
    ("ʛ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.88, 0.0, 0.40, 0.15]),

    ("ʘ",  [0.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.40, 0.50]),
    ("ǀ",  [1.00, 0.0, 0.00, 0.05, 0.00, 0.75, 0.0, 0.40, 0.50]),
    ("ǃ",  [1.00, 0.0, 0.00, 0.25, 0.00, 0.75, 0.0, 0.40, 0.50]),
    ("ǂ",  [1.00, 0.0, 0.00, 0.50, 0.00, 0.75, 0.0, 0.40, 0.50]),
    ("ǁ",  [1.00, 0.0, 0.28, 0.25, 0.00, 0.75, 0.0, 0.40, 0.50]),

    ("tˁ", [1.00, 0.0, 0.00, 0.25, 0.80, 0.98, 0.0, 0.60, 0.50]),
    ("dˁ", [1.00, 0.0, 0.00, 0.25, 0.80, 0.98, 0.0, 0.40, 0.50]),
    ("sˁ", [1.00, 0.0, 0.05, 0.25, 0.80, 0.98, 0.0, 0.65, 0.50]),
    ("zˁ", [1.00, 0.0, 0.05, 0.25, 0.80, 0.98, 0.0, 0.40, 0.50]),
    ("ðˁ", [1.00, 0.0, 0.08, 0.05, 0.80, 0.98, 0.0, 0.40, 0.50]),
]

_ATTRACTOR_SYMBOLS: tuple[str, ...] = tuple(item[0] for item in _CANONICAL_ATTRACTORS)
_ATTRACTOR_MATRIX: np.ndarray = np.array([item[1] for item in _CANONICAL_ATTRACTORS], dtype=np.float32)
_V_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s in _VOWEL_SET], dtype=np.int64)
_C_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s not in _VOWEL_SET], dtype=np.int64)


def resolve_tract_vector(g: np.ndarray, is_nucleus: bool | None = None) -> str:
    g_arr = np.asarray(g, dtype=np.float32)[..., :9]
    idx = _V_IDX if (is_nucleus is True or (is_nucleus is None and is_vowel_gesture(g_arr))) else _C_IDX
    dists = sensory_distance(_ATTRACTOR_MATRIX[idx], g_arr[np.newaxis, :])
    sym = _ATTRACTOR_SYMBOLS[int(idx[int(np.argmin(dists))])]
    if sym in _VOICELESS_STOPS and g_arr[7] >= 0.75:
        sym = f"{sym}ʰ"
    return sym


def readout_batch(G: np.ndarray, chunk: int = 1500) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)[..., :9].reshape(-1, 9)
    vmask = np.asarray(is_vowel_gesture(G), dtype=bool)
    out = np.empty(len(G), dtype=np.int64)
    for idx, sel in ((_V_IDX, vmask), (_C_IDX, ~vmask)):
        rows = np.flatnonzero(sel)
        A = _ATTRACTOR_MATRIX[idx]
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            d = sensory_distance(A[:, None, :], G[r][None, :, :])
            out[r] = idx[np.argmin(d, axis=0)]
    return out


def resolve_chao_tone(pitch: tuple[float, float, float] | np.ndarray, duration: float = 1.0) -> str:
    f0, f1, f2 = float(pitch[0]), float(pitch[1]), float(pitch[2])
    c0 = int(np.clip(np.floor(f0 * 2.50 + 3.50), 1, 5))
    c1 = int(np.clip(np.floor(f1 * 2.50 + 3.50), 1, 5))
    c2 = int(np.clip(np.floor(f2 * 2.50 + 3.50), 1, 5))
    if duration < 0.80:
        return _CHAO_BARS[c1]
    # Long syllables (>= 1.25) can carry 3-target convex/concave contours (e.g. ˨˩˦)
    if duration >= 1.25 and c1 != c0 and c1 != c2:
        return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c1]}{_CHAO_BARS[c2]}"
    if c0 == c2:
        return _CHAO_BARS[c1]
    return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c2]}"


def parse_sonority_prosody(
    X: np.ndarray, directional_bias: float = 0.50
) -> tuple[list[tuple[int, int]], int, list[tuple[float, float, float]], list[float]]:
    L = len(X)
    if L == 0:
        return [], 0, [], []

    son = np.array([
        AcousticEngine.glottal_source_physics(X[i, :9])[1] + 0.4 * float(oral_aperture(X[i, :9]))
        for i in range(L)
    ], dtype=np.float32)
    v_mask = np.asarray(is_vowel_gesture(X[:, :9]), dtype=bool)

    peaks = [i for i in range(L) if v_mask[i] and (i == 0 or son[i] >= son[i - 1]) and (i == L - 1 or son[i] >= son[i + 1])]
    if not peaks:
        peaks = [int(np.argmax(son))]

    boundaries = [0]
    for k in range(len(peaks) - 1):
        ps, pe = peaks[k] + 1, peaks[k + 1]
        boundaries.append(ps if ps == pe else ps + int(np.argmin(son[ps:pe])))
    boundaries.append(L)

    syl_slices = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    taus = X[:, 9] if X.shape[1] > 9 else np.ones(L, dtype=np.float32)
    syl_masses = [
        sum(float(taus[i]) * (1.0 + float(son[i])) for i in range(st, en))
        * (1.0 + (k / max(1, len(syl_slices) - 1)) * (directional_bias - 0.50) * 0.30)
        for k, (st, en) in enumerate(syl_slices)
    ]
    stressed_idx = int(np.argmax(syl_masses))

    trajectories, nuc_durs = [], []
    for k, (st, en) in enumerate(syl_slices):
        pk = peaks[min(k, len(peaks) - 1)]
        f0 = 1.50 * (float(X[pk - 1, 7]) - 0.40) if pk > st else 0.0
        f1 = f0 + (0.35 if k == stressed_idx else 0.0)
        coda_g7 = float(X[en - 1, 7]) if en - 1 > pk else 0.40
        f2 = f0 + 0.90 * (0.40 - coda_g7) - 0.15
        trajectories.append((round(f0, 2), round(f1, 2), round(f2, 2)))
        nuc_durs.append(float(taus[pk]))

    return syl_slices, stressed_idx, trajectories, nuc_durs


def render_and_extract_tones(
    X: np.ndarray, directional_bias: float = 0.50, is_tonal: bool = False
) -> tuple[str, str, str]:
    L = len(X)
    if L == 0:
        return "", "", ""
    syl_slices, stressed_idx, pitches, nuc_durs = parse_sonority_prosody(X, directional_bias=directional_bias)
    parts, plain_parts = [], []
    assigned_tone = ""

    for k, (st, en) in enumerate(syl_slices):
        if k == stressed_idx and len(syl_slices) > 1:
            parts.append("ˈ")
        dedup = []
        for idx in range(st, en):
            sym = resolve_tract_vector(X[idx, :9])
            dur = float(X[idx, 9]) if X.shape[1] > 9 else 1.0
            if dur >= 1.35 and not sym.endswith("ː"):
                sym = f"{sym}ː"
            if dedup and dedup[-1] == sym:
                if not dedup[-1].endswith("ː"):
                    dedup[-1] = f"{dedup[-1]}ː"
                continue
            dedup.append(sym)

        syl_str = "".join(dedup)
        plain_parts.append(syl_str.replace("ː", ""))
        if is_tonal:
            tone = resolve_chao_tone(pitches[k], duration=nuc_durs[k])
            syl_str += tone
            if k == stressed_idx or not assigned_tone:
                assigned_tone = tone
        parts.append(syl_str)

    return "".join(parts), "".join(plain_parts), assigned_tone


def render_word_trajectory(X: np.ndarray, directional_bias: float = 0.50) -> tuple[str, str]:
    return render_and_extract_tones(X, directional_bias=directional_bias, is_tonal=False)[:2]