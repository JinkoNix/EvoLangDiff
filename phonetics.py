"""
phonetics.py — Unified Continuous Tract Aerodynamics, Quantal Metric Tensors,
Ornstein-Uhlenbeck Laryngeal Integration, and Mode-Density Inventory Extraction.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence, Any
from collections import Counter
import numpy as np

INERTIA_DIAG = np.array([1.20, 0.60, 0.80, 0.40, 2.50, 2.00, 0.50, 1.00, 1.60], dtype=np.float32)

_CHAO_BARS = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
_VOWEL_SET = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o', 'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ'
}
_VOICELESS_STOPS = {'p', 't̪', 't', 'ʈ', 'c', 'k', 'q', 'ʔ'}


# -----------------------------------------------------------------------------
# 1. QUANTAL METRICS & CONTINUOUS AERO-ACOUSTICS (Defects 1, 5, 6, 7 Fixed)
# -----------------------------------------------------------------------------

def compute_metric_weights(g: np.ndarray) -> np.ndarray:
    g0, g1, g2, g3 = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    g4, g5, g6, g7 = g[..., 4], g[..., 5], g[..., 6], g[..., 7]

    w0 = 1.0 + 20.0 / (1.0 + np.exp(np.clip(30.0 * (g0 - 0.10), -30.0, 30.0)))
    w1 = 1.0 + 4.0 * (1.0 - g4)
    w2 = 1.0 + 35.0 / (1.0 + np.exp(np.clip(30.0 * (g2 - 0.12), -30.0, 30.0)))
    w3 = 0.05 + 10.0 * (1.0 - g2)
    w4 = 1.0 + 25.0 / (1.0 + np.exp(np.clip(25.0 * (g4 - 0.12), -30.0, 30.0)))
    w5 = 2.0 + 6.0 * ((g5 - 0.50) ** 2)
    w6 = 0.50 + 25.0 / (1.0 + np.exp(np.clip(40.0 * (g6 - 0.05), -30.0, 30.0)))
    w7 = 1.0 + 20.0 / (1.0 + np.exp(np.clip(30.0 * (g7 - 0.10), -30.0, 30.0)))

    weights = [w0, w1, w2, w3, w4, w5, w6, w7]
    if g.shape[-1] >= 9:
        g8 = g[..., 8]
        w8 = 1.5 + 18.0 * ((g8 - 0.50) ** 2)
        weights.append(w8)

    return np.stack(weights, axis=-1)


def minimal_tract_aperture(g: np.ndarray) -> np.ndarray:
    g0 = np.clip(g[..., 0], 0.0, 1.0)
    g2 = np.clip(g[..., 2], 0.0, 1.0)
    g4 = np.clip(g[..., 4], 0.0, 1.0)
    g5 = np.clip(g[..., 5], 0.0, 1.0)

    a_lip = 3.50 * (g0 ** 1.80)
    a_tip = 3.00 * (g2 ** 1.60)
    a_dors = 3.20 * (g4 ** 1.70)
    a_phar = 3.50 * (np.clip(1.0 - (g5 - 0.85) / 0.15, 0.0, 1.0) ** 1.70)
    return np.minimum(np.minimum(a_lip, a_tip), np.minimum(a_dors, a_phar))


def continuous_vocality(g: np.ndarray) -> np.ndarray:
    """
    Defect 1 Fixed: Properly classifies high vowels /i, y, ɨ, ʉ, ɯ, u/ as vowels
    while rejecting /l, ɹ, w, ħ, ʕ/ as consonants.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    a_min = minimal_tract_aperture(g_arr)
    a_lip = 3.50 * (np.clip(g_arr[..., 0], 0.0, 1.0) ** 1.80)
    a_tip = 3.00 * (np.clip(g_arr[..., 2], 0.0, 1.0) ** 1.60)
    g8 = g_arr[..., 8] if g_arr.shape[-1] > 8 else 0.50

    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    gate = sig(30.0 * (a_min - 0.25)) * sig(20.0 * (a_lip - 0.30)) * sig(20.0 * (a_tip - 0.60))
    larynx_stability = np.exp(-((g8 - 0.50) ** 2) / 0.06)
    return gate * larynx_stability


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    voc = continuous_vocality(np.asarray(g, dtype=np.float32))
    return bool(voc >= 0.50) if np.ndim(voc) == 0 else (voc >= 0.50)


def is_click_gesture(g: np.ndarray) -> np.ndarray:
    """
    Defect 6 Fixed: Factored single click detector using bitwise & on arrays,
    correctly including lateral click /ǁ/ (g2=0.28).
    """
    g_arr = np.asarray(g, dtype=np.float32)
    anterior_closure = (g_arr[..., 0] < 0.08) | (g_arr[..., 2] < 0.30)
    dorsal_closure = (g_arr[..., 4] < 0.08) & (g_arr[..., 5] >= 0.70)
    return anterior_closure & dorsal_closure


def sensory_distance(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Defect 7 Fixed: Internally slices to 9 dimensions to prevent broadcasting errors."""
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    g_mid = 0.50 * (a1 + a2)
    w = compute_metric_weights(g_mid)
    return np.sqrt(np.sum(w * ((a1 - a2) ** 2), axis=-1))


def trajectory_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    """Defect 7 Fixed: Internally slices inputs to 9D."""
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    L1, L2 = len(a1), len(a2)
    if L1 == 0 or L2 == 0:
        return 1.0
    if L1 == L2:
        return float(np.mean(sensory_distance(a1, a2)))
    max_L = max(L1, L2)
    idx1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
    idx2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
    return float(np.mean(sensory_distance(a1[idx1], a2[idx2])))


def transglottal_pressure(g: np.ndarray, rho_air: float = 1.0, g8_dot: float = 0.0) -> np.ndarray:
    a_min = minimal_tract_aperture(g)
    a_nasal = 1.80 * (np.clip(g[..., 6], 0.0, 1.0) ** 2)
    total_exhaust = a_min + a_nasal + 0.01

    p_sub = 1.00 * rho_air
    p_oral = p_sub / (1.0 + 8.0 * (total_exhaust ** 1.5))
    cavity_expansion = -1.20 * g8_dot
    return np.maximum(0.0, p_sub - p_oral + cavity_expansion)


def continuous_acoustic_power(g: np.ndarray, rho_air: float = 1.0, env=None) -> np.ndarray:
    """
    Defect 5 Fixed: Stops /p t k/ now receive transient burst acoustic power B,
    preventing cost/benefit selection from punishing oral closures.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    voc = continuous_vocality(g_arr)
    a_min = minimal_tract_aperture(g_arr)
    g7 = g_arr[..., 7]
    delta_p = transglottal_pressure(g_arr, rho_air=rho_air)

    v_eff = np.exp(-((g7 - 0.40) ** 2) / 0.018) * np.tanh(4.0 * delta_p)
    voiced_radiation = voc * v_eff * (0.40 + 1.80 * np.minimum(1.0, a_min))
    fric_noise = (1.0 - voc) * (delta_p ** 1.2) * (a_min / 0.06) * np.exp(-a_min / 0.06) * 2.2
    nasal_radiance = 0.35 * np.clip(g_arr[..., 6], 0.0, 1.0) * v_eff

    # Transient plosive burst power for stops (prevents stop extinction)
    stop_burst = (1.0 - voc) * (delta_p ** 1.4) * np.exp(-a_min / 0.04) * 0.85

    # Clicks: Short impulsive transient burst
    click_flag = is_click_gesture(g_arr)
    veg = getattr(env, "vegetation", 0.3) if env else 0.3
    roughness = getattr(env, "roughness", 0.1) if env else 0.1
    click_propagation = np.exp(-4.5 * veg - 5.5 * roughness)
    click_burst = np.where(click_flag, 0.65 * click_propagation, 0.0)

    return voiced_radiation + fric_noise + nasal_radiance + stop_burst + click_burst


def continuous_sonority(g: np.ndarray) -> float:
    pwr = float(continuous_acoustic_power(g))
    return round(1.0 + 4.5 * math.tanh(pwr * 1.5), 3)


# -----------------------------------------------------------------------------
# 2. PHONEME CONTAINER & BIOMECHANICAL ENERGY
# -----------------------------------------------------------------------------

class PhonemeEnergy(NamedTuple):
    motor_effort: float
    air_consumption: float
    voicing_drag: float
    audibility_payoff: float

    @property
    def acoustic_salience(self) -> float:
        return self.audibility_payoff


class Phoneme:
    __slots__ = ('_vector', '_ipa', '_energy', '_hash')

    def __init__(self, vector: Sequence[float] | np.ndarray, duration: float | None = None):
        v = list(vector)
        if len(v) == 9:
            v.append(float(duration) if duration is not None else 1.0)
        elif len(v) == 10 and duration is not None:
            v[9] = float(duration)
        elif len(v) == 8:
            v.append(0.50)
            v.append(float(duration) if duration is not None else 1.0)

        g_clamped = [max(0.0, min(1.0, float(v[i]))) for i in range(9)]
        tau_clamped = max(0.01, min(3.0, float(v[9])))
        self._vector = np.array(g_clamped + [tau_clamped], dtype=np.float32)
        self._ipa = None
        self._energy = None
        self._hash = hash(tuple(int(x * 100) for x in self._vector))

    @property
    def vector(self) -> np.ndarray: return self._vector
    @property
    def g(self) -> np.ndarray: return self._vector[:9]
    @property
    def duration(self) -> float: return float(self._vector[9])
    @property
    def tau(self) -> float: return self.duration
    @property
    def is_vowel(self) -> bool: return bool(is_vowel_gesture(self.g))

    @property
    def ipa(self) -> str:
        if self._ipa is None:
            base = resolve_tract_vector(self.g)
            self._ipa = f"{base}ː" if (self.is_vowel and self.duration >= 1.35) else base
        return self._ipa

    @property
    def energy(self) -> PhonemeEnergy:
        if self._energy is None:
            g = self.g
            tau = self.duration
            a_min = float(minimal_tract_aperture(g))
            e_displacement = 0.35 * (1.0 - a_min) + 0.80 * ((g[8] - 0.50) ** 2) + 0.40 * math.exp(-25.0 * g[7])
            e_mot = round((0.10 + e_displacement) * (tau ** 1.08), 3)
            air = round((0.10 + 0.35 * a_min + 0.40 * max(0.0, g[7] - 0.40)) * tau, 3)
            v_drag = round(math.exp(-((g[7] - 0.40) ** 2) / 0.02) * math.exp(-10.0 * a_min) * 0.4 * tau, 3)
            aud = round(float(continuous_acoustic_power(g)), 3)
            self._energy = PhonemeEnergy(e_mot, air, v_drag, aud)
        return self._energy

    def drift(self, delta: Sequence[float] | None = None, target: Sequence[float] | None = None) -> Phoneme:
        if target is not None:
            return Phoneme(target)
        if delta is not None:
            new_v = self._vector.copy()
            for i in range(min(len(delta), 10)):
                new_v[i] += float(delta[i])
            return Phoneme(new_v)
        return Phoneme(self._vector.copy())

    def __hash__(self) -> int: return self._hash
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phoneme) and np.array_equal(self._vector, other._vector)
    def __repr__(self) -> str: return f"Phoneme(/{self.ipa}/, tau={self.duration:.2f})"


# -----------------------------------------------------------------------------
# 3. STABLE ORNSTEIN-UHLENBECK SDE INTEGRATOR (Defect 2 Fixed)
# -----------------------------------------------------------------------------

def transition_cost(p1: Phoneme | np.ndarray, p2: Phoneme | np.ndarray, tau1: float = 1.0, tau2: float = 1.0) -> float:
    g1 = p1.g if hasattr(p1, "g") else p1[:9]
    g2 = p2.g if hasattr(p2, "g") else p2[:9]
    t1 = p1.duration if hasattr(p1, "duration") else tau1
    t2 = p2.duration if hasattr(p2, "duration") else tau2

    diff = g1 - g2
    min_tau = max(0.10, min(t1, t2))
    kinetic = float(np.sum(INERTIA_DIAG * (diff ** 2)) / (min_tau ** 0.85))

    voc1 = float(continuous_vocality(g1))
    voc2 = float(continuous_vocality(g2))
    dist = float(sensory_distance(g1, g2))
    ocp_friction = 2.50 * math.exp(-5.0 * dist) * (voc1 * voc2 + (1.0 - voc1) * (1.0 - voc2))
    return round(kinetic + ocp_friction, 3)


def relax_gesture_gradient(
    p: Phoneme,
    adjacent_p: Phoneme | None,
    neighbor_g_pool: list[np.ndarray],
    valence_intensity: float,
    bias: np.ndarray,
    alt: float,
    hum: float,
    temp: float,
    t_drift: float,
    rng: Any,
    word_cost: float = 10.0,
) -> Phoneme:
    """
    Defect 2 Fixed: Integrates via an exact Ornstein-Uhlenbeck decay for the stiff
    laryngeal spring, scaling all drift terms by dt and noise by sqrt(dt).
    Eliminates explicit Euler rail-banging that manufactured fake ejectives.
    """
    g = p.g.copy()
    voc = float(continuous_vocality(g))
    rho = math.exp(-0.85 * max(0.0, alt))
    dt = float(np.clip(t_drift * 0.15, 0.02, 0.25))

    # Exact Ornstein-Uhlenbeck integration for stiff laryngeal spring: g8 -> 0.50
    k_L = 8.5
    g[8] = 0.50 + (g[8] - 0.50) * math.exp(-k_L * dt)

    f_env = np.zeros(9, dtype=np.float32)

    # Hypoxia pulmonary breathing penalty pushes larynx upward only under extreme hypoxia
    hypoxia_strain = (1.0 / (rho ** 1.8) - 1.0) * 0.40
    f_env[8] += hypoxia_strain * (1.0 - voc)

    # Aridity tongue backing
    drying = max(0.0, 0.40 - hum)
    f_env[5] += drying * 0.35 * voc
    f_env[4] += (hum - 0.40) * 0.30 * voc

    target_g7 = 0.40 * voc + 0.60 * (1.0 - voc)
    f_env[7] += -0.30 * (g[7] - target_g7)

    delta_p = float(transglottal_pressure(g, rho_air=rho))
    coronal = math.exp(-((float(g[3]) - 0.25) ** 2) / 0.03)
    groove = math.exp(-((float(g[2]) - 0.06) ** 2) / 0.015)
    f_env[2] += -0.55 * delta_p * (1.0 - voc) * coronal * groove * (float(g[2]) - 0.06)

    f_coartic = 0.06 * (adjacent_p.g - g) if adjacent_p is not None else np.zeros(9, dtype=np.float32)

    f_repulse = np.zeros(9, dtype=np.float32)
    if neighbor_g_pool:
        pool_mat = np.stack(neighbor_g_pool[:16], axis=0)[:, :9]
        diffs = g[np.newaxis, :] - pool_mat
        d_sqs = np.sum(diffs ** 2, axis=-1, keepdims=True)
        f_repulse = np.sum(diffs / (d_sqs + 0.05), axis=0) * 0.015

    drift_sigma = 0.020 * math.sqrt(dt) * math.sqrt(max(0.5, word_cost / 10.0))
    noise = np.array([rng.gauss(0.0, drift_sigma) for _ in range(9)], dtype=np.float32)

    eff_bias = bias[:9] if bias is not None else np.zeros(9, dtype=np.float32)
    dg = (f_env + f_coartic + f_repulse + eff_bias * 0.20) * dt + noise
    return Phoneme(np.clip(g + dg, 0.0, 1.0), duration=p.duration)


def repair_phonemes(phonemes: list[Phoneme], profile=None, max_passes: int = 4) -> list[Phoneme]:
    s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    cluster_budget = 1.20 + 4.50 * s_comp

    for _ in range(max_passes):
        if len(phonemes) <= 1:
            break
        changed = False
        i = 0
        while i < len(phonemes) - 1:
            p1, p2 = phonemes[i], phonemes[i + 1]
            dist = float(sensory_distance(p1.g, p2.g))

            if dist < 0.18:
                blended_g = 0.50 * (p1.g + p2.g)
                coalesced_tau = min(2.5, p1.duration + p2.duration * 0.65)
                phonemes[i] = Phoneme(blended_g, duration=coalesced_tau)
                phonemes.pop(i + 1)
                changed = True
                continue

            cost = transition_cost(p1, p2)
            if cost > cluster_budget:
                tot_tau = p1.duration + p2.duration
                w1 = p1.duration / tot_tau
                w2 = p2.duration / tot_tau

                coalesced_g = w1 * p1.g + w2 * p2.g
                coalesced_g[0] = np.clip(coalesced_g[0] + 0.12, 0.0, 1.0)
                coalesced_g[2] = np.clip(coalesced_g[2] + 0.10, 0.0, 1.0)
                coalesced_g[4] = np.clip(coalesced_g[4] + 0.10, 0.0, 1.0)

                coalesced_tau = min(2.0, tot_tau * 0.70)
                phonemes[i] = Phoneme(coalesced_g, duration=coalesced_tau)
                phonemes.pop(i + 1)
                changed = True
                continue

            i += 1
        if not changed:
            break

    return phonemes


def enforce_acoustic_contrast(
    words: list,
    min_auditory_margin: float = 0.32,
    ambient_noise: float = 0.10,
) -> None:
    """
    Defect 4 Fixed: Riemannian-consistent contrast push.
    Handles identical vowels by random unit vector perturbation, and pushes pitch
    symmetrically along sign(p1 - p2) rather than stack-biasing by word order.
    """
    n = len(words)
    if n < 2:
        return

    for i in range(n):
        w1 = words[i]
        if not w1.phonemes:
            continue
        v_indices1 = [idx for idx, p in enumerate(w1.phonemes) if p.is_vowel]
        if not v_indices1:
            continue
        v_idx1 = v_indices1[0]
        g1 = w1.phonemes[v_idx1].g

        for j in range(i + 1, n):
            w2 = words[j]
            if not w2.phonemes or w1.id == w2.id:
                continue

            d_sem = w1.vector.distance_to(w2.vector)
            if d_sem < 0.15:
                continue

            v_indices2 = [idx for idx, p in enumerate(w2.phonemes) if p.is_vowel]
            if not v_indices2:
                continue
            v_idx2 = v_indices2[0]
            g2 = w2.phonemes[v_idx2].g

            d_aud = float(sensory_distance(g1, g2))
            eff_margin = min_auditory_margin + ambient_noise * 0.50

            if d_aud < eff_margin:
                diff = g1 - g2
                norm_euclid = float(np.linalg.norm(diff))
                if norm_euclid < 1e-4:
                    # Identical vowel collision: perturb with random unit vector
                    h_hat = np.random.normal(0, 1, size=9).astype(np.float32)
                    h_hat /= np.linalg.norm(h_hat)
                else:
                    h_hat = diff / norm_euclid

                w_mid = compute_metric_weights(0.50 * (g1 + g2))
                metric_norm = math.sqrt(float(np.sum(w_mid * (h_hat ** 2))))
                step_a = 0.40 * (eff_margin - d_aud) / (2.0 * metric_norm + 1e-4)

                g1_new = np.clip(g1 + step_a * h_hat, 0.05, 0.95)
                g2_new = np.clip(g2 - step_a * h_hat, 0.05, 0.95)

                w1.phonemes[v_idx1] = Phoneme(g1_new, duration=w1.phonemes[v_idx1].duration)
                w2.phonemes[v_idx2] = Phoneme(g2_new, duration=w2.phonemes[v_idx2].duration)
                w1._invalidate_cache()
                w2._invalidate_cache()

            # Symmetric pitch push
            syls1 = w1.syllables
            syls2 = w2.syllables
            if syls1 and syls2 and hasattr(syls1[0], "pitch") and hasattr(syls2[0], "pitch"):
                p1 = syls1[0].pitch
                p2 = syls2[0].pitch
                pitch_diff = p1[0] - p2[0]
                pitch_dist = math.sqrt(sum((p1[k] - p2[k]) ** 2 for k in range(3)))

                if pitch_dist < 0.42 and d_aud < 0.45:
                    direction = 1.0 if pitch_diff >= 0 else -1.0
                    shift = 0.25 * direction
                    syls1[0].pitch = (round(float(np.clip(p1[0] + shift, -1.2, 1.2)), 2), round(float(np.clip(p1[1] + shift, -1.2, 1.2)), 2), round(float(np.clip(p1[2] + shift, -1.2, 1.2)), 2))
                    syls2[0].pitch = (round(float(np.clip(p2[0] - shift, -1.2, 1.2)), 2), round(float(np.clip(p2[1] - shift, -1.2, 1.2)), 2), round(float(np.clip(p2[2] - shift, -1.2, 1.2)), 2))
                    w1._invalidate_cache()
                    w2._invalidate_cache()


def word_energy(
    word,
    profile=None,
    env=None,
    cultural_attention: Sequence[float] | None = None,
    cohort_sample: Sequence = None,
) -> float:
    phonemes = word.phonemes
    n = len(phonemes)
    if n == 0:
        return 0.0

    alt = getattr(env, "altitude", 0.0) if env else 0.0
    hum = getattr(env, "humidity", 0.50) if env else 0.50
    temp = getattr(env, "temperature", 0.50) if env else 0.50
    veg = getattr(env, "vegetation", 0.30) if env else 0.30
    roughness = getattr(env, "roughness", 0.10) if env else 0.10
    rho = math.exp(-0.85 * max(0.0, alt))

    e_motor = sum(p.energy.motor_effort for p in phonemes)
    for j in range(n - 1):
        e_motor += transition_cost(phonemes[j], phonemes[j + 1]) * 0.40

    e_larynx = 0.0
    click_motor_tax = 0.0
    for p in phonemes:
        g = p.g
        voc = float(continuous_vocality(g))
        a_min = float(minimal_tract_aperture(g))
        g8 = float(g[8])
        g7 = float(g[7])

        e_spring = 8.5 * ((g8 - 0.50) ** 2)

        is_ejective = (g8 > 0.70) and (g7 < 0.05) and not voc
        is_click = is_click_gesture(g)

        if is_ejective:
            e_air = 0.0
        elif is_click:
            click_motor_tax += 1.65
            e_air = 0.05
        else:
            pulmonary_drain = (0.15 + 0.35 * a_min + 0.40 * max(0.0, g7 - 0.35))
            hypoxia_penalty = (1.0 / (rho ** 1.8))
            e_air = pulmonary_drain * hypoxia_penalty

        e_larynx += e_spring + e_air

    click_scattering = 0.0
    for p in phonemes:
        if is_click_gesture(p.g):
            click_scattering += (4.5 * veg + 5.5 * roughness)

    arid_evap = max(0.0, 0.35 - hum) * (1.0 + max(0.0, temp - 0.5)) * sum(p.duration for p in phonemes if p.is_vowel) * 0.45
    s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
    freq_load = math.log1p(getattr(word, "usage_frequency", 1.0))
    e_length = 0.12 * (n ** 1.65) * (1.0 - 0.55 * s_idx) * (0.60 + 0.40 * freq_load)

    e_tone = 0.0
    syls = word.syllables
    if syls:
        for s in syls:
            if hasattr(s, "pitch"):
                p0, p1, p2 = s.pitch
                pitch_excursion = (p1 - p0) ** 2 + (p2 - p1) ** 2 + (p1 ** 2) * 0.2
                e_tone += pitch_excursion * 0.18

    return round(float(e_motor + e_larynx + click_motor_tax + click_scattering + e_length + arid_evap + e_tone), 3)


# -----------------------------------------------------------------------------
# 4. CONTOUR TONE PROSODY & MODE-DENSITY INVENTORY (Defect 3 Fixed)
# -----------------------------------------------------------------------------

def resolve_chao_tone(pitch: tuple[float, float, float] | np.ndarray, duration: float = 1.0) -> str:
    f0, f1, f2 = float(pitch[0]), float(pitch[1]), float(pitch[2])
    c0 = int(np.clip(np.floor(f0 * 2.50 + 3.50), 1, 5))
    c1 = int(np.clip(np.floor(f1 * 2.50 + 3.50), 1, 5))
    c2 = int(np.clip(np.floor(f2 * 2.50 + 3.50), 1, 5))
    if duration < 0.80 or c0 == c2:
        return _CHAO_BARS[c1]
    return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c2]}"


def parse_sonority_prosody(X: np.ndarray, directional_bias: float = 0.50) -> tuple[list[tuple[int, int]], int, list[tuple[float, float, float]], list[float]]:
    """
    Defect 3 Fixed: Gives f2 an independent dynamic target based on coda voicing
    and boundary prosody, allowing contour tones (rising, falling, dipping) to occur.
    """
    L = len(X)
    if L == 0:
        return [], 0, [], []
    son = np.array([continuous_sonority(X[i, :9]) for i in range(L)], dtype=np.float32)
    v_mask = np.array([is_vowel_gesture(X[i, :9]) for i in range(L)], dtype=bool)

    peaks = [i for i in range(L) if v_mask[i] and (i == 0 or son[i] >= son[i - 1]) and (i == L - 1 or son[i] >= son[i + 1])]
    if not peaks:
        peaks = [int(np.argmax(son))]

    boundaries = [0]
    for k in range(len(peaks) - 1):
        p_start, p_end = peaks[k] + 1, peaks[k + 1]
        boundaries.append(p_start if p_start == p_end else p_start + int(np.argmin(son[p_start:p_end])))
    boundaries.append(L)

    syl_slices = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    syl_masses = [sum(X[i, 9] * son[i] for i in range(st, en)) * (1.0 + (k / max(1, len(syl_slices) - 1)) * (directional_bias - 0.50) * 0.30) for k, (st, en) in enumerate(syl_slices)]
    stressed_idx = int(np.argmax(syl_masses))

    trajectories = []
    nuc_durs = []
    for k, (st, en) in enumerate(syl_slices):
        pk = peaks[min(k, len(peaks) - 1)]
        # Onset pitch influence
        f0 = 1.50 * (float(X[pk - 1, 7]) - 0.40) if pk > st else 0.0
        # Peak stress pitch bump
        f1 = f0 + (0.35 if k == stressed_idx else 0.0)
        # Coda pitch target: Glottal closure raises f2; open coda allows prosodic fall
        coda_g7 = float(X[en - 1, 7]) if en - 1 > pk else 0.40
        f2 = f0 + 0.40 * (0.40 - coda_g7) - 0.15  # Natural contour trajectory

        trajectories.append((round(f0, 2), round(f1, 2), round(f2, 2)))
        nuc_durs.append(float(X[pk, 9]))

    return syl_slices, stressed_idx, trajectories, nuc_durs


def render_and_extract_tones(X: np.ndarray, directional_bias: float = 0.50, is_tonal: bool = False) -> tuple[str, str, str]:
    L = len(X)
    if L == 0:
        return "", "", ""
    syl_slices, stressed_idx, pitches, _ = parse_sonority_prosody(X, directional_bias=directional_bias)
    parts, plain_parts = [], []
    assigned_tone = ""

    for k, (st, en) in enumerate(syl_slices):
        if k == stressed_idx and len(syl_slices) > 1:
            parts.append("ˈ")
        syl_sonorities = [continuous_sonority(X[i, :9]) for i in range(st, en)]
        peak_offset = int(np.argmax(syl_sonorities))

        syl_symbols = [
            resolve_tract_vector(X[st + off, :9], is_nucleus=(off == peak_offset))
            for off in range(en - st)
        ]
        dedup = []
        for sym in syl_symbols:
            if dedup and dedup[-1] == sym:
                if sym in _VOWEL_SET and not dedup[-1].endswith("ː"):
                    dedup[-1] = f"{dedup[-1]}ː"
                continue
            dedup.append(sym)
        syl_str = "".join(dedup)
        plain_parts.append(syl_str.replace("ː", ""))
        if is_tonal and (k == stressed_idx or len(syl_slices) == 1):
            tone = resolve_chao_tone(pitches[k])
            syl_str += tone
            assigned_tone = tone
        parts.append(syl_str)

    return "".join(parts), "".join(plain_parts), assigned_tone


def cluster_contrastive_phonemes(words: Sequence) -> tuple[list[str], list[str]]:
    """
    Mode-Density Phoneme Observer.
    Evaluates phonemes directly from spoken usage frequency across roots and morphemes.
    Reports actual contrastive density peaks—including /i/ and /u/.
    """
    v_usage: dict[str, float] = Counter()
    c_usage: dict[str, float] = Counter()
    v_roots: dict[str, set[int]] = {}
    c_roots: dict[str, set[int]] = {}

    for w in words:
        u = float(getattr(w, "usage_frequency", 1.0))
        root_id = getattr(w.derivation, "root_family_id", w.id)
        for p in w.phonemes:
            clean = p.ipa.replace("ː", "").replace("˥", "").replace("˦", "").replace("˧", "").replace("˨", "").replace("˩", "")
            if p.is_vowel:
                v_usage[clean] += u
                v_roots.setdefault(clean, set()).add(root_id)
            else:
                c_usage[clean] += u
                c_roots.setdefault(clean, set()).add(root_id)

    attractor_dict = dict(_CANONICAL_ATTRACTORS)

    def extract_attractor_modes(usage_dict: dict[str, float], root_dict: dict[str, set[int]], min_margin: float) -> list[str]:
        ranked = sorted(usage_dict.keys(), key=lambda sym: usage_dict[sym], reverse=True)
        modes: list[str] = []

        for sym in ranked:
            if len(root_dict.get(sym, set())) < 2 and usage_dict[sym] < 12.0:
                continue
            if sym not in attractor_dict:
                continue

            g_cand = np.array(attractor_dict[sym], dtype=np.float32)
            if not any(float(sensory_distance(g_cand, np.array(attractor_dict[m], dtype=np.float32))) < min_margin for m in modes):
                modes.append(sym)

        return modes

    # 0.22 margin preserves natural high vowel distinctions (/i/ vs /e/, /u/ vs /o/)
    admitted_v = extract_attractor_modes(v_usage, v_roots, min_margin=0.22)
    admitted_c = extract_attractor_modes(c_usage, c_roots, min_margin=0.24)

    return (admitted_v or ["a", "i", "u"]), (admitted_c or ["t", "p", "k", "s", "m", "n"])


def render_word_trajectory(X: np.ndarray, directional_bias: float = 0.50) -> tuple[str, str]:
    return render_and_extract_tones(X, directional_bias=directional_bias, is_tonal=False)[:2]


class Syllable:
    __slots__ = ('onset', 'nucleus', 'coda', 'stressed', 'pitch')
    def __init__(self, onset, nucleus, coda, stressed=False, pitch=(0.0, 0.0, 0.0)):
        self.onset = onset
        self.nucleus = nucleus
        self.coda = coda
        self.stressed = stressed
        self.pitch = pitch

    def to_phonemes(self) -> list[Phoneme]:
        return self.onset + self.nucleus + self.coda

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.nucleus)

    @property
    def mass(self) -> float:
        return sum(p.duration * p.energy.audibility_payoff for p in self.to_phonemes())

    @property
    def form(self) -> str:
        pref = "ˈ" if self.stressed else ""
        return f"{pref}{''.join(p.ipa for p in self.onset)}{''.join(p.ipa for p in self.nucleus)}{''.join(p.ipa for p in self.coda)}"


def syllabify(target, syllable_complexity: float = 0.50) -> list[Syllable]:
    raw = target.phonemes if hasattr(target, "phonemes") else list(target)
    if not raw:
        return []
    X = np.stack([p.vector for p in raw], axis=0)
    slices, stressed_idx, pitches, _ = parse_sonority_prosody(X)
    syls = []
    for k, (st, en) in enumerate(slices):
        syl_p = raw[st:en]
        nuc = [p for p in syl_p if p.is_vowel]
        if not nuc:
            syls.append(Syllable(onset=syl_p, nucleus=[], coda=[], stressed=(k == stressed_idx), pitch=pitches[k]))
        else:
            first_v = syl_p.index(nuc[0])
            last_v = len(syl_p) - 1 - list(reversed(syl_p)).index(nuc[-1])
            syls.append(Syllable(
                onset=syl_p[:first_v],
                nucleus=syl_p[first_v:last_v + 1],
                coda=syl_p[last_v + 1:],
                stressed=(k == stressed_idx),
                pitch=pitches[k],
            ))
    return syls


def assign_stress(syllables: list[Syllable], directional_bias: float = 0.50) -> None:
    if not syllables:
        return
    n = len(syllables)
    scores = [s.mass * (1.0 + (i / max(1, n - 1)) * (directional_bias - 0.50) * 0.40) for i, s in enumerate(syllables)]
    str_idx = int(np.argmax(scores))
    for i, s in enumerate(syllables):
        s.stressed = (i == str_idx)


def apply_prosody(word, directional_bias: float = 0.50, syllable_complexity: float = 0.50):
    assign_stress(word.syllables, directional_bias=directional_bias)
    return word


# -----------------------------------------------------------------------------
# 5. OBSERVATION CODEBOOK
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

    ("t͡s", [1.00, 0.0, 0.02, 0.25, 0.80, 0.50, 0.0, 0.60, 0.50]),
    ("d͡z", [1.00, 0.0, 0.02, 0.25, 0.80, 0.50, 0.0, 0.40, 0.50]),
    ("t͡ʃ", [1.00, 0.3, 0.02, 0.50, 0.60, 0.20, 0.0, 0.60, 0.50]),
    ("d͡ʒ", [1.00, 0.3, 0.02, 0.50, 0.60, 0.20, 0.0, 0.40, 0.50]),
    ("ʈ͡ʂ", [1.00, 0.0, 0.02, 0.90, 0.60, 0.40, 0.0, 0.60, 0.50]),
    ("k͡x", [1.00, 0.0, 1.00, 0.00, 0.03, 0.75, 0.0, 0.60, 0.50]),

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
    ("ʂ",  [1.00, 0.0, 0.05, 0.90, 0.60, 0.40, 0.0, 0.65, 0.50]),
    ("ʐ",  [1.00, 0.0, 0.05, 0.90, 0.60, 0.40, 0.0, 0.40, 0.50]),
    ("ç",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.08, 0.0, 0.65, 0.50]),
    ("ʝ",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.08, 0.0, 0.40, 0.50]),
    ("x",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.75, 0.0, 0.65, 0.50]),
    ("ɣ",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.75, 0.0, 0.40, 0.50]),
    ("χ",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.88, 0.0, 0.65, 0.50]),
    ("ʁ",  [1.00, 0.0, 1.00, 0.00, 0.08, 0.88, 0.0, 0.40, 0.50]),
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
    ("ʎ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.08, 0.00, 0.40, 0.50]),
    ("ʟ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.75, 0.00, 0.40, 0.50]),
    ("ɾ",  [1.00, 0.0, 0.14, 0.25, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("r",  [1.00, 0.0, 0.06, 0.25, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("r̥",  [1.00, 0.0, 0.06, 0.25, 0.60, 0.40, 0.00, 0.60, 0.50]),
    ("ɽ",  [1.00, 0.0, 0.14, 0.90, 0.60, 0.40, 0.00, 0.40, 0.50]),
    ("ʀ",  [1.00, 0.0, 1.00, 0.00, 0.05, 0.88, 0.00, 0.40, 0.50]),
    ("ɹ",  [1.00, 0.0, 0.22, 0.25, 0.60, 0.30, 0.00, 0.40, 0.50]),
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
    ("t͡sʼ", [1.00, 0.0, 0.02, 0.25, 0.80, 0.50, 0.0, 0.02, 0.85]),
    ("t͡ʃʼ", [1.00, 0.3, 0.02, 0.50, 0.60, 0.20, 0.0, 0.02, 0.85]),
    ("xʼ", [1.00, 0.0, 1.00, 0.00, 0.08, 0.75, 0.0, 0.02, 0.85]),

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


def resolve_tract_vector(g: np.ndarray, is_nucleus: bool | None = None) -> str:
    g_arr = np.asarray(g, dtype=np.float32)[..., :9]

    if is_nucleus is True or (is_nucleus is None and is_vowel_gesture(g_arr)):
        candidate_indices = [i for i, (sym, _) in enumerate(_CANONICAL_ATTRACTORS) if sym in _VOWEL_SET]
    else:
        candidate_indices = [i for i, (sym, _) in enumerate(_CANONICAL_ATTRACTORS) if sym not in _VOWEL_SET]

    if not candidate_indices:
        candidate_indices = list(range(len(_CANONICAL_ATTRACTORS)))

    sub_matrix = _ATTRACTOR_MATRIX[candidate_indices]
    g_mid = 0.50 * (sub_matrix + g_arr[np.newaxis, :])
    w = compute_metric_weights(g_mid)
    dists = np.sum(w * ((sub_matrix - g_arr[np.newaxis, :]) ** 2), axis=-1)
    symbol = _ATTRACTOR_SYMBOLS[candidate_indices[int(np.argmin(dists))]]

    if symbol in _VOICELESS_STOPS and g_arr[7] >= 0.75:
        symbol = f"{symbol}ʰ"
    return symbol