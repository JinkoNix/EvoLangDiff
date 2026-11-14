"""
phonetics.py — Unified Continuous Tract Aerodynamics, Quantal Metric Tensors,
Ornstein-Uhlenbeck Laryngeal Integration, Per-Articulator Biomechanical Effort,
and Energy-Driven Sound Change (effort + environment + distinctiveness + inventory cost).
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


def articulator_apertures(g: np.ndarray) -> np.ndarray:
    """Per-articulator aperture [lip, tongue tip, dorsum, pharynx]; 0 = closed, >=1 = open."""
    g = np.asarray(g, dtype=np.float32)
    g0 = np.clip(g[..., 0], 0.0, 1.0)
    g2 = np.clip(g[..., 2], 0.0, 1.0)
    g4 = np.clip(g[..., 4], 0.0, 1.0)
    g5 = np.clip(g[..., 5], 0.0, 1.0)
    a_lip = 3.50 * (g0 ** 1.80)
    a_tip = 3.00 * (g2 ** 1.60)
    a_dors = 3.20 * (g4 ** 1.70)
    a_phar = 3.50 * (np.clip(1.0 - (g5 - 0.85) / 0.15, 0.0, 1.0) ** 1.70)
    return np.stack([a_lip, a_tip, a_dors, a_phar], axis=-1)


def minimal_tract_aperture(g: np.ndarray) -> np.ndarray:
    return np.min(articulator_apertures(g), axis=-1)


_POSTURE_KAPPA = np.array([0.25, 0.35, 0.30, 0.60], dtype=np.float32)  # lip, tip, dorsum, pharynx


def posture_effort(g: np.ndarray) -> np.ndarray:
    """
    Static postural effort, summed over articulators so that double closures cost the sum of their parts.
    c_x = 1 - min(1, a_x) is how far articulator x is from rest. Retroflexion, lip rounding and nasal
    port opening carry their own terms (previously all of these dims were free).
    """
    g = np.asarray(g, dtype=np.float32)
    c = 1.0 - np.minimum(1.0, articulator_apertures(g))
    e = np.sum(_POSTURE_KAPPA * (c ** 1.3), axis=-1)
    e = e + 0.10 * c[..., 1] * ((g[..., 3] - 0.25) ** 2) / 0.25   # retroflexion only costs when the tip is engaged
    e = e + 0.15 * (g[..., 1] ** 2)                                # lip rounding
    e = e + 0.12 * (g[..., 6] ** 2)                                # velopharyngeal port
    return e


def articulatory_cost(g: np.ndarray) -> np.ndarray:
    """Posture + laryngeal displacement + voicelessness/glottal-constriction cost (per unit duration)."""
    g = np.asarray(g, dtype=np.float32)
    return posture_effort(g) + 0.80 * ((g[..., 8] - 0.50) ** 2) + 0.40 * np.exp(-25.0 * g[..., 7])


def articulatory_cost_grad(g: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central-difference gradient of articulatory_cost, batched over a (..., 9, 9) perturbation tensor."""
    g = np.asarray(g, dtype=np.float32)[..., :9]
    eye = np.eye(9, dtype=np.float32)
    hi = np.clip(g[..., None, :] + eps * eye, 0.0, 1.0)
    lo = np.clip(g[..., None, :] - eps * eye, 0.0, 1.0)
    span = np.maximum(np.diagonal(hi, axis1=-2, axis2=-1) - np.diagonal(lo, axis1=-2, axis2=-1), 1e-6)
    return (articulatory_cost(hi) - articulatory_cost(lo)) / span


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
    voicing_gate = np.exp(-((g_arr[..., 7] - 0.40) ** 2) / 0.025)   # excludes /h ɦ ʔ/ (open tract, no modal voicing)
    return gate * larynx_stability * voicing_gate


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    voc = continuous_vocality(np.asarray(g, dtype=np.float32))
    return bool(voc >= 0.50) if np.ndim(voc) == 0 else (voc >= 0.50)


def is_click_gesture(g: np.ndarray) -> np.ndarray:
    """
    Dorsal closure with a velar/uvular seal (g4 < 0.08, g5 >= 0.70) plus an anterior closure:
    labial or tip contact (< 0.08) or the lateral tip setting of /ǁ/ (g2 within 0.06 of 0.28).
    NOTE: without an airstream coordinate a click is geometrically a doubly-articulated stop (/k͡p/).
    The per-articulator effort above prices the double closure, so drift no longer produces clicks for free.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    dorsal = (g_arr[..., 4] < 0.08) & (g_arr[..., 5] >= 0.70)
    anterior = (np.minimum(g_arr[..., 0], g_arr[..., 2]) < 0.08) | (np.abs(g_arr[..., 2] - 0.28) < 0.06)
    return dorsal & anterior


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


def continuous_acoustic_power(g: np.ndarray, rho_air: float = 1.0, env=None, include_transients: bool = True) -> np.ndarray:
    """
    Sustained power (voiced radiation, frication, nasal radiance) plus, optionally, transients:
    the plosive burst B = (1-voc)(1-nasal) p_closed tanh-free release term exp(-(a/0.02)^2), where p_closed is the
    oral pressure built up behind the closure (the closed-state transglottal drop is ~0, which is why the
    previous burst term was dead), and the click burst attenuated by vegetation and roughness.
    Sonority uses include_transients=False so that stops stay sonority minima.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    voc = continuous_vocality(g_arr)
    a_min = minimal_tract_aperture(g_arr)
    g6 = np.clip(g_arr[..., 6], 0.0, 1.0)
    g7 = g_arr[..., 7]
    delta_p = transglottal_pressure(g_arr, rho_air=rho_air)

    v_eff = np.exp(-((g7 - 0.40) ** 2) / 0.018) * np.tanh(4.0 * delta_p)
    voiced_radiation = voc * v_eff * (0.40 + 1.80 * np.minimum(1.0, a_min))
    fric_noise = (1.0 - voc) * (delta_p ** 1.2) * (a_min / 0.06) * np.exp(-a_min / 0.06) * 2.2
    nasal_radiance = 0.35 * g6 * v_eff
    power = voiced_radiation + fric_noise + nasal_radiance
    if not include_transients:
        return power

    a_exhaust = a_min + 1.80 * (g6 ** 2) + 0.01
    p_closed = rho_air / (1.0 + 8.0 * (a_exhaust ** 1.5))
    click_flag = is_click_gesture(g_arr)
    stop_burst = (1.0 - voc) * (1.0 - g6) * p_closed * np.exp(-((a_min / 0.02) ** 2)) * 0.85
    stop_burst = np.where(click_flag, 0.0, stop_burst)

    veg = getattr(env, "vegetation", 0.3) if env else 0.3
    roughness = getattr(env, "roughness", 0.1) if env else 0.1
    click_propagation = np.exp(-4.5 * veg - 5.5 * roughness)
    click_burst = np.where(click_flag, 0.65 * click_propagation, 0.0)
    return power + stop_burst + click_burst


def continuous_sonority(g: np.ndarray) -> float:
    pwr = float(continuous_acoustic_power(g, include_transients=False))
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
            a_min = min(1.0, float(minimal_tract_aperture(g)))
            e_mot = round((0.10 + float(articulatory_cost(g))) * (tau ** 1.08), 3)
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


def coarticulation_force(g, prev_g, next_g, tau: float = 1.0, tau_prev: float = 1.0, tau_next: float = 1.0,
                         k: float = 0.05, d_reach: float = 0.70) -> np.ndarray:
    """
    Negative gradient of the kinetic part of transition_cost with respect to g, over both neighbours:
        f = -k * exp(-(d/d_reach)^2) * INERTIA (.) (g - g_nb) / min(tau, tau_nb)^0.85
    with d the metric distance to the neighbour, so only perceptually close segments assimilate. Heavy articulators (dorsum, pharynx) resist assimilation; short segments assimilate more.
    """
    f = np.zeros(9, dtype=np.float32)
    for nb, t_nb in ((prev_g, tau_prev), (next_g, tau_next)):
        if nb is None:
            continue
        m = max(0.10, min(float(tau), float(t_nb)))
        nb = np.asarray(nb, dtype=np.float32)[:9]
        reach = math.exp(-(float(sensory_distance(g, nb)) / d_reach) ** 2)   # assimilation is partial: /t d/ yes, /k a/ no
        f -= k * reach * INERTIA_DIAG * (g - nb) / (m ** 0.85)
    return f


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
    next_p: Phoneme | None = None,
    effort_gain: float = 0.10,
) -> Phoneme:
    """
    One drift step. Exact OU decay for the stiff laryngeal spring; all other drift terms scale with dt and the
    noise with sqrt(dt). New: (i) a lenition force -effort_gain * G^-1 grad(articulatory_cost), (ii) noise
    preconditioned by G^-1/2 (stable, quantal dimensions are noisier less), (iii) coarticulation derived from the
    transition cost over both neighbours (pass next_p). The Ito correction term T div(G^-1) is omitted.
    """
    g = p.g.copy()
    voc = float(continuous_vocality(g))
    rho = math.exp(-0.85 * max(0.0, alt))
    dt = float(np.clip(t_drift * 0.15, 0.02, 0.25))

    k_L = 8.5
    g[8] = 0.50 + (g[8] - 0.50) * math.exp(-k_L * dt)

    f_env = np.zeros(9, dtype=np.float32)
    hypoxia_strain = (1.0 / (rho ** 1.8) - 1.0) * 0.40
    f_env[8] += hypoxia_strain * (1.0 - voc)

    drying = max(0.0, 0.40 - hum)
    f_env[5] += drying * 0.35 * voc
    f_env[4] += (hum - 0.40) * 0.30 * voc

    target_g7 = 0.40 * voc + 0.60 * (1.0 - voc)
    f_env[7] += -0.30 * (g[7] - target_g7)

    delta_p = float(transglottal_pressure(g, rho_air=rho))
    coronal = math.exp(-((float(g[3]) - 0.25) ** 2) / 0.03)
    groove = math.exp(-((float(g[2]) - 0.06) ** 2) / 0.015)
    f_env[2] += -0.55 * delta_p * (1.0 - voc) * coronal * groove * (float(g[2]) - 0.06)

    f_coartic = coarticulation_force(
        g,
        adjacent_p.g if adjacent_p is not None else None,
        next_p.g if next_p is not None else None,
        p.duration,
        adjacent_p.duration if adjacent_p is not None else 1.0,
        next_p.duration if next_p is not None else 1.0,
    )

    f_repulse = np.zeros(9, dtype=np.float32)
    if neighbor_g_pool:
        pool_mat = np.stack(neighbor_g_pool[:16], axis=0)[:, :9]
        diffs = g[np.newaxis, :] - pool_mat
        d_sqs = np.sum(diffs ** 2, axis=-1, keepdims=True)
        f_repulse = np.sum(diffs / (d_sqs + 0.05), axis=0) * 0.015

    w = np.clip(compute_metric_weights(g), 0.25, 40.0).astype(np.float32)
    grad_c = articulatory_cost_grad(g)
    grad_c[8] = 0.0                       # the larynx is already handled exactly by the OU step
    f_effort = -effort_gain * grad_c / w

    drift_sigma = 0.020 * math.sqrt(dt) * math.sqrt(max(0.5, word_cost / 10.0))
    xi = np.array([rng.gauss(0.0, 1.0) for _ in range(9)], dtype=np.float32)
    noise = drift_sigma * xi / np.sqrt(w)

    eff_bias = bias[:9] if bias is not None else np.zeros(9, dtype=np.float32)
    dg = (f_env + f_coartic + f_repulse + f_effort + eff_bias * 0.20) * dt + noise
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


def _unit_noise(rng, n: int) -> np.ndarray:
    if rng is not None and hasattr(rng, "normal"):
        v = np.asarray(rng.normal(0.0, 1.0, size=n), dtype=np.float32)
    elif rng is not None and hasattr(rng, "gauss"):
        v = np.array([rng.gauss(0.0, 1.0) for _ in range(n)], dtype=np.float32)
    else:
        v = np.random.default_rng().normal(0.0, 1.0, size=n).astype(np.float32)
    return v


def enforce_acoustic_contrast(
    words: list,
    min_auditory_margin: float = 0.32,
    ambient_noise: float = 0.10,
    rng=None,
) -> None:
    """
    Hard-margin contrast on the first vowel of each pair. Once sound_change_step is driving the lexicon, prefer it:
    distinctiveness then comes from the confusability energy over whole words, not from a fixed vowel margin.
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
                    h_hat = _unit_noise(rng, 9)
                    h_hat /= (np.linalg.norm(h_hat) + 1e-8)
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
        a_min = min(1.0, float(minimal_tract_aperture(g)))
        g8 = float(g[8])
        g7 = float(g[7])

        e_spring = 8.5 * ((g8 - 0.50) ** 2)

        is_ejective = (g8 > 0.70) and (g7 < 0.05) and voc < 0.5
        is_click = bool(is_click_gesture(g))

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
        if bool(is_click_gesture(p.g)):
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


# -----------------------------------------------------------------------------
# 6. ENERGY-DRIVEN SOUND CHANGE
#    E = sum_w f_w E_w                                  token energy (effort, air, environment, tone)
#      + lam_c * sum_{i<j} f_i f_j s_ij exp(-D_ij^2/2s^2)  confusability of whole words (distinctiveness)
#      + lam_i * sum_s (1 - exp(-n_s/n0)) c_s              inventory cost: every distinct segment pays a slot
# -----------------------------------------------------------------------------

_CLICK_SYMBOLS = {"ʘ", "ǀ", "ǃ", "ǂ", "ǁ"}
_V_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s in _VOWEL_SET], dtype=np.int64)
_C_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s not in _VOWEL_SET], dtype=np.int64)

# Non-pulmonic airstreams (clicks, ejectives, implosives) need an extra mechanism to learn and control.
_NONPULMONIC = np.array(
    [1.0 if (s in _CLICK_SYMBOLS or (a[8] > 0.70 and a[7] < 0.05) or a[8] < 0.25) else 0.0
     for s, a in zip(_ATTRACTOR_SYMBOLS, _ATTRACTOR_MATRIX)],
    dtype=np.float32,
)
_SLOT_COST = (0.60 + 1.50 * posture_effort(_ATTRACTOR_MATRIX) + 1.20 * _NONPULMONIC).astype(np.float32)


def ambient_from_env(env=None) -> float:
    """Assumed mapping from site environment to masking noise (vegetation and rough ground mask speech)."""
    if env is None:
        return 0.10
    veg = float(getattr(env, "vegetation", 0.3))
    rough = float(getattr(env, "roughness", 0.1))
    return float(np.clip(0.10 + 0.25 * veg + 0.25 * rough, 0.05, 0.60))


def readout_batch(G: np.ndarray, vmask: np.ndarray | None = None, chunk: int = 1500) -> np.ndarray:
    """Nearest-attractor symbol index for each row of G, with the same metric as resolve_tract_vector."""
    G = np.asarray(G, dtype=np.float32)[..., :9].reshape(-1, 9)
    if vmask is None:
        vmask = continuous_vocality(G) >= 0.5
    out = np.empty(len(G), dtype=np.int64)
    for idx, sel in ((_V_IDX, vmask), (_C_IDX, ~vmask)):
        rows = np.flatnonzero(sel)
        A = _ATTRACTOR_MATRIX[idx]
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            g = G[r][:, None, :]
            w = compute_metric_weights(0.5 * (A[None] + g))
            out[r] = idx[np.argmin(np.sum(w * (A[None] - g) ** 2, axis=-1), axis=1)]
    return out


def _collect_tokens(words: Sequence):
    refs, gs, fs = [], [], []
    for wi, w in enumerate(words):
        fw = float(getattr(w, "usage_frequency", 1.0))
        for pi, p in enumerate(w.phonemes):
            refs.append((wi, pi))
            gs.append(p.g)
            fs.append(fw)
    G = np.array(gs, dtype=np.float32).reshape(-1, 9)
    return refs, G, np.array(fs, dtype=np.float32)


def _inventory_cost(counts: np.ndarray, n0: float) -> float:
    return float(np.sum((1.0 - np.exp(-counts / n0)) * _SLOT_COST))


def inventory_counts(words: Sequence) -> np.ndarray:
    _, G, F = _collect_tokens(words)
    if len(G) == 0:
        return np.zeros(len(_ATTRACTOR_SYMBOLS), dtype=np.float32)
    return np.bincount(readout_batch(G), weights=F, minlength=len(_ATTRACTOR_SYMBOLS)).astype(np.float32)


def inventory_size(words: Sequence, min_share: float = 0.005) -> dict:
    """Distinct segments carrying at least min_share of usage mass, split into vowels / consonants / non-pulmonic."""
    n = inventory_counts(words)
    tot = float(n.sum())
    if tot <= 0:
        return {"vowels": 0, "consonants": 0, "nonpulmonic": 0, "total": 0}
    keep = n / tot >= min_share
    v = int(np.sum(keep[_V_IDX])); c = int(np.sum(keep[_C_IDX]))
    return {"vowels": v, "consonants": c, "nonpulmonic": int(np.sum(keep & (_NONPULMONIC > 0))), "total": v + c}


def inventory_energy(words: Sequence, lam_inv: float = 1.0, n0: float = 3.0) -> float:
    return lam_inv * _inventory_cost(inventory_counts(words), n0)


def _word_tensor(w) -> np.ndarray:
    return np.stack([p.g for p in w.phonemes], axis=0)


def word_confusability(word, others: Sequence, sigma: float, d_sem_scale: float = 0.35) -> float:
    """
    Mean over `others` of f_j s_ij exp(-D_ij^2 / 2 sigma^2), with D the whole-word trajectory distance and
    s_ij = 1 - exp(-(d_sem/d_sem_scale)^2), so synonyms may collide and distant meanings may not.
    """
    if not word.phonemes or not others:
        return 0.0
    gi = _word_tensor(word)
    tot = 0.0
    for o in others:
        if o is word or not o.phonemes:
            continue
        try:
            d_sem = float(word.vector.distance_to(o.vector))
        except Exception:
            d_sem = 1.0
        s_ij = 1.0 - math.exp(-((d_sem / d_sem_scale) ** 2))
        if s_ij < 1e-3:
            continue
        D = trajectory_distance(gi, _word_tensor(o))
        tot += float(getattr(o, "usage_frequency", 1.0)) * s_ij * math.exp(-(D ** 2) / (2.0 * sigma ** 2))
    return tot / max(1, len(others))


def _as_np_rng(rng):
    if hasattr(rng, "integers"):
        return rng
    if hasattr(rng, "getrandbits"):
        return np.random.default_rng(rng.getrandbits(32))
    return np.random.default_rng()


def sound_change_step(
    words: list,
    rng,
    T: float = 0.5,
    eps: float = 0.05,
    p_merge: float = 0.35,
    env=None,
    profile=None,
    lam_conf: float = 1.0,
    lam_inv: float | None = None,
    n0: float | None = None,
    ambient_noise: float | None = None,
    n_words_eval: int = 48,
    n_ref: int = 24,
    conf_scale: float = 50.0,
    shift_sigma: float = 0.03,
) -> dict:
    """
    One regular sound change on the whole lexicon: pick the phoneme class that a random token reads out as, then
      drift: shift every member by  -eps G^-1 grad C(mean) + shift_sigma G^-1/2 xi   (lenition / economy), or
      merge: translate every member by the displacement to the nearest same-type attractor.
    Accept with probability min(1, exp(-dE/T)), dE = d(token energy) + d(confusability) + d(inventory cost).
    Token energy and confusability are estimated on up to n_words_eval affected words against n_ref random words
    and scaled to all affected words; the inventory change is exact. This is annealing-style selection: the
    proposals are not symmetric, so detailed balance is not claimed.
    Returns a dict describing the proposal. Mutates `words` in place only when accepted.
    """
    rng = _as_np_rng(rng)
    refs, G, F = _collect_tokens(words)
    if len(G) == 0:
        return {"accepted": False, "kind": "none"}

    mass = float(F.sum())
    lam_inv = 0.04 * mass if lam_inv is None else lam_inv
    n0 = max(1.5, 0.01 * mass) if n0 is None else n0
    sigma = 0.25 + 0.5 * (ambient_from_env(env) if ambient_noise is None else ambient_noise)

    vmask = continuous_vocality(G) >= 0.5
    sym = readout_batch(G, vmask)
    counts = np.bincount(sym, weights=F, minlength=len(_ATTRACTOR_SYMBOLS)).astype(np.float32)

    s = int(sym[int(rng.integers(len(G)))])
    members = np.flatnonzero(sym == s)
    s_is_vowel = _ATTRACTOR_SYMBOLS[s] in _VOWEL_SET

    if rng.random() < p_merge:
        pool = _V_IDX if s_is_vowel else _C_IDX
        cand = pool[pool != s]
        d = sensory_distance(_ATTRACTOR_MATRIX[s][None, :], _ATTRACTOR_MATRIX[cand])
        order = np.argsort(d)[:3]
        pr = np.exp(-d[order]); pr /= pr.sum()
        t = int(cand[order[int(rng.choice(len(order), p=pr))]])
        shift = _ATTRACTOR_MATRIX[t] - _ATTRACTOR_MATRIX[s]
        kind, target = "merge", _ATTRACTOR_SYMBOLS[t]
    else:
        gbar = G[members].mean(axis=0)
        w = np.clip(compute_metric_weights(gbar), 0.25, 40.0)
        grad = articulatory_cost_grad(gbar)
        shift = -eps * grad / w + shift_sigma * rng.normal(0.0, 1.0, size=9) / np.sqrt(w)
        kind, target = "drift", None

    G_new = np.clip(G[members] + shift[None, :], 0.0, 1.0).astype(np.float32)

    # exact inventory change
    sym_new = readout_batch(G_new)
    counts_new = counts.copy()
    np.subtract.at(counts_new, sym[members], F[members])
    np.add.at(counts_new, sym_new, F[members])
    d_inv = lam_inv * (_inventory_cost(counts_new, n0) - _inventory_cost(counts, n0))

    # sampled token energy and confusability on affected words
    aff = sorted({refs[k][0] for k in members})
    sample = [aff[i] for i in rng.choice(len(aff), size=min(n_words_eval, len(aff)), replace=False)]
    scale = len(aff) / len(sample)
    ref_idx = rng.choice(len(words), size=min(n_ref, len(words)), replace=False)
    ref_words = [words[i] for i in ref_idx]

    def local_energy() -> tuple[float, float]:
        e_tok = e_conf = 0.0
        for wi in sample:
            w_ = words[wi]
            fw = float(getattr(w_, "usage_frequency", 1.0))
            e_tok += fw * word_energy(w_, profile=profile, env=env)
            e_conf += lam_conf * conf_scale * fw * word_confusability(w_, ref_words, sigma)
        return e_tok, e_conf

    tok0, conf0 = local_energy()

    undo = []
    for row, k in enumerate(members):
        wi, pi = refs[k]
        old = words[wi].phonemes[pi]
        undo.append((wi, pi, old))
        words[wi].phonemes[pi] = Phoneme(G_new[row], duration=old.duration)
    for wi in aff:
        if hasattr(words[wi], "_invalidate_cache"):
            words[wi]._invalidate_cache()

    tok1, conf1 = local_energy()
    dE = scale * ((tok1 - tok0) + (conf1 - conf0)) + d_inv
    accept = dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-6))

    if not accept:
        for wi, pi, old in undo:
            words[wi].phonemes[pi] = old
        for wi in aff:
            if hasattr(words[wi], "_invalidate_cache"):
                words[wi]._invalidate_cache()

    return {
        "accepted": bool(accept), "kind": kind, "symbol": _ATTRACTOR_SYMBOLS[s], "target": target,
        "n_tokens": int(len(members)), "n_words": len(aff),
        "dE": float(dE), "d_token": float(scale * (tok1 - tok0)), "d_conf": float(scale * (conf1 - conf0)),
        "d_inventory": float(d_inv),
    }