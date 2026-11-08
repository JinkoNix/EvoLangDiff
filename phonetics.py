"""
phonetics.py — Unified Continuous Articulatory Physics, Quantal Acoustics,
Thermodynamic Environmental Work, Metrical Syncope, and Organ-Specific OCP.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import NamedTuple, Sequence
from collections import Counter
import numpy as np

LONG_VOWEL_THRESHOLD = 1.35

# Articulatory Inertia Tensor M (tissue mass in grams / acceleration resistance)
# [Lip_D, Lip_P, Tip_D, Tip_L, Body_D, Body_L, Velum, Glottis]
INERTIA_DIAG = np.array([1.20, 0.60, 0.80, 0.40, 2.50, 2.00, 0.50, 1.00], dtype=np.float32)

_CHAO_BARS = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
_VOWEL_SET = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ'
}
_VOICELESS_STOPS = {'p', 't̪', 't', 'ʈ', 'c', 'k', 'q', 'ʔ'}
_LABIALS = {'p', 'b', 'm', 'ɸ', 'β', 'f', 'v', 'w', 'pʼ', 'ɓ', 'ʘ'}
_CLICK_SET = {'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ'}


# -----------------------------------------------------------------------------
# 1. QUANTAL METRICS & CONTINUOUS ACOUSTICS
# -----------------------------------------------------------------------------

def compute_metric_weights(g: np.ndarray) -> np.ndarray:
    """Computes diagonal weights W(g) of the Quantal metric tensor."""
    g0, g1, g2, g3 = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    g4, g5, g6, g7 = g[..., 4], g[..., 5], g[..., 6], g[..., 7]

    w0 = 1.0 + 25.0 / (1.0 + np.exp(np.clip(30.0 * (g0 - 0.08), -30.0, 30.0)))
    c_vowel = 1.0 / (1.0 + np.exp(np.clip(-25.0 * (g4 - 0.20), -30.0, 30.0)))
    w1 = 1.0 + 8.0 * c_vowel * (1.0 - g4) * (1.0 - 0.50 * g5)
    w2 = 1.0 + 50.0 / (1.0 + np.exp(np.clip(35.0 * (g2 - 0.06), -30.0, 30.0)))

    c_tip = 1.0 / (1.0 + np.exp(np.clip(25.0 * (g2 - 0.15), -30.0, 30.0)))
    w3 = 0.02 + c_tip * (2.0 + 16.0 * (1.0 - g3) ** 2)

    w4_stop = 30.0 / (1.0 + np.exp(np.clip(25.0 * (g4 - 0.10), -30.0, 30.0)))
    w4 = 1.0 + w4_stop + 6.0 * (1.0 - g4)
    w5 = 2.0 + 8.0 * ((g5 - 0.50) ** 2)
    w6 = 0.50 + 35.0 / (1.0 + np.exp(np.clip(40.0 * (g6 - 0.05), -30.0, 30.0)))

    w7_glo = 25.0 / (1.0 + np.exp(np.clip(30.0 * (g7 - 0.08), -30.0, 30.0)))
    w7_phon = 15.0 * np.exp(-((g7 - 0.40) ** 2) / 0.0128)
    w7 = 1.0 + w7_glo + w7_phon

    return np.stack([w0, w1, w2, w3, w4, w5, w6, w7], axis=-1)


def sensory_distance(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Riemannian Quantal distance: ds = sqrt(dg^T * W(g_mid) * dg)."""
    g_mid = 0.50 * (g1 + g2)
    w = compute_metric_weights(g_mid)
    diff_sq = (g1 - g2) ** 2
    return np.sqrt(np.sum(w * diff_sq, axis=-1))


def trajectory_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    """Length-independent Riemannian distance between two word articulatory matrices."""
    L1, L2 = len(g1), len(g2)
    if L1 == 0 or L2 == 0:
        return 1.0
    if L1 == L2:
        return float(np.mean(sensory_distance(g1, g2)))

    max_L = max(L1, L2)
    idx1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
    idx2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
    return float(np.mean(sensory_distance(g1[idx1], g2[idx2])))


def continuous_sonority(g: np.ndarray) -> float:
    """Computes radiated acoustic energy with nasal anti-formant dampening."""
    oral_conductance = min(float(g[0]), float(g[2]), float(g[4]))
    g7 = float(g[7])
    nasal_port = float(g[6])

    fric_aperture = (oral_conductance / 0.04) * math.exp(1.0 - (oral_conductance / 0.04)) if oral_conductance > 0.0 else 0.0
    flow_factor = (1.0 / (1.0 + math.exp(-25.0 * (g7 - 0.20)))) * (0.50 + 0.50 * g7)
    turbulence_noise = 1.20 * fric_aperture * flow_factor

    voicing_factor = math.exp(-((g7 - 0.40) ** 2) / 0.0128)
    exit_aperture = min(1.0, oral_conductance * 3.0 + nasal_port * 1.1)
    # Nasal cavity anti-formants absorb acoustic power relative to wide open oral radiated sound
    nasal_dampening = 1.0 - 0.35 * nasal_port
    voice_radiation = voicing_factor * (0.35 + 1.65 * exit_aperture) * nasal_dampening

    return round(1.0 + oral_conductance * 4.0 + voice_radiation + turbulence_noise, 3)


def vocalic_potential(g: np.ndarray) -> float | np.ndarray:
    """
    Continuous physical potential [0.0, 1.0] of unobstructed vocalic resonance.
    Replaces discrete thresholds with smooth aerodynamic laminar flow and modal phonation.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    if g_arr.ndim == 1:
        oral_ap = min(float(g_arr[0]), float(g_arr[2]), float(g_arr[4]))
        g7 = float(g_arr[7])
        laminar_flow = 1.0 / (1.0 + math.exp(-20.0 * (oral_ap - 0.16)))
        modal_voicing = math.exp(-((g7 - 0.40) ** 2) / 0.0288)
        return laminar_flow * modal_voicing
    else:
        oral_ap = np.min(g_arr[..., [0, 2, 4]], axis=-1)
        g7 = g_arr[..., 7]
        laminar_flow = 1.0 / (1.0 + np.exp(-20.0 * (oral_ap - 0.16)))
        modal_voicing = np.exp(-((g7 - 0.40) ** 2) / 0.0288)
        return laminar_flow * modal_voicing


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    """Statistical classification: True when vocalic resonance dominates over friction/closure."""
    pot = vocalic_potential(g)
    return pot >= 0.50


# -----------------------------------------------------------------------------
# 2. CONTINUOUS 9D PHONEME MODEL
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
        if len(v) == 8:
            v.append(float(duration) if duration is not None else 1.0)
        elif len(v) == 9 and duration is not None:
            v[8] = float(duration)

        g_clamped = [max(0.0, min(1.0, float(v[i]))) for i in range(8)]
        tau_clamped = max(0.08, min(3.0, float(v[8])))
        self._vector = np.array(g_clamped + [tau_clamped], dtype=np.float32)
        self._ipa = None
        self._energy = None
        self._hash = hash(tuple(int(x * 100) for x in self._vector))

    @property
    def vector(self) -> np.ndarray: return self._vector
    @property
    def g(self) -> np.ndarray: return self._vector[:8]
    @property
    def duration(self) -> float: return float(self._vector[8])
    @property
    def tau(self) -> float: return self.duration
    @property
    def is_vowel(self) -> bool: return is_vowel_gesture(self.g)

    @property
    def ipa(self) -> str:
        if self._ipa is None:
            base = resolve_tract_vector(self.g)
            self._ipa = f"{base}ː" if (self.is_vowel and self.duration >= LONG_VOWEL_THRESHOLD) else base
        return self._ipa

    @property
    def energy(self) -> PhonemeEnergy:
        if self._energy is None:
            g = self.g
            g0, g1, g2, g3, g4, g5, g6, g7 = g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]
            tau = self.duration
            lip_s = 0.40 * (1.0 - g0) + 0.15 * max(0.0, g1 - 0.50)
            sib_s = 0.50 * math.exp(-((g2 - 0.04) ** 2) / (2.0 * (0.015 ** 2)))
            tip_s = 0.35 * (1.0 - g2) + sib_s + 0.25 * max(0.0, g3 - 0.65)
            e_mot = round((0.10 + lip_s + tip_s + 0.40 * (1.0 - g4) + 0.55 * math.exp(-30.0 * g7)) * (tau ** 1.15), 3)

            oral_ap = min(g0, g2, g4)
            if g7 <= 0.10:
                air = 0.02
            elif oral_ap < 0.02:
                air = 0.18 if g7 >= 0.55 else 0.10
            else:
                flow = (0.22 + 0.35 / (1.0 + math.exp(-15.0 * (g7 - 0.50)))) * (1.80 if oral_ap <= 0.12 else 1.0) * (2.20 if g7 >= 0.78 else 1.0)
                air = round(flow * tau, 3)

            v_act = math.exp(-((g7 - 0.40) ** 2) / 0.0128)
            v_drag = round(v_act * math.exp(-15.0 * (oral_ap + 0.60 * g6)) * 0.70 * tau, 3)

            if self.is_vowel:
                # Liljencrants-Lindblom acoustic dispersion: peripheral vowels maximize contrast
                disp = math.sqrt((g4 - 0.60) ** 2 + 2.5 * (g5 - 0.50) ** 2)
                aud = round(0.50 + 0.50 * math.tanh(3.5 * disp), 3)
            else:
                aud = round(0.80 if sib_s > 0.10 else (0.55 if oral_ap < 0.02 else 0.40), 3)

            self._energy = PhonemeEnergy(e_mot, air, v_drag, aud)
        return self._energy

    def drift(self, delta: Sequence[float] | None = None, target: Sequence[float] | None = None) -> Phoneme:
        if target is not None:
            return Phoneme(target)
        if delta is not None:
            new_v = self._vector.copy()
            for i in range(min(len(delta), 9)):
                new_v[i] += float(delta[i])
            return Phoneme(new_v)
        return Phoneme(self._vector.copy())

    def __hash__(self) -> int: return self._hash
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phoneme) and np.array_equal(self._vector, other._vector)
    def __repr__(self) -> str: return f"Phoneme(/{self.ipa}/, tau={self.duration:.2f})"


# -----------------------------------------------------------------------------
# 3. GRADIENT DESCENT ON THE SYSTEMIC HAMILTONIAN
# -----------------------------------------------------------------------------

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
    rng: random.Random,
) -> Phoneme:
    """Autonomous gradient flow on the thermodynamic articulatory landscape."""
    g = p.g.copy()
    is_voc = float(is_vowel_gesture(g))

    drift_scale = min(2.5, max(0.1, t_drift))
    noise = np.array([rng.gauss(0.0, 0.020 * drift_scale) for _ in range(8)], dtype=np.float32)

    # Pulmonary, altitude, and climatic forces
    rho = math.exp(-0.90 * max(0.0, alt))
    pulmonary_tax = 1.0 + 3.2 / (1.0 + math.exp(12.0 * (rho - 0.72))) + 1.6 / (1.0 + math.exp(10.0 * (hum - 0.35)))
    chill_exposure = g[4] * (1.0 / (1.0 + math.exp(10.0 * (temp - 0.50)))) * (1.0 / (1.0 + math.exp(10.0 * (hum - 0.40)))) * 2.8
    radiation_payoff = ((g[4] * g[0]) ** 1.4) * (0.20 + 0.25 * hum + 0.15 * temp)

    f_aero = np.zeros(8, dtype=np.float32)
    f_aero[4] = (radiation_payoff - chill_exposure) * 0.10 * is_voc

    oral_ap = min(g[0], g[2], g[4])
    stop_density = math.exp(-(oral_ap ** 2) / 0.0032) * (1.0 - is_voc)
    f_aero[7] = -(1.0 / (1.0 + math.exp(-4.5 * (pulmonary_tax - 2.4)))) * g[7] * 0.20 * stop_density

    # Physiological coarticulation
    f_coartic = np.zeros(8, dtype=np.float32)
    if adjacent_p is not None:
        adj = adjacent_p.g
        w_lab = math.exp(-(adj[0] ** 2) / 0.0128)
        w_cor = math.exp(-(adj[2] ** 2) / 0.0128)
        w_dor = math.exp(-(adj[4] ** 2) / 0.0128)

        f_coartic[1] += is_voc * w_lab * 0.12 * (0.85 - g[1])
        f_coartic[5] += is_voc * (w_lab * 0.10 * (0.85 - g[5]) + w_cor * 0.14 * (0.10 - g[5]) + w_dor * 0.12 * (0.80 - g[5]))
        f_coartic[4] += is_voc * w_cor * 0.06 * (0.42 - g[4])

    # Coulomb repulsion across neighbor pool (vectorized)
    f_repulse = np.zeros(8, dtype=np.float32)
    if neighbor_g_pool:
        pool_mat = np.stack(neighbor_g_pool, axis=0)
        diffs = g[np.newaxis, :] - pool_mat
        d_sqs = np.sum(diffs ** 2, axis=-1, keepdims=True)

        if is_voc > 0.5:
            voc_mask = np.array([is_vowel_gesture(ng) for ng in neighbor_g_pool], dtype=np.float32)[:, np.newaxis]
            w_rep = voc_mask * (diffs / (d_sqs + 0.015))
            f_repulse[4] += float(np.sum(w_rep[:, 4])) * 0.08
            f_repulse[5] += float(np.sum(w_rep[:, 5])) * 0.10
            f_repulse[1] += float(np.sum(w_rep[:, 1])) * 0.06
        else:
            w_rep = diffs / (d_sqs + 0.025)
            f_repulse += np.sum(w_rep, axis=0) * 0.035

    # Empirical Vowel Dispersion: pushes vowels outward from the central murmur basin
    f_disp = np.zeros(8, dtype=np.float32)
    if is_voc > 0.5:
        d_h = g[4] - 0.60
        d_b = g[5] - 0.50
        dist_c = math.sqrt(d_h ** 2 + d_b ** 2)
        if dist_c < 0.32:
            scale = (0.32 - dist_c) * 0.55
            f_disp[4] = (d_h / (dist_c + 0.01)) * scale
            f_disp[5] = (d_b / (dist_c + 0.01)) * scale

    dg = noise + f_aero + f_coartic + f_repulse + f_disp + (bias * 0.25)
    g_new = np.clip(g + dg, 0.0, 1.0)

    if is_voc > 0.5:
        g_new[4] = np.clip(g_new[4], 0.25, 0.96)
        g_new[5] = np.clip(g_new[5], 0.04, 0.96)
        lip_pull = 1.0 / (1.0 + math.exp(-10.0 * (g_new[1] - 0.45)))
        g_new[0] = (1.0 - lip_pull) * 1.00 + lip_pull * 0.35

    return Phoneme(g_new, duration=p.duration)


def apply_syncope_and_clustering(
    phonemes: list[Phoneme],
    usage_frequency: float,
    alt: float,
    hum: float,
    temp: float,
    profile=None,
) -> list[Phoneme]:
    """Applies metrical vowel deletion under environmental stress and frequency load."""
    L = len(phonemes)
    if L <= 2:
        return phonemes

    h = getattr(profile, "head_directionality", 0.50) if profile else 0.50
    X = np.stack([p.vector for p in phonemes], axis=0)
    syl_slices, stressed_idx, _, _ = parse_sonority_prosody(X, directional_bias=h)

    rho = math.exp(-0.90 * max(0.0, alt))
    pulmonary_tax = 1.0 + 3.0 / (1.0 + math.exp(12.0 * (rho - 0.75))) + 1.8 / (1.0 + math.exp(10.0 * (hum - 0.40)))
    thermal_chill = 1.5 / (1.0 + math.exp(10.0 * (temp - 0.50)))
    vowel_drain = 0.10 * math.log1p(usage_frequency) * (pulmonary_tax + thermal_chill) * ((L / 4.0) ** 0.55)

    eroded: list[Phoneme] = []
    for k, (st, en) in enumerate(syl_slices):
        is_stressed = (k == stressed_idx)
        for j in range(st, en):
            p = phonemes[j]
            tau = p.duration
            if not is_stressed and p.is_vowel:
                tau -= vowel_drain
                p = Phoneme(p.g, duration=tau)

            # Energy-based ballistic threshold for deletion
            if tau > 0.15 or (not p.is_vowel and tau > 0.10):
                eroded.append(p)

    if not any(p.is_vowel for p in eroded):
        return phonemes
    return repair_phonemes(eroded, profile=profile)


# -----------------------------------------------------------------------------
# 4. PROSODY, HIATUS RESOLUTION & SYLLABLES
# -----------------------------------------------------------------------------

def resolve_chao_tone(pitch: tuple[float, float, float] | np.ndarray, duration: float = 1.0) -> str:
    f0, f1, f2 = float(pitch[0]), float(pitch[1]), float(pitch[2])
    c0 = int(np.clip(np.floor(f0 * 2.50 + 3.50), 1, 5))
    c1 = int(np.clip(np.floor(f1 * 2.50 + 3.50), 1, 5))
    c2 = int(np.clip(np.floor(f2 * 2.50 + 3.50), 1, 5))

    if duration < 0.85:
        return _CHAO_BARS[c1]
    if (c1 < c0 and c1 < c2) or ((c1 > c0 and c1 > c2) and (c1 - min(c0, c2) >= 2)):
        return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c1]}{_CHAO_BARS[c2]}"
    if c0 == c2:
        return _CHAO_BARS[c1]
    return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c2]}"


def parse_sonority_prosody(X: np.ndarray, directional_bias: float = 0.50) -> tuple[list[tuple[int, int]], int, list[tuple[float, float, float]], list[float]]:
    L = len(X)
    if L == 0:
        return [], 0, [], []
    son = np.array([continuous_sonority(X[i, :8]) for i in range(L)], dtype=np.float32)
    v_mask = np.array([is_vowel_gesture(X[i, :8]) for i in range(L)], dtype=bool)

    peaks = [i for i in range(L) if v_mask[i] and (i == 0 or son[i] >= son[i - 1]) and (i == L - 1 or son[i] >= son[i + 1])]
    if not peaks:
        return [(0, L)], 0, [(0.0, 0.0, 0.0)], [1.0]

    boundaries = [0]
    for k in range(len(peaks) - 1):
        p_start, p_end = peaks[k] + 1, peaks[k + 1]
        if p_start == p_end:
            boundaries.append(p_start)
        else:
            trough_offset = int(np.argmin(son[p_start:p_end]))
            boundaries.append(p_start + trough_offset)
    boundaries.append(L)
    syl_slices = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]

    syl_masses = [sum(X[i, 8] * son[i] for i in range(st, en)) * (1.0 + (k / max(1, len(syl_slices) - 1)) * (directional_bias - 0.50) * 0.40) for k, (st, en) in enumerate(syl_slices)]
    stressed_idx = int(np.argmax(syl_masses))

    trajectories = []
    nuc_durs = []
    for k, (st, en) in enumerate(syl_slices):
        pk = peaks[k]
        f0 = 2.20 * (float(X[pk - 1, 7]) - 0.50) if pk > st else 0.0
        f1 = f0 + (0.30 if k == stressed_idx else 0.0)
        f2 = f0
        if en > pk + 1:
            g7, tau = float(X[pk + 1, 7]), float(X[pk + 1, 8])
            if g7 <= 0.15:
                f2 = min(1.0, f0 + 0.80 * max(0.1, 1.20 - tau))
            elif g7 >= 0.65:
                f2 = max(-1.0, f0 - 0.80 * max(0.1, 1.20 - tau))
        tau_nuc = float(X[pk, 8])
        if tau_nuc >= 1.30 and f2 > f0:
            f1 = min(f0, f2) - 0.55 * (tau_nuc - 0.80) + (0.20 if k == stressed_idx else 0.0)
        trajectories.append((round(max(-1.0, min(1.0, f0)), 3), round(max(-1.0, min(1.0, f1)), 3), round(max(-1.0, min(1.0, f2)), 3)))
        nuc_durs.append(tau_nuc)

    return syl_slices, stressed_idx, trajectories, nuc_durs


class Syllable:
    __slots__ = ('onset', 'nucleus', 'coda', 'stressed', 'pitch')

    def __init__(
        self,
        onset: Sequence[Phoneme] = (),
        nucleus: Sequence[Phoneme] | Phoneme | None = None,
        coda: Sequence[Phoneme] = (),
        stressed: bool = False,
        pitch: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.onset = list(onset)
        self.nucleus = [nucleus] if isinstance(nucleus, Phoneme) else list(nucleus or ())
        self.coda = list(coda)
        self.stressed = stressed
        self.pitch = tuple(float(x) for x in pitch)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.nucleus)

    @property
    def mass(self) -> float:
        return sum(p.duration * p.energy.audibility_payoff * (1.4 if p.is_vowel else 0.8) for p in self.to_phonemes())

    def to_phonemes(self) -> list[Phoneme]:
        return self.onset + self.nucleus + self.coda

    @property
    def form(self) -> str:
        pref = "ˈ" if self.stressed else ""
        s = f"{pref}{''.join(p.ipa for p in self.onset)}{''.join(p.ipa for p in self.nucleus)}{''.join(p.ipa for p in self.coda)}"
        tone = resolve_chao_tone(self.pitch, duration=self.duration) if self.nucleus else ""
        return f"{s}{tone}" if any(abs(x) > 0.15 for x in self.pitch) else s


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
                nucleus=syl_p[first_v : last_v + 1],
                coda=syl_p[last_v + 1 :],
                stressed=(k == stressed_idx),
                pitch=pitches[k],
            ))
    return syls


def assign_stress(syllables: list[Syllable], directional_bias: float = 0.50) -> None:
    if not syllables:
        return
    for s in syllables:
        s.stressed = False
    if len(syllables) == 1:
        syllables[0].stressed = True
        return
    n = len(syllables)
    scores = [s.mass * (1.0 + (i / (n - 1)) * (directional_bias - 0.50) * 0.40) for i, s in enumerate(syllables)]
    syllables[int(np.argmax(scores))].stressed = True


def apply_prosody(word, directional_bias: float = 0.50, syllable_complexity: float = 0.50):
    assign_stress(word.syllables, directional_bias=directional_bias)
    return word


# -----------------------------------------------------------------------------
# 5. KINETIC WORK & ORGAN-SPECIFIC OCP
# -----------------------------------------------------------------------------

def transition_cost(p1: Phoneme | np.ndarray, p2: Phoneme | np.ndarray, tau1: float = 1.0, tau2: float = 1.0) -> float:
    """
    Computes kinetic motor work plus organ-specific OCP contact penalties.
    Heterorganic gestures (/pt/, /km/) move freely; homorganic sequences (/td/, /ɐə/) are taxed.
    """
    g1 = p1.g if hasattr(p1, "g") else p1[:8]
    g2 = p2.g if hasattr(p2, "g") else p2[:8]
    t1 = p1.duration if hasattr(p1, "duration") else tau1
    t2 = p2.duration if hasattr(p2, "duration") else tau2

    diff = g1 - g2
    min_tau = max(0.18, min(t1, t2))
    kinetic = np.sum(INERTIA_DIAG * (diff ** 2)) / (min_tau ** 0.80)

    v1, v2 = is_vowel_gesture(g1), is_vowel_gesture(g2)
    dist = sensory_distance(g1, g2)

    # Organ-Specific OCP overlap
    labial_overlap = max(0.0, 0.25 - max(g1[0], g2[0])) * 4.0
    coronal_overlap = max(0.0, 0.25 - max(g1[2], g2[2])) * 4.0
    dorsal_overlap = max(0.0, 0.25 - max(g1[4], g2[4])) * 4.0

    stutter_tax = 0.0
    if v1 and v2:
        # Vowel hiatus stutter penalty (penalizes consecutive flat vowels like /ɐə/)
        stutter_tax = 2.50 * math.exp(-3.5 * max(0.0, float(dist) - 0.10))
    elif not v1 and not v2:
        # Consonant homorganic contact penalty
        homorganic_factor = max(labial_overlap, coronal_overlap, dorsal_overlap)
        stutter_tax = 2.50 * homorganic_factor * math.exp(-3.0 * float(dist))

    if v1 and v2:
        cost = kinetic + stutter_tax + 0.45
    elif v1 != v2:
        cost = kinetic * 0.35
    else:
        cost = kinetic + stutter_tax
    return round(float(cost), 3)


def repair_phonemes(phonemes: list[Phoneme], profile=None, max_passes: int = 4) -> list[Phoneme]:
    """Phonotactic repair engine: resolves hiatus and excessive consonant clusters."""
    s_complex = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    max_cost = 1.80 + (4.50 * s_complex)

    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(phonemes) - 1:
            p1, p2 = phonemes[i], phonemes[i + 1]

            # 1. Hiatus resolution
            if p1.is_vowel and p2.is_vowel:
                dist = sensory_distance(p1.g, p2.g)
                if dist < 0.35:
                    # Coalesce near-identical vowels into a single lengthened vowel
                    coalesced_dur = min(2.0, p1.duration + p2.duration * 0.60)
                    coalesced_g = 0.50 * (p1.g + p2.g)
                    phonemes[i] = Phoneme(coalesced_g, duration=coalesced_dur)
                    phonemes.pop(i + 1)
                    changed = True
                    continue
                else:
                    # Dissimilar vowels receive an organic intervening glide derived from V1
                    glide_g = p1.g.copy()
                    glide_g[4] = 0.20
                    if glide_g[5] > 0.55:  # Back vowel triggers labiovelar glide [w]
                        glide_g[0] = 0.15
                        glide_g[1] = 1.00
                    else:  # Front/central vowel triggers palatal glide [j]
                        glide_g[0] = 1.00
                        glide_g[1] = 0.00
                        glide_g[5] = 0.05
                    phonemes.insert(i + 1, Phoneme(glide_g, duration=0.45))
                    changed = True
                    i += 2
                    continue

            # 2. Harmonic consonant cluster epenthesis
            if not p1.is_vowel and not p2.is_vowel:
                if transition_cost(p1, p2) > max_cost:
                    # Inherit place harmony from the preceding consonant
                    if p1.g[5] < 0.30 or p1.g[3] < 0.20:
                        # Palatal/coronal context -> front [i]-like epenthesis
                        ep_g = np.array([1.0, 0.0, 1.0, 0.0, 0.35, 0.10, 0.0, 0.40], dtype=np.float32)
                    elif p1.g[0] < 0.20 or p1.g[5] > 0.70:
                        # Labial/velar context -> back [u]-like epenthesis
                        ep_g = np.array([0.35, 0.9, 1.0, 0.0, 0.35, 0.85, 0.0, 0.40], dtype=np.float32)
                    else:
                        # Neutral context -> central [ə]
                        ep_g = np.array([1.0, 0.0, 1.0, 0.25, 0.55, 0.50, 0.0, 0.40], dtype=np.float32)

                    phonemes.insert(i + 1, Phoneme(ep_g, duration=0.65))
                    changed = True
                    i += 2
                    continue

            i += 1
        if not changed:
            break

    return phonemes


def word_energy(
    word,
    profile=None,
    env=None,
    cultural_attention: Sequence[float] | None = None,
    cohort_sample: Sequence = None,
) -> float:
    """
    Unified Thermodynamic Cost:
      E_total = E_biomechanical + E_temporal_modulation + E_ambiguity
    """
    phonemes = word.phonemes
    n = len(phonemes)
    if n == 0:
        return 0.0
    attn = cultural_attention or getattr(profile, "cultural_attention", [1.0] * 7)
    alt = getattr(env, "altitude", 0.0) if env else 0.0
    temp = getattr(env, "temperature", 0.5) if env else 0.5
    hum = getattr(env, "humidity", 0.5) if env else 0.5

    rho = math.exp(-0.90 * max(0.0, alt))
    air_cold = max(0.0, 0.45 - temp)
    aridity = max(0.0, 0.45 - hum)
    pulmonary_tax = 1.0 + max(0.0, (1.0 - rho) - 0.20) * 2.5 + aridity * 1.2

    # 1. Biomechanical work
    biomech_joules = 0.0
    for p in phonemes:
        e = p.energy
        net = (e.motor_effort * (1.0 + air_cold * 0.8) + e.air_consumption * pulmonary_tax + e.voicing_drag * 1.4) - (e.audibility_payoff * 0.45)
        biomech_joules += max(0.05, net)

    tempo = 1.0 + 0.30 * max(0.0, attn[4] - 1.0)
    for j in range(n - 1):
        biomech_joules += transition_cost(phonemes[j], phonemes[j + 1]) * 0.50 * tempo

    s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
    synthesis_tolerance = 1.0 - 0.75 * s_idx
    freq_load = math.log(getattr(word, "usage_frequency", 1.0) + 1.0)
    biomech_joules += round(0.10 * (n ** 1.20) * synthesis_tolerance * (0.60 + 0.40 * freq_load), 3)

    # 2. Intra-word temporal modulation cost: penalizes flat vocalic murmurs
    sonorities = [continuous_sonority(p.g) for p in phonemes]
    sonority_variance = float(np.var(sonorities)) if len(sonorities) > 1 else 1.0
    modulation_tax = 2.00 / (sonority_variance + 0.30)

    # 3. Acoustic ambiguity cost against cohort
    ambiguity_tax = 0.0
    if cohort_sample and len(cohort_sample) > 1:
        w_tensor = word.tensor[:, :8]
        w_vec = word.vector
        for other in cohort_sample[:30]:
            if other.id == word.id or len(other.phonemes) == 0:
                continue
            d_acoust = trajectory_distance(w_tensor, other.tensor[:, :8])
            d_sem = w_vec.distance_to(other.vector)
            ambiguity_tax += (d_sem ** 2) / (d_acoust ** 2 + 0.06)
        ambiguity_tax = 0.12 * ambiguity_tax

    return round(float(biomech_joules + modulation_tax + ambiguity_tax), 3)


# -----------------------------------------------------------------------------
# 6. SERIALIZATION & ATTRACTOR QUANTIZATION
# -----------------------------------------------------------------------------

def render_and_extract_tones(
    X: np.ndarray,
    directional_bias: float = 0.50,
    is_tonal: bool = False,
) -> tuple[str, str, str]:
    L = len(X)
    if L == 0:
        return "", "", ""

    syl_slices, stressed_idx, pitches, _ = parse_sonority_prosody(X, directional_bias=directional_bias)
    parts = []
    plain_parts = []
    assigned_tone = ""

    def quantize_tone(p_traj: tuple[float, float, float]) -> str:
        f0, f1, f2 = p_traj[0], p_traj[1], p_traj[2]
        delta = f2 - f0
        if delta <= -0.28:
            return "˥˩"
        elif delta >= 0.28:
            return "˩˥"
        elif f1 < min(f0, f2) - 0.18:
            return "˨˩˦"
        elif f0 >= 0.22:
            return "˥"
        elif f0 <= -0.22:
            return "˩"
        return "˧"

    for k, (st, en) in enumerate(syl_slices):
        if k == stressed_idx and len(syl_slices) > 1:
            parts.append("ˈ")

        syl_symbols = [resolve_tract_vector(X[i, :8]) for i in range(st, en)]
        coalesced: list[str] = []
        for sym in syl_symbols:
            if coalesced and coalesced[-1] == sym:
                if sym in _VOWEL_SET:
                    if not coalesced[-1].endswith("ː"):
                        coalesced[-1] = f"{coalesced[-1]}ː"
                    continue
                else:
                    continue
            coalesced.append(sym)

        syl_str = "".join(coalesced)
        plain_parts.append(syl_str.replace("ː", ""))

        if is_tonal and (k == stressed_idx or len(syl_slices) == 1):
            tone_cat = quantize_tone(pitches[k])
            syl_str = f"{syl_str}{tone_cat}"
            assigned_tone = tone_cat

        parts.append(syl_str)

    return "".join(parts), "".join(plain_parts), assigned_tone


def cluster_contrastive_phonemes(words: Sequence) -> tuple[list[str], list[str]]:
    v_counts = Counter()
    c_counts = Counter()

    for w in words:
        for p in w.phonemes:
            clean = p.ipa.replace("ː", "").replace("˥", "").replace("˦", "").replace("˧", "").replace("˨", "").replace("˩", "")
            if p.is_vowel:
                v_counts[clean] += w.usage_frequency
            else:
                c_counts[clean] += w.usage_frequency

    tot_v = sum(v_counts.values()) or 1.0
    tot_c = sum(c_counts.values()) or 1.0

    v_thresh = max(0.02, 0.40 / (len(v_counts) + 1.0))
    c_thresh = max(0.015, 0.35 / (len(c_counts) + 1.0))

    contrastive_v = sorted([v for v, cnt in v_counts.items() if (cnt / tot_v) >= v_thresh])
    contrastive_c = sorted([c for c, cnt in c_counts.items() if (cnt / tot_c) >= c_thresh])

    if not contrastive_v and v_counts:
        contrastive_v = [max(v_counts, key=v_counts.get)]
    if not contrastive_c and c_counts:
        contrastive_c = [max(c_counts, key=c_counts.get)]

    return contrastive_v, contrastive_c


def render_word_trajectory(X: np.ndarray, directional_bias: float = 0.50) -> tuple[str, str]:
    form_w, plain_w, _ = render_and_extract_tones(X, directional_bias=directional_bias, is_tonal=False)
    return form_w, plain_w


# -----------------------------------------------------------------------------
# 7. EXPANDED INTERNATIONAL PHONETIC ATTRACTOR MANIFOLD
# -----------------------------------------------------------------------------

_CANONICAL_ATTRACTORS: list[tuple[str, list[float]]] = [
    # --- VOWELS (Full Quadrilateral) ---
    # Close
    ("i",  [1.00, 0.0, 1.0, 0.0, 0.28, 0.05, 0.0, 0.40]), ("y",  [0.35, 1.0, 1.0, 0.0, 0.28, 0.05, 0.0, 0.40]),
    ("ɨ",  [1.00, 0.0, 1.0, 0.0, 0.30, 0.50, 0.0, 0.40]), ("ʉ",  [0.40, 0.8, 1.0, 0.0, 0.30, 0.50, 0.0, 0.40]),
    ("ɯ",  [1.00, 0.0, 1.0, 0.0, 0.30, 0.92, 0.0, 0.40]), ("u",  [0.30, 1.0, 1.0, 0.0, 0.28, 0.92, 0.0, 0.40]),
    # Near-Close
    ("ɪ",  [1.00, 0.0, 1.0, 0.0, 0.42, 0.15, 0.0, 0.40]), ("ʏ",  [0.40, 0.8, 1.0, 0.0, 0.42, 0.15, 0.0, 0.40]),
    ("ʊ",  [0.40, 0.8, 1.0, 0.0, 0.42, 0.80, 0.0, 0.40]),
    # Close-Mid
    ("e",  [1.00, 0.0, 1.0, 0.0, 0.52, 0.10, 0.0, 0.40]), ("ø",  [0.40, 0.8, 1.0, 0.0, 0.52, 0.10, 0.0, 0.40]),
    ("ɘ",  [1.00, 0.0, 1.0, 0.0, 0.52, 0.50, 0.0, 0.40]), ("ɵ",  [0.40, 0.8, 1.0, 0.0, 0.52, 0.50, 0.0, 0.40]),
    ("ɤ",  [1.00, 0.0, 1.0, 0.0, 0.52, 0.90, 0.0, 0.40]), ("o",  [0.35, 0.9, 1.0, 0.0, 0.52, 0.90, 0.0, 0.40]),
    # Mid
    ("ə",  [1.00, 0.0, 1.0, 0.0, 0.60, 0.50, 0.0, 0.40]),
    # Open-Mid
    ("ɛ",  [1.00, 0.0, 1.0, 0.0, 0.75, 0.15, 0.0, 0.40]), ("œ",  [0.50, 0.7, 1.0, 0.0, 0.75, 0.15, 0.0, 0.40]),
    ("ɜ",  [1.00, 0.0, 1.0, 0.0, 0.75, 0.50, 0.0, 0.40]), ("ɞ",  [0.40, 0.8, 1.0, 0.0, 0.75, 0.50, 0.0, 0.40]),
    ("ʌ",  [1.00, 0.0, 1.0, 0.0, 0.75, 0.85, 0.0, 0.40]), ("ɔ",  [0.40, 0.9, 1.0, 0.0, 0.75, 0.90, 0.0, 0.40]),
    # Near-Open & Open
    ("æ",  [1.00, 0.0, 1.0, 0.0, 0.88, 0.15, 0.0, 0.40]), ("ɐ",  [1.00, 0.0, 1.0, 0.0, 0.85, 0.50, 0.0, 0.40]),
    ("a",  [1.00, 0.0, 1.0, 0.0, 0.95, 0.40, 0.0, 0.40]), ("ɶ",  [0.40, 0.8, 1.0, 0.0, 0.95, 0.15, 0.0, 0.40]),
    ("ɑ",  [1.00, 0.0, 1.0, 0.0, 0.95, 0.85, 0.0, 0.40]), ("ɒ",  [0.40, 0.8, 1.0, 0.0, 0.95, 0.85, 0.0, 0.40]),

    # --- PLOSIVES ---
    ("p",  [0.00, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.60]), ("b",  [0.00, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("t̪",  [1.00, 0.0, 0.0, 0.05, 0.80, 0.50, 0.0, 0.60]), ("d̪",  [1.00, 0.0, 0.0, 0.05, 0.80, 0.50, 0.0, 0.40]),
    ("t",  [1.00, 0.0, 0.0, 0.25, 0.80, 0.50, 0.0, 0.60]), ("d",  [1.00, 0.0, 0.0, 0.25, 0.80, 0.50, 0.0, 0.40]),
    ("ʈ",  [1.00, 0.0, 0.0, 0.95, 0.80, 0.50, 0.0, 0.60]), ("ɖ",  [1.00, 0.0, 0.0, 0.95, 0.80, 0.50, 0.0, 0.40]),
    ("c",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.05, 0.0, 0.60]), ("ɟ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.05, 0.0, 0.40]),
    ("k",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.75, 0.0, 0.60]), ("ɡ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.75, 0.0, 0.40]),
    ("q",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.98, 0.0, 0.60]), ("ɢ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.98, 0.0, 0.40]),
    ("ʔ",  [1.00, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.02]),

    # --- FRICATIVES ---
    ("ɸ",  [0.05, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.65]), ("β",  [0.05, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("f",  [0.08, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.65]), ("v",  [0.08, 0.0, 1.0, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("θ",  [1.00, 0.0, 0.06, 0.05, 0.80, 0.50, 0.0, 0.65]), ("ð",  [1.00, 0.0, 0.06, 0.05, 0.80, 0.50, 0.0, 0.40]),
    ("s",  [1.00, 0.0, 0.03, 0.25, 0.80, 0.50, 0.0, 0.65]), ("z",  [1.00, 0.0, 0.03, 0.25, 0.80, 0.50, 0.0, 0.40]),
    ("ʃ",  [1.00, 0.5, 0.04, 0.55, 0.50, 0.20, 0.0, 0.65]), ("ʒ",  [1.00, 0.5, 0.04, 0.55, 0.50, 0.20, 0.0, 0.40]),
    ("ʂ",  [1.00, 0.0, 0.04, 0.95, 0.50, 0.50, 0.0, 0.65]), ("ʐ",  [1.00, 0.0, 0.04, 0.95, 0.50, 0.50, 0.0, 0.40]),
    ("ç",  [1.00, 0.0, 1.00, 0.00, 0.10, 0.05, 0.0, 0.65]), ("ʝ",  [1.00, 0.0, 1.00, 0.00, 0.10, 0.05, 0.0, 0.40]),
    ("x",  [1.00, 0.0, 1.00, 0.00, 0.12, 0.75, 0.0, 0.65]), ("ɣ",  [1.00, 0.0, 1.00, 0.00, 0.12, 0.75, 0.0, 0.40]),
    ("χ",  [1.00, 0.0, 1.00, 0.00, 0.12, 0.98, 0.0, 0.65]), ("ʁ",  [1.00, 0.0, 1.00, 0.00, 0.12, 0.98, 0.0, 0.40]),
    ("ħ",  [1.00, 0.0, 1.00, 0.00, 0.25, 1.00, 0.0, 0.85]), ("ʕ",  [1.00, 0.0, 1.00, 0.00, 0.25, 1.00, 0.0, 0.45]),
    ("h",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.90]), ("ɦ",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.55]),

    # --- NASALS ---
    ("m",  [0.00, 0.0, 1.0, 0.00, 0.80, 0.50, 0.60, 0.40]),
    ("n̪",  [1.00, 0.0, 0.0, 0.05, 0.80, 0.50, 0.60, 0.40]),
    ("n",  [1.00, 0.0, 0.0, 0.25, 0.80, 0.50, 0.60, 0.40]),
    ("ɳ",  [1.00, 0.0, 0.0, 0.95, 0.80, 0.50, 0.60, 0.40]),
    ("ɲ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.05, 0.60, 0.40]),
    ("ŋ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.75, 0.60, 0.40]),
    ("ɴ",  [1.00, 0.0, 1.0, 0.00, 0.00, 0.98, 0.60, 0.40]),

    # --- APPROXIMANTS & LIQUIDS ---
    ("ʋ",  [0.20, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("ɹ",  [1.00, 0.0, 0.18, 0.25, 0.70, 0.30, 0.0, 0.40]),
    ("j",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.05, 0.0, 0.40]),
    ("ɰ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.85, 0.0, 0.40]),
    ("w",  [0.15, 1.0, 1.00, 0.00, 0.20, 0.92, 0.0, 0.40]),
    ("l",  [1.00, 0.0, 0.25, 0.25, 0.70, 0.30, 0.0, 0.40]),
    ("ɭ",  [1.00, 0.0, 0.25, 0.95, 0.70, 0.30, 0.0, 0.40]),
    ("ʎ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.05, 0.0, 0.40]),
    ("r",  [1.00, 0.0, 0.02, 0.25, 0.80, 0.50, 0.0, 0.40]),
    ("ɾ",  [1.00, 0.0, 0.06, 0.25, 0.80, 0.50, 0.0, 0.40]),

    # --- NON-PULMONIC (Ejectives, Implosives, Clicks) ---
    ("pʼ", [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.02]),
    ("tʼ", [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.02]),
    ("kʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.02]),
    ("qʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.98, 0.0, 0.02]),
    ("sʼ", [1.00, 0.0, 0.03, 0.25, 0.80, 0.50, 0.0, 0.02]),
    ("ɓ",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.15]),
    ("ɗ",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.15]),
    ("ɠ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.15]),
    ("ʛ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.98, 0.0, 0.15]),
    ("ʘ",  [0.00, 0.0, 1.00, 0.00, 0.00, 0.85, 0.0, 0.04]),
    ("ǀ",  [1.00, 0.0, 0.06, 0.05, 0.00, 0.85, 0.0, 0.04]),
    ("ǃ",  [1.00, 0.0, 0.00, 0.25, 0.00, 0.85, 0.0, 0.04]),
    ("ǂ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.15, 0.0, 0.04]),
    ("ǁ",  [1.00, 0.0, 0.25, 0.25, 0.00, 0.85, 0.0, 0.04]),
]

_ATTRACTOR_SYMBOLS: tuple[str, ...] = tuple(item[0] for item in _CANONICAL_ATTRACTORS)
_ATTRACTOR_MATRIX: np.ndarray = np.array([item[1] for item in _CANONICAL_ATTRACTORS], dtype=np.float32)


def resolve_tract_vector(g: np.ndarray, is_nucleus: bool | None = None) -> str:
    """
    Maps 8D tract vector to IPA symbol.
    If is_nucleus is specified, syllable prosody determines vowel vs glide distinction.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    
    # Syllabic role takes precedence; fallback to continuous potential
    if is_nucleus is not None:
        is_voc = is_nucleus
    else:
        is_voc = bool(is_vowel_gesture(g_arr))

    candidate_indices = [
        i for i, (sym, _) in enumerate(_CANONICAL_ATTRACTORS)
        if (sym in _VOWEL_SET) == is_voc and (sym not in _CLICK_SET or g_arr[7] <= 0.05)
    ]
    if not candidate_indices:
        candidate_indices = list(range(len(_CANONICAL_ATTRACTORS)))

    sub_matrix = _ATTRACTOR_MATRIX[candidate_indices]
    g_mid = 0.50 * (sub_matrix + g_arr[np.newaxis, :])
    w = compute_metric_weights(g_mid)
    dists = np.sum(w * ((sub_matrix - g_arr[np.newaxis, :]) ** 2), axis=-1)
    symbol = _ATTRACTOR_SYMBOLS[candidate_indices[int(np.argmin(dists))]]

    if symbol in _VOWEL_SET and g_arr[6] >= 0.22:
        symbol = f"{symbol}\u0303"
    elif g_arr[7] >= 0.80 and symbol in _VOICELESS_STOPS:
        symbol = f"{symbol}ʰ"
    return symbol