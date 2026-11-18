"""
phonetics.py — Minimal-Parameter Zone-Energy + Airflow Phonetics,
Acoustic Formants & Bark Filterbank Integration, and Autonomous Sound Change.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence
from collections import Counter
import numpy as np

from acoustics import AcousticEngine, AuditoryFeatures, BARK_CENTERS

# =============================================================================
# 14 CORE PHYSICAL PARAMETERS
# =============================================================================
K_STOP     = 0.20   # Oral closure effort (lips, tip, dorsum)
K_BODY     = 0.45   # Tongue-blade / palatal / pharyngeal displacement effort
K_GLOT     = 0.55   # Glottal squeeze / abduction effort
K_LARYNX   = 2.60   # Vertical larynx shift (ejectives / implosives)
K_FLOW     = 0.42   # Turbulent & unsealed airflow cost
K_VDRAG    = 0.24   # Oral pressure drag opposing voicing during sealed stops
K_AUD      = 0.24   # Base perceptibility reward (scaled up by societal noise/potency)
K_HYPOXIA  = 0.85   # Altitude air-density decay rate
K_ARID     = 0.65   # Moisture/heat loss on open unrounded vowels
W_CLOSURE  = 16.0   # Metric sharpness of oral closures (stop vs fricative vs open)
W_CAT      = 12.0   # Metric sharpness of categorical voicing & nasality boundaries
W_FORMANT  = 2.20   # Acoustic (F1, F2) vowel-triangle perceptual metric weight
LAM_INV    = 0.003  # Inventory slot cost (fraction of lexicon usage mass)
CONF_SCALE = 28.0   # Whole-word distinctiveness pressure
# =============================================================================

INERTIA_DIAG = np.array([0.90, 0.15, 0.80, 0.40, 2.50, 2.00, 0.50, 1.00, 1.60], dtype=np.float32)
_BOUNDARY_REST_G = np.array([1.00, 0.00, 1.00, 0.00, 0.70, 0.50, 0.00, 0.75, 0.50], dtype=np.float32)

_CHAO_BARS = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
_VOWEL_SET = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o', 'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ'
}
_VOICELESS_STOPS = {'p', 't̪', 't', 'ʈ', 'c', 'k', 'q', 'ʔ'}


# -----------------------------------------------------------------------------
# 1. METRIC TENSORS, ZONE ENERGY, AIRFLOW & PERCEPTIBILITY
# -----------------------------------------------------------------------------

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.minimum(np.maximum(x, -30.0), 30.0)))


def compute_metric_weights(g: np.ndarray) -> np.ndarray:
    g0, g2, g4, g5, g6, g7 = g[..., 0], g[..., 2], g[..., 4], g[..., 5], g[..., 6], g[..., 7]
    w0 = 1.0 + W_CLOSURE / (1.0 + np.exp(np.clip(30.0 * (g0 - 0.11), -30.0, 30.0)))
    w1 = 1.0 + 3.0 * (1.0 - g4)
    w2 = 1.0 + 1.5 * W_CLOSURE / (1.0 + np.exp(np.clip(30.0 * (g2 - 0.12), -30.0, 30.0)))
    w3 = 0.05 + 8.0 * (1.0 - g2)
    w4 = 1.5 + 1.2 * W_CLOSURE / (1.0 + np.exp(np.clip(25.0 * (g4 - 0.12), -30.0, 30.0)))
    w5 = 2.0 + 4.0 * ((g5 - 0.50) ** 2)
    w6 = 0.50 + 0.5 * W_CAT * np.exp(-((g6 - 0.25) ** 2) / 0.04) + 0.8 * W_CLOSURE / (1.0 + np.exp(np.clip(35.0 * (g6 - 0.05), -30.0, 30.0)))
    w7 = 1.0 + W_CAT * np.exp(-((g7 - 0.50) ** 2) / 0.015) + 0.9 * W_CLOSURE / (1.0 + np.exp(np.clip(30.0 * (g7 - 0.10), -30.0, 30.0)))

    weights = [w0, w1, w2, w3, w4, w5, w6, w7]
    if g.shape[-1] >= 9:
        weights.append(1.5 + 16.0 * ((g[..., 8] - 0.50) ** 2))
    return np.stack(weights, axis=-1)


def continuous_vocality(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32)
    g8 = g[..., 8] if g.shape[-1] > 8 else 0.50
    oral_open = _sig(30.0 * (g[..., 0] - 0.25)) * _sig(18.0 * (g[..., 2] - 0.60)) * _sig(35.0 * (g[..., 4] - 0.215))
    not_phar = _sig(35.0 * (0.91 - g[..., 5]))
    modal_larynx = np.exp(-((g[..., 7] - 0.40) ** 2) / 0.025) * np.exp(-((g8 - 0.50) ** 2) / 0.06)
    return oral_open * not_phar * modal_larynx


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    voc = continuous_vocality(np.asarray(g, dtype=np.float32))
    return bool(voc >= 0.50) if np.ndim(voc) == 0 else (voc >= 0.50)


def is_click_gesture(g: np.ndarray) -> np.ndarray:
    g_arr = np.asarray(g, dtype=np.float32)
    dorsal = (g_arr[..., 4] < 0.08) & (g_arr[..., 5] >= 0.70)
    anterior = (np.minimum(g_arr[..., 0], g_arr[..., 2]) < 0.08) | (np.abs(g_arr[..., 2] - 0.28) < 0.06)
    return dorsal & anterior


def click_gate(g: np.ndarray) -> np.ndarray:
    g_arr = np.asarray(g, dtype=np.float32)
    dorsal = _sig(45.0 * (0.08 - g_arr[..., 4])) * _sig(25.0 * (g_arr[..., 5] - 0.68))
    ant_close = _sig(45.0 * (0.08 - np.minimum(g_arr[..., 0], g_arr[..., 2])))
    ant_lat = np.exp(-((g_arr[..., 2] - 0.28) ** 2) / 0.0025)
    return dorsal * np.maximum(ant_close, ant_lat)


def _vowel_formants(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Acoustic triangle: Normalized F1 and F2 in [0, 1].
    Directly mirrors the perturbation acoustics in AcousticEngine.
    """
    g0 = g[..., 0]
    g1 = g[..., 1]
    g4 = g[..., 4]
    g5 = g[..., 5]

    f1 = (730.0 * (g4 ** 1.35) * (0.35 + 0.65 * g0)) / 750.0
    raw_f2 = 1.0 - g5 * (1.0 - 0.52 * g1)
    f2 = 0.50 + (1.0 - 0.50 * g4) * (raw_f2 - 0.50)
    return f1, f2


def sensory_distance(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    w = compute_metric_weights(0.50 * (a1 + a2))
    d9_sq = np.sum(w * ((a1 - a2) ** 2), axis=-1)

    v_pair = continuous_vocality(a1) * continuous_vocality(a2)
    f1_1, f2_1 = _vowel_formants(a1)
    f1_2, f2_2 = _vowel_formants(a2)
    d_formant_sq = W_FORMANT * (((f1_1 - f1_2) ** 2) + ((f2_1 - f2_2) ** 2))
    return np.sqrt(d9_sq + v_pair * d_formant_sq)


def trajectory_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    a1 = np.asarray(g1, dtype=np.float32)[..., :9]
    a2 = np.asarray(g2, dtype=np.float32)[..., :9]
    if a1.ndim == 1: a1 = a1[None, :]
    if a2.ndim == 1: a2 = a2[None, :]
    L1, L2 = len(a1), len(a2)
    if L1 == 0 or L2 == 0:
        return 1.0
    if L1 == L2:
        return float(np.mean(sensory_distance(a1, a2)))
    max_L = max(L1, L2)
    idx1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
    idx2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
    return float(np.mean(sensory_distance(a1[idx1], a2[idx2])))


def _oral_opening(g: np.ndarray) -> np.ndarray:
    phar_open = np.clip(1.0 - (g[..., 5] - 0.85) / 0.15, 0.0, 1.0)
    return np.minimum(np.minimum(g[..., 0], g[..., 2]), np.minimum(g[..., 4], phar_open))


def zone_energy(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32)
    voc = continuous_vocality(g)
    v_reg = _sig(25.0 * (voc - 0.50))
    g0, g1, g2, g3, g4, g5, g6, g7, g8 = (g[..., i] for i in range(9))

    lip_e = K_STOP * (1.0 - v_reg) * (np.maximum(0.0, 1.0 - g0) ** 1.5) + 0.005 * (g1 ** 2) + 0.22 * g1 * np.maximum(0.0, g4 - 0.55) * v_reg
    lip_e = lip_e + 0.12 * (1.0 - v_reg) * (1.0 - g1) * np.exp(-(((g0 - 0.18) / 0.10) ** 2))
    tip_e = K_STOP * (np.maximum(0.0, 1.0 - g2) ** 1.5) + K_BODY * (1.0 - g2) * (((g3 - 0.25) ** 2) / 0.25)
    dors_e = 1.25 * K_STOP * (1.0 - v_reg) * _sig(40.0 * (0.22 - g4))
    body_e = (0.12 + 2.2 * K_BODY * (1.0 - v_reg) * (1.0 - g4)) * ((g5 - 0.50) ** 2) + 8.0 * K_BODY * (np.maximum(0.0, g5 - 0.80) ** 1.5)
    nasal_e = 0.08 * (g6 ** 1.5)
    glot_e = K_GLOT * (np.exp(-20.0 * g7) + (1.0 - v_reg) * (np.maximum(0.0, g7 - 0.40) ** 1.3)) + K_LARYNX * ((g8 - 0.50) ** 2)
    return lip_e + tip_e + dors_e + body_e + nasal_e + glot_e


def continuous_acoustic_power(g: np.ndarray, rho_air: float = 1.0, env=None, include_transients: bool = True) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32)
    voc = continuous_vocality(g)
    v_reg = _sig(25.0 * (voc - 0.50))
    a_oral = _oral_opening(g)
    g0, g2, g3, g6, g7 = g[..., 0], g[..., 2], g[..., 3], np.clip(g[..., 6], 0.0, 1.0), g[..., 7]

    f1, f2 = _vowel_formants(g)
    disp = 1.50 * ((f1 - 0.55) ** 2) + 1.20 * ((f2 - 0.50) ** 2)
    modal_larynx = np.exp(-((g7 - 0.40) ** 2) / 0.015)
    vowel_rad = v_reg * modal_larynx * (0.45 + 0.85 * a_oral + 1.60 * disp)

    sonorant_rad = (1.0 - v_reg) * modal_larynx * (
        0.70 * g6 + 0.42 * _sig(20.0 * (a_oral - 0.14)) * np.exp(-((g2 - 0.25) ** 2) / 0.08)
    )
    sib_turb = (1.0 - v_reg) * np.exp(-(((a_oral - 0.06) / 0.030) ** 2))
    hf_noise = 0.85 * np.tanh(1.15 * sib_turb * _sig(10.0 * (g7 - 0.42)) * (0.35 + 0.65 * np.exp(-((g3 - 0.25) ** 2) / 0.04)))
    power = (vowel_rad + sonorant_rad + hf_noise) * math.sqrt(max(0.1, rho_air))
    if not include_transients:
        return power

    c_gate = click_gate(g)
    cavity_res = 0.75 + 0.25 * _sig(12.0 * (g0 - 0.30))
    burst_sharp = 0.78 + 0.22 * _sig(12.0 * (g7 - 0.50))
    stop_burst = (1.0 - v_reg) * (1.0 - g6) * np.exp(-((a_oral / 0.015) ** 2)) * 0.80 * cavity_res * burst_sharp * (1.0 - c_gate)

    veg = getattr(env, "vegetation", 0.3) if env else 0.3
    roughness = getattr(env, "roughness", 0.1) if env else 0.1
    click_burst = c_gate * 0.65 * np.exp(-4.5 * veg - 5.5 * roughness)
    return power + stop_burst + click_burst


def sonority_batch(G: np.ndarray) -> np.ndarray:
    pwr = continuous_acoustic_power(np.asarray(G, dtype=np.float32)[..., :9], include_transients=False)
    return np.round(1.0 + 4.5 * np.tanh(pwr * 1.5), 3)


def _societal_drives(profile=None, env=None, cultural_attention: Sequence[float] | None = None) -> tuple[float, float]:
    amb = ambient_from_env(env)
    pop = float(getattr(env, "population", 0.5)) if env else 0.5
    if cultural_attention is not None and len(cultural_attention) >= 5:
        potency = float(cultural_attention[3])
        velocity = float(cultural_attention[4])
    else:
        potency = float(getattr(profile, "potency", 1.0)) if profile else 1.0
        velocity = float(getattr(profile, "velocity", 1.0)) if profile else 1.0
    s_comp = float(getattr(profile, "syllable_complexity", 0.50)) if profile else 0.50

    power_gain = K_AUD + 0.25 * max(0.0, amb - 0.10) + 0.06 * math.tanh(max(0.0, pop)) + 0.07 * math.tanh(potency - 0.85)
    speed_gain = 1.00 + 0.25 * math.tanh(velocity - 1.00) + 0.15 * (s_comp - 0.50)
    return float(np.clip(power_gain, 0.18, 0.45)), float(np.clip(speed_gain, 0.75, 1.40))


def _segment_biomech_energy(
    G: np.ndarray,
    taus: np.ndarray | float = 1.0,
    env=None,
    profile=None,
    cultural_attention: Sequence[float] | None = None,
) -> np.ndarray:
    alt = float(getattr(env, "altitude", 0.0)) if env else 0.0
    hum = float(getattr(env, "humidity", 0.50)) if env else 0.50
    temp = float(getattr(env, "temperature", 0.50)) if env else 0.50
    veg = float(getattr(env, "vegetation", 0.30)) if env else 0.30
    rough = float(getattr(env, "roughness", 0.10)) if env else 0.10
    rho = math.exp(-K_HYPOXIA * max(0.0, alt))
    power_gain, _ = _societal_drives(profile=profile, env=env, cultural_attention=cultural_attention)

    G = np.asarray(G, dtype=np.float32)[..., :9]
    voc = continuous_vocality(G)
    v_reg = _sig(25.0 * (voc - 0.50))
    a_oral = _oral_opening(G)
    g1, g4, g6, g7, g8 = G[..., 1], G[..., 4], np.clip(G[..., 6], 0.0, 1.0), G[..., 7], G[..., 8]

    e_zone = zone_energy(G) * (taus ** 1.08)

    turb = (1.0 - v_reg) * np.exp(-(((a_oral - 0.06) / 0.065) ** 2))
    raw_flow = (0.14 + 0.50 * a_oral * (1.0 - 0.85 * v_reg) + 0.10 * g6) * (0.35 + g7) + K_FLOW * turb * (0.50 + g7)

    ejective_gate = _sig(25.0 * (0.15 - g7)) * _sig(15.0 * (g8 - 0.65)) * (1.0 - v_reg)
    implosive_gate = _sig(15.0 * (0.35 - g8)) * (1.0 - v_reg)
    c_gate = click_gate(G)
    e_air = ((1.0 - c_gate) * (1.0 - ejective_gate) * (1.0 - 0.60 * implosive_gate) * (raw_flow / (rho ** 1.8)) + c_gate * 0.05) * taus

    v_drag = K_VDRAG * np.exp(-((g7 - 0.40) ** 2) / 0.02) * np.exp(-16.0 * a_oral) * ((1.0 - g6) ** 2) * (0.75 + 0.50 * (1.0 - g4)) * taus

    click_cost = c_gate * (1.40 + 4.5 * veg + 5.5 * rough)
    oral_exposure = a_oral * (1.0 - 0.65 * g1)
    dry_cold = max(0.0, 0.35 - hum) * (1.0 + max(0.0, temp - 0.50)) + max(0.0, 0.35 - temp) * 0.80
    arid = v_reg * dry_cold * oral_exposure * taus * K_ARID

    aud = continuous_acoustic_power(G, rho_air=rho, env=env)
    return e_zone + e_air + v_drag + click_cost + arid - power_gain * aud


def _local_segment_energy_grad(
    g: np.ndarray,
    env=None,
    profile=None,
    cultural_attention: Sequence[float] | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32)[..., :9]
    eye = np.eye(9, dtype=np.float32)
    hi = np.clip(g[..., None, :] + eps * eye, 0.0, 1.0)
    lo = np.clip(g[..., None, :] - eps * eye, 0.0, 1.0)
    span = np.maximum(np.diagonal(hi, axis1=-2, axis2=-1) - np.diagonal(lo, axis1=-2, axis2=-1), 1e-6)
    e_hi = _segment_biomech_energy(hi, 1.0, env=env, profile=profile, cultural_attention=cultural_attention)
    e_lo = _segment_biomech_energy(lo, 1.0, env=env, profile=profile, cultural_attention=cultural_attention)
    return (e_hi - e_lo) / span


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
    __slots__ = ('_vector', '_ipa', '_energy', '_hash', '_voc')

    def __init__(self, vector: Sequence[float] | np.ndarray | Phoneme, duration: float | None = None):
        if isinstance(vector, Phoneme):
            v = list(vector._vector)
        else:
            v = list(vector)

        if len(v) == 9:
            v.append(float(duration) if duration is not None else 1.0)
        elif len(v) == 10 and duration is not None:
            v[9] = float(duration)
        elif len(v) == 8:
            v.append(0.50)  # neutral vertical larynx height
            v.append(float(duration) if duration is not None else 1.0)

        g_clamped = [max(0.0, min(1.0, float(v[i]))) for i in range(9)]
        tau_clamped = max(0.01, min(3.0, float(v[9])))
        vec = np.array(g_clamped + [tau_clamped], dtype=np.float32)
        vec.flags.writeable = False  # Enforce true immutability for hash safety
        self._vector = vec

        self._ipa = None
        self._energy = None
        self._voc = None
        self._hash = hash(self._vector.tobytes())

    @property
    def vector(self) -> np.ndarray: return self._vector
    @property
    def g(self) -> np.ndarray: return self._vector[:9]
    @property
    def duration(self) -> float: return float(self._vector[9])
    @property
    def tau(self) -> float: return self.duration

    @property
    def voc(self) -> float:
        if self._voc is None:
            self._voc = float(continuous_vocality(self._vector[:9]))
        return self._voc

    @property
    def is_vowel(self) -> bool: return self.voc >= 0.50

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
            a_oral = float(_oral_opening(g))
            v_reg = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, 25.0 * (self.voc - 0.50)))))
            e_mot = round(float(zone_energy(g)) * (tau ** 1.08), 3)

            turb = (1.0 - v_reg) * math.exp(-(((a_oral - 0.06) / 0.065) ** 2))
            raw_flow = (0.14 + 0.50 * a_oral * (1.0 - 0.85 * v_reg) + 0.10 * g[6]) * (0.35 + g[7]) + K_FLOW * turb * (0.50 + g[7])
            ejective_gate = float(_sig(25.0 * (0.15 - g[7])) * _sig(15.0 * (g[8] - 0.65)) * (1.0 - v_reg))
            implosive_gate = float(_sig(15.0 * (0.35 - g[8])) * (1.0 - v_reg))
            c_gate = float(click_gate(g))
            air = round(((1.0 - c_gate) * (1.0 - ejective_gate) * (1.0 - 0.60 * implosive_gate) * raw_flow + c_gate * 0.05) * tau, 3)

            v_drag = round(K_VDRAG * math.exp(-((g[7] - 0.40) ** 2) / 0.02) * math.exp(-16.0 * a_oral) * ((1.0 - g[6]) ** 2) * (0.75 + 0.50 * (1.0 - g[4])) * tau, 3)
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

    def synthesize_wav(self, base_f0: float = 120.0) -> bytes:
        """Synthesizes playable 16-bit 16kHz WAV audio bytes for this phoneme."""
        return AcousticEngine.synthesize_wav_pcm([self._vector], base_f0=base_f0)

    def __hash__(self) -> int: return self._hash
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phoneme) and np.array_equal(self._vector, other._vector)
    def __len__(self) -> int: return len(self._vector)
    def __getitem__(self, idx: int) -> float: return float(self._vector[idx])
    def __iter__(self): return iter(self._vector)
    def __repr__(self) -> str: return f"Phoneme(/{self.ipa}/, tau={self.duration:.2f})"


# -----------------------------------------------------------------------------
# 3. TRANSITION COST, COARTICULATION & SYLLABLE REPAIR
# -----------------------------------------------------------------------------

def transition_cost(
    p1: Phoneme | np.ndarray,
    p2: Phoneme | np.ndarray,
    tau1: float = 1.0,
    tau2: float = 1.0,
    s_comp: float = 0.50,
) -> float:
    g1 = p1.g if hasattr(p1, "g") else np.asarray(p1, dtype=np.float32)[:9]
    g2 = p2.g if hasattr(p2, "g") else np.asarray(p2, dtype=np.float32)[:9]
    t1 = p1.duration if hasattr(p1, "duration") else tau1
    t2 = p2.duration if hasattr(p2, "duration") else tau2

    voc1 = float(p1.voc) if hasattr(p1, "voc") else float(continuous_vocality(g1))
    voc2 = float(p2.voc) if hasattr(p2, "voc") else float(continuous_vocality(g2))
    same_class = voc1 * voc2 + (1.0 - voc1) * (1.0 - voc2)

    diff = g1 - g2
    eff_inertia = INERTIA_DIAG.copy()
    cv_oral = 0.04 + 0.96 * same_class
    cv_larynx = 0.25 + 0.75 * same_class
    eff_inertia[:6] *= cv_oral
    eff_inertia[6:] *= cv_larynx

    min_tau = max(0.10, min(t1, t2))
    kinetic = float(np.sum(eff_inertia * (diff ** 2)) / (min_tau ** 0.85))
    dist = float(sensory_distance(g1, g2))
    cluster_weight = 1.0 + same_class * (0.45 - 0.85 * s_comp)
    ocp_friction = 2.50 * math.exp(-5.0 * dist) * same_class
    return round(float((kinetic + ocp_friction) * cluster_weight), 3)


def coarticulation_force(
    g: Phoneme | np.ndarray,
    prev_g: Phoneme | np.ndarray | None,
    next_g: Phoneme | np.ndarray | None,
    tau: float = 1.0,
    tau_prev: float = 1.0,
    tau_next: float = 1.0,
    k: float = 0.05,
    d_reach: float = 0.70,
) -> np.ndarray:
    g_arr = np.asarray(g.g if hasattr(g, "g") else g, dtype=np.float32)[:9]
    f = np.zeros(9, dtype=np.float32)

    for nb, t_nb in ((prev_g, tau_prev), (next_g, tau_next)):
        if nb is None:
            continue
        m = max(0.10, min(float(tau), float(t_nb)))
        nb_arr = np.asarray(nb.g if hasattr(nb, "g") else nb, dtype=np.float32)[:9]
        reach = math.exp(-(float(sensory_distance(g_arr, nb_arr)) / d_reach) ** 2)
        f -= k * reach * INERTIA_DIAG * (g_arr - nb_arr) / (m ** 0.85)

    return f


def _transition_costs(G1, G2, t1, t2, voc1, voc2, s_comp: float = 0.50) -> np.ndarray:
    diff = G1 - G2
    same_class = voc1 * voc2 + (1.0 - voc1) * (1.0 - voc2)
    cv_oral = (0.04 + 0.96 * same_class)[:, None]
    cv_larynx = (0.25 + 0.75 * same_class)[:, None]

    kinetic_raw = INERTIA_DIAG[None, :] * (diff ** 2)
    kinetic_raw[:, :6] *= cv_oral
    kinetic_raw[:, 6:] *= cv_larynx

    min_tau = np.maximum(0.10, np.minimum(t1, t2))
    kinetic = np.sum(kinetic_raw, axis=-1) / (min_tau ** 0.85)

    dist = sensory_distance(G1, G2)
    cluster_weight = 1.0 + same_class * (0.45 - 0.85 * s_comp)
    ocp_friction = 2.50 * np.exp(-5.0 * dist) * same_class
    return (kinetic + ocp_friction) * cluster_weight


def repair_phonemes(phonemes: list[Phoneme], profile=None, max_passes: int = 4) -> list[Phoneme]:
    s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    cluster_budget = 0.90 + 3.50 * s_comp
    phonemes = list(phonemes)

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

            same_class = p1.voc * p2.voc + (1.0 - p1.voc) * (1.0 - p2.voc)
            cost = transition_cost(p1, p2, s_comp=s_comp) * (0.15 + 0.85 * same_class)
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
    rng=None,
) -> None:
    n = len(words)
    if n < 2:
        return
    r = _as_np_rng(rng)

    for i in range(n):
        w1 = words[i]
        v1 = [idx for idx, p in enumerate(w1.phonemes) if p.is_vowel]
        if not v1:
            continue
        v1_idx = v1[0]
        g1 = w1.phonemes[v1_idx].g

        for j in range(i + 1, n):
            w2 = words[j]
            if w1.id == w2.id or w1.vector.distance_to(w2.vector) < 0.15:
                continue
            v2 = [idx for idx, p in enumerate(w2.phonemes) if p.is_vowel]
            if not v2:
                continue
            v2_idx = v2[0]
            g2 = w2.phonemes[v2_idx].g

            d_aud = float(sensory_distance(g1, g2))
            eff_margin = min_auditory_margin + ambient_noise * 0.50
            if d_aud < eff_margin:
                diff = g1 - g2
                norm_e = float(np.linalg.norm(diff))
                if norm_e < 1e-4:
                    h_hat = r.normal(0.0, 1.0, size=9).astype(np.float32)
                    h_hat /= (np.linalg.norm(h_hat) + 1e-8)
                else:
                    h_hat = diff / norm_e

                w_mid = compute_metric_weights(0.50 * (g1 + g2))
                step_a = 0.40 * (eff_margin - d_aud) / (2.0 * math.sqrt(float(np.sum(w_mid * (h_hat ** 2)))) + 1e-4)

                p1_new = Phoneme(np.clip(g1 + step_a * h_hat, 0.05, 0.95), duration=w1.phonemes[v1_idx].duration)
                p2_new = Phoneme(np.clip(g2 - step_a * h_hat, 0.05, 0.95), duration=w2.phonemes[v2_idx].duration)

                w1_ph = list(w1.phonemes)
                w1_ph[v1_idx] = p1_new
                w1.phonemes = w1_ph

                w2_ph = list(w2.phonemes)
                w2_ph[v2_idx] = p2_new
                w2.phonemes = w2_ph

                g1 = p1_new.g


def word_energy(
    word,
    profile=None,
    env=None,
    cultural_attention: Sequence[float] | None = None,
    base_f0: float = 120.0,
    **kwargs,
) -> float:
    """
    Computes total biomechanical and acoustic communication cost.
    High auditory transmission clarity directly offsets motor effort on the Pareto frontier.
    """
    phonemes = word.phonemes
    n = len(phonemes)
    if n == 0:
        return 0.0

    G = np.stack([p.g for p in phonemes], axis=0)
    taus = np.array([p.duration for p in phonemes], dtype=np.float32)
    voc = continuous_vocality(G)

    s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
    s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    power_gain, speed_gain = _societal_drives(profile=profile, env=env, cultural_attention=cultural_attention)

    e_seg = float(np.sum(_segment_biomech_energy(G, taus=taus, env=env, profile=profile, cultural_attention=cultural_attention)))
    e_trans = 0.0
    if n > 1:
        e_trans = 0.40 * speed_gain * float(np.sum(_transition_costs(G[:-1], G[1:], taus[:-1], taus[1:], voc[:-1], voc[1:], s_comp=s_comp)))

    freq_load = math.log1p(getattr(word, "usage_frequency", 1.0))
    e_length = 0.12 * speed_gain * (n ** 1.65) * (1.0 - 0.55 * s_idx) * (0.60 + 0.40 * freq_load)

    e_tone = 0.0
    cached_syls = getattr(word, "_syllables", None)
    if cached_syls:
        for s in cached_syls:
            if hasattr(s, "pitch"):
                p0, p1, p2 = s.pitch
                e_tone += ((p1 - p0) ** 2 + (p2 - p1) ** 2 + (p1 ** 2) * 0.2) * 0.18
    else:
        _, _, pitches, _ = parse_sonority_prosody(np.column_stack([G, taus]))
        for p0, p1, p2 in pitches:
            e_tone += ((p1 - p0) ** 2 + (p2 - p1) ** 2 + (p1 ** 2) * 0.2) * 0.18

    # Auditory Clarity Reward from AcousticEngine:
    # Formant prominence and high-frequency bursts directly offset motor effort
    af = AcousticEngine.get_auditory_features([p.vector for p in phonemes], base_f0=base_f0)
    clarity_payoff = 0.35 * power_gain * af.clarity_score

    return round(float(e_seg + e_trans + e_length + e_tone - clarity_payoff), 3)


def synthesize_word_wav(word, base_f0: float = 120.0) -> bytes:
    """Synthesizes playable 16-bit 16kHz WAV audio bytes directly from word phonemes."""
    vectors = [p.vector for p in getattr(word, "phonemes", [])]
    return AcousticEngine.synthesize_wav_pcm(vectors, base_f0=base_f0)


# -----------------------------------------------------------------------------
# 4. CONTOUR TONE PROSODY & MODE-DENSITY INVENTORY
# -----------------------------------------------------------------------------

def resolve_chao_tone(pitch: tuple[float, float, float] | np.ndarray, duration: float = 1.0) -> str:
    f0, f1, f2 = float(pitch[0]), float(pitch[1]), float(pitch[2])
    c0 = int(np.clip(np.floor(f0 * 2.50 + 3.50), 1, 5))
    c1 = int(np.clip(np.floor(f1 * 2.50 + 3.50), 1, 5))
    c2 = int(np.clip(np.floor(f2 * 2.50 + 3.50), 1, 5))
    if duration < 0.80 or c0 == c2:
        return _CHAO_BARS[c1]
    return f"{_CHAO_BARS[c0]}{_CHAO_BARS[c2]}"


def parse_sonority_prosody(
    X: np.ndarray, directional_bias: float = 0.50
) -> tuple[list[tuple[int, int]], int, list[tuple[float, float, float]], list[float]]:
    L = len(X)
    if L == 0:
        return [], 0, [], []
    son = sonority_batch(X[:, :9]).astype(np.float32)
    v_mask = np.asarray(continuous_vocality(np.asarray(X[:, :9], dtype=np.float32)) >= 0.50, dtype=bool)

    peaks = [i for i in range(L) if v_mask[i] and (i == 0 or son[i] >= son[i - 1]) and (i == L - 1 or son[i] >= son[i + 1])]
    if not peaks:
        peaks = [int(np.argmax(son))]

    boundaries = [0]
    for k in range(len(peaks) - 1):
        p_start, p_end = peaks[k] + 1, peaks[k + 1]
        boundaries.append(p_start if p_start == p_end else p_start + int(np.argmin(son[p_start:p_end])))
    boundaries.append(L)

    syl_slices = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    syl_masses = [
        sum(X[i, 9] * son[i] for i in range(st, en))
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
        f2 = f0 + 0.40 * (0.40 - coda_g7) - 0.15
        trajectories.append((round(f0, 2), round(f1, 2), round(f2, 2)))
        nuc_durs.append(float(X[pk, 9]) if X.shape[1] > 9 else 1.0)

    return syl_slices, stressed_idx, trajectories, nuc_durs


def render_and_extract_tones(
    X: np.ndarray, directional_bias: float = 0.50, is_tonal: bool = False
) -> tuple[str, str, str]:
    L = len(X)
    if L == 0:
        return "", "", ""
    syl_slices, stressed_idx, pitches, nuc_durs = parse_sonority_prosody(X, directional_bias=directional_bias)
    son_all = sonority_batch(X[:, :9])
    parts, plain_parts = [], []
    assigned_tone = ""

    for k, (st, en) in enumerate(syl_slices):
        if k == stressed_idx and len(syl_slices) > 1:
            parts.append("ˈ")
        peak_offset = int(np.argmax(son_all[st:en]))
        dedup = []
        for off in range(en - st):
            curr_idx = st + off
            sym = resolve_tract_vector(X[curr_idx, :9], is_nucleus=(off == peak_offset))
            dur = float(X[curr_idx, 9]) if X.shape[1] > 9 else 1.0

            # Preserve length mark for long monophthongs
            if sym in _VOWEL_SET and dur >= 1.35 and not sym.endswith("ː"):
                sym = f"{sym}ː"

            if dedup and dedup[-1] == sym:
                if sym in _VOWEL_SET and not dedup[-1].endswith("ː"):
                    dedup[-1] = f"{dedup[-1]}ː"
                continue
            dedup.append(sym)

        syl_str = "".join(dedup)
        plain_parts.append(syl_str.replace("ː", ""))

        if is_tonal and (k == stressed_idx or len(syl_slices) == 1):
            tone = resolve_chao_tone(pitches[k], duration=nuc_durs[k])
            syl_str += tone
            assigned_tone = tone

        parts.append(syl_str)

    return "".join(parts), "".join(plain_parts), assigned_tone


def cluster_contrastive_phonemes(words: Sequence) -> tuple[list[str], list[str]]:
    v_usage: dict[str, float] = Counter()
    c_usage: dict[str, float] = Counter()
    v_roots: dict[str, set[int]] = {}
    c_roots: dict[str, set[int]] = {}

    for w in words:
        u = float(getattr(w, "usage_frequency", 1.0))
        root_id = getattr(w.derivation, "root_family_id", w.id)
        for p in w.phonemes:
            clean = (
                p.ipa.replace("ː", "")
                .replace("˥", "")
                .replace("˦", "")
                .replace("˧", "")
                .replace("˨", "")
                .replace("˩", "")
            )
            if p.is_vowel:
                v_usage[clean] += u
                v_roots.setdefault(clean, set()).add(root_id)
            else:
                c_usage[clean] += u
                c_roots.setdefault(clean, set()).add(root_id)

    attractor_dict = dict(_CANONICAL_ATTRACTORS)

    def extract_attractor_modes(
        usage_dict: dict[str, float], root_dict: dict[str, set[int]], min_margin: float, is_v: bool
    ) -> list[str]:
        ranked = sorted(usage_dict.keys(), key=lambda sym: usage_dict[sym], reverse=True)
        modes: list[str] = []
        mode_vecs: list[np.ndarray] = []

        for sym in ranked:
            min_u = 1.8 if not is_v else 2.5
            if len(root_dict.get(sym, set())) < 2 and usage_dict[sym] < min_u:
                continue

            base_sym = sym.replace("ʰ", "")
            if base_sym not in attractor_dict:
                continue

            g_cand = np.array(attractor_dict[base_sym], dtype=np.float32)
            if "ʰ" in sym:
                g_cand[7] = 0.75  # Aspiration offset in glottal aperture

            if not any(float(sensory_distance(g_cand, m_vec)) < min_margin for m_vec in mode_vecs):
                modes.append(sym)
                mode_vecs.append(g_cand)

        return modes

    return (
        extract_attractor_modes(v_usage, v_roots, min_margin=0.36, is_v=True) or ["a", "i", "u"],
        extract_attractor_modes(c_usage, c_roots, min_margin=0.18, is_v=False) or ["t", "p", "k", "s", "m", "n"],
    )


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
        return (
            f"{pref}"
            f"{''.join(p.ipa for p in self.onset)}"
            f"{''.join(p.ipa for p in self.nucleus)}"
            f"{''.join(p.ipa for p in self.coda)}"
        )


def syllabify(target, syllable_complexity: float = 0.50, directional_bias: float = 0.50) -> list[Syllable]:
    raw = target.phonemes if hasattr(target, "phonemes") else list(target)
    if not raw:
        return []
    X = np.stack([p.vector for p in raw], axis=0)
    slices, stressed_idx, pitches, _ = parse_sonority_prosody(X, directional_bias=directional_bias)
    syls = []
    for k, (st, en) in enumerate(slices):
        syl_p = raw[st:en]
        nuc = [p for p in syl_p if p.is_vowel]
        if not nuc:
            syls.append(Syllable(onset=syl_p, nucleus=[], coda=[], stressed=(k == stressed_idx), pitch=pitches[k]))
        else:
            first_v = syl_p.index(nuc[0])
            last_v = len(syl_p) - 1 - list(reversed(syl_p)).index(nuc[-1])
            syls.append(
                Syllable(
                    onset=syl_p[:first_v],
                    nucleus=syl_p[first_v : last_v + 1],
                    coda=syl_p[last_v + 1 :],
                    stressed=(k == stressed_idx),
                    pitch=pitches[k],
                )
            )
    return syls


def assign_stress(syllables: list[Syllable], directional_bias: float = 0.50) -> None:
    if not syllables:
        return
    n = len(syllables)
    scores = [
        s.mass * (1.0 + (i / max(1, n - 1)) * (directional_bias - 0.50) * 0.40)
        for i, s in enumerate(syllables)
    ]
    str_idx = int(np.argmax(scores))
    for i, s in enumerate(syllables):
        s.stressed = (i == str_idx)


def apply_prosody(word, directional_bias: float = 0.50, syllable_complexity: float = 0.50):
    assign_stress(word.syllables, directional_bias=directional_bias)
    return word


# -----------------------------------------------------------------------------
# 5. OBSERVATION CODEBOOK & QUANTAL READOUT BASINS
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
    ("ʎ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.08, 0.00, 0.40, 0.50]),
    ("ʟ",  [1.00, 0.0, 1.00, 0.00, 0.20, 0.75, 0.00, 0.40, 0.50]),
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


def resolve_tract_vector(g: np.ndarray, is_nucleus: bool | None = None) -> str:
    """Quantal basin readout using the exact same Riemannian + Formant metric as sensory_distance."""
    g_arr = np.asarray(g, dtype=np.float32)[..., :9]
    if is_nucleus is True or (is_nucleus is None and is_vowel_gesture(g_arr)):
        candidate_indices = [i for i, (sym, _) in enumerate(_CANONICAL_ATTRACTORS) if sym in _VOWEL_SET]
    else:
        candidate_indices = [i for i, (sym, _) in enumerate(_CANONICAL_ATTRACTORS) if sym not in _VOWEL_SET]

    if not candidate_indices:
        candidate_indices = list(range(len(_CANONICAL_ATTRACTORS)))

    sub_matrix = _ATTRACTOR_MATRIX[candidate_indices]
    dists = sensory_distance(sub_matrix, g_arr[np.newaxis, :])
    symbol = _ATTRACTOR_SYMBOLS[candidate_indices[int(np.argmin(dists))]]

    if symbol in _VOICELESS_STOPS and g_arr[7] >= 0.75:
        symbol = f"{symbol}ʰ"
    return symbol


# -----------------------------------------------------------------------------
# 6. ENERGY-DRIVEN SOUND CHANGE
# -----------------------------------------------------------------------------

_V_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s in _VOWEL_SET], dtype=np.int64)
_C_IDX = np.array([i for i, s in enumerate(_ATTRACTOR_SYMBOLS) if s not in _VOWEL_SET], dtype=np.int64)

_NONPULMONIC = np.maximum(
    click_gate(_ATTRACTOR_MATRIX),
    np.maximum(
        _sig(25.0 * (0.15 - _ATTRACTOR_MATRIX[:, 7])) * _sig(15.0 * (_ATTRACTOR_MATRIX[:, 8] - 0.65)),
        _sig(15.0 * (0.35 - _ATTRACTOR_MATRIX[:, 8])),
    ),
).astype(np.float32)

_SLOT_VOC = continuous_vocality(_ATTRACTOR_MATRIX)
_SLOT_VREG = _sig(25.0 * (_SLOT_VOC - 0.50))
_SLOT_AORAL = _oral_opening(_ATTRACTOR_MATRIX)
_SLOT_TURB = (1.0 - _SLOT_VREG) * np.exp(-(((_SLOT_AORAL - 0.06) / 0.065) ** 2))
_SLOT_FLOW = (0.14 + 0.50 * _SLOT_AORAL * (1.0 - 0.85 * _SLOT_VREG) + 0.10 * _ATTRACTOR_MATRIX[:, 6]) * (0.35 + _ATTRACTOR_MATRIX[:, 7]) + K_FLOW * _SLOT_TURB * (0.50 + _ATTRACTOR_MATRIX[:, 7])
_SLOT_COST = (0.65 + 0.35 * (zone_energy(_ATTRACTOR_MATRIX) + _SLOT_FLOW) + 0.65 * _NONPULMONIC).astype(np.float32)
SLOT_COST = _SLOT_COST
ATTRACTOR_SYMBOLS = _ATTRACTOR_SYMBOLS


def ambient_from_env(env=None) -> float:
    if env is None:
        return 0.10
    veg = float(getattr(env, "vegetation", 0.3))
    rough = float(getattr(env, "roughness", 0.1))
    return float(np.clip(0.10 + 0.25 * veg + 0.25 * rough, 0.05, 0.60))


def readout_batch(G: np.ndarray, vmask: np.ndarray | None = None, chunk: int = 1500) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)[..., :9].reshape(-1, 9)
    if vmask is None:
        vmask = continuous_vocality(G) >= 0.5
    out = np.empty(len(G), dtype=np.int64)
    for idx, sel in ((_V_IDX, vmask), (_C_IDX, ~vmask)):
        rows = np.flatnonzero(sel)
        A = _ATTRACTOR_MATRIX[idx]
        for s in range(0, len(rows), chunk):
            r = rows[s : s + chunk]
            g = G[r]
            d = sensory_distance(A[:, None, :], g[None, :, :])
            out[r] = idx[np.argmin(d, axis=0)]
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
    n = inventory_counts(words)
    tot = float(n.sum())
    if tot <= 0:
        return {"vowels": 0, "consonants": 0, "nonpulmonic": 0, "total": 0}
    keep = n / tot >= min_share
    v = int(np.sum(keep[_V_IDX]))
    c = int(np.sum(keep[_C_IDX]))
    return {
        "vowels": v,
        "consonants": c,
        "nonpulmonic": int(np.sum(keep & (_NONPULMONIC > 0.5))),
        "total": v + c,
    }


def _as_np_rng(rng):
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
    eps: float = 0.05,
    p_merge: float = 0.30,
    p_cond: float = 0.35,
    env=None,
    profile=None,
    cultural_attention: Sequence[float] | None = None,
    lam_conf: float = 1.0,
    lam_inv: float | None = None,
    n0: float | None = None,
    ambient_noise: float | None = None,
    n_words_eval: int = 48,
    n_ref: int = 24,
    conf_scale: float = CONF_SCALE,
    shift_sigma: float = 0.018,
    ctx_sigma: float = 0.55,
) -> dict:
    rng = _as_np_rng(rng)
    refs, G, F = _collect_tokens(words)
    if len(G) == 0:
        return {"accepted": False, "kind": "none"}

    mass = float(F.sum())
    lam_inv = LAM_INV * mass if lam_inv is None else lam_inv
    n0 = max(1.5, 0.01 * mass) if n0 is None else n0
    env_fn = env if callable(env) else (lambda _w: env)
    if ambient_noise is None:
        ambient_noise = 0.10 if callable(env) else ambient_from_env(env)
    sigma = 0.38 + 0.45 * ambient_noise

    vmask = continuous_vocality(G) >= 0.5
    sym = readout_batch(G, vmask)
    counts = np.bincount(sym, weights=F, minlength=len(_ATTRACTOR_SYMBOLS)).astype(np.float32)
    active_syms = np.flatnonzero(counts > 0)

    if rng.random() < 0.50 and len(active_syms) > 0:
        s = int(active_syms[int(rng.integers(len(active_syms)))])
        members = np.flatnonzero(sym == s)
        pick_k = int(members[int(rng.integers(len(members)))])
    else:
        pick_k = int(rng.integers(len(G)))
        s = int(sym[pick_k])
        members = np.flatnonzero(sym == s)

    s_is_vowel = _ATTRACTOR_SYMBOLS[s] in _VOWEL_SET
    env_ref = env_fn(words[refs[pick_k][0]])

    pool = _V_IDX if s_is_vowel else _C_IDX
    cand_active = pool[(pool != s) & (counts[pool] > 0)]

    roll = rng.random()
    if roll < p_merge and len(cand_active) > 0:
        d = sensory_distance(_ATTRACTOR_MATRIX[s][None, :], _ATTRACTOR_MATRIX[cand_active])
        order = np.argsort(d)[:4]
        pr = np.exp(-d[order])
        pr /= pr.sum()
        t = int(cand_active[order[int(rng.choice(len(order), p=pr))]])
        shift = (_ATTRACTOR_MATRIX[t] - _ATTRACTOR_MATRIX[s])[None, :]
        G_new = np.clip(G[members] + shift, 0.0, 1.0).astype(np.float32)
        kind, target = "merge", _ATTRACTOR_SYMBOLS[t]

    elif roll < (p_merge + p_cond) and len(members) >= 2:
        k0 = int(members[int(rng.integers(len(members)))])
        wi0, pi0 = refs[k0]
        ph0 = words[wi0].phonemes
        prev0 = ph0[pi0 - 1].g if pi0 > 0 else _BOUNDARY_REST_G
        next0 = ph0[pi0 + 1].g if (pi0 + 1) < len(ph0) else _BOUNDARY_REST_G

        prev_mat = np.empty((len(members), 9), dtype=np.float32)
        next_mat = np.empty((len(members), 9), dtype=np.float32)
        shifts = np.empty((len(members), 9), dtype=np.float32)

        for m_i, k in enumerate(members):
            wi, pi = refs[k]
            phs = words[wi].phonemes
            p_prev = phs[pi - 1] if pi > 0 else None
            p_next = phs[pi + 1] if (pi + 1) < len(phs) else None
            prev_mat[m_i] = p_prev.g if p_prev is not None else _BOUNDARY_REST_G
            next_mat[m_i] = p_next.g if p_next is not None else _BOUNDARY_REST_G
            shifts[m_i] = coarticulation_force(
                G[k],
                p_prev.g if p_prev is not None else None,
                p_next.g if p_next is not None else None,
                phs[pi].duration,
                p_prev.duration if p_prev is not None else 1.0,
                p_next.duration if p_next is not None else 1.0,
                k=0.08,
                d_reach=0.65,
            )

        d_prev = sensory_distance(prev_mat, prev0[None, :])
        d_next = sensory_distance(next_mat, next0[None, :])
        w_ctx = np.exp(-((d_prev ** 2 + d_next ** 2) / (2.0 * ctx_sigma * ctx_sigma)))

        keep = w_ctx > 0.18
        if not np.any(keep):
            keep[0] = True
        members = members[keep]
        w_ctx = w_ctx[keep][:, None]

        g_sub = G[members]
        w_met = np.clip(compute_metric_weights(g_sub), 0.25, 40.0)
        grad = _local_segment_energy_grad(g_sub, env=env_ref, profile=profile, cultural_attention=cultural_attention)
        noise = shift_sigma * rng.normal(0.0, 1.0, size=(len(members), 9)) / np.sqrt(w_met)
        delta = w_ctx * (-eps * grad / w_met + 0.06 * shifts[keep] + noise)
        G_new = np.clip(g_sub + delta, 0.0, 1.0).astype(np.float32)
        kind, target = "conditioned", None

    else:
        mean_a = float(_oral_opening(G[members].mean(axis=0)))
        if not s_is_vowel and mean_a < 0.12 and rng.random() < 0.12:
            g78 = rng.beta(0.4, 0.4, size=2).astype(np.float32)
            G_new = G[members].copy()
            G_new[:, 7] = g78[0]
            G_new[:, 8] = g78[1]
            kind, target = "airstream", None
        else:
            gbar = G[members].mean(axis=0)
            w = np.clip(compute_metric_weights(gbar), 0.25, 40.0)
            grad = _local_segment_energy_grad(gbar, env=env_ref, profile=profile, cultural_attention=cultural_attention)
            shift = (-eps * grad / w + shift_sigma * rng.normal(0.0, 1.0, size=9) / np.sqrt(w))[None, :]
            G_new = np.clip(G[members] + shift, 0.0, 1.0).astype(np.float32)
            kind, target = "drift", None

    sym_new = readout_batch(G_new)
    counts_new = counts.copy()
    np.subtract.at(counts_new, sym[members], F[members])
    np.add.at(counts_new, sym_new, F[members])
    d_inv = lam_inv * (_inventory_cost(counts_new, n0) - _inventory_cost(counts, n0))

    aff = sorted({refs[k][0] for k in members})
    sample = [aff[i] for i in rng.choice(len(aff), size=min(n_words_eval, len(aff)), replace=False)]
    scale = len(aff) / len(sample)
    ref_idx = rng.choice(len(words), size=min(n_ref, len(words)), replace=False)
    ref_words = [words[i] for i in ref_idx]

    sample_words = [words[wi] for wi in sample]
    n_ref_total = max(1, len(ref_words))
    inv_two_sig2 = 1.0 / (2.0 * sigma * sigma)

    if all(hasattr(w, "vector") and hasattr(w.vector, "data") for w in sample_words) and all(hasattr(w, "vector") and hasattr(w.vector, "data") for w in ref_words):
        S_vecs = np.stack([w.vector.data for w in sample_words], axis=0)
        R_vecs = np.stack([w.vector.data for w in ref_words], axis=0)
        d_sem_mat = np.sqrt(np.sum((S_vecs[:, None, :] - R_vecs[None, :, :]) ** 2, axis=-1))
    else:
        d_sem_mat = np.ones((len(sample_words), len(ref_words)), dtype=np.float32)

    s_ij_mat = 1.0 - np.exp(-((d_sem_mat / 0.35) ** 2))
    s_ij_mat = np.where(s_ij_mat < 1e-3, 0.0, s_ij_mat)
    ref_freqs = np.array([float(getattr(o, "usage_frequency", 1.0)) for o in ref_words], dtype=np.float32)
    sample_freqs = np.array([float(getattr(w, "usage_frequency", 1.0)) for w in sample_words], dtype=np.float32)
    pair_weights = s_ij_mat * ref_freqs[None, :]
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
        e_tok = 0.0
        for a_i, w_ in enumerate(sample_words):
            e_tok += float(sample_freqs[a_i]) * word_energy(w_, profile=profile, env=env_fn(w_), cultural_attention=cultural_attention)

        r_tensors_by_len = {
            L2: np.stack([np.stack([p.g for p in ref_words[j].phonemes], axis=0) for j in r_idx], axis=0)
            for L2, r_idx in r_by_len.items()
        }
        conf_per_sample = np.zeros(len(sample_words), dtype=np.float64)
        for L1, s_idx in s_by_len.items():
            A1 = np.stack([np.stack([p.g for p in sample_words[i].phonemes], axis=0) for i in s_idx], axis=0)
            for L2, r_idx in r_by_len.items():
                A2 = r_tensors_by_len[L2]
                if L1 == L2:
                    S1, S2 = A1[:, None, :, :], A2[None, :, :, :]
                else:
                    max_L = max(L1, L2)
                    idx1 = np.linspace(0, L1 - 1, max_L).astype(np.int32)
                    idx2 = np.linspace(0, L2 - 1, max_L).astype(np.int32)
                    S1, S2 = A1[:, idx1, :][:, None, :, :], A2[:, idx2, :][None, :, :, :]
                D = np.mean(sensory_distance(S1, S2), axis=-1)
                W_sub = pair_weights[np.ix_(s_idx, r_idx)]
                conf_per_sample[s_idx] += np.sum(W_sub * np.exp(-(D * D) * inv_two_sig2), axis=1)

        e_conf = float(lam_conf * conf_scale * np.sum(sample_freqs * (conf_per_sample / n_ref_total)))
        return e_tok, e_conf

    tok0, conf0 = local_energy()

    undo = []
    for row, k in enumerate(members):
        wi, pi = refs[k]
        old = words[wi].phonemes[pi]
        undo.append((wi, pi, old))
        words[wi].phonemes[pi] = Phoneme(G_new[row], duration=old.duration)

    for wi in aff:
        words[wi].phonemes = words[wi].phonemes  # Synchronizes _morphemes and invalidates syllables/form

    tok1, conf1 = local_energy()
    dE = scale * ((tok1 - tok0) + (conf1 - conf0)) + d_inv
    accept = dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-6))

    if not accept:
        for wi, pi, old in undo:
            words[wi].phonemes[pi] = old
        for wi in aff:
            words[wi].phonemes = words[wi].phonemes  # Restores _morphemes and invalidates caches

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
        "d_inventory": float(d_inv),
    }


# -----------------------------------------------------------------------------
# 7. INTRINSIC-PROMINENCE SEGMENT SAMPLING (No Initial Inventory Needed)
# -----------------------------------------------------------------------------

_SEG_ENERGY_CACHE: dict[tuple, np.ndarray] = {}


def segment_energy_table(
    env=None,
    tau: float = 1.0,
    profile=None,
    cultural_attention: Sequence[float] | None = None,
) -> np.ndarray:
    alt = round(float(getattr(env, "altitude", 0.0)), 3) if env else 0.0
    hum = round(float(getattr(env, "humidity", 0.50)), 3) if env else 0.50
    temp = round(float(getattr(env, "temperature", 0.50)), 3) if env else 0.50
    veg = round(float(getattr(env, "vegetation", 0.30)), 3) if env else 0.30
    rough = round(float(getattr(env, "roughness", 0.10)), 3) if env else 0.10
    pg, sg = _societal_drives(profile=profile, env=env, cultural_attention=cultural_attention)

    key = (alt, hum, temp, veg, rough, round(float(tau), 3), round(pg, 3), round(sg, 3))
    cached = _SEG_ENERGY_CACHE.get(key)
    if cached is not None:
        return cached

    res = _segment_biomech_energy(
        _ATTRACTOR_MATRIX,
        taus=float(tau),
        env=env,
        profile=profile,
        cultural_attention=cultural_attention,
    ).astype(np.float32)

    if len(_SEG_ENERGY_CACHE) > 2048:
        _SEG_ENERGY_CACHE.clear()
    _SEG_ENERGY_CACHE[key] = res
    return res


def sample_segment(
    vowel: bool,
    counts: np.ndarray | None = None,
    env=None,
    population: float = 0.5,
    rng=None,
    T: float = 0.07,
    p_innovate: float = 0.025,
    profile=None,
    cultural_attention: Sequence[float] | None = None,
) -> int:
    """
    Draws a segment from an initially empty inventory:
      - Uses intrinsic acoustic clarity and physical energy to seed foundational phonemes (/a, i, u, p, t, k, m, n, s, l/)
      - When expanding, repels near-duplicate sounds in acoustic distance space so contrasts stay crisp.
    """
    rng = _as_np_rng(rng)
    idx = _V_IDX if vowel else _C_IDX
    E = segment_energy_table(env, profile=profile, cultural_attention=cultural_attention)[idx]
    slot = _SLOT_COST[idx] * (1.0 + 0.5 * math.tanh(max(0.0, population)))

    if counts is not None and len(counts) >= len(_ATTRACTOR_SYMBOLS):
        c = np.asarray(counts, dtype=np.float64)[idx]
    else:
        c = np.zeros(len(idx), dtype=np.float64)

    active_mask = c >= 0.5
    n_active = int(np.sum(active_mask))
    k_target = 0.70 if vowel else 3.0
    p_eff = p_innovate + (1.0 - p_innovate) * math.exp(-n_active / k_target)

    if c.sum() > 0 and rng.random() >= p_eff:
        temp_scale = 0.35 if vowel else 1.25
        w = c * np.exp(-(E - E.min()) / temp_scale)
    else:
        cand_mat = _ATTRACTOR_MATRIX[idx]
        pwr = continuous_acoustic_power(cand_mat, env=env, include_transients=True)
        if vowel:
            f1, f2 = _vowel_formants(cand_mat)
            disp_corner = 1.50 * ((f1 - 0.55) ** 2) + 1.20 * ((f2 - 0.50) ** 2)
            prominence = 0.35 * disp_corner + 0.15 * pwr
        else:
            prominence = 0.28 * pwr

        repulsion = np.zeros(len(idx), dtype=np.float64)
        if n_active > 0:
            used_mat = _ATTRACTOR_MATRIX[idx[active_mask]]
            d_mat = sensory_distance(cand_mat[:, None, :], used_mat[None, :, :])
            rep_sigma = 0.32 if vowel else 0.26
            repulsion = np.sum(np.exp(-(d_mat ** 2) / (2.0 * rep_sigma * rep_sigma)), axis=1)

        x = (E - prominence) + slot + 0.25 * repulsion
        w = np.exp(-(x - x.min()) / T)

    w = w / w.sum()
    return int(idx[int(rng.choice(len(idx), p=w))])