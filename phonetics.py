"""
phonetics.py — Unified Continuous Articulatory Physics, Metric Tensors,
Thermodynamic Work, Metrical Syncope, and Syllable Phonotactics.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple, Sequence
from collections import Counter
import numpy as np

LONG_VOWEL_THRESHOLD = 1.35

INERTIA_DIAG = np.array([1.20, 0.60, 0.80, 0.40, 2.50, 2.00, 0.50, 1.00], dtype=np.float32)

_CHAO_BARS = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
# Natural 7-vowel inventory
_VOWEL_SET = {'i', 'u', 'e', 'o', 'a', 'ə', 'ɨ'}
_VOICELESS_STOPS = {'p', 't', 'ʈ', 'c', 'k', 'q', 'ʔ'}

# -----------------------------------------------------------------------------
# 1. QUANTAL METRICS & CONTINUOUS ACOUSTICS
# -----------------------------------------------------------------------------

def compute_metric_weights(g: np.ndarray) -> np.ndarray:
    """Computes sensitivity weights W(g) across the articulatory space."""
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

    return np.stack([w0, w1, w2, w3, w4, w5, w6, w7], axis=-1)


def sensory_distance(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Riemannian Quantal distance: ds = sqrt(dg^T * W(g_mid) * dg)."""
    g_mid = 0.50 * (g1 + g2)
    w = compute_metric_weights(g_mid)
    return np.sqrt(np.sum(w * ((g1 - g2) ** 2), axis=-1))


def trajectory_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    """Length-normalized distance between two word articulatory trajectories."""
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
    """Derives radiating acoustic power from vocal tract geometry."""
    oral_ap = min(float(g[0]), float(g[2]), float(g[4]))
    g7 = float(g[7])
    nasal_port = float(g[6])

    fric_noise = (oral_ap / 0.04) * math.exp(1.0 - (oral_ap / 0.04)) if 0.0 < oral_ap < 0.20 else 0.0
    voice_radiation = math.exp(-((g7 - 0.40) ** 2) / 0.02) * (0.30 + 1.70 * min(1.0, oral_ap * 3.0))
    nasal_damp = 1.0 - 0.25 * nasal_port

    return round(1.0 + oral_ap * 4.0 + voice_radiation * nasal_damp + fric_noise * 1.2, 3)


def is_vowel_gesture(g: np.ndarray) -> bool | np.ndarray:
    """
    True strictly when jaw is open and tongue tip is relaxed on the floor (g2 >= 0.55).
    Active coronal constrictions (t, d, s, z, l, r, n where g2 <= 0.35) are strictly consonants.
    """
    g_arr = np.asarray(g, dtype=np.float32)
    if g_arr.ndim == 1:
        return bool(float(g_arr[4]) >= 0.18 and float(g_arr[2]) >= 0.55 and float(g_arr[0]) >= 0.12)
    else:
        return (g_arr[..., 4] >= 0.18) & (g_arr[..., 2] >= 0.55) & (g_arr[..., 0] >= 0.12)


# -----------------------------------------------------------------------------
# 2. PHONEME CONTAINER
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
        tau_clamped = max(0.05, min(2.5, float(v[8])))
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
            tau = self.duration
            lip_e = 0.30 * (1.0 - g[0]) + 0.15 * max(0.0, g[1] - 0.50)
            tip_e = 0.35 * (1.0 - g[2]) + 0.20 * max(0.0, g[3] - 0.60)
            e_mot = round((0.10 + lip_e + tip_e + 0.35 * (1.0 - g[4]) + 0.40 * math.exp(-25.0 * g[7])) * (tau ** 1.1), 3)

            oral_ap = min(g[0], g[2], g[4])
            air = round((0.18 if oral_ap < 0.05 else 0.40 * oral_ap) * tau, 3)
            v_drag = round(math.exp(-((g[7] - 0.40) ** 2) / 0.02) * math.exp(-12.0 * oral_ap) * 0.5 * tau, 3)

            if self.is_vowel:
                disp = math.sqrt((g[4] - 0.55) ** 2 + 2.0 * (g[5] - 0.50) ** 2)
                aud = round(0.50 + 0.50 * math.tanh(3.0 * disp), 3)
            else:
                # Sibilant and liquid acoustic payoffs
                aud = round(0.75 if (g[2] <= 0.10 and g[7] >= 0.60) else (0.65 if g[2] <= 0.30 else 0.45), 3)

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
# 3. DYNAMICS, DISPERSION & DRIFT
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
    word_cost: float = 10.0,
) -> Phoneme:
    """Continuous gradient flow with physical laryngeal restoring forces."""
    g = p.g.copy()
    is_voc = float(is_vowel_gesture(g))

    cost_pressure = max(0.7, min(2.5, word_cost / 10.0))
    drift_scale = min(2.8, max(0.1, t_drift * cost_pressure))
    noise = np.array([rng.gauss(0.0, 0.018 * drift_scale) for _ in range(8)], dtype=np.float32)

    rho = math.exp(-0.80 * max(0.0, alt))
    chill = max(0.0, 0.50 - temp) * 1.5
    h_eff = hum + 0.08

    f_env = np.zeros(8, dtype=np.float32)
    f_env[4] = (0.25 * hum - 0.30 * chill) * 0.10 * is_voc

    # Continuous Laryngeal Physics
    if is_voc > 0.5:
        f_env[7] = -0.35 * (g[7] - 0.40)
    else:
        grad_rest = -0.08 * (g[7] - 0.40)
        pump_delta = max(0.0, 0.40 - float(g[7]))
        grad_pump = rho * pump_delta * 1.40

        shear_act = math.exp(-((g[7] - 0.18) ** 2) / 0.0128)
        grad_shear = (1.0 / h_eff) * shear_act * (float(g[7]) - 0.18) * 4.0

        flow_delta = max(0.0, float(g[7]) - 0.40)
        grad_flow = -flow_delta * ((1.0 / rho) + (0.6 / h_eff) + chill) * 0.20

        f_env[7] = grad_rest + grad_pump + grad_shear + grad_flow

    effective_bias = bias.copy()
    if is_voc > 0.5 and effective_bias[7] < 0.0:
        effective_bias[7] = 0.0

    # Coarticulation & Intervocalic Lenition (Stops naturally spirantize / tap between vowels)
    f_coartic = np.zeros(8, dtype=np.float32)
    if adjacent_p is not None:
        adj = adjacent_p.g
        adj_is_voc = is_vowel_gesture(adj)
        if not is_voc and adj_is_voc:
            # Undershoot: consonants flanked by open vowels undergo natural opening drift toward sibilants/liquids
            f_coartic[2] += 0.04 * (0.15 - g[2])
        elif is_voc:
            f_coartic[1] += 0.08 * (adj[1] - g[1])
            f_coartic[5] += 0.08 * (adj[5] - g[5])

    f_disp = np.zeros(8, dtype=np.float32)
    if is_voc > 0.5:
        dh = g[4] - 0.55
        db = g[5] - 0.50
        dc = math.sqrt(dh ** 2 + db ** 2)
        if dc < 0.30:
            scale = (0.30 - dc) * 0.45
            f_disp[4] = (dh / (dc + 0.01)) * scale
            f_disp[5] = (db / (dc + 0.01)) * scale

    f_repulse = np.zeros(8, dtype=np.float32)
    if neighbor_g_pool:
        pool_mat = np.stack(neighbor_g_pool[:20], axis=0)
        diffs = g[np.newaxis, :] - pool_mat
        d_sqs = np.sum(diffs ** 2, axis=-1, keepdims=True)
        f_repulse += np.sum(diffs / (d_sqs + 0.03), axis=0) * 0.025

    dg = noise + f_env + f_coartic + f_disp + f_repulse + (effective_bias * 0.20)
    g_new = np.clip(g + dg, 0.0, 1.0)

    if is_voc > 0.5 and g_new[4] > 0.75:
        g_new[1] = min(g_new[1], 1.2 - g_new[4])

    return Phoneme(g_new, duration=p.duration)


def apply_syncope_and_clustering(
    phonemes: list[Phoneme],
    usage_frequency: float,
    alt: float,
    hum: float,
    temp: float,
    profile=None,
    word_cost: float = 0.0,
) -> list[Phoneme]:
    """Elides unstressed vowels and performs variational truncation on high-energy words."""
    L = len(phonemes)
    if L <= 2:
        return phonemes

    effective_cost = word_cost if word_cost > 0.0 else (L * 1.75)
    s_complex = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    max_cluster = 2 if s_complex < 0.60 else 3
    h = getattr(profile, "head_directionality", 0.50) if profile else 0.50

    X = np.stack([p.vector for p in phonemes], axis=0)
    syl_slices, stressed_idx, _, _ = parse_sonority_prosody(X, directional_bias=h)

    while (effective_cost > 15.0 or len(phonemes) > 8) and len(syl_slices) >= 3:
        son = np.array([continuous_sonority(X[i, :8]) for i in range(len(phonemes))], dtype=np.float32)
        syl_masses = [sum(X[i, 8] * son[i] for i in range(st, en)) for st, en in syl_slices]
        weakest_k = min(
            (k for k in range(len(syl_slices)) if k != stressed_idx and k != 0),
            key=lambda k: syl_masses[k],
            default=None
        )
        if weakest_k is None:
            break
        w_st, w_en = syl_slices[weakest_k]
        phonemes = [p for idx, p in enumerate(phonemes) if not (w_st <= idx < w_en)]
        if len(phonemes) <= 2:
            return phonemes
        effective_cost *= 0.65
        X = np.stack([p.vector for p in phonemes], axis=0)
        syl_slices, stressed_idx, _, _ = parse_sonority_prosody(X, directional_bias=h)

    L = len(phonemes)
    rho = math.exp(-0.90 * max(0.0, alt))
    pulmonary_tax = 1.0 + 3.0 / (1.0 + math.exp(12.0 * (rho - 0.75))) + 1.8 / (1.0 + math.exp(10.0 * (hum - 0.40)))
    thermal_chill = 1.5 / (1.0 + math.exp(10.0 * (temp - 0.50)))

    cost_mult = max(1.0, min(3.0, (effective_cost / 10.0) ** 1.3))
    freq_boost = (math.log1p(usage_frequency) ** 1.25)
    drain = (0.08 + 0.05 * freq_boost) * (pulmonary_tax + thermal_chill) * ((L / 4.0) ** 0.55) * cost_mult

    eroded: list[Phoneme] = []
    for k, (st, en) in enumerate(syl_slices):
        is_stressed = (k == stressed_idx)
        for j in range(st, en):
            p = phonemes[j]
            tau = p.duration
            if not is_stressed and p.is_vowel:
                tau -= drain
                p = Phoneme(p.g, duration=tau)

            if tau <= 0.15 and p.is_vowel and not is_stressed:
                prev_cons = sum(1 for x in reversed(eroded) if not x.is_vowel)
                next_cons = sum(1 for x in phonemes[j + 1:j + 3] if not x.is_vowel)

                if prev_cons + next_cons >= max_cluster:
                    eroded.append(Phoneme(p.g, duration=0.20))
                    continue

                if len(eroded) == 1 and not eroded[0].is_vowel and j + 1 < len(phonemes) and not phonemes[j + 1].is_vowel:
                    son1 = continuous_sonority(eroded[0].g)
                    son2 = continuous_sonority(phonemes[j + 1].g)
                    if s_complex < 0.65 and son2 <= son1 + 0.3:
                        eroded.append(Phoneme(p.g, duration=0.20))
                        continue

                continue

            eroded.append(p)

    if not any(p.is_vowel for p in eroded):
        return phonemes
    return repair_phonemes(eroded, profile=profile)


# -----------------------------------------------------------------------------
# 4. PROSODY, HIATUS & SYLLABIFICATION
# -----------------------------------------------------------------------------

def resolve_chao_tone(pitch: tuple[float, float, float] | np.ndarray, duration: float = 1.0) -> str:
    f0, f1, f2 = float(pitch[0]), float(pitch[1]), float(pitch[2])
    c0 = int(np.clip(np.floor(f0 * 2.50 + 3.50), 1, 5))
    c1 = int(np.clip(np.floor(f1 * 2.50 + 3.50), 1, 5))
    c2 = int(np.clip(np.floor(f2 * 2.50 + 3.50), 1, 5))

    if duration < 0.80:
        return _CHAO_BARS[c1]
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
        peaks = [int(np.argmax(son))]

    boundaries = [0]
    for k in range(len(peaks) - 1):
        p_start, p_end = peaks[k] + 1, peaks[k + 1]
        boundaries.append(p_start if p_start == p_end else p_start + int(np.argmin(son[p_start:p_end])))
    boundaries.append(L)

    syl_slices = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    syl_masses = [sum(X[i, 8] * son[i] for i in range(st, en)) * (1.0 + (k / max(1, len(syl_slices) - 1)) * (directional_bias - 0.50) * 0.30) for k, (st, en) in enumerate(syl_slices)]
    stressed_idx = int(np.argmax(syl_masses))

    trajectories = []
    nuc_durs = []
    for k, (st, en) in enumerate(syl_slices):
        pk = peaks[min(k, len(peaks) - 1)]
        f0 = 1.50 * (float(X[pk - 1, 7]) - 0.40) if pk > st else 0.0
        f1 = f0 + (0.25 if k == stressed_idx else 0.0)
        f2 = f0
        if en > pk + 1 and float(X[pk + 1, 7]) <= 0.15:
            f2 = min(1.0, f0 + 0.60)
        trajectories.append((round(f0, 2), round(f1, 2), round(f2, 2)))
        nuc_durs.append(float(X[pk, 8]))

    return syl_slices, stressed_idx, trajectories, nuc_durs


class Syllable:
    __slots__ = ('onset', 'nucleus', 'coda', 'stressed', 'pitch')

    def __init__(
        self,
        onset: list[Phoneme],
        nucleus: list[Phoneme],
        coda: list[Phoneme],
        stressed: bool = False,
        pitch: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.onset = onset
        self.nucleus = nucleus
        self.coda = coda
        self.stressed = stressed
        self.pitch = pitch

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.nucleus)

    @property
    def mass(self) -> float:
        return sum(p.duration * p.energy.audibility_payoff for p in self.to_phonemes())

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
# 5. RECURSIVE DEGEMINATION & PHONOTACTIC REPAIR
# -----------------------------------------------------------------------------

def transition_cost(p1: Phoneme | np.ndarray, p2: Phoneme | np.ndarray, tau1: float = 1.0, tau2: float = 1.0) -> float:
    """Computes kinetic work plus organ-specific OCP contact friction."""
    g1 = p1.g if hasattr(p1, "g") else p1[:8]
    g2 = p2.g if hasattr(p2, "g") else p2[:8]
    t1 = p1.duration if hasattr(p1, "duration") else tau1
    t2 = p2.duration if hasattr(p2, "duration") else tau2

    diff = g1 - g2
    min_tau = max(0.15, min(t1, t2))
    kinetic = float(np.sum(INERTIA_DIAG * (diff ** 2)) / (min_tau ** 0.80))

    v1, v2 = is_vowel_gesture(g1), is_vowel_gesture(g2)
    dist = float(sensory_distance(g1, g2))

    stutter_tax = 0.0
    if v1 and v2:
        stutter_tax = 2.50 * math.exp(-4.0 * max(0.0, dist - 0.10))
    elif not v1 and not v2:
        same_organ = (g1[0] < 0.2 and g2[0] < 0.2) or (g1[2] < 0.35 and g2[2] < 0.35) or (g1[4] < 0.2 and g2[4] < 0.2)
        if same_organ:
            stutter_tax = 2.00 * math.exp(-3.0 * dist)

    cost = kinetic + stutter_tax + (0.40 if (v1 and v2) else 0.0)
    return round(float(cost), 3)


def _harmonic_vowel_vector(p: Phoneme) -> np.ndarray:
    """Synthesizes harmonic epenthetic vowel vector matching preceding consonant place."""
    if p.g[5] < 0.30 or p.g[3] < 0.20:
        return np.array([1.0, 0.0, 1.0, 0.0, 0.25, 0.10, 0.0, 0.40], dtype=np.float32)  # [i]
    elif p.g[0] < 0.20 or p.g[5] > 0.70:
        return np.array([0.3, 0.9, 1.0, 0.0, 0.25, 0.90, 0.0, 0.40], dtype=np.float32)  # [u]
    return np.array([1.0, 0.0, 1.0, 0.0, 0.55, 0.50, 0.0, 0.40], dtype=np.float32)       # [ə]


def repair_phonemes(phonemes: list[Phoneme], profile=None, max_passes: int = 4) -> list[Phoneme]:
    """Phonotactic repair engine: recursive degemination, hiatus resolution, and cluster repairs."""
    s_complex = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    max_cluster = 2 if s_complex < 0.60 else 3

    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(phonemes) - 1:
            p1, p2 = phonemes[i], phonemes[i + 1]

            # 1. Full-Word Degemination: merge adjacent identical or near-identical consonants or glides
            if not p1.is_vowel and not p2.is_vowel:
                dist = sensory_distance(p1.g, p2.g)
                same_organ = (p1.g[0] < 0.25 and p2.g[0] < 0.25) or (p1.g[2] < 0.35 and p2.g[2] < 0.35) or (p1.g[4] < 0.25 and p2.g[4] < 0.25)
                if dist < 0.22 or (same_organ and dist < 0.32):
                    coalesced_dur = min(1.5, p1.duration * 1.10)
                    phonemes[i] = Phoneme(p1.g, duration=coalesced_dur)
                    phonemes.pop(i + 1)
                    changed = True
                    continue

            # 2. Hiatus resolution
            if p1.is_vowel and p2.is_vowel:
                dist = sensory_distance(p1.g, p2.g)
                if dist < 0.30:
                    coalesced_g = 0.50 * (p1.g + p2.g)
                    phonemes[i] = Phoneme(coalesced_g, duration=min(2.0, p1.duration + p2.duration * 0.6))
                    phonemes.pop(i + 1)
                    changed = True
                    continue
                else:
                    glide_g = p1.g.copy()
                    glide_g[4] = 0.20
                    if glide_g[5] > 0.50 or glide_g[1] > 0.50:
                        glide_g[0], glide_g[1] = 0.15, 1.00  # [w]
                    else:
                        glide_g[0], glide_g[1], glide_g[5] = 1.00, 0.00, 0.05  # [j]
                    phonemes.insert(i + 1, Phoneme(glide_g, duration=0.40))
                    changed = True
                    i += 2
                    continue

            # 3. Trapped Sonority Peak (e.g. liquid between stops /t-l-t/)
            if i + 2 < len(phonemes):
                p_prev, p_curr, p_next = phonemes[i], phonemes[i + 1], phonemes[i + 2]
                if not p_prev.is_vowel and not p_curr.is_vowel and not p_next.is_vowel:
                    s_prev = continuous_sonority(p_prev.g)
                    s_curr = continuous_sonority(p_curr.g)
                    s_next = continuous_sonority(p_next.g)
                    if s_curr > s_prev + 0.4 and s_curr > s_next + 0.4:
                        ep_g = _harmonic_vowel_vector(p_prev)
                        phonemes.insert(i + 1, Phoneme(ep_g, duration=0.50))
                        changed = True
                        i += 2
                        continue

            # 4. Word-Initial Onset Sonority Violation (e.g. Stop + Stop #tb-)
            if i == 0 and len(phonemes) >= 2 and not p1.is_vowel and not p2.is_vowel:
                s1 = continuous_sonority(p1.g)
                s2 = continuous_sonority(p2.g)
                if s_complex < 0.65 and s2 <= s1 + 0.2:
                    ep_g = _harmonic_vowel_vector(p1)
                    phonemes.insert(1, Phoneme(ep_g, duration=0.50))
                    changed = True
                    continue

            # 5. Consonant cluster budget repair
            if not p1.is_vowel and not p2.is_vowel:
                streak = 1
                k = i + 1
                while k < len(phonemes) and not phonemes[k].is_vowel:
                    streak += 1
                    k += 1

                if streak > max_cluster:
                    ep_g = _harmonic_vowel_vector(p1)
                    phonemes.insert(i + 1, Phoneme(ep_g, duration=0.60))
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
    """Evaluates total communicative dissipation: E = E_motor + E_larynx + E_length + E_mod + E_ambig."""
    phonemes = word.phonemes
    n = len(phonemes)
    if n == 0:
        return 0.0

    alt = getattr(env, "altitude", 0.0) if env else 0.0
    hum = getattr(env, "humidity", 0.5) if env else 0.5
    temp = getattr(env, "temperature", 0.5) if env else 0.5

    rho = math.exp(-0.80 * max(0.0, alt))
    chill = max(0.0, 0.50 - temp) * 1.5
    h_eff = hum + 0.08

    # 1. Base motor effort & transition costs
    e_motor = sum(p.energy.motor_effort + p.energy.air_consumption for p in phonemes)
    for j in range(n - 1):
        e_motor += transition_cost(phonemes[j], phonemes[j + 1]) * 0.45

    # 2. Continuous Atmospheric-Laryngeal Potentials
    e_larynx = 0.0
    for p in phonemes:
        g = p.g
        if not is_vowel_gesture(g):
            e_pump = rho * (max(0.0, 0.40 - float(g[7])) ** 2) * 5.5
            e_shear = (1.5 / h_eff) * math.exp(-((float(g[7]) - 0.18) ** 2) / 0.0128)
            e_flow = (max(0.0, float(g[7]) - 0.40) ** 2) * ((0.8 / rho) + (0.5 / h_eff) + chill)
            e_larynx += e_pump + e_shear + e_flow

    # 3. Superlinear metabolic word length tax
    s_idx = getattr(profile, "synthesis_index", 0.50) if profile else 0.50
    synthesis_tolerance = 1.0 - 0.60 * s_idx
    freq_load = math.log(getattr(word, "usage_frequency", 1.0) + 1.0)
    e_length = 0.14 * (n ** 1.68) * synthesis_tolerance * (0.60 + 0.40 * freq_load)

    # 4. Intra-word sonority modulation
    sonorities = [continuous_sonority(p.g) for p in phonemes]
    var_son = float(np.var(sonorities)) if len(sonorities) > 1 else 0.5
    e_mod = 1.80 / (var_son + 0.30)

    # 5. Lexical ambiguity cost against cohort
    e_ambig = 0.0
    if cohort_sample and len(cohort_sample) > 1:
        w_tensor = word.tensor[:, :8]
        w_vec = word.vector
        for other in cohort_sample[:25]:
            if other.id == word.id or len(other.phonemes) == 0:
                continue
            d_acoust = trajectory_distance(w_tensor, other.tensor[:, :8])
            d_sem = w_vec.distance_to(other.vector)
            e_ambig += (d_sem ** 2) / (d_acoust ** 2 + 0.06)
        e_ambig *= 0.10

    return round(float(e_motor + e_larynx + e_length + e_mod + e_ambig), 3)


# -----------------------------------------------------------------------------
# 6. SERIALIZATION & BALANCED IPA QUANTIZATION
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
    parts, plain_parts = [], []
    assigned_tone = ""

    for k, (st, en) in enumerate(syl_slices):
        if k == stressed_idx and len(syl_slices) > 1:
            parts.append("ˈ")

        syl_sonorities = [continuous_sonority(X[i, :8]) for i in range(st, en)]
        peak_offset = int(np.argmax(syl_sonorities))

        syl_symbols = [
            resolve_tract_vector(X[st + off, :8], is_nucleus=(off == peak_offset))
            for off in range(en - st)
        ]

        dedup_symbols: list[str] = []
        for sym in syl_symbols:
            if dedup_symbols and dedup_symbols[-1] == sym:
                if sym in _VOWEL_SET:
                    if not dedup_symbols[-1].endswith("ː"):
                        dedup_symbols[-1] = f"{dedup_symbols[-1]}ː"
                continue
            dedup_symbols.append(sym)

        syl_str = "".join(dedup_symbols)
        plain_parts.append(syl_str.replace("ː", ""))

        if is_tonal and (k == stressed_idx or len(syl_slices) == 1):
            tone_cat = resolve_chao_tone(pitches[k])
            syl_str += tone_cat
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

    contrastive_v = sorted([v for v, cnt in v_counts.items() if (cnt / tot_v) >= 0.03])
    contrastive_c = sorted([c for c, cnt in c_counts.items() if (cnt / tot_c) >= 0.02])

    return (contrastive_v or ["a"]), (contrastive_c or ["t"])


def render_word_trajectory(X: np.ndarray, directional_bias: float = 0.50) -> tuple[str, str]:
    return render_and_extract_tones(X, directional_bias=directional_bias, is_tonal=False)[:2]


# -----------------------------------------------------------------------------
# 7. NATURAL 7-VOWEL CANONICAL IPA ATTRACTORS (Strict Separation)
# -----------------------------------------------------------------------------

_CANONICAL_ATTRACTORS: list[tuple[str, list[float]]] = [
    # --- NATURAL 7-VOWEL SYSTEM (Tongue tip relaxed on floor: g2 = 1.0) ---
    ("i",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.08, 0.0, 0.40]),
    ("u",  [0.30, 1.0, 1.00, 0.00, 0.25, 0.90, 0.0, 0.40]),
    ("ɨ",  [1.00, 0.0, 1.00, 0.00, 0.25, 0.50, 0.0, 0.40]),
    ("e",  [1.00, 0.0, 1.00, 0.00, 0.50, 0.10, 0.0, 0.40]),
    ("o",  [0.35, 0.9, 1.00, 0.00, 0.50, 0.90, 0.0, 0.40]),
    ("ə",  [1.00, 0.0, 1.00, 0.00, 0.55, 0.50, 0.0, 0.40]),
    ("a",  [1.00, 0.0, 1.00, 0.00, 0.95, 0.40, 0.0, 0.40]),

    # --- STOPS ---
    ("p",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.60]),
    ("b",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("t",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.60]),
    ("d",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.40]),
    ("ʈ",  [1.00, 0.0, 0.00, 0.85, 0.80, 0.50, 0.0, 0.60]),
    ("c",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.08, 0.0, 0.60]),
    ("k",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.60]),
    ("ɡ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.40]),
    ("q",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.98, 0.0, 0.60]),
    ("ʔ",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.02]),

    # --- FRICATIVES ---
    ("ɸ",  [0.05, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.65]),
    ("β",  [0.05, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("f",  [0.08, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.65]),
    ("v",  [0.08, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.40]),
    ("s",  [1.00, 0.0, 0.06, 0.25, 0.80, 0.50, 0.0, 0.65]),  # Distinct sibilant groove
    ("z",  [1.00, 0.0, 0.06, 0.25, 0.80, 0.50, 0.0, 0.40]),
    ("ʃ",  [1.00, 0.5, 0.04, 0.60, 0.50, 0.20, 0.0, 0.65]),
    ("ʒ",  [1.00, 0.5, 0.04, 0.60, 0.50, 0.20, 0.0, 0.40]),
    ("x",  [1.00, 0.0, 1.00, 0.00, 0.10, 0.75, 0.0, 0.65]),
    ("ɣ",  [1.00, 0.0, 1.00, 0.00, 0.10, 0.75, 0.0, 0.40]),
    ("h",  [1.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.90]),

    # --- NASALS & LIQUIDS (Strictly distinct g2 and manner) ---
    ("m",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.60, 0.40]),
    ("n",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.60, 0.40]),
    ("ŋ",  [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.60, 0.40]),
    ("l",  [1.00, 0.0, 0.28, 0.25, 0.60, 0.30, 0.00, 0.40]),  # Lateral Liquid
    ("r",  [1.00, 0.0, 0.14, 0.25, 0.60, 0.40, 0.00, 0.40]),  # Rhotic Tap
    ("j",  [1.00, 0.0, 1.00, 0.00, 0.15, 0.08, 0.00, 0.40]),
    ("w",  [0.15, 1.0, 1.00, 0.00, 0.15, 0.90, 0.00, 0.40]),

    # --- LARYNGEAL SERIES ---
    ("pʼ", [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.02]),
    ("tʼ", [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.02]),
    ("kʼ", [1.00, 0.0, 1.00, 0.00, 0.00, 0.75, 0.0, 0.02]),
    ("ɓ",  [0.00, 0.0, 1.00, 0.00, 0.80, 0.50, 0.0, 0.15]),
    ("ɗ",  [1.00, 0.0, 0.00, 0.25, 0.80, 0.50, 0.0, 0.15]),
]

_ATTRACTOR_SYMBOLS: tuple[str, ...] = tuple(item[0] for item in _CANONICAL_ATTRACTORS)
_ATTRACTOR_MATRIX: np.ndarray = np.array([item[1] for item in _CANONICAL_ATTRACTORS], dtype=np.float32)


def resolve_tract_vector(g: np.ndarray, is_nucleus: bool | None = None) -> str:
    """Vectorized Quantal lookup. Syllable nuclei strictly map to vowels."""
    g_arr = np.asarray(g, dtype=np.float32)

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

    if symbol in _VOICELESS_STOPS and g_arr[7] >= 0.80:
        symbol = f"{symbol}ʰ"
    return symbol