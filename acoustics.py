"""
acoustics.py — Continuous 10D Gestural Acoustic Modeling & Auditory Perception.
Maps continuous vocal-tract tensors [g0..g8, tau] to:
  1. Source-Filter Physics: F0, Phonation vs. Turbulence, Formants (F1..F4), Nasality
  2. 16-Band Bark Filterbank Spectrograms for Cochlear Representation
  3. Continuous Auditory Clarity & Distance (Without Symbolic Thresholds)
  4. On-Demand 16-bit PCM RIFF/WAVE Audio Synthesis
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence
import numpy as np

# 16 Critical-Band Bark Centers of the human auditory system (Zwicker & Fastl)
BARK_CENTERS = np.array([
    100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,
    1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0
], dtype=np.float32)

BARK_BANDWIDTHS = np.array([
    100.0, 100.0, 100.0, 100.0, 110.0, 120.0, 140.0, 150.0,
    160.0, 190.0, 210.0, 240.0, 280.0, 320.0, 380.0, 450.0
], dtype=np.float32)

_AF_CACHE: dict[tuple[bytes, float], AuditoryFeatures] = {}

@dataclass
class AuditoryFeatures:
    """Compact summary of auditory perception across a word trajectory."""
    spectrogram: np.ndarray        # (T, 16) Bark critical-band spectral energy
    pitch_contour: np.ndarray      # (T,) F0 in Hz over time
    clarity_score: float           # Intrinsic transmission salience
    harmonicity_ratio: float       # Ratio of periodic harmonic vs aperiodic noise energy
    temporal_flux: float           # Spectral edge sharpness across boundaries
    duration_seconds: float        # Total physical duration


class AcousticEngine:
    SAMPLE_RATE = 16000  # 16 kHz sampling rate for audio synthesis

    # -------------------------------------------------------------------------
    # 1. CONTINUOUS VOCAL TRACT ACOUSTICS (Tract -> Formant Resonances)
    # -------------------------------------------------------------------------

    @classmethod
    def tract_to_formants(cls, g: np.ndarray) -> tuple[float, float, float, float, float]:
        g0 = float(np.clip(g[0], 0.0, 1.0))
        g1 = float(np.clip(g[1], 0.0, 1.0))
        g3 = float(np.clip(g[3], 0.0, 1.0))
        g4 = float(np.clip(g[4], 0.0, 1.0))
        g5 = float(np.clip(g[5], 0.0, 1.0))
        g6 = float(np.clip(g[6], 0.0, 1.0))
        g8 = float(np.clip(g[8], 0.0, 1.0))

        f1 = 220.0 + 730.0 * (g4 ** 1.35) * (0.75 + 0.25 * g0)
        raw_f2 = 550.0 + 1850.0 * (1.0 - g5) * (1.0 - 0.65 * g1)
        f2 = 1500.0 + (1.0 - 0.65 * g4) * (raw_f2 - 1500.0)
        f3 = 2550.0 + 450.0 * (1.0 - g5) - 950.0 * (g3 ** 1.5)
        f4 = 3600.0 + 350.0 * (g8 - 0.50)
        nasal_damping = 1.0 - 0.40 * g6
        return f1, f2, f3, f4, nasal_damping

    @classmethod
    def glottal_source_physics(cls, g: np.ndarray, base_f0: float = 120.0) -> tuple[float, float, float, float]:
        g0 = float(np.clip(g[0], 0.0, 1.0))
        g1 = float(np.clip(g[1], 0.0, 1.0))
        g2 = float(np.clip(g[2], 0.0, 1.0))
        g4 = float(np.clip(g[4], 0.0, 1.0))
        g5 = float(np.clip(g[5], 0.0, 1.0))
        g6 = float(np.clip(g[6], 0.0, 1.0))
        g7 = float(np.clip(g[7], 0.0, 1.0))
        g8 = float(np.clip(g[8], 0.0, 1.0))

        a_oral = min(g0, g2, g4)
        modal_glottis = math.exp(-((g7 - 0.40) ** 2) / 0.025)
        transglottal = 1.0 - math.exp(-14.0 * a_oral) * ((1.0 - g6) ** 2)

        # Oral vowel radiation only scales when oral tract is actually open (prevents nasal leak boost)
        oral_open_gate = min(1.0, a_oral / 0.15)
        vowel_rad = oral_open_gate * max(a_oral, 0.62 * g1, 0.52 * (1.0 - g5) * g0)
        radiation_eff = 0.68 + 0.32 * min(1.0, vowel_rad + 0.30 * g6)
        voicing_amp = modal_glottis * transglottal * radiation_eff

        # Sustained frication requires pulmonic airflow (choked off if glottis is sealed or velum open)
        pulmonic_drive = min(1.0, g7 / 0.25) * max(0.25, 1.0 - g6)
        frication_noise = math.exp(-(((a_oral - 0.06) / 0.030) ** 2)) * (1.0 - 0.65 * voicing_amp) * (0.15 + 0.85 * pulmonic_drive)
        aspiration_noise = max(0.0, g7 - 0.52) * 0.55
        noise_amp = float(np.clip(frication_noise + aspiration_noise, 0.0, 1.0))

        # Oral burst at full stop closure; glottal burst (/ʔ/) requires open oral tract
        oral_burst = math.exp(-((a_oral / 0.020) ** 2)) * (1.0 - g6) * (0.65 + 0.35 * min(1.0, g7 / 0.60))
        glottal_burst = math.exp(-((g7 / 0.055) ** 2)) * 0.55 * min(1.0, a_oral / 0.30)
        burst_potential = float(np.clip(max(oral_burst, glottal_burst), 0.0, 1.0))

        f0 = max(55.0, min(390.0, base_f0 * (0.80 + 0.60 * g7 + 0.50 * (g8 - 0.50))))
        return f0, float(voicing_amp), noise_amp, burst_potential

    @classmethod
    def get_auditory_features(
        cls,
        phoneme_vectors: Sequence[np.ndarray] | np.ndarray,
        base_f0: float = 120.0,
    ) -> AuditoryFeatures:
        if phoneme_vectors is None or len(phoneme_vectors) == 0:
            return AuditoryFeatures(np.zeros((1, 16), dtype=np.float32), np.zeros(1), 0.0, 0.0, 0.0, 0.0)

        arr = np.asarray(phoneme_vectors, dtype=np.float32)
        key = (arr.tobytes(), round(float(base_f0), 1))
        cached = _AF_CACHE.get(key)
        if cached is not None:
            return cached

        frames_bark: list[np.ndarray] = []
        frames_f0: list[float] = []
        seg_barks: list[np.ndarray] = []
        total_periodic = 0.0
        total_aperiodic = 0.0
        time_step = 0.01
        bw_sq_2 = 2.0 * (np.array([135.0, 200.0, 250.0, 310.0], dtype=np.float32) ** 2)

        for vec in arr:
            g = vec[:9]
            duration = max(0.02, min(0.60, float(vec[9]) * 0.12))
            n_frames = max(1, int(round(duration / time_step)))

            f1, f2, f3, f4, nasal_damp = cls.tract_to_formants(g)
            f0, v_amp, n_amp, burst = cls.glottal_source_physics(g, base_f0=base_f0)

            poles = np.array([f1, f2, f3, f4], dtype=np.float32)
            diff_sq = (BARK_CENTERS[:, None] - poles[None, :]) ** 2
            transfer = np.sum(np.exp(-diff_sq / bw_sq_2), axis=1) * nasal_damp

            a_oral = float(min(g[0], g[2], g[4]))
            g6 = float(np.clip(g[6], 0.0, 1.0))
            oral_open_rad = 0.0 if (g6 > 0.30 and a_oral < 0.12) else min(1.0, (a_oral / 0.18) ** 0.95)
            blade_damp = 0.50 + 0.50 * min(1.0, float(g[2]) / 0.55)
            lip_damp = 0.40 + 0.60 * min(1.0, (float(g[0]) + 0.45 * float(g[1])) / 0.60)

            nasal_murmur = g6 * v_amp * (
                3.10 * np.exp(-((BARK_CENTERS - 250.0) ** 2) / (2.0 * 115.0 ** 2)) +
                0.95 * np.exp(-((BARK_CENTERS - 1000.0) ** 2) / (2.0 * 180.0 ** 2))
            )

            anterior_jet = min(1.0, float(g[0]) / 0.50) * min(1.0, float(g[4]) / 0.40) * max(0.30, min(1.0, (0.80 - float(g[5])) / 0.30))
            sibilant_peak = 1.85 * anterior_jet * np.exp(-((BARK_CENTERS - 2700.0) ** 2) / (2.0 * 520.0 ** 2))
            noise_profile = (0.35 + 0.45 * (BARK_CENTERS / 2000.0)) + sibilant_peak
            burst_profile = np.where(BARK_CENTERS > 1600.0, 1.8, 0.35)

            bark_frame = (
                v_amp * (1.45 * oral_open_rad * blade_damp * lip_damp * transfer) +
                nasal_murmur +
                n_amp * (0.78 * noise_profile) +
                burst * (0.85 * burst_profile)
            )
            seg_barks.append(bark_frame)

            total_periodic += v_amp * n_frames
            total_aperiodic += (n_amp + burst) * n_frames

            f0_val = f0 if v_amp > 0.06 else 0.0
            for _ in range(n_frames):
                frames_bark.append(bark_frame)
                frames_f0.append(f0_val)

        spectrogram = np.array(frames_bark, dtype=np.float32)
        pitch_contour = np.array(frames_f0, dtype=np.float32)

        mean_power = float(np.mean(spectrogram))
        peak_contrast = float(np.mean(np.max(spectrogram, axis=1) - np.min(spectrogram, axis=1)))
        temporal_flux = float(np.mean(np.abs(np.diff(spectrogram, axis=0)))) if len(spectrogram) > 1 else 0.1

        if len(seg_barks) > 1:
            seg_mat = np.asarray(seg_barks, dtype=np.float32)
            d_self = np.sqrt(np.sum((seg_mat[:, None, :] - seg_mat[None, :, :]) ** 2, axis=-1))
            novelties = [1.0]
            for i in range(1, len(seg_mat)):
                priors = int(np.sum(d_self[i, :i] < 0.70))
                if priors == 0:
                    novelties.append(1.0)
                elif priors == 1:
                    novelties.append(0.85)
                else:
                    novelties.append(0.45 ** (priors - 1))
            mean_novelty = sum(novelties) / len(novelties)
        else:
            mean_novelty = 1.0

        raw_clarity = (mean_power * 1.6) + (peak_contrast * 2.0) + (temporal_flux * 4.2)
        clarity = raw_clarity * (0.35 + 0.65 * mean_novelty)
        clarity_bounded = float(np.clip(clarity, 0.05, 10.0))
        harmonicity = float(total_periodic / (total_periodic + total_aperiodic + 1e-5))

        res = AuditoryFeatures(
            spectrogram=spectrogram,
            pitch_contour=pitch_contour,
            clarity_score=clarity_bounded,
            harmonicity_ratio=harmonicity,
            temporal_flux=temporal_flux,
            duration_seconds=float(len(frames_bark) * time_step),
        )
        if len(_AF_CACHE) > 4096:
            _AF_CACHE.clear()
        _AF_CACHE[key] = res
        return res

    @classmethod
    def auditory_distance(cls, af1: AuditoryFeatures, af2: AuditoryFeatures) -> float:
        """
        Cochlear perceived distance between two words across time-aligned Bark bands
        and pitch contours. Replaces heuristic margin thresholds.
        """
        S1, S2 = af1.spectrogram, af2.spectrogram
        if len(S1) == 0 or len(S2) == 0:
            return 1.0

        t_max = max(len(S1), len(S2))
        idx1 = np.linspace(0, len(S1) - 1, t_max).astype(np.int32)
        idx2 = np.linspace(0, len(S2) - 1, t_max).astype(np.int32)

        # Root-mean-square Bark critical band spectral difference
        bark_diff = S1[idx1] - S2[idx2]
        d_spectral = float(np.mean(np.sqrt(np.sum(bark_diff ** 2, axis=1))))

        # Intonation contour contrast (tonal distinction)
        p1, p2 = af1.pitch_contour[idx1], af2.pitch_contour[idx2]
        d_pitch = float(np.mean(np.abs(p1 - p2)) / 75.0)

        return d_spectral + 0.35 * d_pitch

    @classmethod
    def evaluate_pareto_fitness(
        cls,
        candidate_af: AuditoryFeatures,
        competitor_afs: Sequence[AuditoryFeatures],
        motor_effort: float,
        lambda_distinct: float = 2.4,
        lambda_effort: float = 0.8
    ) -> float:
        """
        Autonomous Pareto Selection:
        Peaks at LOW expense and HIGH clarity. Distinction dominates energy conservation.
        """
        clarity = candidate_af.clarity_score

        if competitor_afs:
            distances = [cls.auditory_distance(candidate_af, comp) for comp in competitor_afs]
            min_dist = min(distances)
            mean_dist = sum(distances) / len(distances)
            distinctiveness = 0.70 * min_dist + 0.30 * mean_dist
        else:
            distinctiveness = 1.0

        fitness = clarity + (lambda_distinct * distinctiveness) - (lambda_effort * motor_effort)
        return float(fitness)

    # -------------------------------------------------------------------------
    # 5. ON-DEMAND TIME-DOMAIN WAVEFORM SYNTHESIZER (Lazy Execution)
    # -------------------------------------------------------------------------

    @classmethod
    def synthesize_wav_pcm(
        cls,
        phoneme_vectors: Sequence[np.ndarray] | np.ndarray,
        base_f0: float = 120.0
    ) -> bytes:
        """
        Generates standard 16-bit 16kHz mono RIFF/WAVE audio bytes directly from 10D vectors.
        Run only on-demand when a user inspects or clicks a word in the browser UI.
        """
        if phoneme_vectors is None or len(phoneme_vectors) == 0:
            return cls._pack_wav_header(np.zeros(0, dtype=np.int16))

        audio_chunks: list[np.ndarray] = []
        phase = 0.0

        for vec in phoneme_vectors:
            g = vec[:9]
            duration = max(0.03, min(0.60, float(vec[9]) * 0.12))
            n_samples = int(duration * cls.SAMPLE_RATE)
            if n_samples <= 0:
                continue

            f1, f2, f3, _, nasal_damp = cls.tract_to_formants(g)
            f0, v_amp, n_amp, burst = cls.glottal_source_physics(g, base_f0=base_f0)

            t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)

            # 1. Periodic Glottal Pulse Generator (Rosenberg Pulse)
            if v_amp > 0.05:
                pitch_period = cls.SAMPLE_RATE / max(50.0, f0)
                phase_arr = (np.arange(n_samples, dtype=np.float32) + phase) % pitch_period
                duty = 0.42 * pitch_period
                glottal = np.where(phase_arr < duty, 0.5 * (1.0 - np.cos(np.pi * phase_arr / duty)), 0.0)
                phase = (phase + n_samples) % pitch_period
            else:
                glottal = np.zeros(n_samples, dtype=np.float32)

            # 2. Aperiodic Turbulence & Plosive Bursts
            noise = np.random.normal(0.0, 1.0, n_samples).astype(np.float32) * (n_amp * 0.40)
            if burst > 0.05:
                # Plosive burst transient in initial 12 ms
                burst_len = min(n_samples, int(0.012 * cls.SAMPLE_RATE))
                burst_env = np.linspace(1.0, 0.0, burst_len, dtype=np.float32) ** 2
                noise[:burst_len] += np.random.normal(0.0, 1.0, burst_len).astype(np.float32) * (burst * 0.90) * burst_env

            source = glottal * v_amp + noise

            # 3. Formant Resonance Oscillators (F1, F2, F3)
            sig = np.zeros(n_samples, dtype=np.float32)
            for f_pole, gain in zip((f1, f2, f3), (1.0, 0.70, 0.40)):
                damp = 0.05 * (f_pole / 1000.0)
                sig += np.sin(2.0 * np.pi * f_pole * t, dtype=np.float32) * np.exp(-damp * t * cls.SAMPLE_RATE * 0.001) * gain

            chunk = (source + sig * source * 0.85) * nasal_damp
            audio_chunks.append(chunk)

        if not audio_chunks:
            return cls._pack_wav_header(np.zeros(0, dtype=np.int16))

        full_signal = np.concatenate(audio_chunks)

        # Smooth edges to prevent clicks
        fade_len = min(120, len(full_signal))
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            full_signal[:fade_len] *= fade_in
            full_signal[-fade_len:] *= fade_out

        # Peak normalize to 16-bit integer range
        peak = float(np.max(np.abs(full_signal))) + 1e-5
        scaled = np.int16((full_signal / peak) * 27000)
        return cls._pack_wav_header(scaled)

    @classmethod
    def _pack_wav_header(cls, pcm_data: np.ndarray) -> bytes:
        """Packs raw 16-bit mono PCM samples into a standard RIFF/WAVE container."""
        byte_data = pcm_data.tobytes()
        num_channels = 1
        bytes_per_sample = 2
        block_align = num_channels * bytes_per_sample
        byte_rate = cls.SAMPLE_RATE * block_align
        data_size = len(byte_data)
        riff_chunk_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            riff_chunk_size,
            b'WAVE',
            b'fmt ',
            16,               # Subchunk1Size for PCM
            1,                # AudioFormat (1 = PCM)
            num_channels,
            cls.SAMPLE_RATE,
            byte_rate,
            block_align,
            bytes_per_sample * 8,
            b'data',
            data_size
        )
        return header + byte_data