"""
word.py — Phoneme, Morpheme, and Word containers with vectorized diachronic history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Sequence
import numpy as np

from acoustics import AcousticEngine
from phonetics import (
    REST_POSTURE,
    STIFFNESS,
    is_vowel_gesture,
    oral_aperture,
    render_word_trajectory,
    resolve_tract_vector,
    sensory_distance,
    trajectory_distance,
    transition_cost,
    word_energy,
)
from semantics import ClineStage, DerivationRecord, DerivationType, SemanticVector


class LexicalCategory(Enum):
    CONTENT_OPEN = "content_open"
    FUNCTIONAL_CLOSED = "functional"


@dataclass
class PhoneticHistoryEntry:
    epoch: int
    tensor: np.ndarray
    event: str


class PhonemeEnergy(NamedTuple):
    motor_effort: float
    air_consumption: float
    voicing_drag: float
    audibility_payoff: float

    @property
    def acoustic_salience(self) -> float:
        return self.audibility_payoff


class Phoneme:
    __slots__ = ('_vector', '_ipa', '_energy', '_hash', '_is_v')

    def __init__(self, vector: Sequence[float] | np.ndarray | Phoneme, duration: float | None = None):
        if isinstance(vector, Phoneme):
            vec = vector._vector.copy()
            if duration is not None:
                vec[9] = max(0.01, min(3.0, float(duration)))
        else:
            v = list(vector)
            n = len(v)
            if n == 9:
                v.append(float(duration) if duration is not None else 1.0)
            elif n == 10:
                if duration is not None:
                    v[9] = float(duration)
            elif n == 8:
                v.append(0.50)
                v.append(float(duration) if duration is not None else 1.0)
            else:
                raise ValueError(f"Expected 8, 9, or 10 elements for Phoneme vector, got {n}")

            vec = np.empty(10, dtype=np.float32)
            for i in range(9):
                x = float(v[i])
                vec[i] = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            tau = float(v[9])
            vec[9] = 0.01 if tau < 0.01 else (3.0 if tau > 3.0 else tau)

        vec.flags.writeable = False
        self._vector = vec
        self._ipa = None
        self._energy = None
        self._is_v = None
        self._hash = hash(self._vector.tobytes())

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    @property
    def g(self) -> np.ndarray:
        return self._vector[:9]

    @property
    def duration(self) -> float:
        return float(self._vector[9])

    @property
    def tau(self) -> float:
        return float(self._vector[9])

    @property
    def is_vowel(self) -> bool:
        if self._is_v is None:
            self._is_v = bool(is_vowel_gesture(self._vector[:9]))
        return self._is_v

    @property
    def voc(self) -> float:
        return 1.0 if self.is_vowel else 0.0

    @property
    def ipa(self) -> str:
        if self._ipa is None:
            base = resolve_tract_vector(self._vector[:9])
            self._ipa = f"{base}ː" if self._vector[9] >= 1.35 else base
        return self._ipa

    @property
    def energy(self) -> PhonemeEnergy:
        if self._energy is None:
            g = self._vector[:9]
            tau = float(self._vector[9])
            a_oral = float(oral_aperture(g))
            e_mot = round(float(np.sum(STIFFNESS * ((g - REST_POSTURE) ** 2))) * tau, 3)
            air = round(float((0.10 + 0.40 * a_oral + 0.12 * g[6]) * (0.30 + g[7]) * tau), 3)
            v_drag = round(
                0.24 * math.exp(-((g[7] - 0.40) / 0.14) ** 2) * math.exp(-15.0 * a_oral) * ((1.0 - g[6]) ** 2) * tau,
                3,
            )
            af = AcousticEngine.get_auditory_features([self._vector])
            self._energy = PhonemeEnergy(e_mot, air, v_drag, round(af.clarity_score, 3))
        return self._energy

    def drift(self, delta: Sequence[float] | np.ndarray | None = None, target: Sequence[float] | np.ndarray | None = None) -> Phoneme:
        if target is not None:
            return Phoneme(target)
        if delta is not None:
            new_v = self._vector.copy()
            k = min(len(delta), 10)
            for i in range(k):
                new_v[i] += float(delta[i])
            return Phoneme(new_v)
        return Phoneme(self)

    def synthesize_wav(self, base_f0: float = 120.0) -> bytes:
        return AcousticEngine.synthesize_wav_pcm([self._vector], base_f0=base_f0)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phoneme) and np.array_equal(self._vector, other._vector)

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> float:
        return float(self._vector[idx])

    def __iter__(self):
        return iter(self._vector)

    def __repr__(self) -> str:
        return f"Phoneme(/{self.ipa}/, tau={self.duration:.2f})"


def repair_phonemes(phonemes: Sequence[Phoneme], profile=None, max_passes: int = 3) -> list[Phoneme]:
    s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
    budget = 0.85 + 2.20 * s_comp
    out = list(phonemes)

    for _ in range(max_passes):
        if len(out) <= 1:
            break
        changed = False
        i = 0
        while i < len(out) - 1:
            p1, p2 = out[i], out[i + 1]
            if float(sensory_distance(p1.g, p2.g)) < 0.15:
                out[i] = Phoneme(0.50 * (p1.g + p2.g), duration=min(2.5, p1.duration + 0.6 * p2.duration))
                out.pop(i + 1)
                changed = True
                continue
            if p1.is_vowel == p2.is_vowel and transition_cost(p1, p2) > budget:
                tot = p1.duration + p2.duration
                g_mix = (p1.duration * p1.g + p2.duration * p2.g) / tot
                out[i] = Phoneme(g_mix, duration=min(2.0, tot * 0.70))
                out.pop(i + 1)
                changed = True
                continue
            i += 1
        if not changed:
            break
    return out


class Morpheme:
    __slots__ = ('phonemes', 'root_family_id', 'is_grammatical', 'stage')

    def __init__(
        self,
        phonemes: Sequence[Phoneme],
        root_family_id: int,
        is_grammatical: bool = False,
        stage: ClineStage = ClineStage.FREE_ROOT,
    ):
        self.phonemes = list(phonemes)
        self.root_family_id = int(root_family_id)
        self.is_grammatical = bool(is_grammatical)
        self.stage = stage

    @property
    def ipa(self) -> str:
        return "".join(p.ipa for p in self.phonemes)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phonemes)

    def copy(self) -> Morpheme:
        return Morpheme(
            phonemes=[p.drift() for p in self.phonemes],
            root_family_id=self.root_family_id,
            is_grammatical=self.is_grammatical,
            stage=self.stage,
        )

    def __repr__(self) -> str:
        cat = "GRAM" if self.is_grammatical else "LEX"
        return f"Morpheme(/{self.ipa}/, fam={self.root_family_id}, {cat})"


class Word:
    __slots__ = (
        'id', 'vector', '_phonemes', '_morphemes',
        'derivation', 'usage_frequency', 'category', 'generation_born',
        'epochs_idle', 'senses', '_form', '_plain_form',
        '_cached_cost', '_cost_env_key', 'is_archaic', 'replaced_by_id',
        '_history'
    )

    _ID_COUNTER = 1000
    _EPHEMERAL_COUNTER = -1

    def __init__(
        self,
        vector: SemanticVector,
        phonemes: Sequence[Phoneme] | None = None,
        morphemes: Sequence[Morpheme] | None = None,
        derivation: DerivationRecord | None = None,
        word_id: int | None = None,
        usage_frequency: float = 1.0,
        category: LexicalCategory = LexicalCategory.CONTENT_OPEN,
        generation_born: int = 0,
        senses: Sequence[SemanticVector] | tuple[SemanticVector, ...] | None = None,
        is_ephemeral: bool = False,
        is_archaic: bool = False,
        history: list[PhoneticHistoryEntry] | None = None,
    ):
        if word_id is not None:
            self.id = word_id
        elif is_ephemeral:
            Word._EPHEMERAL_COUNTER -= 1
            self.id = Word._EPHEMERAL_COUNTER
        else:
            Word._ID_COUNTER += 1
            self.id = Word._ID_COUNTER

        self.vector = vector
        self._morphemes = list(morphemes) if morphemes else []
        self._form = None
        self._plain_form = None
        self._cached_cost = None
        self._cost_env_key = None
        self.is_archaic = is_archaic
        self.replaced_by_id = None

        if phonemes is not None:
            self._phonemes = list(phonemes)
        elif self._morphemes:
            self._phonemes = [p for m in self._morphemes for p in m.phonemes]
        else:
            self._phonemes = []

        self.derivation = derivation or DerivationRecord(root_family_id=self.id)
        self.usage_frequency = max(0.01, float(usage_frequency))
        self.category = category
        self.generation_born = int(generation_born)
        self.epochs_idle = 0
        self.senses = senses if isinstance(senses, tuple) else (tuple(senses) if senses else (self.vector,))

        if not self._morphemes and self._phonemes:
            self._morphemes = [
                Morpheme(
                    phonemes=list(self._phonemes),
                    root_family_id=self.derivation.root_family_id,
                    is_grammatical=(self.category == LexicalCategory.FUNCTIONAL_CLOSED),
                )
            ]

        if history is not None:
            self._history = list(history)
        else:
            event_tag = "Proto-Root Coinage" if self.generation_born == 0 else "Productive Coinage"
            self._history = [PhoneticHistoryEntry(epoch=self.generation_born, tensor=self.tensor.copy(), event=event_tag)]

    def _invalidate_cache(self) -> None:
        self._form = None
        self._plain_form = None
        self._cached_cost = None
        self._cost_env_key = None

    def record_evolution(
        self,
        epoch: int,
        new_phonemes: Sequence[Phoneme],
        event: str = "Phonetic Drift",
        min_divergence: float = 0.08,
        force: bool = False,
    ) -> bool:
        if not new_phonemes:
            return False
        new_tensor = np.stack([p.vector for p in new_phonemes], axis=0)
        if not force and self._history:
            last_tensor = self._history[-1].tensor
            div = trajectory_distance(last_tensor[:, :9], new_tensor[:, :9])
            if div < min_divergence and (epoch - self._history[-1].epoch) < 30:
                return False
        self._history.append(PhoneticHistoryEntry(epoch=epoch, tensor=new_tensor, event=event))
        return True

    def get_history_log(self, directional_bias: float = 0.50) -> list[dict]:
        log = []
        for idx, entry in enumerate(self._history):
            if len(entry.tensor) > 0:
                form, plain = render_word_trajectory(entry.tensor, directional_bias=directional_bias)
            else:
                form, plain = "Ø", "Ø"
            log.append({
                "stage_idx": idx,
                "epoch": entry.epoch,
                "form": form,
                "plain": plain,
                "event": entry.event,
            })
        return log

    @property
    def history(self) -> list[PhoneticHistoryEntry]:
        return self._history

    @property
    def phonemes(self) -> list[Phoneme]:
        return self._phonemes

    @phonemes.setter
    def phonemes(self, new_phonemes: Sequence[Phoneme]) -> None:
        self._phonemes = list(new_phonemes)
        self._invalidate_cache()

        n_m = len(self._morphemes)
        if n_m <= 1:
            if n_m == 0:
                self._morphemes = [Morpheme(self._phonemes, self.derivation.root_family_id)]
            else:
                self._morphemes[0].phonemes = list(self._phonemes)
        elif not self._phonemes:
            for m in self._morphemes:
                m.phonemes = []
        else:
            offset = 0
            n_p = len(self._phonemes)
            for idx, m in enumerate(self._morphemes):
                if idx == n_m - 1:
                    m.phonemes = list(self._phonemes[offset:])
                else:
                    take = min(len(m.phonemes), max(0, n_p - offset - (n_m - 1 - idx)))
                    m.phonemes = list(self._phonemes[offset : offset + take])
                    offset += take

    @property
    def morphemes(self) -> list[Morpheme]:
        return self._morphemes

    @morphemes.setter
    def morphemes(self, new_morphemes: Sequence[Morpheme]) -> None:
        self._morphemes = list(new_morphemes)
        self._phonemes = [p for m in self._morphemes for p in m.phonemes]
        self._invalidate_cache()

    @property
    def tensor(self) -> np.ndarray:
        if not self._phonemes:
            return np.zeros((0, 10), dtype=np.float32)
        return np.stack([p.vector for p in self._phonemes], axis=0)

    @property
    def form(self) -> str:
        if self._form is None:
            if self._phonemes:
                self._form, self._plain_form = render_word_trajectory(self.tensor)
            else:
                self._form, self._plain_form = "", ""
        return self._form

    @property
    def plain_form(self) -> str:
        if self._plain_form is None:
            _ = self.form
        return self._plain_form

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self._phonemes)

    @property
    def is_proto_root(self) -> bool:
        return self.derivation.derivation_type == DerivationType.ROOT and self.generation_born == 0

    @property
    def is_ephemeral(self) -> bool:
        return self.id < 0

    def energetic_cost(self, profile=None, env=None, **kwargs) -> float:
        env_key = (
            getattr(profile, "synthesis_index", 0.50),
            getattr(env, "altitude", 0.0),
            getattr(env, "temperature", 0.50),
            getattr(env, "humidity", 0.50),
        )
        if self._cached_cost is None or self._cost_env_key != env_key:
            self._cached_cost = word_energy(self, profile=profile, env=env)
            self._cost_env_key = env_key
        return self._cached_cost

    def record_usage(self, boost: float = 1.0) -> None:
        self.usage_frequency += max(0.0, float(boost))
        self.epochs_idle = 0
        self.is_archaic = False

    def copy(self) -> Word:
        w = Word(
            word_id=self.id,
            vector=self.vector,
            phonemes=[p.drift() for p in self._phonemes],
            morphemes=[m.copy() for m in self._morphemes],
            derivation=self.derivation,
            usage_frequency=self.usage_frequency,
            category=self.category,
            generation_born=self.generation_born,
            senses=self.senses,
            is_archaic=self.is_archaic,
            history=[PhoneticHistoryEntry(e.epoch, e.tensor.copy(), e.event) for e in self._history],
        )
        w._form = self._form
        w._plain_form = self._plain_form
        w._cached_cost = self._cached_cost
        w._cost_env_key = self._cost_env_key
        w.epochs_idle = self.epochs_idle
        w.replaced_by_id = self.replaced_by_id
        return w

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Word) and (self is other if (self.id < 0 or other.id < 0) else self.id == other.id)

    def __len__(self) -> int:
        return len(self._phonemes)

    def __getitem__(self, idx: int) -> Phoneme:
        return self._phonemes[idx]

    def __iter__(self):
        return iter(self._phonemes)

    def __repr__(self) -> str:
        cat = "ARCH" if self.is_archaic else ("FUNC" if self.category == LexicalCategory.FUNCTIONAL_CLOSED else "CONTENT")
        return f"Word(#{self.id}, /{self.form}/, U={self.usage_frequency:.2f}, {cat})"