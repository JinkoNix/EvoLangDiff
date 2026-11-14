"""
world_grid.py — Continuous 2D Tensor Geography, Planetary Climate Thermodynamics,
9-Point Isotropic Diffusion, Universal Faction Coverage (distance-transform seeding),
Urban Scaling Dynamism, and Sunk Infrastructure Milestones.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import colorsys
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.ndimage import distance_transform_edt as _edt

from phonetics import Phoneme, repair_phonemes, sensory_distance, trajectory_distance
from word import Word, LexicalCategory
from semantics import SemanticVector, DerivationRecord, DerivationType


# -----------------------------------------------------------------------------
# 1. ISOTROPIC 9-POINT CONSERVATIVE DIFFUSION
# -----------------------------------------------------------------------------

_CARDINAL_SLICES = [
    (slice(None, -2), slice(1, -1)),  # North
    (slice(2, None), slice(1, -1)),   # South
    (slice(1, -1), slice(None, -2)),  # West
    (slice(1, -1), slice(2, None)),   # East
]

_DIAGONAL_SLICES = [
    (slice(None, -2), slice(None, -2)),  # North-West
    (slice(None, -2), slice(2, None)),   # North-East
    (slice(2, None), slice(None, -2)),   # South-West
    (slice(2, None), slice(2, None)),    # South-East
]


def diffuse_masked_2d(f: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """9-point conservative diffusion ∇·(D∇f); weights 2/3 cardinal, 1/6 diagonal; no flux into masked cells."""
    m = mask.astype(np.float32)
    fp = np.pad(f, 1, mode='edge')
    Dp = np.pad(D, 1, mode='edge')
    mp = np.pad(m, 1, mode='edge')

    cardinal_flux = sum(0.5 * (Dp[s] + D) * mp[s] * (fp[s] - f) for s in _CARDINAL_SLICES)
    diagonal_flux = sum(0.5 * (Dp[s] + D) * mp[s] * (fp[s] - f) for s in _DIAGONAL_SLICES)
    return ((2.0 / 3.0) * cardinal_flux + (1.0 / 6.0) * diagonal_flux) * m


def laplacian_isotropic_2d(f: np.ndarray) -> np.ndarray:
    fp = np.pad(f, 1, mode='edge')
    cardinal = sum(fp[s] for s in _CARDINAL_SLICES)
    diagonal = sum(fp[s] for s in _DIAGONAL_SLICES)
    return (2.0 / 3.0) * cardinal + (1.0 / 6.0) * diagonal - (10.0 / 3.0) * f


_NB_3D = [
    (slice(None), slice(None, -2), slice(1, -1)),
    (slice(None), slice(2, None), slice(1, -1)),
    (slice(None), slice(1, -1), slice(None, -2)),
    (slice(None), slice(1, -1), slice(2, None)),
]


def diffuse_masked_3d_spatial(U: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = m[np.newaxis, :, :]
    if D.ndim == 2:
        D = D[np.newaxis, :, :]

    Up = np.pad(U, ((0, 0), (1, 1), (1, 1)), mode='edge')
    Dp = np.pad(D, ((0, 0), (1, 1), (1, 1)), mode='edge')
    mp = np.pad(m, ((0, 0), (1, 1), (1, 1)), mode='edge')
    flux_sum = sum(0.5 * (Dp[s] + D) * mp[s] * (Up[s] - U) for s in _NB_3D)
    return flux_sum * m


def conservative_flux_divergence_2d(p: np.ndarray, potential: np.ndarray, mobility: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(potential)
    jx = mobility * p * grad_x
    jy = mobility * p * grad_y
    return -(np.gradient(jx, axis=1) + np.gradient(jy, axis=0))


def upwind_shadow(b: np.ndarray, wind: np.ndarray, K: int = 8, lam: float = 0.75) -> np.ndarray:
    H, W = b.shape
    P = K + 1
    bp = np.pad(b, P, mode='edge')
    out = np.zeros_like(b, dtype=np.float32)
    for k in range(1, K + 1):
        dr = int(round(-wind[1] * k))
        dc = int(round(-wind[0] * k))
        out += (lam ** k) * bp[P + dr: P + dr + H, P + dc: P + dc + W]
    return out * (1.0 - lam)


def chaikin_smooth(points: list[list[float]], iterations: int = 2) -> list[list[float]]:
    if len(points) < 3:
        return points
    pts = points
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            q = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1], p0[2]]
            r = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1], p1[2]]
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


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
    canonical_roots: dict[int, list[Phoneme]] = field(default_factory=dict)
    is_decapitated: bool = False
    last_secession_epoch: int = 0


class WorldGrid:
    MAX_POLITIES = 40  # caps the O(H·W·N²) conflict tensor

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
            indexing='ij'
        )
        self.X_grid = x_coords
        self.Y_grid = y_coords

        self.elev = np.zeros((self.H, self.W), dtype=np.float32)
        self.slope = np.zeros((self.H, self.W), dtype=np.float32)      # raw |∇z| per cell (mobility, permeability)
        self.roughness = np.zeros((self.H, self.W), dtype=np.float32)  # normalised [0,1] (clicks, defense)
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

        self._grid_y, self._grid_x = np.indices((self.H, self.W), dtype=np.float32)

        self.lexical_manifold: list[Word] = [w.copy() for w in proto_roots] if proto_roots else []
        n_seeds = max(1, len(self.lexical_manifold))
        self.word_presence = np.zeros((n_seeds, self.H, self.W), dtype=np.float32)

        self._init_geography()
        self._init_hydrology()
        self._init_demographics()
        self._init_word_presence()

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

                gx = grad_x[iy, ix]
                gy = grad_y[iy, ix]
                slope_vec = np.array([-gx, -gy], dtype=np.float32)
                norm_s = np.linalg.norm(slope_vec)

                if norm_s < 0.015:
                    if not any(math.hypot(pos[0] - lk.x, pos[1] - lk.y) < 3.0 for lk in lakes):
                        lakes.append(Lake(x=float(pos[0]), y=float(pos[1]),
                                          radius=0.8 + 0.3 * math.sqrt(acc_discharge), discharge=acc_discharge))

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
        self._update_climate(epoch=0)

    # -------------------------------------------------------------------------
    # PLANETARY BIOCLIMATOLOGY
    # -------------------------------------------------------------------------

    def _update_climate(self, epoch: int) -> None:
        season = math.sin(2.0 * math.pi * (epoch % 12) / 12.0)
        y = self.Y_grid

        insolation = np.clip(np.sin(np.pi * np.clip(y + season * 0.08, 0.0, 1.0)), 0.0, 1.0)
        base_thermal = 0.06 + 0.88 * (insolation ** 1.35)

        sea_thermal = np.clip(0.05 + 0.72 * (insolation ** 1.2), 0.02, 0.82)
        lapse_cooling = 1.25 * np.maximum(0.0, self.elev)
        land_temp = np.clip(base_thermal - lapse_cooling, 0.02, 1.0)
        self.temp = np.where(self.elev <= 0.0, sea_thermal, land_temp)

        grad_y, grad_x = np.gradient(self.elev)
        slope_lift = grad_x * self.wind[0] + grad_y * self.wind[1]
        orographic_rain = np.where(slope_lift > 0, np.minimum(0.45, slope_lift * 5.0), 0.0)

        high_barrier = np.maximum(0.0, self.elev - 0.20)
        rainshadow_drying = upwind_shadow(high_barrier, self.wind, K=8, lam=0.75) * 0.70

        ambient_moisture = 0.35 + 0.22 * np.sin(self.X_grid * 3.2 + y * 2.6)
        raw_hum = np.where(self.elev <= 0.0, 0.88, ambient_moisture + orographic_rain - rainshadow_drying)
        self.hum = np.clip(raw_hum, 0.05, 0.98)

        thermal_growth_potential = np.maximum(0.0, self.temp - 0.18)
        thermal_gate = np.tanh(4.5 * thermal_growth_potential)

        pet = 0.20 + 0.90 * self.temp
        river_boost = np.clip(self.discharge / 2.5, 0.0, 2.5)
        water_availability = (self.hum + river_boost * 0.35) / np.maximum(0.10, pet)

        climax_veg = 0.95 * thermal_gate * (water_availability / (0.85 + water_availability))

        timber_depletion = 0.010 * np.clip(self.pop / 3.0, 0.0, 1.0) * (self.attention[..., 0] / 1.5)
        self.soil_health = np.clip(self.soil_health + (0.006 - timber_depletion), 0.35, 1.0)

        emergent_veg = np.where(self.elev <= 0.0, 0.0, climax_veg * self.soil_health)
        self.veg = np.clip(emergent_veg * np.exp(-0.20 * (self.pop ** 0.8)), 0.005, 0.98)

        thermal_fauna_peak = np.exp(-((self.temp - 0.55) ** 2) / 0.12)
        raw_fauna = 2.5 * (self.veg ** 0.8) * thermal_fauna_peak * (1.0 + 0.5 * np.tanh(self.discharge))
        human_hunting_pressure = np.exp(-0.35 * self.pop * (1.0 + self.attention[..., 1] * 0.4))
        self.fauna = np.where(self.elev <= 0.0, 0.0, np.clip(raw_fauna * human_hunting_pressure, 0.01, 3.5))

        fluvial_silt = 6.0 * (np.clip(self.discharge / 2.0, 0.0, 3.0) ** 1.6)
        coastal_silt = np.where((self.elev > 0.0) & (self.elev < 0.10), 3.0, 0.0)
        base_food = (self.veg * 2.0 + self.fauna * 1.5 + fluvial_silt + coastal_silt)

        self.base_capacity = np.where(self.elev <= 0.0, 0.0, np.clip(0.12 + 0.65 * (base_food ** 1.35), 0.05, 25.0))
        infra_scaling = 1.0 + 0.60 * np.log1p(self.infrastructure * 0.40)
        self.capacity = np.clip(self.base_capacity * infra_scaling, 0.05, 45.0)
        self.resource_value = np.where(self.elev <= 0.0, 0.0, base_food * (1.0 + self.elev * 0.2) * (1.0 + 0.3 * np.log1p(self.infrastructure)))

    def assign_polity_chroma(self, polity_id: int) -> list[int]:
        GOLDEN_RATIO = 0.618033988749895
        hue = (0.08 + polity_id * GOLDEN_RATIO) % 1.0
        r_f, g_f, b_f = colorsys.hls_to_rgb(hue, 0.54, 0.85)
        return [int(r_f * 255), int(g_f * 255), int(b_f * 255)]

    def evaluate_site_semantic_vector(self, x: float, y: float) -> list[float]:
        ix = int(np.clip(round(x), 0, self.W - 1))
        iy = int(np.clip(round(y), 0, self.H - 1))

        disch = self.discharge[iy, ix]
        elev = self.elev[iy, ix]
        veg = self.veg[iy, ix]
        pop = self.pop[iy, ix]
        infra = self.infrastructure[iy, ix]

        c = float(np.clip(0.20 + 0.60 * max(0.0, elev) + 0.40 * (1.0 - veg), 0.0, 1.0))
        a = float(np.clip(0.15 + 0.70 * veg + 0.30 * math.tanh(pop * 0.4), 0.0, 1.0))
        v = float(np.clip(0.30 + 0.50 * veg - 0.25 * max(0.0, elev), 0.0, 1.0))
        p = float(np.clip(0.20 + 0.60 * math.tanh(infra * 0.25) + 0.35 * max(0.0, elev), 0.0, 1.0))
        d = float(np.clip(0.15 + 0.65 * math.tanh(disch * 0.35) + 0.30 * math.tanh(pop * 0.25), 0.0, 1.0))
        s = float(np.clip(0.10 + 0.80 * math.tanh(pop * 0.35) + 0.20 * math.tanh(infra * 0.2), 0.0, 1.0))
        e = float(np.clip(0.20 + 0.60 * (1.0 - self.permeability[iy, ix]), 0.0, 1.0))
        return [c, a, v, p, d, s, e]

    # -------------------------------------------------------------------------
    # UNIVERSAL FACTION COVERAGE
    # -------------------------------------------------------------------------

    def _spawn_clan(self, sx: int, sy: int, footprint: np.ndarray, name_hint: str = "") -> None:
        nid = self.next_polity_id
        self.next_polity_id += 1
        if nid >= self.hegemony.shape[-1]:
            self.hegemony = np.pad(self.hegemony, ((0, 0), (0, 0), (0, 8)), mode='constant')

        dist = np.hypot(self._grid_x - sx, self._grid_y - sy)
        self.polities.append(Polity(
            id=nid, name=name_hint, literal_meaning="", x=int(sx), y=int(sy), prestige=1.2,
            color=self.assign_polity_chroma(nid), age=self.epoch, dist_cells=dist,
            semantic_vector=self.evaluate_site_semantic_vector(float(sx), float(sy)),
            government_type="Nomadic Clan", centralization=0.30,
            infrastructure_bias=0.25, sacred_focus=0.45, tolerance=0.65,
        ))
        self.hegemony[..., nid] = np.where(footprint, 5.0 * np.exp(-dist ** 2 / 6.0), 0.0)
        self.events.append(f"Epoch {self.epoch}: Autonomous settlers formed Clan #{nid} at ({sx}, {sy}).")

    def _seed_unclaimed_clans(self, k_max: int = 4, r_min: float = 6.0) -> None:
        """Seeds clans in the interior of unclaimed settled regions: argmax of pop · dist-to-nearest-claimed-cell."""
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

    def _init_demographics(self) -> None:
        fertility = np.where(self.elev > 0.10, self.capacity * self.veg, -1.0)
        sorted_indices = np.argsort(fertility.ravel())[::-1]

        hearth_nodes: list[tuple[int, int]] = []
        for idx in sorted_indices:
            cy, cx = np.unravel_index(idx, fertility.shape)
            if not any(math.hypot(cx - hx, cy - hy) < 11.0 for hy, hx in hearth_nodes):
                hearth_nodes.append((int(cy), int(cx)))
                if len(hearth_nodes) >= 6:
                    break

        self.polities = []
        for pid, (cy, cx) in enumerate(hearth_nodes):
            dy, dx = self.Y_grid - self.Y_grid[cy, cx], self.X_grid - self.X_grid[cy, cx]
            band_footprint = 0.90 * np.exp(-(dx ** 2 + dy ** 2) / 0.003).astype(np.float32)
            self.pop += band_footprint
            self.infrastructure[cy, cx] = 0.8

            chroma = self.assign_polity_chroma(pid)
            sem_vec = self.evaluate_site_semantic_vector(float(cx), float(cy))
            dist_field = np.sqrt((self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2)

            env_label = "Valley Clan" if self.discharge[cy, cx] > 1.0 else ("Upland Band" if self.elev[cy, cx] > 0.35 else "Coastal Tribe")

            self.polities.append(Polity(
                id=pid, name=f"Tribe #{pid}", literal_meaning=env_label,
                x=int(cx), y=int(cy), prestige=1.2, color=chroma, age=0,
                dist_cells=dist_field, semantic_vector=sem_vec,
                government_type="Nomadic Clan", centralization=0.30,
                infrastructure_bias=0.25, sacred_focus=0.45, tolerance=0.65
            ))
            self.hegemony[cy, cx, pid] = 4.5

        self.next_polity_id = len(hearth_nodes)
        self._discover_and_update_landmarks()

    def _init_word_presence(self) -> None:
        if not self.polities or not self.lexical_manifold:
            return
        for p in self.polities:
            d2 = (self._grid_x - p.x) ** 2 + (self._grid_y - p.y) ** 2
            footprint = np.exp(-d2 / 20.0).astype(np.float32)
            self.word_presence[:len(self.lexical_manifold)] += footprint[np.newaxis] * 1.5

    # -------------------------------------------------------------------------
    # SETTLEMENT ACCUMULATION TIERS
    # -------------------------------------------------------------------------

    def _discover_and_update_landmarks(self) -> None:
        tot_heg = np.sum(self.hegemony, axis=-1)
        dom_owners = np.where(tot_heg > 0.25, np.argmax(self.hegemony, axis=-1), -1)

        is_urban_site = (self.infrastructure >= 3.0) & (self.pop >= 1.4) & (self.elev > 0.0)
        py_arr, px_arr = np.nonzero(is_urban_site)

        for py, px in zip(py_arr, px_arr):
            existing_city = next(
                (lm for lm in self.landmarks.values()
                 if lm.feature_type == "city" and math.hypot(lm.x - px, lm.y - py) < 3.2),
                None
            )
            owner_id = int(dom_owners[py, px])  # -1 if unclaimed

            if existing_city is None:
                sem_vec = self.evaluate_site_semantic_vector(float(px), float(py))
                new_lm = Landmark(
                    id=self.next_landmark_id, name_endonym="", gloss="", feature_type="city",
                    x=float(px), y=float(py), source_culture_id=owner_id, epoch_named=self.epoch,
                    semantic_vector=sem_vec, infrastructure_level=float(self.infrastructure[py, px]),
                    is_named=False,
                )
                self.landmarks[self.next_landmark_id] = new_lm
                self.next_landmark_id += 1
                self.new_city_events.append({"x": px, "y": py, "name": "Settlement Established", "color": [255, 215, 0]})
            else:
                existing_city.infrastructure_level = float(self.infrastructure[py, px])

        pad_elev = np.pad(self.elev, 1, mode='edge')
        is_m_peak = (
            (self.elev >= pad_elev[:-2, 1:-1]) & (self.elev >= pad_elev[2:, 1:-1]) &
            (self.elev >= pad_elev[1:-1, :-2]) & (self.elev >= pad_elev[1:-1, 2:]) &
            (self.elev > 0.52)
        )
        my_arr, mx_arr = np.nonzero(is_m_peak)
        for py, px in zip(my_arr, mx_arr):
            existing_m = next(
                (lm for lm in self.landmarks.values()
                 if lm.feature_type == "mountain" and math.hypot(lm.x - px, lm.y - py) < 4.5),
                None
            )
            if existing_m is None:
                cid = int(dom_owners[py, px])
                sem_vec = self.evaluate_site_semantic_vector(float(px), float(py))
                new_m = Landmark(
                    id=self.next_landmark_id, name_endonym="", gloss="", feature_type="mountain",
                    x=float(px), y=float(py), source_culture_id=cid, epoch_named=self.epoch,
                    semantic_vector=sem_vec, is_named=False,
                )
                self.landmarks[self.next_landmark_id] = new_m
                self.next_landmark_id += 1

    # -------------------------------------------------------------------------
    # CONSERVATIVE CONFLICT, RIVER-CONDUIT EXPANSION, SECESSION
    # -------------------------------------------------------------------------

    def _step_reaction_diffusion_hegemony(self, dt: float) -> None:
        land_mask = (self.elev > 0.0)
        res_pull = 1.0 + 0.85 * (self.resource_value / (1.0 + self.resource_value))
        river_conduit = np.clip(self.discharge / 3.0, 0.0, 2.5)

        active_polities: list[Polity] = []
        for p in self.polities:
            pid = p.id
            px, py = p.x, p.y
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
                p.standard_epoch = self.epoch
                p.canonical_roots = {
                    w.id: [phon.drift() for phon in w.phonemes]
                    for w in self.lexical_manifold[:min(80, len(self.lexical_manifold))]
                }
                self.events.append(f"Epoch {self.epoch}: {p.name or ('#' + str(pid))} codified its Classical Canonical Standard Register.")

        self.polities = active_polities
        num_p = len(self.polities)
        if num_p == 0:
            self._seed_unclaimed_clans()
            return

        active_ids = [p.id for p in self.polities]
        U_stack = np.stack([self.hegemony[..., pid] for pid in active_ids], axis=-1)

        # 1. Local combat reach field
        combat_reach_stack = np.zeros_like(U_stack)
        for idx, p in enumerate(self.polities):
            cap_E = float(self.attention[p.y, p.x, 6])
            cap_S = float(self.attention[p.y, p.x, 5])
            cost_dist = p.dist_cells * (1.0 + 1.8 * (1.0 - self.permeability)) / (1.0 + 0.30 * np.log1p(self.infrastructure))
            combat_reach_stack[..., idx] = np.exp(-cost_dist / (6.0 + 6.0 * cap_E + 4.0 * cap_S))

        terrain_defense_bonus = 1.0 + 0.8 * self.roughness + 0.4 * self.veg

        prest_arr = np.array([p.prestige for p in self.polities], dtype=np.float32)
        tau = 1.2
        adv = 1.0 / (1.0 + np.exp(-(prest_arr[np.newaxis, :] - prest_arr[:, np.newaxis]) / tau))
        np.fill_diagonal(adv, 0.0)

        A = U_stack * combat_reach_stack
        Abar = np.maximum(0.0, A + 0.125 * np.stack([laplacian_isotropic_2d(A[..., i]) for i in range(num_p)], axis=-1))

        kappa = 0.08
        T = kappa * np.einsum('hwj,hwi,ji->hwji', U_stack, Abar, adv) / terrain_defense_bonus[..., np.newaxis, np.newaxis]
        out_loss = T.sum(axis=3)

        T *= np.minimum(1.0, U_stack / (dt * out_loss + 1e-6))[..., :, np.newaxis]
        gain = T.sum(axis=2)
        loss = T.sum(axis=3)

        combat_devastation = np.sum(loss, axis=-1)
        war_casualties = np.clip(0.16 * np.tanh(combat_devastation * 0.4) * self.pop, 0.0, self.pop * 0.5)
        self.pop = np.maximum(0.0, self.pop - war_casualties * dt)
        infra_ruin = np.clip(0.20 * np.tanh(combat_devastation * 0.5) * self.infrastructure, 0.0, self.infrastructure)
        self.infrastructure = np.maximum(0.0, self.infrastructure - infra_ruin * dt)

        unclaimed_mask = (U_stack.sum(axis=-1) < 1.0) & land_mask
        colonization = 0.040 * Abar * np.clip(self.pop[..., np.newaxis] / 1.5, 0.1, 1.5) * res_pull[..., np.newaxis]
        colonization = np.where(unclaimed_mask[..., np.newaxis], colonization, 0.0)

        # Diffusivity: permeability plus a river-conduit floor so valleys remain traversable
        D_h = 0.045 * res_pull * (self.permeability + 0.4 * river_conduit)

        for idx, p in enumerate(self.polities):
            pid = p.id
            px, py = p.x, p.y
            u = U_stack[..., idx]
            tot_u = float(np.sum(u))

            source = np.zeros((self.H, self.W), dtype=np.float32)
            if not p.is_decapitated:
                cap_pop_factor = float(np.tanh(self.pop[py, px] * 0.40))
                source[py, px] = max(0.5, p.prestige * cap_pop_factor * (1.2 + p.centralization * 0.6))

            reach_field = combat_reach_stack[..., idx]
            logistical_cap = max(14.0, p.prestige * 14.0 * float(np.tanh(self.pop[py, px] * 0.40)) * (1.0 + p.infrastructure_bias * 0.5))

            overextension = max(0.0, (tot_u - logistical_cap) / (logistical_cap + 0.1))
            overstretch_decay = 0.035 * (overextension ** 1.3)

            k_carrying = np.clip(1.5 + 0.5 * p.prestige + 0.8 * self.pop, 1.5, 20.0)
            growth = 0.10 * u * (1.0 - (u / k_carrying))

            diff_flux = diffuse_masked_2d(u, D_h, land_mask)
            decay = (0.008 + 0.015 * (1.0 - reach_field) + overstretch_decay) * u

            new_u = np.where(
                land_mask,
                np.clip(u + (diff_flux + source + growth + gain[..., idx] + colonization[..., idx] - decay - loss[..., idx]) * dt, 0.0, 30.0),
                0.0
            )
            self.hegemony[..., pid] = np.where(new_u < 0.01, 0.0, new_u)

        # -------------------------------------------------------------
        # SECESSION IN OVEREXTENDED EMPIRES (runs before seeding so idx matches combat_reach_stack)
        # -------------------------------------------------------------
        if self.epoch % 6 == 0 and self.epoch > 12 and len(self.polities) < self.MAX_POLITIES:
            tot_h = np.sum(self.hegemony, axis=-1)
            dom_owner = np.where(tot_h > 0.25, np.argmax(self.hegemony, axis=-1), -1)

            for idx, p in enumerate(list(self.polities)):
                pid = p.id
                px, py = p.x, p.y
                local_u = self.hegemony[..., pid]
                if float(np.sum(local_u)) < 5.0:
                    continue
                if (self.epoch - p.last_secession_epoch) < 14:
                    continue

                cap_A = self.attention[py, px]
                delta_A = np.sum(np.abs(self.attention - cap_A), axis=-1)
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

                    new_color = self.assign_polity_chroma(new_id)
                    sem_vec = self.evaluate_site_semantic_vector(float(sx), float(sy))
                    dist_field = np.sqrt((self._grid_x - sx) ** 2 + (self._grid_y - sy) ** 2)

                    infra_val = float(self.infrastructure[sy, sx])
                    sacred_val = float(self.attention[sy, sx, 2] + self.attention[sy, sx, 3]) / 2.0
                    gov_label = "Domain" if infra_val > 3.0 else ("Order" if sacred_val > 1.3 else "Clan")

                    new_polity = Polity(
                        id=new_id, name="", literal_meaning="", x=int(sx), y=int(sy),
                        prestige=float(self.prestige[sy, sx]) * 1.5, color=new_color,
                        age=self.epoch, dist_cells=dist_field, semantic_vector=sem_vec,
                        government_type=gov_label,
                        centralization=float(np.clip(0.40 + 0.10 * infra_val, 0.2, 0.9)),
                        infrastructure_bias=float(np.clip(infra_val / 8.0, 0.2, 0.9)),
                        sacred_focus=float(np.clip(sacred_val / 2.5, 0.1, 0.9)),
                        tolerance=float(np.clip(1.0 - infra_val / 10.0, 0.2, 0.8))
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

        # -------------------------------------------------------------
        # UNIVERSAL COVERAGE: seed clans in the interior of unclaimed settled regions
        # -------------------------------------------------------------
        if self.epoch % 4 == 0:
            self._seed_unclaimed_clans()

    # -------------------------------------------------------------------------
    # LEXICAL S-CURVE & PRESENCE DIFFUSION
    # -------------------------------------------------------------------------

    def _step_lexical_presence_and_drift(self, dt: float) -> None:
        n_words = len(self.lexical_manifold)
        land_mask = (self.elev > 0.0)

        rho = np.exp(-0.85 * np.maximum(0.0, self.elev))
        target_b8 = np.clip(0.40 * (1.0 - rho) - 0.50 * self.hum * np.maximum(0.0, 0.30 - self.elev), -0.30, 0.30)
        target_b7 = np.clip(-0.50 * (1.0 - rho) * (1.0 - self.hum), -0.60, 0.0)

        self.phonetic_bias[..., 8] += 0.05 * (target_b8 - self.phonetic_bias[..., 8]) * dt
        self.phonetic_bias[..., 7] += 0.05 * (target_b7 - self.phonetic_bias[..., 7]) * dt

        for p in self.polities:
            if p.is_standardized and p.id < self.hegemony.shape[-1]:
                h_share = self.hegemony[..., p.id]
                decay_dist = 1.0 / (1.0 + p.dist_cells * 0.08)
                standard_pull = 0.03 * p.centralization * h_share * decay_dist * self.permeability
                for d in range(9):
                    self.phonetic_drift_field[..., d] *= np.clip(1.0 - standard_pull * dt, 0.0, 1.0)

                if self.epoch % 4 == 0:
                    for root_id in p.canonical_roots.keys():
                        if root_id < n_words:
                            self.word_presence[root_id] += 0.02 * h_share * p.centralization * self.permeability

        for d in range(9):
            d_flux = diffuse_masked_2d(self.phonetic_drift_field[..., d], 0.015 * self.permeability, land_mask)
            self.phonetic_drift_field[..., d] = np.clip(self.phonetic_drift_field[..., d] + d_flux * dt, -0.45, 0.45)

        if n_words > 0 and self.epoch % 3 == 0:
            if self.word_presence.shape[0] < n_words:
                needed = n_words - self.word_presence.shape[0]
                self.word_presence = np.pad(self.word_presence, ((0, needed), (0, 0), (0, 0)), mode='constant')

            presence_peaks = np.max(self.word_presence[:n_words], axis=(1, 2))
            active_word_indices = np.where(presence_peaks > 0.02)[0]

            if len(active_word_indices) > 0:
                U_active = self.word_presence[active_word_indices]
                pop_density = np.clip(self.pop / 2.0, 0.0, 2.5)[np.newaxis, :, :]

                diff_U = diffuse_masked_3d_spatial(U_active, 0.035 * self.permeability, land_mask)
                logistic_maintenance = 0.05 * U_active * (1.0 - (U_active / (pop_density + 0.3)))

                self.word_presence[active_word_indices] = np.where(
                    land_mask[np.newaxis, :, :],
                    np.clip(U_active + (diff_U + logistic_maintenance) * 2.0, 0.0, 5.0),
                    0.0
                )

        from lexicon import LexiconGenerator
        from language import LanguageProfile

        total_global_pop = float(np.sum(self.pop[self.elev > 0.0]))
        num_polities = max(1, len(self.polities))
        global_lexical_target = int(350 + 2500 * math.tanh(total_global_pop * 0.005) + 35 * num_polities)

        settled_mask = (self.pop > 0.25) & (self.elev > 0.0)
        settled_y, settled_x = np.nonzero(settled_mask)

        if len(settled_y) > 0 and n_words < global_lexical_target and self.epoch % 2 == 0:
            growth_pressure = (global_lexical_target - n_words) / float(global_lexical_target)
            batch_target = max(1, min(14, int(math.ceil(growth_pressure * 16.0))))

            pop_weights = self.pop[settled_y, settled_x].astype(np.float64)
            p_dist = pop_weights / pop_weights.sum()
            sample_indices = self.rng.choice(len(settled_y), size=min(batch_target, len(settled_y)), replace=False, p=p_dist)

            for s_idx in sample_indices:
                sy, sx = settled_y[s_idx], settled_x[s_idx]
                cell_pop = float(self.pop[sy, sx])

                env_site = type("E", (), {
                    "altitude": float(self.elev[sy, sx]),
                    "humidity": float(self.hum[sy, sx]),
                    "temperature": float(self.temp[sy, sx]),
                    "vegetation": float(self.veg[sy, sx]),
                    "fauna": float(self.fauna[sy, sx]),
                    "roughness": float(self.roughness[sy, sx]),
                    "population": cell_pop,
                })()
                prof_site = LanguageProfile(
                    name=f"Site_{sx}_{sy}",
                    head_directionality=float(self.typology[sy, sx, 0]),
                    synthesis_index=float(self.typology[sy, sx, 1]),
                    syllable_complexity=float(self.typology[sy, sx, 2])
                )

                local_scores = self.word_presence[:n_words, sy, sx]
                active_local_indices = np.where(local_scores > 0.04)[0]
                active_local = [self.lexical_manifold[idx] for idx in active_local_indices]

                new_word = None
                coin_roll = self.rng.random()
                next_family_id = len(self.lexical_manifold)

                if coin_roll < 0.65 and len(active_local) >= 2:
                    base_candidate = active_local[int(self.rng.integers(len(active_local)))]
                    deriv_mode = self.rng.choice(["agent", "instrument", "locative", "diminutive", "augmentative"])
                    new_word = LexiconGenerator.derive_productive_concept(
                        base_candidate, deriv_mode, active_local, profile=prof_site, env=env_site
                    )

                elif coin_roll < 0.85 and len(active_local) >= 1:
                    is_marine = (self.elev[sy, sx] > 0.0) and (self.elev[sy, sx] < 0.08)
                    is_alpine = (self.elev[sy, sx] > 0.45)
                    is_river = (self.discharge[sy, sx] > 1.2)

                    if is_marine:
                        target_vec = SemanticVector(0.9, 0.4, 0.7, 0.6, 0.8, 0.2, 0.9)
                    elif is_alpine:
                        target_vec = SemanticVector(1.0, 0.0, 0.4, 0.9, 0.1, 0.1, 0.8)
                    elif is_river:
                        target_vec = SemanticVector(0.9, 0.5, 0.8, 0.5, 0.9, 0.3, 0.8)
                    else:
                        target_vec = SemanticVector(0.7, 0.6, 0.5, 0.5, 0.5, 0.7, 0.5)

                    new_phonemes = LexiconGenerator.generate_gestural_root(profile=prof_site, env=env_site, existing_words=self.lexical_manifold[-35:])
                    if new_phonemes and any(p.is_vowel for p in new_phonemes):
                        new_word = Word(
                            vector=target_vec,
                            phonemes=new_phonemes,
                            derivation=DerivationRecord(root_family_id=next_family_id),
                            category=LexicalCategory.CONTENT_OPEN,
                            generation_born=self.epoch,
                        )

                else:
                    new_phonemes = LexiconGenerator.generate_gestural_root(profile=prof_site, env=env_site, existing_words=self.lexical_manifold[-35:])
                    if new_phonemes and any(p.is_vowel for p in new_phonemes):
                        coords = [float(np.clip(self.rng.normal(0.5, 0.22), 0.05, 0.95)) for _ in range(7)]
                        new_word = Word(
                            vector=SemanticVector(*coords),
                            phonemes=new_phonemes,
                            derivation=DerivationRecord(root_family_id=next_family_id),
                            category=LexicalCategory.CONTENT_OPEN,
                            generation_born=self.epoch,
                        )

                if new_word is None:
                    continue

                cand_tensor = np.stack([p.g for p in new_word.phonemes], axis=0)
                is_clash = any(
                    trajectory_distance(cand_tensor, e.tensor[:, :9]) < 0.28 and new_word.vector.distance_to(e.vector) < 0.15
                    for e in self.lexical_manifold[-30:] if len(e.phonemes) > 0
                )
                if is_clash:
                    continue

                self.lexical_manifold.append(new_word)
                new_idx = len(self.lexical_manifold) - 1

                if new_idx >= self.word_presence.shape[0]:
                    self.word_presence = np.pad(self.word_presence, ((0, 32), (0, 0), (0, 0)), mode='constant')

                d2 = (self._grid_x - sx) ** 2 + (self._grid_y - sy) ** 2
                self.word_presence[new_idx] = float(np.clip(2.5 * math.log1p(cell_pop * 1.5), 1.2, 4.0)) * np.exp(-d2 / 16.0)

    # -------------------------------------------------------------------------
    # MAIN SIMULATION STEP
    # -------------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> None:
        self.epoch += 1
        self._probe_cache.clear()
        self._update_climate(self.epoch)

        land_mask = (self.elev > 0.0)
        # Raw slope (not the normalised roughness): constants below were tuned on per-cell |∇z|
        mobility = 0.065 * np.exp(-5.0 * self.slope) * np.where(land_mask, 1.0, 0.0)

        is_shallow = (self.elev <= 0.0) & (self.elev > -0.16)
        maritime_mobility = np.where(is_shallow, 0.02 * np.tanh(self.infrastructure * 0.15), 0.0)
        river_conduit = np.clip(self.discharge / 3.0, 0.0, 2.5)
        eff_mobility = mobility + maritime_mobility + 0.025 * river_conduit

        # Urban agglomeration potential; log1p keeps a gradient where discharge saturates the old clip
        river_junction = 1.2 * np.log1p(self.discharge)
        agglomeration_potential = (
            np.tanh(self.prestige * 0.25) * 1.8 +
            np.log1p(self.infrastructure * 0.30) * 0.8 +
            river_junction
        )
        urban_flux = conservative_flux_divergence_2d(self.pop, agglomeration_potential, eff_mobility * 0.50)

        C_matter = self.attention[..., 0]
        A_bio = self.attention[..., 1]
        S_coord = self.attention[..., 5]

        yield_agri = (2.2 * self.veg + 3.5 * np.tanh(self.discharge)) * (0.6 + 0.4 * C_matter / 1.5) * (0.6 + 0.4 * S_coord / 1.5) * np.maximum(0.2, self.temp)
        yield_pastoral = (1.8 * self.fauna + 1.0 * self.veg) * (0.6 + 0.4 * A_bio / 1.5)

        cold_strain = np.maximum(0.0, 0.20 - self.temp) * 3.5
        exposure_deficit = cold_strain * np.maximum(0.0, 1.0 - C_matter / 1.2)

        nomadic_forage_cap = np.where(self.infrastructure < 1.0, 0.45, 99.0)

        caloric_yield = np.maximum(0.10, yield_agri + yield_pastoral - exposure_deficit) * (1.0 + 0.40 * np.log1p(self.infrastructure))
        caloric_demand = self.pop * 1.05
        net_calories = caloric_yield - caloric_demand

        # Starvation only on a real deficit; the nomad ceiling acts through logistic K_eff (soft, not a wipe-out)
        K_eff = np.maximum(0.05, np.minimum(self.capacity, nomadic_forage_cap))
        deficit = np.maximum(0.0, -net_calories) / np.maximum(0.1, caloric_demand)
        starvation = 0.15 * self.pop * deficit ** 1.3
        pop_growth = np.where(net_calories >= 0, 0.08 * self.pop * (1.0 - self.pop / K_eff), 0.0)

        tot_heg = np.sum(self.hegemony, axis=-1)
        threat_gy, threat_gx = np.gradient(tot_heg)
        pasture_gy, pasture_gx = np.gradient(self.veg * 0.6 + self.fauna * 0.8 + np.tanh(self.discharge) * 0.6)

        is_threatened = (np.sqrt(threat_gx ** 2 + threat_gy ** 2) > 0.3) & (self.pop < 1.2)
        self.migration_vx = np.where(is_threatened, -threat_gx * 0.55, pasture_gx * 0.35)
        self.migration_vy = np.where(is_threatened, -threat_gy * 0.55, pasture_gy * 0.35)

        refugee_mobility = eff_mobility * np.clip(self.attention[..., 6] / 1.5, 0.5, 2.0)
        refugee_jx = refugee_mobility * self.pop * self.migration_vx
        refugee_jy = refugee_mobility * self.pop * self.migration_vy
        refugee_flux = -(np.gradient(refugee_jx, axis=1) + np.gradient(refugee_jy, axis=0))

        diff_flux = diffuse_masked_2d(self.pop, eff_mobility, land_mask)
        self.pop = np.where(
            land_mask,
            np.clip(self.pop + (diff_flux + pop_growth - starvation + urban_flux + refugee_flux) * dt, 0.0, 25.0),
            0.0
        )

        labor_surplus = np.clip(net_calories / np.maximum(0.1, caloric_demand), 0.0, 2.0)
        infra_growth = 0.045 * self.pop * (0.2 + 0.8 * labor_surplus) * (C_matter / 1.2) * (S_coord / 1.2)
        infra_decay = 0.007 * self.infrastructure + 0.0010 * (self.infrastructure ** 2)
        self.infrastructure = np.where(
            land_mask,
            np.clip(self.infrastructure + (infra_growth - infra_decay) * dt, 0.0, 50.0),
            0.0
        )
        self.prestige = np.where(land_mask, self.pop * (1.0 + 0.40 * np.log1p(self.infrastructure)), 0.0)

        self._step_reaction_diffusion_hegemony(dt)
        self._step_lexical_presence_and_drift(dt)

        if self.epoch % 6 == 0:
            self._discover_and_update_landmarks()

        # Attention dynamics
        coast_edge = np.sqrt(np.gradient(land_mask.astype(np.float32), axis=0) ** 2 + np.gradient(land_mask.astype(np.float32), axis=1) ** 2)
        real_coast = np.clip(coast_edge * 3.0, 0.0, 1.0)

        target_A = np.zeros_like(self.attention)
        target_A[..., 0] = 0.20 + 0.90 * np.maximum(0.0, self.elev) + 0.50 * (1.0 - self.veg)
        target_A[..., 1] = 0.20 + 1.20 * self.fauna * 0.6 + 0.60 * self.veg
        target_A[..., 2] = 0.25 + 0.85 * (1.0 - np.abs(self.hum - 0.55)) * (1.0 - np.abs(self.temp - 0.55))
        target_A[..., 3] = 0.20 + 0.80 * np.maximum(0.0, self.elev) + 0.80 * np.tanh(self.infrastructure * 0.20)

        urban_activity = 0.70 * np.tanh(self.pop * 0.35) + 0.50 * np.tanh(self.infrastructure * 0.25)
        target_A[..., 4] = 0.20 + urban_activity + 0.40 * np.tanh(self.discharge * 0.40) + 0.40 * real_coast

        target_A[..., 5] = 0.15 + 1.20 * np.tanh(self.pop * 0.35) + 0.40 * np.tanh(self.infrastructure * 0.20)
        target_A[..., 6] = 0.20 + 1.10 * real_coast + 0.60 * np.tanh(self.infrastructure * 0.25)

        target_sum = np.sum(target_A, axis=-1, keepdims=True)
        target_A = 7.0 * (target_A / np.maximum(0.1, target_sum))

        prestige_conductance = self.permeability * (0.010 + 0.025 * np.tanh(self.prestige * 0.20))
        for d in range(7):
            d_lap = diffuse_masked_2d(self.attention[..., d], prestige_conductance, land_mask)
            d_env = 0.06 * (target_A[..., d] - self.attention[..., d])
            self.attention[..., d] = np.clip(self.attention[..., d] + (d_env + d_lap) * dt, 0.02, 7.0)

        norm_A = np.sum(self.attention, axis=-1, keepdims=True)
        self.attention = 7.0 * (self.attention / np.maximum(0.1, norm_A))

        # Typology dynamics
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

        jespersen_erosion = -0.020 * (self.typology[..., 1] ** 2) - 0.025 * np.tanh(self.pop * 0.35)
        urban_simplification = -0.030 * np.tanh(self.pop * 0.40)

        targets = [target_h + advect_h, target_s + advect_s + jespersen_erosion, target_sigma + urban_simplification]

        for i in range(3):
            t_lap = diffuse_masked_2d(self.typology[..., i], 0.02 * self.permeability, land_mask)
            self.typology[..., i] = np.clip(self.typology[..., i] + (0.04 * (targets[i] - self.typology[..., i]) + t_lap) * dt, 0.05, 0.95)

        for d in range(9):
            bias_lap = diffuse_masked_2d(self.phonetic_bias[..., d], 0.015 * self.permeability, land_mask)
            self.phonetic_bias[..., d] = np.clip(self.phonetic_bias[..., d] + bias_lap * dt, -0.45, 0.45)

        self.border_barrier *= 0.97
        self.danger_zones *= 0.90

        grad_hx, grad_hy = np.gradient(self.typology[..., 0])
        grad_sx, grad_sy = np.gradient(self.typology[..., 1])
        ling_grad = np.sqrt(grad_hx ** 2 + grad_hy ** 2 + grad_sx ** 2 + grad_sy ** 2)

        # Raw slope in the exponent (was: normalised roughness, which collapsed permeability to ~0.1 on gentle terrain)
        base_perm = np.exp(-6.0 * self.slope) / (1.0 + np.exp(16.0 * (ling_grad - 0.22)))
        self.permeability = np.clip(base_perm * (1.0 - self.border_barrier), 0.01, 1.0)

    def get_cell_state(self, x: int, y: int) -> dict:
        r, c = int(np.clip(y, 0, self.H - 1)), int(np.clip(x, 0, self.W - 1))
        h_vals = self.hegemony[r, c]
        tot_h = float(np.sum(h_vals))
        max_h = float(np.max(h_vals))

        dom_id = int(np.argmax(h_vals)) if max_h >= 0.25 else -1
        dom_polity = next((p for p in self.polities if p.id == dom_id), None)

        is_danger = float(self.danger_zones[r, c]) > 0.35

        infra = float(self.infrastructure[r, c])
        cell_p = float(self.pop[r, c])
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

        c_matter = self.attention[r, c, 0]
        a_bio = self.attention[r, c, 1]
        s_coord = self.attention[r, c, 5]
        yield_agri = (2.2 * self.veg[r, c] + 3.5 * math.tanh(self.discharge[r, c])) * (0.6 + 0.4 * c_matter / 1.5) * (0.6 + 0.4 * s_coord / 1.5) * max(0.2, self.temp[r, c])
        yield_pastoral = (1.8 * self.fauna[r, c] + 1.0 * self.veg[r, c]) * (0.6 + 0.4 * a_bio / 1.5)
        cold_str = max(0.0, 0.20 - self.temp[r, c]) * 3.5
        exp_def = cold_str * max(0.0, 1.0 - c_matter / 1.2)
        c_yield = max(0.10, yield_agri + yield_pastoral - exp_def) * (1.0 + 0.40 * math.log1p(infra))
        c_demand = cell_p * 1.05
        net_cal = c_yield - c_demand

        mvx = float(self.migration_vx[r, c])
        mvy = float(self.migration_vy[r, c])
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
            "population": round(float(self.pop[r, c]), 3),
            "carrying_capacity": round(float(self.capacity[r, c]), 3),
            "infrastructure": round(float(self.infrastructure[r, c]), 2),
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
            }
        }