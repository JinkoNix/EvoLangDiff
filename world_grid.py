"""
world_grid.py — Continuous 2D Tensor Geography, Urban Agglomeration Gravity,
and Orographic Rainshadow Deserts.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def laplacian_2d(f: np.ndarray) -> np.ndarray:
    """Discrete 2D Laplacian with exact Neumann (zero-flux) boundary conditions."""
    f_pad = np.pad(f, pad_width=1, mode='edge')
    return (
        f_pad[:-2, 1:-1] + f_pad[2:, 1:-1] +
        f_pad[1:-1, :-2] + f_pad[1:-1, 2:] - 4.0 * f
    )


class WorldGrid:
    def __init__(self, size: int = 64, seed: int = 42):
        self.H = size
        self.W = size
        self.epoch = 0

        self.wind = np.array([0.85, -0.15], dtype=np.float32)
        self.wind /= np.linalg.norm(self.wind)

        y_coords, x_coords = np.meshgrid(
            np.linspace(0.0, 1.0, self.H, dtype=np.float32),
            np.linspace(0.0, 1.0, self.W, dtype=np.float32),
            indexing='ij'
        )
        self.X_grid = x_coords
        self.Y_grid = y_coords

        self.elev = np.zeros((self.H, self.W), dtype=np.float32)
        self.temp = np.zeros((self.H, self.W), dtype=np.float32)
        self.hum = np.zeros((self.H, self.W), dtype=np.float32)
        self.veg = np.zeros((self.H, self.W), dtype=np.float32)
        self.capacity = np.zeros((self.H, self.W), dtype=np.float32)

        self.pop = np.zeros((self.H, self.W), dtype=np.float32)
        self.infrastructure = np.zeros((self.H, self.W), dtype=np.float32)
        self.prestige = np.zeros((self.H, self.W), dtype=np.float32)

        self.attention = np.full((self.H, self.W, 7), 1.0, dtype=np.float32)
        self.typology = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.typology[..., 0] = 0.50
        self.typology[..., 1] = 0.50
        self.typology[..., 2] = 0.50

        self.phonetic_bias = np.zeros((self.H, self.W, 8), dtype=np.float32)
        self.permeability = np.ones((self.H, self.W), dtype=np.float32)

        self._init_geography()
        self._init_demographics()

    def _init_geography(self) -> None:
        x, y = self.X_grid, self.Y_grid

        c1 = np.sin(x * 3.8 + 0.4) * np.cos(y * 3.2 - 0.2)
        c2 = 0.50 * np.sin(x * 7.5 - 1.2) * np.sin(y * 6.8 + 0.8)
        c3 = 0.25 * np.cos(x * 14.0 + y * 12.0)
        plate = 0.42 * (c1 + c2 + c3)

        spine_d2 = ((x - 0.62) * 1.7) ** 2 + ((y - 0.70) * 1.3) ** 2
        mountains = 0.88 * np.exp(-spine_d2 / 0.028)

        trench = 0.40 * np.exp(-((x - 0.76) ** 2) / 0.003)
        island_d2 = ((x - 0.84) * 2.2) ** 2 + ((y - 0.48) * 1.1) ** 2
        island_arc = 0.36 * np.exp(-island_d2 / 0.020)
        west_shelf = np.maximum(0.0, (0.16 - x) * 2.8)
        east_abyss = np.maximum(0.0, (x - 0.90) * 3.6)

        self.elev = np.clip(plate + mountains - trench + island_arc - west_shelf - east_abyss, -1.0, 1.2)
        self._update_climate(epoch=0)

    def _update_climate(self, epoch: int) -> None:
        season = math.sin(2.0 * math.pi * (epoch % 12) / 12.0)
        x, y = self.X_grid, self.Y_grid

        base_temp = 0.96 - (y * 0.58) + (season * 0.08)
        self.temp = np.clip(base_temp - np.maximum(0.0, self.elev) * 0.55, 0.0, 1.0)

        dist_to_coast = np.minimum(x, np.minimum(np.abs(x - 0.76), 1.0 - x))
        ocean_moist = np.exp(-3.8 * dist_to_coast)

        grad_y, grad_x = np.gradient(self.elev)
        slope_lift = grad_x * self.wind[0] + grad_y * self.wind[1]

        windward_rain = np.where(slope_lift > 0, np.minimum(0.35, slope_lift * 3.5), 0.0)
        leeward_rainshadow = np.where(slope_lift < 0, np.maximum(-0.55, slope_lift * 7.5), 0.0)
        subtropical_subsidence = 0.35 * np.exp(-((y - 0.35) ** 2) / 0.012)

        raw_hum = ocean_moist + windward_rain + leeward_rainshadow - subtropical_subsidence
        min_hum = np.where(self.elev <= 0.0, 0.60, np.where(dist_to_coast < 0.06, 0.35, 0.04))
        self.hum = np.clip(np.maximum(min_hum, raw_hum), 0.03, 0.98)

        pet = np.maximum(0.08, (np.maximum(0.0, self.temp) ** 1.30) * 1.40)
        aridity_idx = self.hum / pet

        # Continuous logistic vegetation function
        climax_veg = 0.95 / (1.0 + np.exp(-6.0 * (aridity_idx - 0.45)))
        treeline = np.exp(-3.50 * (np.maximum(0.0, self.elev - 0.35) ** 2))
        thermal_growth = 1.0 / (1.0 + np.exp(-8.0 * (self.temp - 0.22)))
        emergent_veg = np.where(self.elev <= 0.0, 0.0, climax_veg * treeline * thermal_growth)
        depletion = np.exp(-0.35 * (np.maximum(0.0, self.pop) ** 1.20))
        self.veg = np.clip(emergent_veg * depletion, 0.01, 0.98)

        arable_d2 = ((self.hum - 0.55) / 0.22) ** 2 + ((self.veg - 0.45) / 0.22) ** 2 + ((self.temp - 0.55) / 0.22) ** 2
        grain_yield = np.exp(-arable_d2) * np.maximum(0.0, 1.0 - self.elev * 1.5) * (1.0 + 0.45 * np.log1p(self.infrastructure))

        is_coastal_land = (self.elev > 0.0) & (dist_to_coast < 0.12)
        coastal_marine = np.where(is_coastal_land, (0.12 - dist_to_coast) * 22.0, 0.0)
        alpine_tax = np.exp(-4.0 * np.maximum(0.0, self.elev - 0.25))

        total_food = (grain_yield * 2.5 + coastal_marine * 2.8) * alpine_tax
        land_capacity = np.clip(0.08 + 2.50 * np.log1p(total_food), 0.05, 15.0)
        self.capacity = np.where(self.elev <= 0.0, 0.0, land_capacity)

    def _init_demographics(self) -> None:
        emerged_mask = (self.elev >= 0.14) & (self.elev <= 0.35)
        fertility_scores = np.where(emerged_mask, self.capacity * self.veg, -1.0)
        cradle_y, cradle_x = np.unravel_index(np.argmax(fertility_scores), fertility_scores.shape)

        dy = self.Y_grid - self.Y_grid[cradle_y, cradle_x]
        dx = self.X_grid - self.X_grid[cradle_y, cradle_x]
        self.pop = 1.20 * np.exp(-(dx ** 2 + dy ** 2) / 0.002).astype(np.float32)
        self.infrastructure[cradle_y, cradle_x] = 2.0

    def step(self, dt: float = 1.0) -> None:
        self.epoch += 1
        self._update_climate(epoch=self.epoch)

        grad_ey, grad_ex = np.gradient(self.elev)
        terrain_roughness = np.sqrt(grad_ex ** 2 + grad_ey ** 2)
        mobility = 0.035 * np.exp(-18.0 * terrain_roughness) * np.where(self.elev <= 0.0, 0.0, 1.0)

        growth = 0.08 * self.pop * (1.0 - (self.pop / np.maximum(0.05, self.capacity)))
        diffusive_flux = mobility * laplacian_2d(self.pop)

        urban_potential = np.log1p(self.infrastructure * 0.8) + np.log1p(self.capacity * 0.5)
        u_lap = 0.025 * mobility * laplacian_2d(self.pop * np.tanh(urban_potential * 0.5))

        self.pop = np.where(
            self.elev <= 0.0,
            0.0,
            np.clip(self.pop + (diffusive_flux + growth + u_lap) * dt, 0.0, 20.0)
        )
        self.infrastructure = np.where(
            self.elev <= 0.0,
            0.0,
            np.clip(self.infrastructure + (self.pop * 0.05 - 0.012) * dt, 0.0, 40.0)
        )

        self.prestige = np.where(
            self.elev <= 0.0,
            0.0,
            self.pop * (1.0 + 0.40 * np.log1p(self.infrastructure))
        )

        dist_to_coast = np.minimum(self.X_grid, np.minimum(np.abs(self.X_grid - 0.76), 1.0 - self.X_grid))
        is_coastal = (dist_to_coast < 0.10)

        target_A = np.zeros_like(self.attention)
        target_A[..., 0] = 0.50 + 0.60 * np.maximum(0.0, self.elev) + 0.40 * (1.0 - self.veg)
        target_A[..., 1] = 0.50 + 0.80 * self.veg * (1.0 - np.minimum(1.0, self.pop * 0.2))
        target_A[..., 2] = 0.50 + 0.70 * (1.0 - self.hum)
        target_A[..., 3] = 0.50 + 0.70 * np.maximum(0.0, self.elev) + 0.50 * np.tanh(self.pop * 0.5)
        target_A[..., 4] = 0.50 + 0.60 * self.hum + 0.40 * np.where(is_coastal, 0.8, 0.2)
        target_A[..., 5] = 0.50 + 0.80 * np.tanh(self.pop * 0.5) + 0.20 * self.temp
        target_A[..., 6] = 0.50 + 0.60 * np.where(is_coastal, 0.9, 0.3) + 0.30 * np.tanh(self.pop * 0.5)

        target_sum = np.sum(target_A, axis=-1, keepdims=True)
        target_A = 7.0 * (target_A / np.maximum(0.1, target_sum))

        prestige_conductance = self.permeability * (0.015 + 0.035 * np.tanh(self.prestige * 0.20))

        for d in range(7):
            d_lap = prestige_conductance * laplacian_2d(self.attention[..., d])
            d_env = 0.04 * (target_A[..., d] - self.attention[..., d])
            new_val = self.attention[..., d] + (d_env + d_lap) * dt
            self.attention[..., d] = np.clip(new_val, 0.02, 7.0)

        norm_A = np.sum(self.attention, axis=-1, keepdims=True)
        self.attention = 7.0 * (self.attention / np.maximum(0.1, norm_A))

        h_target = 0.50 + 0.25 * np.tanh((self.attention[..., 6] - self.attention[..., 5]) * 0.5)
        s_target = 0.15 + 0.70 / (1.0 + np.exp(-2.5 * (np.tanh(self.pop * 0.5) - np.maximum(0.0, self.elev) * 0.5)))
        canopy = self.veg * 0.60 + self.hum * 0.40
        aridity = np.maximum(0.0, 1.0 - self.hum)
        cold = np.maximum(0.0, 0.45 - self.temp) * 1.5
        comp_driver = (np.maximum(0.0, self.elev) * 1.4 + aridity * 0.8 + cold * 0.7) - (canopy * 2.2)
        sig_target = 0.12 + 0.73 / (1.0 + np.exp(-4.5 * comp_driver))

        for i, target in enumerate([h_target, s_target, sig_target]):
            t_lap = 0.02 * self.permeability * laplacian_2d(self.typology[..., i])
            self.typology[..., i] = np.clip(
                self.typology[..., i] + (0.04 * (target - self.typology[..., i]) + t_lap) * dt,
                0.12, 0.88
            )

        # Continuous Biome Acoustic Bias Vector
        rho_air = np.exp(-0.90 * np.maximum(0.0, self.elev))
        drying = np.maximum(0.0, 1.0 - self.hum)

        self.phonetic_bias[..., 1] = np.clip(0.18 * (self.veg - 0.50), -0.25, 0.25)
        self.phonetic_bias[..., 4] = np.clip(0.25 * (self.hum - 0.45) + 0.15 * (self.temp - 0.50), -0.25, 0.25)
        self.phonetic_bias[..., 5] = np.clip(0.15 * (self.attention[..., 5] - self.attention[..., 0]), -0.25, 0.25)

        # Smooth continuous alpine laryngeal ejective driver
        ejective_drive = 1.0 / (1.0 + np.exp(12.0 * (rho_air - 0.75))) * (1.0 / (1.0 + np.exp(8.0 * (self.hum - 0.40))))
        self.phonetic_bias[..., 7] = np.clip(-0.65 * ejective_drive, -0.70, 0.0)

        for d in range(8):
            lap_bias = 0.015 * self.permeability * laplacian_2d(self.phonetic_bias[..., d])
            self.phonetic_bias[..., d] = np.clip(self.phonetic_bias[..., d] + lap_bias * dt, -0.45, 0.45)

        grad_hx, grad_hy = np.gradient(self.typology[..., 0])
        grad_sx, grad_sy = np.gradient(self.typology[..., 1])
        ling_dist = np.sqrt((grad_hx * 2.0) ** 2 + (grad_hy * 2.0) ** 2 + grad_sx ** 2 + grad_sy ** 2)
        terrain_conductance = np.exp(-12.0 * terrain_roughness) * np.where(self.elev <= 0.0, 0.10, 1.0)
        self.permeability = np.clip(terrain_conductance / (1.0 + np.exp(15.0 * (ling_dist - 0.25))), 0.0, 1.0)

    def get_cell_state(self, x: int, y: int) -> dict:
        r = int(np.clip(y, 0, self.H - 1))
        c = int(np.clip(x, 0, self.W - 1))
        return {
            "x": c, "y": r,
            "elevation": round(float(self.elev[r, c]), 3),
            "temperature": round(float(self.temp[r, c]), 3),
            "humidity": round(float(self.hum[r, c]), 3),
            "vegetation": round(float(self.veg[r, c]), 3),
            "population": round(float(self.pop[r, c]), 3),
            "carrying_capacity": round(float(self.capacity[r, c]), 3),
            "infrastructure": round(float(self.infrastructure[r, c]), 2),
            "prestige": round(float(self.prestige[r, c]), 2),
            "attention": [round(float(v), 3) for v in self.attention[r, c]],
            "head_directionality": round(float(self.typology[r, c, 0]), 3),
            "synthesis_index": round(float(self.typology[r, c, 1]), 3),
            "syllable_complexity": round(float(self.typology[r, c, 2]), 3),
            "phonetic_bias": [round(float(v), 3) for v in self.phonetic_bias[r, c]],
            "permeability": round(float(self.permeability[r, c]), 3),
        }