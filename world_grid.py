"""
world_grid.py — Continuous 2D Tensor Geography, Urban Agglomeration Gravity,
Dynamic Linguistic Territory Colors, and Provincial Secession.
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
    MAX_POLITIES = 8

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

        # Geographic Resource Fields
        self.minerals = np.zeros((self.H, self.W), dtype=np.float32)
        self.fisheries = np.zeros((self.H, self.W), dtype=np.float32)
        self.resource_value = np.zeros((self.H, self.W), dtype=np.float32)

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

        # Continuous Societal Hegemony Tensor: [H, W, K]
        self.hegemony = np.zeros((self.H, self.W, self.MAX_POLITIES), dtype=np.float32)
        self.polity_centers: list[dict] = []
        self.events: list[str] = []

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

        climax_veg = 0.95 / (1.0 + np.exp(-6.0 * (aridity_idx - 0.45)))
        treeline = np.exp(-3.50 * (np.maximum(0.0, self.elev - 0.35) ** 2))
        thermal_growth = 1.0 / (1.0 + np.exp(-8.0 * (self.temp - 0.22)))
        emergent_veg = np.where(self.elev <= 0.0, 0.0, climax_veg * treeline * thermal_growth)
        depletion = np.exp(-0.35 * (np.maximum(0.0, self.pop) ** 1.20))
        self.veg = np.clip(emergent_veg * depletion, 0.01, 0.98)

        arable_d2 = ((self.hum - 0.55) / 0.22) ** 2 + ((self.veg - 0.45) / 0.22) ** 2 + ((self.temp - 0.55) / 0.22) ** 2
        grain_yield = np.exp(-arable_d2) * np.maximum(0.0, 1.0 - self.elev * 1.5) * (1.0 + 0.45 * np.log1p(self.infrastructure))

        is_coastal_land = (self.elev > 0.0) & (dist_to_coast < 0.12)
        self.fisheries = np.where(is_coastal_land, (0.12 - dist_to_coast) * 22.0, 0.0)
        self.minerals = np.where((self.elev > 0.35) & (self.elev < 0.95), (self.elev - 0.35) * 3.5, 0.0)

        alpine_tax = np.exp(-4.0 * np.maximum(0.0, self.elev - 0.25))
        total_food = (grain_yield * 2.5 + self.fisheries * 2.8) * alpine_tax
        self.resource_value = np.where(self.elev <= 0.0, 0.0, (grain_yield * 1.5 + self.fisheries * 1.8 + self.minerals * 1.4))

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

        self.polity_centers = [{
            "id": 0,
            "name": "Ancestral Cradle",
            "x": int(cradle_x),
            "y": int(cradle_y),
            "age": 0,
            "prestige": 2.0,
            "color": [210, 70, 70],
        }]
        self.hegemony[cradle_y, cradle_x, 0] = 5.0

    def _update_societies(self, dt: float) -> None:
        land_mask = (self.elev > 0.0) & (self.pop > 0.20)
        score_matrix = np.where(land_mask, self.prestige * (1.0 + 0.3 * np.log1p(self.infrastructure)), 0.0)

        # 1. Prune completely absorbed/extinct polities
        active_polities = []
        for p in self.polity_centers:
            pid = p["id"]
            tot_h = float(np.sum(self.hegemony[..., pid]))
            if tot_h > 0.5 or p["id"] == 0:
                active_polities.append(p)
            else:
                self.hegemony[..., pid] = 0.0
                self.events.append(f"Epoch {self.epoch}: {p['name']} has fallen into oblivion.")
        self.polity_centers = active_polities

        # 2. Dynamic Linguistic Secession & Emergent Breakaway Polities
        old_hegemony_owners = np.argmax(self.hegemony, axis=-1)

        # Check for provincial fracture in distant overextended territory
        if self.epoch % 20 == 0 and len(self.polity_centers) < self.MAX_POLITIES:
            for p in list(self.polity_centers):
                pid = p["id"]
                cap_x, cap_y = p["x"], p["y"]
                cap_typo = self.typology[cap_y, cap_x]

                # Find territory owned by this polity with high linguistic divergence
                owned_mask = (old_hegemony_owners == pid) & (self.pop > 0.40) & (self.elev > 0.0)
                if not np.any(owned_mask):
                    continue

                diff_h = np.abs(self.typology[..., 0] - cap_typo[0])
                diff_s = np.abs(self.typology[..., 1] - cap_typo[1])
                diff_sig = np.abs(self.typology[..., 2] - cap_typo[2])
                ling_divergence = np.where(owned_mask, diff_h + diff_s + diff_sig, 0.0)

                # Distance from capital adds administrative friction
                dy, dx = np.ogrid[:self.H, :self.W]
                dist_cells = np.sqrt((dx - cap_x) ** 2 + (dy - cap_y) ** 2)
                rebellion_potential = ling_divergence * (dist_cells / 16.0)

                if np.max(rebellion_potential) > 0.42:
                    ry, rx = np.unravel_index(np.argmax(rebellion_potential), rebellion_potential.shape)
                    # Assign unused ID
                    used_ids = {pol["id"] for pol in self.polity_centers}
                    new_id = next(i for i in range(self.MAX_POLITIES) if i not in used_ids)
                    
                    reb_typo = self.typology[ry, rx]
                    # Dynamic color computed directly from local linguistic typology: R=h, G=s, B=sigma
                    reb_color = [
                        int(np.clip(reb_typo[0] * 220 + 25, 20, 245)),
                        int(np.clip(reb_typo[1] * 210 + 35, 20, 245)),
                        int(np.clip(reb_typo[2] * 220 + 25, 20, 245)),
                    ]

                    tier_names = ["Secessionist League", "Free Realm", "Breakaway Kingdom", "Frontier Domain"]
                    name_tier = tier_names[new_id % len(tier_names)]
                    new_name = f"{name_tier} of ({rx},{ry})"

                    self.polity_centers.append({
                        "id": new_id,
                        "name": new_name,
                        "x": int(rx),
                        "y": int(ry),
                        "age": self.epoch,
                        "prestige": float(self.prestige[ry, rx]) * 1.2,
                        "color": reb_color,
                    })

                    # Carve out local breakaway hegemony
                    self.hegemony[ry, rx, new_id] = 6.0
                    self.hegemony[ry, rx, pid] *= 0.2
                    self.events.append(f"Epoch {self.epoch}: {new_name} seceded from {p['name']} due to linguistic divergence!")
                    break

        # 3. Lotka-Volterra Territorial Competition
        for p in self.polity_centers:
            pid = p["id"]
            px, py = p["x"], p["y"]
            p["prestige"] = float(self.prestige[py, px])

            # Update society's territorial color dynamically from current capital typology
            cur_t = self.typology[py, px]
            p["color"] = [
                int(np.clip(cur_t[0] * 220 + 25, 20, 245)),
                int(np.clip(cur_t[1] * 210 + 35, 20, 245)),
                int(np.clip(cur_t[2] * 220 + 25, 20, 245)),
            ]

            source = np.zeros((self.H, self.W), dtype=np.float32)
            source[py, px] = max(1.0, float(self.prestige[py, px]) * 1.5)

            resource_pull = 0.03 * self.resource_value * self.hegemony[..., pid]
            diff_flux = 0.04 * self.permeability * laplacian_2d(self.hegemony[..., pid])

            # Imperial Overextension: Decay increases with distance from capital
            dy, dx = np.ogrid[:self.H, :self.W]
            dist_decay = 1.0 + 0.04 * np.sqrt((dx - px) ** 2 + (dy - py) ** 2)
            decay = 0.015 * dist_decay * self.hegemony[..., pid]

            # Warfare displacement
            warfare_loss = np.zeros((self.H, self.W), dtype=np.float32)
            for rival in self.polity_centers:
                rid = rival["id"]
                if rid == pid:
                    continue
                rival_prestige = rival["prestige"]
                if rival_prestige > p["prestige"] + 0.3:
                    warfare_loss += 0.05 * (rival_prestige - p["prestige"]) * self.hegemony[..., pid] * self.hegemony[..., rid]

            self.hegemony[..., pid] = np.clip(
                self.hegemony[..., pid] + (diff_flux + source + resource_pull - decay - warfare_loss) * dt,
                0.0, 50.0
            )

        for pid in range(self.MAX_POLITIES):
            self.hegemony[..., pid] = np.where(self.elev <= 0.0, 0.0, self.hegemony[..., pid])

        # Log conquest events
        new_hegemony_owners = np.argmax(self.hegemony, axis=-1)
        flips = (old_hegemony_owners != new_hegemony_owners) & (self.elev > 0.0) & (self.pop > 0.50)
        if np.any(flips) and self.epoch % 15 == 0:
            fy, fx = np.unravel_index(np.argmax(flips), flips.shape)
            invader_id = new_hegemony_owners[fy, fx]
            invader = next((p for p in self.polity_centers if p["id"] == invader_id), None)
            if invader:
                self.events.append(f"Epoch {self.epoch}: {invader['name']} annexed resource province at ({fx}, {fy}).")

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

        self._update_societies(dt)

        grad_hx, grad_hy = np.gradient(np.max(self.hegemony, axis=-1))
        imperial_turnover = np.sqrt(grad_hx ** 2 + grad_hy ** 2)

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
        s_base = 0.15 + 0.70 / (1.0 + np.exp(-2.5 * (np.tanh(self.pop * 0.5) - np.maximum(0.0, self.elev) * 0.5)))
        s_target = np.clip(s_base - 0.30 * np.tanh(imperial_turnover * 2.0), 0.12, 0.88)

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

        rho_air = np.exp(-0.90 * np.maximum(0.0, self.elev))
        self.phonetic_bias[..., 1] = np.clip(0.18 * (self.veg - 0.50), -0.25, 0.25)
        self.phonetic_bias[..., 4] = np.clip(0.25 * (self.hum - 0.45) + 0.15 * (self.temp - 0.50), -0.25, 0.25)
        self.phonetic_bias[..., 5] = np.clip(0.15 * (self.attention[..., 5] - self.attention[..., 0]), -0.25, 0.25)

        ejective_drive = 1.0 / (1.0 + np.exp(12.0 * (rho_air - 0.70))) * (1.0 / (1.0 + np.exp(8.0 * (self.hum - 0.35))))
        self.phonetic_bias[..., 7] = np.clip(-0.50 * ejective_drive, -0.60, 0.0)

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

        h_vals = self.hegemony[r, c]
        tot_h = float(np.sum(h_vals))
        if tot_h > 0.01:
            dom_id = int(np.argmax(h_vals))
            dom_share = float(h_vals[dom_id] / tot_h)
            dom_polity = next((p for p in self.polity_centers if p["id"] == dom_id), None)
            dom_name = dom_polity["name"] if dom_polity else f"Polity {dom_id}"
            dom_color = dom_polity["color"] if dom_polity else [150, 150, 150]
        else:
            dom_id = -1
            dom_share = 0.0
            dom_name = "Independent Tribes (Ungoverned)"
            dom_color = [100, 100, 100]

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
            "polity": {
                "dominant_id": dom_id,
                "name": dom_name,
                "control_percentage": round(dom_share * 100.0, 1),
                "is_capital": any(p["x"] == c and p["y"] == r for p in self.polity_centers),
                "color": dom_color,
            }
        }