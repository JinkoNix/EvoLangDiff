"""
gpu_accelerator.py — Vectorized & GPU Acceleration Engine for EvoLang.
Includes batched 7D semantic filtering, territorial propagation, and Perceptual Listener Wells.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"


class GPUSemanticEngine:
    device = DEVICE
    has_cuda = HAS_TORCH and torch.cuda.is_available()

    @classmethod
    def get_hardware_info(cls) -> str:
        if cls.has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"PyTorch CUDA Engine ACTIVE [GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB]"
        elif HAS_TORCH:
            return "PyTorch CPU Vectorized Engine ACTIVE"
        return "NumPy SIMD Vectorized Engine ACTIVE"

    @classmethod
    def decode_listener_wells_batch(
        cls,
        tokens_coords: np.ndarray,
        wells_mu: np.ndarray,
        wells_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Maps emitted continuous speech tokens into attractor wells with continuous
        probabilistic boundary entropy calculation.
        """
        if cls.has_cuda:
            t_t = torch.tensor(tokens_coords, dtype=torch.float32, device=cls.device)
            w_mu = torch.tensor(wells_mu, dtype=torch.float32, device=cls.device)
            w_w = torch.tensor(wells_weights, dtype=torch.float32, device=cls.device)

            dist_sq = torch.cdist(t_t, w_mu, p=2.0) ** 2
            potential = 0.5 * dist_sq - torch.log(torch.clamp(w_w, min=0.01)).unsqueeze(0)

            sorted_pot, sorted_indices = torch.sort(potential, dim=1)
            winning_idx = sorted_indices[:, 0]

            margin = sorted_pot[:, 1] - sorted_pot[:, 0]
            # Continuous probabilistic ambiguity: logistic gate replaces boolean threshold
            ambiguity_prob = 1.0 / (1.0 + torch.exp(25.0 * (margin - 0.08)))
            return winning_idx.cpu().numpy(), ambiguity_prob.cpu().numpy()

        else:
            diff = tokens_coords[:, np.newaxis, :] - wells_mu[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            potential = 0.5 * dist_sq - np.log(np.maximum(wells_weights, 0.01))[np.newaxis, :]

            sorted_indices = np.argsort(potential, axis=1)
            sorted_pot = np.take_along_axis(potential, sorted_indices, axis=1)

            winning_idx = sorted_indices[:, 0]
            margin = sorted_pot[:, 1] - sorted_pot[:, 0]
            ambiguity_prob = 1.0 / (1.0 + np.exp(25.0 * (margin - 0.08)))
            return winning_idx, ambiguity_prob

    @classmethod
    def filter_candidate_density_gpu(
        cls,
        candidates_coords: Sequence[Sequence[float]],
        existing_coords: Sequence[Sequence[float]],
        weights: Sequence[float],
        d_min: float,
    ) -> list[bool]:
        if not candidates_coords:
            return []
        if not existing_coords:
            return [True] * len(candidates_coords)

        d_min_sq = d_min ** 2

        if cls.has_cuda:
            cand_t = torch.tensor(candidates_coords, dtype=torch.float32, device=cls.device)
            exist_t = torch.tensor(existing_coords, dtype=torch.float32, device=cls.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=cls.device).view(1, 1, 7)

            diff = cand_t.unsqueeze(1) - exist_t.unsqueeze(0)
            weighted_d_sq = torch.sum(w_t * (diff ** 2), dim=-1)
            min_dists, _ = torch.min(weighted_d_sq, dim=1)
            return (min_dists >= d_min_sq).cpu().tolist()
        else:
            cand_arr = np.asarray(candidates_coords, dtype=np.float32)
            exist_arr = np.asarray(existing_coords, dtype=np.float32)
            w_arr = np.asarray(weights, dtype=np.float32).reshape(1, 1, 7)

            diff = cand_arr[:, np.newaxis, :] - exist_arr[np.newaxis, :, :]
            weighted_d_sq = np.sum(w_arr * (diff ** 2), axis=-1)
            min_dists = np.min(weighted_d_sq, axis=1)
            return (min_dists >= d_min_sq).tolist()

    @classmethod
    def find_nearest_semantic_neighbor(
        cls,
        target_coords: Sequence[float],
        candidate_pool: Sequence[Sequence[float]],
        weights: Sequence[float],
    ) -> Tuple[int, float]:
        if not candidate_pool:
            return 0, 0.0

        if cls.has_cuda:
            t_t = torch.tensor(target_coords, dtype=torch.float32, device=cls.device).view(1, 7)
            p_t = torch.tensor(candidate_pool, dtype=torch.float32, device=cls.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=cls.device).view(1, 7)
            dists = torch.sqrt(torch.sum(((p_t - t_t) ** 2) * w_t, dim=1))
            best_idx = int(torch.argmin(dists).item())
            return best_idx, float(dists[best_idx].item())
        else:
            t_arr = np.asarray(target_coords, dtype=np.float32)
            p_arr = np.asarray(candidate_pool, dtype=np.float32)
            w_arr = np.asarray(weights, dtype=np.float32)
            dists = np.sqrt(np.sum(((p_arr - t_arr) ** 2) * w_arr, axis=1))
            best_idx = int(np.argmin(dists))
            return best_idx, float(dists[best_idx])