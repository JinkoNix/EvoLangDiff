"""
gpu_accelerator.py — Vectorized & GPU Acceleration Engine for EvoLang.
Includes batched 7D semantic filtering, Perceptual Listener Wells,
and GPU-resident 3D word-presence reaction-diffusion.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"


def _as_f32_tensor(arr: np.ndarray, device):
    return torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32), device=device)


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
        Maps continuous speech tokens into attractor wells with probabilistic
        boundary entropy calculation.
        """
        if cls.has_cuda:
            t_t = _as_f32_tensor(tokens_coords, cls.device)
            w_mu = _as_f32_tensor(wells_mu, cls.device)
            w_w = _as_f32_tensor(wells_weights, cls.device)

            dist_sq = torch.cdist(t_t, w_mu, p=2.0) ** 2
            potential = 0.5 * dist_sq - torch.log(torch.clamp(w_w, min=0.01)).unsqueeze(0)

            sorted_pot, sorted_indices = torch.sort(potential, dim=1)
            winning_idx = sorted_indices[:, 0]

            margin = sorted_pot[:, 1] - sorted_pot[:, 0]
            ambiguity_prob = 1.0 / (1.0 + torch.exp(25.0 * (margin - 0.08)))
            return winning_idx.cpu().numpy(), ambiguity_prob.cpu().numpy()

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

        t_arr = np.asarray(target_coords, dtype=np.float32)
        p_arr = np.asarray(candidate_pool, dtype=np.float32)
        w_arr = np.asarray(weights, dtype=np.float32)
        dists = np.sqrt(np.sum(((p_arr - t_arr) ** 2) * w_arr, axis=1))
        best_idx = int(np.argmin(dists))
        return best_idx, float(dists[best_idx])

    # -------------------------------------------------------------------------
    # GPU-RESIDENT 3D WORD PRESENCE REACTION-DIFFUSION
    # -------------------------------------------------------------------------

    @classmethod
    def alloc_presence_buffer(cls, capacity: int, h: int, w: int):
        if HAS_TORCH:
            return torch.zeros((capacity, h, w), dtype=torch.float32, device=cls.device)
        return np.zeros((capacity, h, w), dtype=np.float32)

    @classmethod
    def ensure_presence_capacity(cls, buf, needed: int):
        cap, h, w = buf.shape
        if needed <= cap:
            return buf
        new_cap = max(needed + 64, cap * 2)
        if HAS_TORCH and isinstance(buf, torch.Tensor):
            new_buf = torch.zeros((new_cap, h, w), dtype=torch.float32, device=buf.device)
            new_buf[:cap].copy_(buf)
            return new_buf
        new_buf = np.zeros((new_cap, h, w), dtype=np.float32)
        new_buf[:cap] = buf
        return new_buf

    @classmethod
    def add_to_presence_slice(cls, buf, idx_end: int, footprint_2d: np.ndarray, scale: float = 1.0) -> None:
        if idx_end <= 0:
            return
        if HAS_TORCH and isinstance(buf, torch.Tensor):
            fp_t = _as_f32_tensor(footprint_2d, buf.device)
            buf[:idx_end].add_(fp_t.unsqueeze(0), alpha=scale)
        else:
            buf[:idx_end] += footprint_2d[np.newaxis] * scale

    @classmethod
    def set_presence_slice(cls, buf, idx: int, arr_2d: np.ndarray) -> None:
        if HAS_TORCH and isinstance(buf, torch.Tensor):
            buf[idx].copy_(_as_f32_tensor(arr_2d, buf.device))
        else:
            buf[idx] = arr_2d

    @classmethod
    def add_presence_point_masked(cls, buf, idx: int, delta_2d: np.ndarray) -> None:
        if HAS_TORCH and isinstance(buf, torch.Tensor):
            buf[idx].add_(_as_f32_tensor(delta_2d, buf.device))
        else:
            buf[idx] += delta_2d

    @classmethod
    def get_presence_column(cls, buf, n_words: int, x: int, y: int) -> np.ndarray:
        n = min(int(n_words), int(buf.shape[0]))
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        if HAS_TORCH and isinstance(buf, torch.Tensor):
            return buf[:n, y, x].detach().cpu().numpy()
        return buf[:n, y, x].copy()

    @classmethod
    def step_word_presence(
        cls,
        buf,
        n_words: int,
        D_2d: np.ndarray,
        land_mask_2d: np.ndarray,
        pop_2d: np.ndarray,
        dt_scale: float = 2.0,
    ) -> None:
        n = min(int(n_words), int(buf.shape[0]))
        if n <= 0:
            return

        if HAS_TORCH and isinstance(buf, torch.Tensor):
            U = buf[:n]
            D = _as_f32_tensor(D_2d, buf.device)
            m = _as_f32_tensor(land_mask_2d, buf.device)
            pop_cap = torch.clamp(_as_f32_tensor(pop_2d, buf.device) / 2.0, 0.0, 2.5) + 0.3

            wN = 0.5 * (D[:-1, :] + D[1:, :]) * m[:-1, :] * m[1:, :]
            wW = 0.5 * (D[:, :-1] + D[:, 1:]) * m[:, :-1] * m[:, 1:]

            peaks = U.amax(dim=(1, 2))
            active = torch.nonzero(peaks > 0.02, as_tuple=True)[0]
            n_act = active.numel()
            if n_act == 0:
                return

            Ua = U if n_act == n else U[active]
            flux_y = wN.unsqueeze(0) * (Ua[:, :-1, :] - Ua[:, 1:, :])
            flux_x = wW.unsqueeze(0) * (Ua[:, :, :-1] - Ua[:, :, 1:])

            diff_U = torch.zeros_like(Ua)
            diff_U[:, 1:, :] += flux_y
            diff_U[:, :-1, :] -= flux_y
            diff_U[:, :, 1:] += flux_x
            diff_U[:, :, :-1] -= flux_x

            logistic = 0.05 * Ua * (1.0 - Ua / pop_cap.unsqueeze(0))
            U_new = torch.clamp(Ua + (diff_U + logistic) * dt_scale, 0.0, 5.0) * m.unsqueeze(0)
            if n_act == n:
                U.copy_(U_new)
            else:
                U[active] = U_new
            return

        # NumPy 2-face flux fallback (no 3D np.pad allocation)
        U = buf[:n]
        m = land_mask_2d.astype(np.float32)
        wN = 0.5 * (D_2d[:-1, :] + D_2d[1:, :]) * m[:-1, :] * m[1:, :]
        wW = 0.5 * (D_2d[:, :-1] + D_2d[:, 1:]) * m[:, :-1] * m[:, 1:]

        peaks = np.max(U, axis=(1, 2))
        active = np.where(peaks > 0.02)[0]
        if len(active) == 0:
            return

        Ua = U if len(active) == n else U[active]
        flux_y = wN[np.newaxis] * (Ua[:, :-1, :] - Ua[:, 1:, :])
        flux_x = wW[np.newaxis] * (Ua[:, :, :-1] - Ua[:, :, 1:])

        diff_U = np.zeros_like(Ua)
        diff_U[:, 1:, :] += flux_y
        diff_U[:, :-1, :] -= flux_y
        diff_U[:, :, 1:] += flux_x
        diff_U[:, :, :-1] -= flux_x

        pop_cap = np.clip(pop_2d / 2.0, 0.0, 2.5)[np.newaxis] + 0.3
        logistic = 0.05 * Ua * (1.0 - Ua / pop_cap)
        U_new = np.clip(Ua + (diff_U + logistic) * dt_scale, 0.0, 5.0) * m[np.newaxis]
        if len(active) == n:
            U[:] = U_new
        else:
            U[active] = U_new