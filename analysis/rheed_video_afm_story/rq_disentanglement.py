from __future__ import annotations

import numpy as np
import torch


def center_and_rq(
    height_nm: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, float]:
    arr = np.asarray(height_nm, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    centered = arr - float(arr.mean())
    q = float(np.sqrt(np.mean(centered**2)))
    if q <= epsilon:
        raise ValueError(f"AFM map has near-zero Rq: {q}")
    return centered.astype(np.float32), q


def unit_shape(
    height_nm: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, float]:
    centered, q = center_and_rq(height_nm, epsilon=epsilon)
    shape = centered / (q + epsilon)
    shape = project_unit_rq_np(shape, epsilon=epsilon)
    return centered, shape.astype(np.float32), q


def rq_np(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a - np.nanmean(a)
    return float(np.sqrt(np.nanmean(a**2)))


def ra_np(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a - np.nanmean(a)
    return float(np.nanmean(np.abs(a)))


def project_unit_rq_np(arr: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = a - float(a.mean())
    q = float(np.sqrt(np.mean(a**2)))
    return (a / (q + epsilon)).astype(np.float32)


def project_unit_rq_torch(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    centered = x - x.mean(dim=dims, keepdim=True)
    q = torch.sqrt(torch.mean(centered.square(), dim=dims, keepdim=True) + epsilon)
    return centered / q


def physical_from_q(
    shape: np.ndarray, q_nm: float, epsilon: float = 1e-6
) -> np.ndarray:
    return float(q_nm) * project_unit_rq_np(shape, epsilon=epsilon)
