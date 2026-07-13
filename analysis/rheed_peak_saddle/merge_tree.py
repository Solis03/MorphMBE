"""Maximum-bottleneck saddle measurement for grayscale spot pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SaddleResult:
    saddle_intensity: float
    activated_pixel_count: int
    connected: bool


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.uint8)
        self.has_a = np.zeros(n, dtype=bool)
        self.has_b = np.zeros(n, dtype=bool)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.has_a[rx] = self.has_a[rx] or self.has_a[ry]
        self.has_b[rx] = self.has_b[rx] or self.has_b[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return rx


def maximum_bottleneck_saddle(
    intensity: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
    corridor_mask: np.ndarray | None = None,
) -> SaddleResult:
    """Return the superlevel-set merge intensity connecting two seed regions.

    Pixels inside the corridor are activated from high intensity to low.  The
    saddle is the first intensity level where a component contains both spot
    core seeds.
    """
    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2:
        raise ValueError("intensity must be a 2D array")
    a = np.asarray(seed_a, dtype=bool)
    b = np.asarray(seed_b, dtype=bool)
    if a.shape != values.shape or b.shape != values.shape:
        raise ValueError("seed masks must match intensity shape")
    mask = np.ones(values.shape, dtype=bool) if corridor_mask is None else np.asarray(corridor_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("corridor mask must match intensity shape")
    active_domain = mask & np.isfinite(values)
    if not np.any(active_domain & a) or not np.any(active_domain & b):
        return SaddleResult(float("nan"), 0, False)

    flat_indices = np.flatnonzero(active_domain.ravel())
    order = flat_indices[np.argsort(values.ravel()[flat_indices])[::-1]]
    uf = _UnionFind(values.size)
    active = np.zeros(values.size, dtype=bool)
    flat_a = a.ravel()
    flat_b = b.ravel()
    width = values.shape[1]
    neighbor_offsets = (-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1)

    for count, idx in enumerate(order, start=1):
        active[idx] = True
        uf.has_a[idx] = bool(flat_a[idx])
        uf.has_b[idx] = bool(flat_b[idx])
        y, x = divmod(int(idx), width)
        for offset in neighbor_offsets:
            nbr = int(idx) + offset
            if nbr < 0 or nbr >= values.size or not active[nbr]:
                continue
            ny, nx = divmod(nbr, width)
            if abs(ny - y) > 1 or abs(nx - x) > 1:
                continue
            root = uf.union(int(idx), nbr)
            if uf.has_a[root] and uf.has_b[root]:
                return SaddleResult(float(values.ravel()[idx]), count, True)
        root = uf.find(int(idx))
        if uf.has_a[root] and uf.has_b[root]:
            return SaddleResult(float(values.ravel()[idx]), count, True)
    return SaddleResult(float("nan"), int(order.size), False)

