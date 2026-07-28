from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.spectral import load_unit_map
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np


LEVELS = (0.55, 0.70, 0.82)
ISLAND_FEATURE_COLUMNS = [
    *(f"log_component_count_q{int(level * 100):02d}" for level in LEVELS),
    *(f"log_median_area_q{int(level * 100):02d}" for level in LEVELS),
    "median_solidity_q70",
    "median_eccentricity_q70",
    "boundary_gradient_ratio_q55",
    "boundary_gradient_ratio_q70",
    "boundary_gradient_ratio_q82",
    "log_valley_component_count_q18",
    "log_valley_median_area_q18",
    "gradient_p90",
    "laplacian_rms",
    "flat_fraction",
]


def _interior_regions(mask: np.ndarray, minimum_area: int = 5) -> list:
    labels = measure.label(mask, connectivity=2)
    height, width = mask.shape
    return [
        region
        for region in measure.regionprops(labels)
        if region.area >= minimum_area
        and not (
            region.bbox[0] == 0
            or region.bbox[1] == 0
            or region.bbox[2] == height
            or region.bbox[3] == width
        )
    ]


def _component_summary(
    array: np.ndarray,
    gradient: np.ndarray,
    *,
    quantile: float,
    high: bool,
) -> dict[str, float]:
    threshold = float(np.quantile(array, quantile))
    mask = array >= threshold if high else array <= threshold
    regions = _interior_regions(mask)
    areas = np.asarray([region.area for region in regions], dtype=float)
    boundary = mask ^ morphology.erosion(mask)
    suffix = f"q{int(quantile * 100):02d}"
    prefix = "" if high else "valley_"
    result = {
        f"log_{prefix}component_count_{suffix}": float(
            np.log1p(len(regions))
        ),
        f"log_{prefix}median_area_{suffix}": float(
            np.log1p(np.median(areas) if len(areas) else 0.0)
        ),
    }
    if high:
        result[f"boundary_gradient_ratio_{suffix}"] = float(
            np.mean(gradient[boundary])
            / max(float(np.mean(gradient)), 1e-8)
        )
        result[f"median_solidity_{suffix}"] = float(
            np.median([region.solidity for region in regions])
            if regions
            else 0.0
        )
        result[f"median_eccentricity_{suffix}"] = float(
            np.median([region.eccentricity for region in regions])
            if regions
            else 0.0
        )
    return result


def extract_island_features(array: np.ndarray) -> dict[str, float]:
    """Measure topology that radial PSD and Rq cannot identify.

    Component counts and areas are measured on smoothed unit-Rq level sets.
    Boundary-gradient ratios capture the mesa/island edge contrast that is
    visibly absent from Gaussian cloud fields.
    """

    unit = project_unit_rq_np(np.asarray(array, dtype=np.float32))
    smooth = ndimage.gaussian_filter(unit, sigma=0.8, mode="reflect")
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    laplacian = ndimage.laplace(smooth, mode="reflect")
    features: dict[str, float] = {}
    for level in LEVELS:
        features.update(
            _component_summary(
                smooth, gradient, quantile=level, high=True
            )
        )
    features.update(
        _component_summary(
            smooth, gradient, quantile=0.18, high=False
        )
    )
    features["gradient_p90"] = float(np.quantile(gradient, 0.90))
    features["laplacian_rms"] = float(
        np.sqrt(np.mean(np.square(laplacian)))
    )
    # An absolute unit-Rq threshold retains information; a gradient quantile
    # would make this feature constant by construction.
    features["flat_fraction"] = float(np.mean(gradient < 0.075))
    return {
        column: float(features[column]) for column in ISLAND_FEATURE_COLUMNS
    }


def scan_island_feature_table(
    rows: pd.DataFrame, *, resolution: int
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for _, row in rows.iterrows():
        record: dict[str, float | str] = {
            "sample_id": str(row["sample_id"]),
            "growth_run_id": str(row["growth_run_id"]),
        }
        record.update(
            extract_island_features(load_unit_map(row, resolution))
        )
        records.append(record)
    return pd.DataFrame(records)


def group_island_feature_table(
    rows: pd.DataFrame, *, resolution: int
) -> pd.DataFrame:
    scans = scan_island_feature_table(rows, resolution=resolution)
    table = scans.groupby("growth_run_id")[ISLAND_FEATURE_COLUMNS].median()
    table.index = table.index.astype(str)
    return table.sort_index()


@dataclass
class IslandConditionModel:
    """Small-data map from AFM morphology condition to island statistics."""

    ridge: Ridge
    output_scaler: StandardScaler
    feature_columns: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    alpha: float
    train_groups: list[str]

    def predict(self, condition_z: np.ndarray) -> dict[str, float]:
        condition = np.asarray(condition_z, dtype=float).reshape(1, -1)
        standardized = self.ridge.predict(condition)
        raw = self.output_scaler.inverse_transform(standardized)[0]
        raw = np.clip(raw, self.lower_bounds, self.upper_bounds)
        return {
            column: float(value)
            for column, value in zip(self.feature_columns, raw)
        }


def fit_island_condition_model(
    *,
    train_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    resolution: int,
    alphas: Iterable[float] = (0.3, 1.0, 3.0, 10.0, 30.0),
) -> tuple[IslandConditionModel, pd.DataFrame, pd.DataFrame]:
    targets = group_island_feature_table(
        train_rows, resolution=resolution
    )
    groups = list(targets.index.astype(str))
    raw_conditions = (
        train_rows.groupby("growth_run_id")[condition_scaler.columns]
        .median()
        .loc[groups]
        .to_numpy(float)
    )
    conditions = condition_scaler.transform(raw_conditions, clip=False)
    output_scaler = StandardScaler().fit(
        targets[ISLAND_FEATURE_COLUMNS].to_numpy(float)
    )
    target_z = output_scaler.transform(
        targets[ISLAND_FEATURE_COLUMNS].to_numpy(float)
    )
    records: list[dict[str, float]] = []
    for alpha in alphas:
        predictions = np.zeros_like(target_z)
        for held in range(len(groups)):
            keep = np.arange(len(groups)) != held
            model = Ridge(alpha=float(alpha)).fit(
                conditions[keep], target_z[keep]
            )
            predictions[held] = model.predict(
                conditions[held : held + 1]
            )[0]
        records.append(
            {
                "alpha": float(alpha),
                "loo_island_feature_mae_z": float(
                    np.mean(np.abs(predictions - target_z))
                ),
                "loo_island_feature_rmse_z": float(
                    np.sqrt(np.mean(np.square(predictions - target_z)))
                ),
                "loo_prediction_variance": float(
                    np.mean(np.var(predictions, axis=0))
                ),
            }
        )
    cv = pd.DataFrame(records).sort_values(
        ["loo_island_feature_mae_z", "alpha"]
    )
    alpha = float(cv.iloc[0]["alpha"])
    ridge = Ridge(alpha=alpha).fit(conditions, target_z)
    values = targets[ISLAND_FEATURE_COLUMNS].to_numpy(float)
    spread = np.maximum(np.std(values, axis=0), 1e-6)
    lower = np.quantile(values, 0.02, axis=0) - 0.20 * spread
    upper = np.quantile(values, 0.98, axis=0) + 0.20 * spread
    return (
        IslandConditionModel(
            ridge=ridge,
            output_scaler=output_scaler,
            feature_columns=list(ISLAND_FEATURE_COLUMNS),
            lower_bounds=lower,
            upper_bounds=upper,
            alpha=alpha,
            train_groups=groups,
        ),
        cv.reset_index(drop=True),
        targets,
    )


def _count(target: dict[str, float], level: int) -> float:
    return float(
        max(
            np.expm1(target[f"log_component_count_q{level:02d}"]),
            1.0,
        )
    )


def _area(target: dict[str, float], level: int) -> float:
    return float(
        max(
            np.expm1(target[f"log_median_area_q{level:02d}"]),
            5.0,
        )
    )


def _periodic_delta(axis: np.ndarray, center: float, size: int) -> np.ndarray:
    delta = np.abs(axis - center)
    return np.minimum(delta, size - delta)


@dataclass(frozen=True)
class IslandPrimitiveGenerator:
    """Generate AFM topography from sampled island/valley primitives.

    The generator learns only distributions of object statistics. It never
    copies a measured AFM patch. Random object locations, Fourier boundary
    perturbations, heights and orientations create novel realizations.
    """

    resolution: int = 128
    residual_weight: float = 0.08
    laguerre_count_factor: float = 3.0
    fine_count_factor: float = 3.0

    def _superellipse_field(
        self, target: dict[str, float], rng: np.random.Generator
    ) -> np.ndarray:
        n = self.resolution
        yy, xx = np.mgrid[:n, :n]
        count = int(np.clip(np.round(1.35 * _count(target, 70)), 7, 70))
        median_area = _area(target, 70)
        eccentricity = float(
            np.clip(target["median_eccentricity_q70"], 0.15, 0.94)
        )
        aspect = float(
            np.clip(1.0 / np.sqrt(1.0 - eccentricity**2), 1.0, 3.0)
        )
        solidity = float(
            np.clip(target["median_solidity_q70"], 0.72, 0.98)
        )
        edge_ratio = float(
            np.clip(target["boundary_gradient_ratio_q70"], 0.9, 1.8)
        )
        radius = np.sqrt(median_area / np.pi)
        field = np.full((n, n), -0.35, dtype=np.float64)
        for _ in range(count):
            cx, cy = rng.uniform(0, n, size=2)
            area_scale = float(rng.lognormal(0.0, 0.30))
            local_aspect = float(
                np.clip(aspect * rng.lognormal(0.0, 0.16), 1.0, 3.5)
            )
            a = np.clip(radius * np.sqrt(local_aspect) * area_scale, 2.0, n / 5)
            b = np.clip(radius / np.sqrt(local_aspect) * area_scale, 1.7, n / 5)
            angle = rng.uniform(0.0, np.pi)
            dx = _periodic_delta(xx, cx, n)
            dy = _periodic_delta(yy, cy, n)
            # Restore signs after selecting the periodic distance magnitude.
            dx *= np.sign(((xx - cx + n / 2) % n) - n / 2)
            dy *= np.sign(((yy - cy + n / 2) % n) - n / 2)
            xr = np.cos(angle) * dx + np.sin(angle) * dy
            yr = -np.sin(angle) * dx + np.cos(angle) * dy
            theta = np.arctan2(yr / max(b, 1e-6), xr / max(a, 1e-6))
            irregularity = (1.0 - solidity) * (
                0.80 * np.cos(3 * theta + rng.uniform(0, 2 * np.pi))
                + 0.45 * np.cos(5 * theta + rng.uniform(0, 2 * np.pi))
            )
            exponent = float(rng.uniform(2.5, 5.0))
            distance = (
                np.abs(xr / a) ** exponent
                + np.abs(yr / b) ** exponent
            ) ** (1.0 / exponent)
            distance /= np.clip(1.0 + irregularity, 0.65, 1.35)
            edge_width = np.clip(0.24 / edge_ratio, 0.10, 0.28)
            mask = 1.0 / (
                1.0 + np.exp(np.clip((distance - 1.0) / edge_width, -30, 30))
            )
            dome = np.clip(1.0 - distance**2, 0.0, 1.0)
            height = float(rng.lognormal(0.0, 0.28))
            primitive = -0.35 + height * mask * (0.68 + 0.32 * dome)
            field = np.maximum(field, primitive)
        valley_count = int(
            np.clip(
                np.round(
                    1.25
                    * np.expm1(
                        target["log_valley_component_count_q18"]
                    )
                ),
                3,
                70,
            )
        )
        valley_area = max(
            float(
                np.expm1(target["log_valley_median_area_q18"])
            ),
            5.0,
        )
        valley_radius = np.sqrt(valley_area / np.pi)
        for _ in range(valley_count):
            cx, cy = rng.uniform(0, n, size=2)
            sigma = np.clip(
                valley_radius * rng.lognormal(0.0, 0.3), 1.2, n / 8
            )
            dx = _periodic_delta(xx, cx, n)
            dy = _periodic_delta(yy, cy, n)
            valley = -float(rng.uniform(0.35, 0.9)) * np.exp(
                -(dx**2 + dy**2) / (2.0 * sigma**2)
            )
            field += valley
        return field

    def _laguerre_field(
        self, target: dict[str, float], rng: np.random.Generator
    ) -> np.ndarray:
        n = self.resolution
        yy, xx = np.mgrid[:n, :n]
        # The 70th-percentile components are the most stable proxy for the
        # number of capture zones. q55 merges neighboring terraces and made
        # early prototypes unrealistically coarse for intermediate/rough AFM.
        # Only 30% of pixels survive the q70 threshold. Roughly three capture
        # zones are therefore needed per observed q70 component; using a
        # one-to-one count produced only 4--8 high components and oversized
        # plateaus in v1.
        count = int(
            np.clip(
                np.round(self.laguerre_count_factor * _count(target, 70)),
                18,
                180,
            )
        )
        median_area = _area(target, 70)
        expected_radius = np.sqrt(median_area / np.pi)
        centers = rng.uniform(0, n, size=(count, 2))
        weights = rng.lognormal(
            mean=np.log(max(expected_radius, 1.0)),
            sigma=0.30,
            size=count,
        )
        distances = []
        for (cy, cx), weight in zip(centers, weights):
            dx = _periodic_delta(xx, cx, n)
            dy = _periodic_delta(yy, cy, n)
            distances.append(np.hypot(dx, dy) / max(weight, 1e-3))
        stack = np.stack(distances)
        order = np.argsort(stack, axis=0)
        nearest = np.take_along_axis(stack, order[:1], axis=0)[0]
        second = np.take_along_axis(stack, order[1:2], axis=0)[0]
        labels = order[0]
        heights = rng.normal(0.25, 0.75, size=count)
        field = heights[labels]
        # Each capture zone becomes a gently crowned terrace; the gap between
        # first and second weighted distances identifies coalescence valleys.
        field += 0.30 * np.clip(1.0 - nearest, 0.0, 1.0)
        edge_ratio = float(
            np.clip(target["boundary_gradient_ratio_q55"], 0.9, 1.8)
        )
        groove_width = 0.11 / edge_ratio
        boundary_gap = np.maximum(second - nearest, 0.0)
        field -= 0.65 * np.exp(-boundary_gap / groove_width)
        field = ndimage.gaussian_filter(
            field, sigma=np.clip(1.55 / edge_ratio, 0.65, 1.7), mode="wrap"
        )
        # A second, lower-amplitude capture-zone population represents
        # continued nucleation on/among coalescing terraces. Real smooth 6022
        # has a large flat fraction and receives little of this layer; rougher
        # 6056/6080 receive visibly more small islands.
        fine_count = int(
            np.clip(
                np.round(self.fine_count_factor * _count(target, 82)),
                16,
                220,
            )
        )
        fine_centers = rng.uniform(0, n, size=(fine_count, 2))
        fine_distances = []
        for cy, cx in fine_centers:
            dx = _periodic_delta(xx, cx, n)
            dy = _periodic_delta(yy, cy, n)
            fine_distances.append(np.hypot(dx, dy))
        fine_stack = np.stack(fine_distances)
        fine_order = np.argsort(fine_stack, axis=0)
        fine_nearest = np.take_along_axis(
            fine_stack, fine_order[:1], axis=0
        )[0]
        fine_second = np.take_along_axis(
            fine_stack, fine_order[1:2], axis=0
        )[0]
        fine_labels = fine_order[0]
        fine_heights = rng.normal(0.0, 1.0, size=fine_count)
        fine = fine_heights[fine_labels]
        fine_scale = max(
            np.sqrt(_area(target, 82) / np.pi), 1.0
        )
        fine += 0.20 * np.clip(
            1.0 - fine_nearest / fine_scale, 0.0, 1.0
        )
        fine -= 0.30 * np.exp(
            -np.maximum(fine_second - fine_nearest, 0.0)
            / max(0.20 * fine_scale, 0.4)
        )
        fine = ndimage.gaussian_filter(fine, sigma=0.65, mode="wrap")
        fine -= float(np.mean(fine))
        fine /= max(float(np.std(fine)), 1e-8)
        fine_weight = float(
            np.clip(
                0.34 - (target["flat_fraction"] - 0.08),
                0.12,
                0.32,
            )
        )
        field += fine_weight * fine
        return field

    def generate(
        self,
        target: dict[str, float],
        *,
        seed: int,
        mode: str,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        if mode == "superellipse":
            field = self._superellipse_field(target, rng)
        elif mode == "laguerre":
            field = self._laguerre_field(target, rng)
        elif mode == "hybrid":
            islands = self._superellipse_field(target, rng)
            terraces = self._laguerre_field(target, rng)
            # Large median level-set areas indicate stronger coalescence and
            # therefore favor terrace/capture-zone morphology.
            coalescence = float(
                np.clip((_area(target, 55) - 45.0) / 115.0, 0.20, 0.68)
            )
            field = (1.0 - coalescence) * islands + coalescence * terraces
        else:
            raise ValueError(f"Unknown island generator mode: {mode}")
        noise = rng.normal(size=(self.resolution, self.resolution))
        residual = ndimage.gaussian_filter(noise, sigma=0.65, mode="wrap")
        field += self.residual_weight * residual
        return project_unit_rq_np(field).astype(np.float32)

    def generate_ensemble(
        self,
        target: dict[str, float],
        *,
        draws: int,
        seed: int,
        mode: str,
    ) -> list[np.ndarray]:
        return [
            self.generate(target, seed=seed + index, mode=mode)
            for index in range(int(draws))
        ]
