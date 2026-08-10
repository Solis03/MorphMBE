from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    aggregate_group_conditions,
)
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
        aggregate_group_conditions(
            train_rows,
            condition_scaler.columns,
        )
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


def _periodic_signed_delta(
    axis: np.ndarray, center: float, size: int
) -> np.ndarray:
    """Shortest signed displacement on a periodic image domain."""

    return (axis - center + size / 2.0) % size - size / 2.0


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

    def _dense_microellipse_field(
        self,
        target: dict[str, float],
        rng: np.random.Generator,
        *,
        layout: str,
    ) -> np.ndarray:
        """Generate a dense mosaic of finite round/oval smooth-stage islands.

        The legacy smooth renderer mixes stationary spectral fields with
        capture-zone relief.  At Sq around 1 nm that mixture can join local
        maxima into fibres even when the measured AFM contains individually
        legible grains.  Here each grain is an explicit elliptical radial
        mound.  Repulsive centres make the distribution spatially uniform and
        Gaussian shoulders meet in narrow channels, so uncovered regions do
        not become broad flat dark pools.
        """

        presets = {
            "fine": {
                "count_scale": 1.92,
                "area_scale": 0.66,
                "separation": 0.51,
                "aspect_scale": 0.82,
                "size_sigma": 0.16,
                "height_sigma": 0.28,
                "tip_sigma": 0.46,
            },
            "balanced": {
                "count_scale": 1.72,
                "area_scale": 0.76,
                "separation": 0.53,
                "aspect_scale": 0.86,
                "size_sigma": 0.18,
                "height_sigma": 0.30,
                "tip_sigma": 0.50,
            },
            "dense": {
                "count_scale": 2.12,
                "area_scale": 0.60,
                "separation": 0.47,
                "aspect_scale": 0.80,
                "size_sigma": 0.20,
                "height_sigma": 0.30,
                "tip_sigma": 0.48,
            },
        }
        if layout not in presets:
            raise ValueError(f"Unknown dense-microellipse layout: {layout}")
        settings = presets[layout]
        n = self.resolution
        yy, xx = np.mgrid[:n, :n]
        conditioning_sq_nm = float(target.get("conditioning_sq_nm", 1.0))
        stage = float(
            np.clip((conditioning_sq_nm - 0.70) / (3.60 - 0.70), 0.0, 1.0)
        )
        stage = stage * stage * (3.0 - 2.0 * stage)
        count = int(
            np.clip(
                np.round(
                    float(settings["count_scale"])
                    * (1.0 - 0.24 * stage)
                    * _count(target, 70)
                ),
                46,
                108,
            )
        )
        core_area = max(_area(target, 70), 1.25 * _area(target, 82))
        learned_footprint = float(
            np.clip(
                1.82
                * core_area
                * float(settings["area_scale"])
                * (0.86 + 0.48 * stage),
                48.0,
                190.0,
            )
        )
        capture_area = float(n * n) / float(count)
        median_footprint = float(
            0.25 * learned_footprint + 0.75 * 0.84 * capture_area
        )
        areas = median_footprint * rng.lognormal(
            0.0, float(settings["size_sigma"]), size=count
        )
        radii = np.sqrt(areas / np.pi)
        order = np.argsort(radii)[::-1]
        areas = areas[order]
        radii = radii[order]

        centers: list[tuple[float, float]] = []
        placed_radii: list[float] = []
        separation = float(settings["separation"])
        for radius in radii:
            candidates: list[tuple[float, float, float]] = []
            for _ in range(96):
                cy, cx = rng.uniform(0.0, n, size=2)
                if not centers:
                    candidates.append((np.inf, float(cy), float(cx)))
                    break
                clearances = []
                for (py, px), previous_radius in zip(
                    centers, placed_radii
                ):
                    dy = min(abs(cy - py), n - abs(cy - py))
                    dx = min(abs(cx - px), n - abs(cx - px))
                    clearances.append(
                        float(np.hypot(dx, dy))
                        - separation * (radius + previous_radius)
                    )
                minimum = float(np.min(clearances))
                candidates.append((minimum, float(cy), float(cx)))
                # Keep sampling after the first admissible position.  A
                # high-clearance candidate prevents broad uncovered pools,
                # while sampling among the best few avoids a crystalline
                # farthest-point layout.
            candidates.sort(reverse=True)
            candidate_rank = int(
                np.clip(rng.geometric(0.48) - 1, 0, min(7, len(candidates) - 1))
            )
            _, selected_y, selected_x = candidates[candidate_rank]
            centers.append((selected_y, selected_x))
            placed_radii.append(float(radius))

        # Intermediate growth is represented as another partial deposition
        # layer.  These smaller islands fill and subdivide primary channels;
        # their prevalence fades to almost zero at the smoothest endpoint.
        secondary_count = int(np.round(0.36 * stage * count))
        secondary_centers = [
            tuple(map(float, rng.uniform(0.0, n, size=2)))
            for _ in range(secondary_count)
        ]
        secondary_areas = median_footprint * rng.lognormal(
            np.log(0.34), 0.24, size=secondary_count
        )

        eccentricity = float(
            np.clip(
                target["median_eccentricity_q70"]
                * float(settings["aspect_scale"]),
                0.35,
                0.82,
            )
        )
        aspect = float(
            np.clip(1.0 / np.sqrt(1.0 - eccentricity**2), 1.08, 1.90)
        )
        field = np.full((n, n), 1e-6, dtype=np.float64)
        primitives = [
            (center, float(area), 1.0)
            for center, area in zip(centers, areas)
        ]
        primitives.extend(
            (center, float(area), 0.72)
            for center, area in zip(secondary_centers, secondary_areas)
        )
        for (cy, cx), area, height_scale in primitives:
            local_aspect = float(
                np.clip(aspect * rng.lognormal(0.0, 0.10), 1.03, 2.05)
            )
            a = float(np.sqrt(area * local_aspect / np.pi))
            b = float(np.sqrt(area / (np.pi * local_aspect)))
            angle = float(rng.uniform(0.0, np.pi))
            dx = _periodic_signed_delta(xx, cx, n)
            dy = _periodic_signed_delta(yy, cy, n)
            xr = np.cos(angle) * dx + np.sin(angle) * dy
            yr = -np.sin(angle) * dx + np.cos(angle) * dy
            base_distance_sq = np.square(xr / a) + np.square(yr / b)
            polar_angle = np.arctan2(yr / b, xr / a)
            boundary_amplitude = float(rng.uniform(0.025, 0.075))
            boundary_scale = (
                1.0
                + boundary_amplitude
                * np.cos(3.0 * polar_angle + rng.uniform(0.0, 2.0 * np.pi))
                + 0.55
                * boundary_amplitude
                * np.cos(5.0 * polar_angle + rng.uniform(0.0, 2.0 * np.pi))
            )
            distance_sq = base_distance_sq / np.square(boundary_scale)
            height = float(
                height_scale
                * rng.lognormal(0.0, float(settings["height_sigma"]))
            )
            distance = np.sqrt(distance_sq)
            edge = 1.0 / (
                1.0
                + np.exp(
                    np.clip((distance - 1.0) / 0.12, -30.0, 30.0)
                )
            )
            dome = np.clip(1.0 - distance_sq, 0.0, 1.0) ** 0.62
            # A broad finite top and rounded AFM-tip shoulder produce a
            # cobble-like island rather than a saturated Gaussian point.
            profile = height * (
                edge * (0.52 + 0.48 * dome)
                + 0.10 * np.exp(-0.38 * distance_sq)
            )
            field = np.maximum(field, profile)

        substrate = ndimage.gaussian_filter(
            rng.normal(size=(n, n)), sigma=1.0, mode="wrap"
        )
        substrate -= float(np.mean(substrate))
        substrate /= max(float(np.std(substrate)), 1e-8)
        field += 0.020 * substrate
        # Later smooth-stage deposition raises valleys faster than summits.
        # This monotone response preserves every island and channel ordering
        # while changing a sparse bright-on-black field into the high-coverage
        # orange pebble layer observed in the measured low-Sq AFM.
        low, high = np.quantile(field, (0.002, 0.998))
        span = max(float(high - low), 1e-8)
        normalized = (field - low) / span
        within = np.clip(normalized, 0.0, 1.0)
        response_power = 1.65 - 0.18 * stage
        coalesced = 1.0 - (1.0 - within) ** response_power
        coalesced = np.where(normalized < 0.0, normalized, coalesced)
        coalesced = np.where(
            normalized > 1.0,
            1.0 + (normalized - 1.0) / response_power,
            coalesced,
        )
        field = low + span * coalesced
        return ndimage.gaussian_filter(
            field, sigma=float(settings["tip_sigma"]), mode="wrap"
        )

    def _separated_island_field(
        self,
        target: dict[str, float],
        rng: np.random.Generator,
        *,
        layout: str,
    ) -> np.ndarray:
        """Place finite round/elliptical mounds on a connected deep base.

        Laguerre cells assign a height to every pixel and can therefore turn a
        low capture zone into a broad flat basin.  Rough AFM instead contains
        finite island footprints separated by a connected substrate.  This
        generator models that distinction explicitly: repulsive centers form
        a denuded zone, each island has a closed domed footprint, and overlap
        is limited but not prohibited so that early coalescence remains
        possible.
        """

        presets = {
            "balanced": {
                "count_factor": 1.90,
                "area_factor": 1.05,
                "separation": 0.52,
                "eccentricity_scale": 0.92,
                "size_sigma": 0.24,
                "secondary_fraction": 0.10,
                "secondary_scale": 0.50,
                "secondary_height": 0.58,
                "tip_sigma": 0.55,
            },
            "sparse": {
                "count_factor": 1.55,
                "area_factor": 1.35,
                "separation": 0.65,
                "eccentricity_scale": 0.88,
                "size_sigma": 0.22,
                "secondary_fraction": 0.0,
                "secondary_scale": 0.50,
                "secondary_height": 0.55,
                "tip_sigma": 0.62,
            },
            "round": {
                "count_factor": 1.80,
                "area_factor": 1.15,
                "separation": 0.58,
                "eccentricity_scale": 0.70,
                "size_sigma": 0.20,
                "secondary_fraction": 0.04,
                "secondary_scale": 0.48,
                "secondary_height": 0.55,
                "tip_sigma": 0.65,
            },
            "hierarchical": {
                "count_factor": 1.50,
                "area_factor": 1.15,
                "separation": 0.57,
                "eccentricity_scale": 0.88,
                "size_sigma": 0.28,
                "secondary_fraction": 0.34,
                "secondary_scale": 0.46,
                "secondary_height": 0.62,
                "tip_sigma": 0.52,
            },
            "strict_sparse": {
                "count_factor": 2.05,
                "area_factor": 1.25,
                "separation": 0.75,
                "eccentricity_scale": 0.90,
                "size_sigma": 0.20,
                "secondary_fraction": 0.0,
                "secondary_scale": 0.50,
                "secondary_height": 0.55,
                "tip_sigma": 0.72,
            },
        }
        topology_strength = {
            "strict_sparse_weak": 0.55,
            "strict_sparse_strong": 1.35,
        }.get(layout, 1.0)
        growth_layer_strength = {
            "growth_layered_weak": 0.55,
            "growth_layered": 0.85,
            "growth_layered_strong": 1.15,
            "growth_layered_gapfill_weak": 1.15,
            "growth_layered_gapfill": 1.15,
            "growth_layered_gapfill_strong": 1.15,
        }.get(layout, 0.0)
        growth_layer_scale = {
            "growth_layered_weak": 0.74,
            "growth_layered": 0.85,
            "growth_layered_strong": 0.95,
            "growth_layered_gapfill_weak": 0.95,
            "growth_layered_gapfill": 0.95,
            "growth_layered_gapfill_strong": 0.95,
        }.get(layout, float(presets["strict_sparse"]["secondary_scale"]))
        gap_completion_strength = {
            "growth_layered_gapfill_weak": 0.72,
            "growth_layered_gapfill": 1.00,
            "growth_layered_gapfill_strong": 1.28,
        }.get(layout, 0.0)
        preset_layout = {
            "strict_sparse_weak": "strict_sparse",
            "strict_sparse_strong": "strict_sparse",
            "growth_layered_weak": "strict_sparse",
            "growth_layered": "strict_sparse",
            "growth_layered_strong": "strict_sparse",
            "growth_layered_gapfill_weak": "strict_sparse",
            "growth_layered_gapfill": "strict_sparse",
            "growth_layered_gapfill_strong": "strict_sparse",
        }.get(layout, layout)
        if preset_layout not in presets:
            raise ValueError(f"Unknown separated-island layout: {layout}")
        settings = presets[preset_layout]
        isolation = float(
            np.clip(target.get("rheed_spot_isolation_score", 0.50), 0.0, 1.0)
        )
        # RHEED spot topology controls the same physical degrees of freedom
        # that define a rough AFM surface.  Isolated, round diffraction spots
        # produce fewer, larger and more widely separated islands; bridged or
        # streak-like spots retain denser islands and a shallower substrate.
        count_factor = float(settings["count_factor"]) * (
            1.0 + topology_strength * 0.24 * (0.50 - isolation)
        )
        area_factor = float(settings["area_factor"]) * (
            1.0 + topology_strength * 0.25 * (isolation - 0.50)
        )
        separation_factor = float(settings["separation"]) * (
            1.0 + topology_strength * 0.35 * (isolation - 0.50)
        )
        # A single repulsive island layer describes the early, high-Sq growth
        # stage well, but it stays artificially sparse after more material has
        # arrived.  The predicted Sq is target-blind in outer LOO and provides
        # a physical growth-stage coordinate: as Sq falls from the rough tail,
        # a second population nucleates across gaps and existing islands.  A
        # smooth transition avoids a hard visual regime boundary.  Below the
        # separate renderer's smooth threshold the legacy fine-grain branch is
        # still used, so this term only affects the intermediate regime.
        conditioning_sq_nm = float(target.get("conditioning_sq_nm", np.inf))
        growth_progress = float(
            np.clip((7.6 - conditioning_sq_nm) / (7.6 - 4.5), 0.0, 1.0)
        )
        growth_progress = growth_progress**2 * (3.0 - 2.0 * growth_progress)
        gap_completion_progress = growth_progress * gap_completion_strength
        growth_progress *= growth_layer_strength
        count_factor *= 1.0 + 0.12 * growth_progress
        # Intermediate islands have continued to grow laterally as well as
        # nucleating.  Increasing footprint while retaining the dense count
        # lets neighbouring ovals impinge into the large pebble-like domains
        # seen in real Sq=4--6 nm AFM instead of yielding only fine grains.
        area_factor *= 1.0 + 0.55 * gap_completion_progress
        separation_factor *= 1.0 - 0.28 * growth_progress
        n = self.resolution
        yy, xx = np.mgrid[:n, :n]

        # q55 tracks the visible footprint while q70 is more stable when
        # neighboring mounds begin to touch.  Their maximum avoids the small,
        # under-covered islands produced by the original superellipse mode.
        footprint_area = max(_area(target, 55), 1.55 * _area(target, 70))
        primary_count = int(
            np.clip(
                np.round(count_factor * _count(target, 70)),
                8,
                84,
            )
        )
        secondary_fraction = float(settings["secondary_fraction"])
        if growth_layer_strength > 0.0:
            secondary_fraction += 1.55 * growth_progress
        secondary_count = int(np.round(secondary_fraction * primary_count))
        areas = footprint_area * area_factor * rng.lognormal(
            0.0, settings["size_sigma"], size=primary_count
        )
        if secondary_count:
            secondary_areas = (
                footprint_area
                * area_factor
                * growth_layer_scale**2
                * rng.lognormal(0.0, 0.20, size=secondary_count)
            )
            areas = np.concatenate([areas, secondary_areas])
        secondary_flags = np.arange(len(areas)) >= primary_count
        # Place larger objects first.  This gives the primary islands a stable
        # denuded zone rather than letting small objects crowd them out.
        order = np.argsort(areas)[::-1]
        areas = areas[order]
        secondary_flags = secondary_flags[order]
        radii = np.sqrt(areas / np.pi)

        centers: list[tuple[float, float]] = []
        placed_radii: list[float] = []
        placed_secondary: list[bool] = []
        for radius, is_secondary in zip(radii, secondary_flags):
            best: tuple[float, float] | None = None
            best_clearance = -np.inf
            accepted = False
            for _ in range(320):
                cy, cx = rng.uniform(0.0, n, size=2)
                if not centers:
                    best = (cy, cx)
                    accepted = True
                    break
                clearances = []
                for (py, px), previous_radius, previous_secondary in zip(
                    centers, placed_radii, placed_secondary
                ):
                    dy = min(abs(cy - py), n - abs(cy - py))
                    dx = min(abs(cx - px), n - abs(cx - px))
                    distance = float(np.hypot(dx, dy))
                    separation = separation_factor
                    if is_secondary and previous_secondary:
                        # Second-layer nuclei remain dispersed relative to one
                        # another, but can overlap enough to form a dense coat.
                        separation *= 0.58
                    elif is_secondary or previous_secondary:
                        # Prefer nucleation in first-layer gaps while retaining
                        # enough footprint overlap for physical coalescence.
                        # This turns broad uncovered regions into narrow cracks
                        # instead of wasting most new nuclei on existing peaks.
                        gap_preference = float(
                            np.clip((0.65 - isolation) / 0.30, 0.0, 1.0)
                        )
                        separation *= 0.16 + 0.22 * gap_preference
                    clearances.append(
                        distance - separation * (radius + previous_radius)
                    )
                minimum = float(np.min(clearances))
                if minimum > best_clearance:
                    best_clearance = minimum
                    best = (cy, cx)
                if minimum >= 0.0:
                    accepted = True
                    break
            # At high learned coverages an exact Poisson packing may be
            # impossible.  The max-clearance fallback adds the least-overlap
            # candidate instead of abandoning an island or clustering it.
            assert best is not None
            centers.append(best)
            placed_radii.append(float(radius))
            placed_secondary.append(bool(is_secondary))
            if not accepted:
                continue

        eccentricity = float(
            np.clip(
                target["median_eccentricity_q70"]
                * settings["eccentricity_scale"],
                0.30,
                0.90,
            )
        )
        aspect = float(
            np.clip(1.0 / np.sqrt(1.0 - eccentricity**2), 1.05, 2.35)
        )
        solidity = float(
            np.clip(target["median_solidity_q70"], 0.80, 0.97)
        )
        edge_ratio = float(
            np.clip(target["boundary_gradient_ratio_q70"], 0.92, 1.65)
        )

        substrate_noise = ndimage.gaussian_filter(
            rng.normal(size=(n, n)), sigma=1.10, mode="wrap"
        )
        substrate_noise -= float(np.mean(substrate_noise))
        substrate_noise /= max(float(np.std(substrate_noise)), 1e-8)
        field = -0.92 + 0.035 * substrate_noise
        for (cy, cx), area, is_secondary in zip(
            centers, areas, secondary_flags
        ):
            local_aspect = float(
                np.clip(aspect * rng.lognormal(0.0, 0.12), 1.02, 2.55)
            )
            a = float(
                np.clip(
                    np.sqrt(area * local_aspect / np.pi), 2.4, n / 4.5
                )
            )
            b = float(
                np.clip(
                    np.sqrt(area / (np.pi * local_aspect)), 2.1, n / 5.0
                )
            )
            angle = float(rng.uniform(0.0, np.pi))
            dx = _periodic_signed_delta(xx, cx, n)
            dy = _periodic_signed_delta(yy, cy, n)
            xr = np.cos(angle) * dx + np.sin(angle) * dy
            yr = -np.sin(angle) * dx + np.cos(angle) * dy
            theta = np.arctan2(yr / b, xr / a)
            irregularity = min(0.11, 0.65 * (1.0 - solidity)) * (
                np.cos(3.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
                + 0.42
                * np.cos(5.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
            )
            exponent = float(rng.uniform(1.85, 2.45))
            distance = (
                np.abs(xr / a) ** exponent
                + np.abs(yr / b) ** exponent
            ) ** (1.0 / exponent)
            distance /= np.clip(1.0 + irregularity, 0.84, 1.16)
            edge_width = float(np.clip(0.13 / edge_ratio, 0.065, 0.14))
            edge = 1.0 / (
                1.0
                + np.exp(
                    np.clip((distance - 1.0) / edge_width, -30.0, 30.0)
                )
            )
            dome_power = float(rng.uniform(0.48, 0.76))
            dome = np.clip(1.0 - distance**2, 0.0, 1.0) ** dome_power
            height = float(
                rng.lognormal(
                    0.48
                    + topology_strength * 0.28 * (isolation - 0.50),
                    0.20,
                )
            )
            if is_secondary:
                height *= float(settings["secondary_height"])
            # A small edge shoulder keeps AFM-tip-rounded islands finite while
            # retaining a clear oval/elliptical dome rather than a mesa.
            profile = height * edge * (0.24 + 0.76 * dome)
            if is_secondary and growth_layer_strength > 0.0:
                # Add rather than max-compose the overlayer.  A nucleus in a
                # gap partially fills the substrate; one landing on an older
                # island creates the stacked/coalesced relief seen at Sq~5 nm.
                field += profile
            else:
                primitive = -0.92 + profile
                field = np.maximum(field, primitive)

        # Continued growth should not leave the broad, connected substrate
        # lakes produced by random second-layer placement.  In intermediate
        # AFM the remaining base is observed as narrow channels between a
        # dense layer of islands.  For the gap-completion layouts, repeatedly
        # locate the centre of the widest low-height region and nucleate a
        # finite oval there.  This is a growth construction, not an image
        # filter: every added feature retains a closed domed footprint and a
        # randomly sampled orientation/height.  Periodic tiling makes the
        # largest-gap search consistent with the generator boundaries.
        if gap_completion_progress > 0.0:
            maximum_gap_radius = float(
                np.clip(
                    5.35 - 1.75 * gap_completion_progress,
                    2.65,
                    5.10,
                )
            )
            completion_budget = int(
                np.clip(
                    np.ceil(
                        primary_count
                        * (0.42 + 0.58 * gap_completion_progress)
                    ),
                    8,
                    72,
                )
            )
            for _ in range(completion_budget):
                display_low, display_high = np.quantile(
                    field, (0.005, 0.995)
                )
                low_mask = field < (
                    display_low + 0.30 * (display_high - display_low)
                )
                tiled_distance = ndimage.distance_transform_edt(
                    np.tile(low_mask, (3, 3))
                )
                gap_distance = tiled_distance[n : 2 * n, n : 2 * n]
                flat_index = int(np.argmax(gap_distance))
                gap_radius = float(gap_distance.flat[flat_index])
                if gap_radius <= maximum_gap_radius:
                    break
                cy, cx = np.unravel_index(flat_index, gap_distance.shape)
                cy = float(cy) + float(rng.uniform(-0.35, 0.35))
                cx = float(cx) + float(rng.uniform(-0.35, 0.35))
                local_aspect = float(rng.uniform(1.08, 1.48))
                a = float(
                    np.clip(
                        1.32 * gap_radius * np.sqrt(local_aspect),
                        3.0,
                        10.5,
                    )
                )
                b = float(
                    np.clip(
                        1.32 * gap_radius / np.sqrt(local_aspect),
                        2.7,
                        9.0,
                    )
                )
                angle = float(rng.uniform(0.0, np.pi))
                dx = _periodic_signed_delta(xx, cx, n)
                dy = _periodic_signed_delta(yy, cy, n)
                xr = np.cos(angle) * dx + np.sin(angle) * dy
                yr = -np.sin(angle) * dx + np.cos(angle) * dy
                distance = np.hypot(xr / a, yr / b)
                edge = 1.0 / (
                    1.0
                    + np.exp(
                        np.clip((distance - 1.0) / 0.10, -30.0, 30.0)
                    )
                )
                dome = np.clip(1.0 - distance**2, 0.0, 1.0) ** float(
                    rng.uniform(0.52, 0.72)
                )
                height = float(rng.lognormal(0.20, 0.15))
                field += height * edge * (0.22 + 0.78 * dome)

            # Once islands occupy most capture zones, incoming material raises
            # low terraces faster than the already highest summits.  Apply a
            # monotone coalescence response between robust height limits: it
            # retains the rank ordering and the narrow deepest channels, but
            # changes the intermediate distribution from isolated positive
            # peaks to a predominantly high island layer with sparse valleys.
            # This directly models the negative height skew observed in the
            # measured Sq=4--6 nm maps.
            response_low, response_high = np.quantile(
                field, (0.005, 0.995)
            )
            response_span = max(float(response_high - response_low), 1e-8)
            normalized = (field - response_low) / response_span
            response_power = 1.0 + 0.80 * gap_completion_progress
            within = np.clip(normalized, 0.0, 1.0)
            coalesced = 1.0 - (1.0 - within) ** response_power
            coalesced = np.where(normalized < 0.0, normalized, coalesced)
            coalesced = np.where(
                normalized > 1.0,
                1.0 + (normalized - 1.0) / response_power,
                coalesced,
            )
            field = response_low + response_span * coalesced
        return ndimage.gaussian_filter(
            field, sigma=float(settings["tip_sigma"]), mode="wrap"
        )

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
        elif mode == "separated_ellipse":
            field = self._separated_island_field(
                target, rng, layout="balanced"
            )
        elif mode == "separated_ellipse_sparse":
            field = self._separated_island_field(
                target, rng, layout="sparse"
            )
        elif mode == "separated_ellipse_round":
            field = self._separated_island_field(
                target, rng, layout="round"
            )
        elif mode == "separated_ellipse_hierarchical":
            field = self._separated_island_field(
                target, rng, layout="hierarchical"
            )
        elif mode == "separated_ellipse_strict_sparse":
            field = self._separated_island_field(
                target, rng, layout="strict_sparse"
            )
        elif mode == "separated_ellipse_strict_sparse_weak":
            field = self._separated_island_field(
                target, rng, layout="strict_sparse_weak"
            )
        elif mode == "separated_ellipse_strict_sparse_strong":
            field = self._separated_island_field(
                target, rng, layout="strict_sparse_strong"
            )
        elif mode == "separated_ellipse_growth_layered_weak":
            field = self._separated_island_field(
                target, rng, layout="growth_layered_weak"
            )
        elif mode == "separated_ellipse_growth_layered":
            field = self._separated_island_field(
                target, rng, layout="growth_layered"
            )
        elif mode == "separated_ellipse_growth_layered_strong":
            field = self._separated_island_field(
                target, rng, layout="growth_layered_strong"
            )
        elif mode == "separated_ellipse_growth_layered_gapfill_weak":
            field = self._separated_island_field(
                target, rng, layout="growth_layered_gapfill_weak"
            )
        elif mode == "separated_ellipse_growth_layered_gapfill":
            field = self._separated_island_field(
                target, rng, layout="growth_layered_gapfill"
            )
        elif mode == "separated_ellipse_growth_layered_gapfill_strong":
            field = self._separated_island_field(
                target, rng, layout="growth_layered_gapfill_strong"
            )
        elif mode == "dense_microellipse_fine":
            field = self._dense_microellipse_field(
                target, rng, layout="fine"
            )
        elif mode == "dense_microellipse_balanced":
            field = self._dense_microellipse_field(
                target, rng, layout="balanced"
            )
        elif mode == "dense_microellipse_dense":
            field = self._dense_microellipse_field(
                target, rng, layout="dense"
            )
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
