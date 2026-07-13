"""Stage 1C semantic synthetic renderer and independent oracle."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage, stats
from sklearn.metrics import roc_auc_score

from analysis.rheed_peak_saddle.pair_features import pair_masks
from analysis.rheed_peak_saddle.spot_detection import SpotEstimate
from analysis.rheed_peak_saddle.synthetic import SyntheticPairTruth, SyntheticRheed, SyntheticSpotTruth


TARGET_VISUAL_ADHESION_GRID = tuple(round(i * 0.05, 2) for i in range(20))
DEVELOPMENT_V3_SEEDS = tuple(2026081100 + i for i in range(20))
HOLDOUT_V3_SEEDS = tuple(2026083100 + i for i in range(20))


@dataclass(frozen=True)
class SemanticTemplate:
    template_id: str
    seed: int
    split: str
    image_shape: tuple[int, int]
    row_count: int
    spots_per_row: int
    spacing: float
    width_scale: float
    profile_family: str
    amplitude_ratio: float
    psf_blur_sigma: float
    row_tilt_degrees: float
    row_curvature: float
    bridge_width: float
    nuisance: dict[str, float | str | int]


@dataclass(frozen=True)
class SemanticRender:
    image_id: str
    split: str
    template: SemanticTemplate
    target_visual_adhesion: float
    nominal_bridge_control: float
    spot_signal_clean: np.ndarray
    explicit_bridge_signal_clean: np.ndarray
    morphology_signal_clean: np.ndarray
    smooth_background: np.ndarray
    direct_beam_or_halo: np.ndarray
    noisy_observed_linear: np.ndarray
    displayed_image: np.ndarray
    valid_mask: np.ndarray
    spots: tuple[SyntheticSpotTruth, ...]
    pairs: tuple[SyntheticPairTruth, ...]
    oracle_rows: tuple[dict[str, Any], ...]
    solver_row: dict[str, Any]


def _profile(xx: np.ndarray, yy: np.ndarray, spot: SyntheticSpotTruth, angle_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    dx = xx - spot.center_x
    dy = yy - spot.center_y
    along = math.cos(theta) * dx + math.sin(theta) * dy
    across = -math.sin(theta) * dx + math.cos(theta) * dy
    r2 = (along / max(spot.sigma_x, 1e-6)) ** 2 + (across / max(spot.sigma_y, 1e-6)) ** 2
    if spot.profile_family == "moffat":
        beta = 2.4
        return spot.amplitude * np.power(1.0 + r2 / beta, -beta)
    return spot.amplitude * np.exp(-0.5 * r2)


def _line_geometry(shape: tuple[int, int], left: SyntheticSpotTruth, right: SyntheticSpotTruth, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=float)
    dx = right.center_x - left.center_x
    dy = right.center_y - left.center_y
    length = math.hypot(dx, dy)
    ux = dx / max(length, 1e-6)
    uy = dy / max(length, 1e-6)
    px = -uy
    py = ux
    along = (xx - left.center_x) * ux + (yy - left.center_y) * uy
    across = (xx - left.center_x) * px + (yy - left.center_y) * py
    segment = (along >= 0.0) & (along <= length)
    return segment & (np.abs(across) <= width), segment, across


def make_semantic_templates(split: str, *, count: int = 8) -> tuple[SemanticTemplate, ...]:
    if split not in {"development_v3", "holdout_v3"}:
        raise ValueError("split must be development_v3 or holdout_v3")
    seeds = DEVELOPMENT_V3_SEEDS if split == "development_v3" else HOLDOUT_V3_SEEDS
    templates: list[SemanticTemplate] = []
    for index, seed in enumerate(seeds[:count]):
        holdout = split == "holdout_v3"
        templates.append(
            SemanticTemplate(
                template_id=f"{split}_template_{index:02d}",
                seed=seed,
                split=split,
                image_shape=(160, 240),
                row_count=2 + int(index % 4 == 3),
                spots_per_row=6 + int(index % 3 == 1),
                spacing=29.0 + 1.7 * (index % 4) + (0.9 if holdout else 0.0),
                width_scale=0.92 + 0.08 * (index % 5) + (0.03 if holdout else 0.0),
                profile_family="moffat" if (index + int(holdout)) % 3 == 0 else "gaussian",
                amplitude_ratio=0.72 + 0.07 * (index % 4),
                psf_blur_sigma=0.0 if index % 4 == 0 else 0.10 * (index % 4),
                row_tilt_degrees=(-1.4, -0.3, 0.8, 1.6)[index % 4] + (0.25 if holdout else 0.0),
                row_curvature=0.012 * (index % 3) + (0.006 if holdout else 0.0),
                bridge_width=2.0 + 0.25 * (index % 3),
                nuisance={
                    "exposure": (0.82, 1.0, 1.22, 1.35)[index % 4],
                    "additive_offset": (0.0, 0.02, 0.045, 0.06)[index % 4],
                    "display_gamma": (0.97, 1.0, 1.03, 1.05)[index % 4],
                    "smooth_background_strength": 0.025 + 0.01 * (index % 3),
                    "halo_strength": 0.018 + 0.008 * (index % 4),
                    "read_noise_sigma": 0.0008 + 0.0005 * (index % 3),
                    "poisson_noise_scale": 0.0008 + 0.0004 * (index % 2),
                    "rotation_degrees": (-0.8, 0.0, 0.7, 1.2)[index % 4],
                    "translation_x": (-0.8, 0.4, 1.1, -0.3)[index % 4],
                    "translation_y": (0.4, -0.7, 0.9, -0.2)[index % 4],
                },
            )
        )
    return tuple(templates)


def template_spots(template: SemanticTemplate) -> tuple[SyntheticSpotTruth, ...]:
    rng = np.random.default_rng(template.seed)
    height, width = template.image_shape
    row_gap = 34.0
    base_y = height * 0.50 - row_gap * (template.row_count - 1) / 2.0
    start_x = width * 0.50 - template.spacing * (template.spots_per_row - 1) / 2.0
    spots: list[SyntheticSpotTruth] = []
    for row_id in range(template.row_count):
        for site in range(template.spots_per_row):
            centered = site - (template.spots_per_row - 1) / 2.0
            cx = start_x + site * template.spacing + rng.normal(0.0, 0.45)
            cy = base_y + row_gap * row_id + math.tan(math.radians(template.row_tilt_degrees)) * centered * template.spacing + template.row_curvature * centered**2
            sx = template.width_scale * rng.uniform(4.5, 6.0)
            sy = template.width_scale * rng.uniform(2.7, 3.9)
            amp = rng.uniform(0.92, 1.10) * (template.amplitude_ratio if site % 2 else 1.0)
            spots.append(
                SyntheticSpotTruth(
                    spot_id=len(spots),
                    row_id=row_id,
                    site_index=site,
                    center_x=float(cx),
                    center_y=float(cy),
                    sigma_x=float(sx),
                    sigma_y=float(sy),
                    amplitude=float(amp),
                    profile_family=template.profile_family,
                    missing=0,
                    edge_or_crop_flag=0,
                    saturation_flag=0,
                )
            )
    return tuple(spots)


def template_pairs(template: SemanticTemplate, spots: Sequence[SyntheticSpotTruth], nominal: float) -> tuple[SyntheticPairTruth, ...]:
    pairs: list[SyntheticPairTruth] = []
    by_row: dict[int, list[SyntheticSpotTruth]] = {}
    for spot in spots:
        by_row.setdefault(spot.row_id, []).append(spot)
    for row_id, row_spots in by_row.items():
        row_spots = sorted(row_spots, key=lambda spot: spot.site_index)
        for left, right in zip(row_spots[:-1], row_spots[1:]):
            pairs.append(
                SyntheticPairTruth(
                    pair_id=f"{template.template_id}_r{row_id}_{left.spot_id}_{right.spot_id}",
                    spot_i=left.spot_id,
                    spot_j=right.spot_id,
                    row_id=row_id,
                    site_i=left.site_index,
                    site_j=right.site_index,
                    true_bridge_strength=float(nominal),
                    bridge_width=template.bridge_width,
                    valid_expected=1,
                )
            )
    return tuple(pairs)


def render_semantic_template(template: SemanticTemplate, *, target: float, nominal_bridge_control: float, image_id: str) -> SemanticRender:
    target_seed = 0 if not math.isfinite(float(target)) else int(round(float(target) * 1000))
    rng = np.random.default_rng(template.seed + target_seed + int(round(nominal_bridge_control * 10000)))
    shape = template.image_shape
    yy, xx = np.indices(shape, dtype=float)
    spots = template_spots(template)
    spot_signal = np.zeros(shape, dtype=float)
    for spot in spots:
        spot_signal += _profile(xx, yy, spot, template.row_tilt_degrees)
    pairs = template_pairs(template, spots, nominal_bridge_control)
    bridge = np.zeros(shape, dtype=float)
    for pair in pairs:
        left = spots[pair.spot_i]
        right = spots[pair.spot_j]
        mask, _, across = _line_geometry(shape, left, right, template.bridge_width)
        core_r = 1.35 * max(left.sigma_y, right.sigma_y)
        core = ((xx - left.center_x) ** 2 + (yy - left.center_y) ** 2 <= core_r**2) | ((xx - right.center_x) ** 2 + (yy - right.center_y) ** 2 <= core_r**2)
        bridge = np.maximum(bridge, nominal_bridge_control * min(left.amplitude, right.amplitude) * np.exp(-0.5 * (across / max(template.bridge_width, 1e-6)) ** 2) * mask * ~core)
    morphology = spot_signal + bridge
    if template.psf_blur_sigma > 0:
        morphology = ndimage.gaussian_filter(morphology, template.psf_blur_sigma)
    height, width = shape
    bg_strength = float(template.nuisance["smooth_background_strength"])
    halo_strength = float(template.nuisance["halo_strength"])
    smooth_background = bg_strength * (xx / max(width - 1, 1))
    halo = halo_strength * np.exp(-0.5 * (((xx - width * 0.50) / (width * 0.36)) ** 2 + ((yy - height * 0.52) / (height * 0.45)) ** 2))
    exposure = float(template.nuisance["exposure"])
    observed = exposure * (morphology + smooth_background + halo) + float(template.nuisance["additive_offset"])
    observed += rng.normal(0.0, float(template.nuisance["poisson_noise_scale"]) * np.sqrt(np.maximum(observed, 0.0) + 1e-6), size=shape)
    observed += rng.normal(0.0, float(template.nuisance["read_noise_sigma"]), size=shape)
    observed = np.clip(observed, 0.0, None)
    display = observed - float(np.min(observed))
    display /= max(float(np.max(display)), 1e-8)
    display = np.power(display, float(template.nuisance["display_gamma"]))
    valid = np.ones(shape, dtype=bool)
    oracle_rows = tuple(oracle_rows_for_render(morphology, spots, pairs, valid))
    solver_row = {
        "image_id": image_id,
        "split": template.split,
        "template_id": template.template_id,
        "target_visual_adhesion": target,
        "solved_nominal_bridge_amplitude": nominal_bridge_control,
        "achieved_oracle_visual_adhesion": float(np.median([row["oracle_visual_adhesion_clean"] for row in oracle_rows])) if oracle_rows else float("nan"),
        "solver_iterations": 0,
        "solver_status": "direct_render",
        "attainable_min": "",
        "attainable_max": "",
    }
    return SemanticRender(
        image_id=image_id,
        split=template.split,
        template=template,
        target_visual_adhesion=target,
        nominal_bridge_control=nominal_bridge_control,
        spot_signal_clean=spot_signal.astype(np.float32),
        explicit_bridge_signal_clean=bridge.astype(np.float32),
        morphology_signal_clean=morphology.astype(np.float32),
        smooth_background=smooth_background.astype(np.float32),
        direct_beam_or_halo=halo.astype(np.float32),
        noisy_observed_linear=observed.astype(np.float32),
        displayed_image=display.astype(np.float32),
        valid_mask=valid,
        spots=spots,
        pairs=pairs,
        oracle_rows=oracle_rows,
        solver_row=solver_row,
    )


def independent_maximin_saddle(intensity: np.ndarray, seed_a: np.ndarray, seed_b: np.ndarray, corridor: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    values = np.asarray(intensity, dtype=float)
    domain = np.asarray(corridor, dtype=bool) & np.isfinite(values)
    starts = list(zip(*np.nonzero(seed_a & domain), strict=False))
    targets = set(zip(*np.nonzero(seed_b & domain), strict=False))
    if not starts or not targets:
        return float("nan"), []
    best = np.full(values.shape, -np.inf, dtype=float)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, int, int]] = []
    for y, x in starts:
        cap = float(values[y, x])
        best[y, x] = cap
        heapq.heappush(heap, (-cap, int(y), int(x)))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        neg_cap, y, x = heapq.heappop(heap)
        cap = -neg_cap
        if cap < best[y, x] - 1e-12:
            continue
        if (y, x) in targets:
            path = [(y, x)]
            cur = (y, x)
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return cap, path
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= values.shape[0] or nx >= values.shape[1] or not domain[ny, nx]:
                continue
            next_cap = min(cap, float(values[ny, nx]))
            if next_cap > best[ny, nx]:
                best[ny, nx] = next_cap
                parent[(ny, nx)] = (y, x)
                heapq.heappush(heap, (-next_cap, ny, nx))
    return float("nan"), []


def spot_estimate_from_truth_v3(spot: SyntheticSpotTruth, index: int) -> SpotEstimate:
    return SpotEstimate(
        spot_id=index,
        center_x=spot.center_x,
        center_y=spot.center_y,
        peak_intensity=spot.amplitude,
        sigma_x=spot.sigma_x,
        sigma_y=spot.sigma_y,
        equivalent_width=math.sqrt(max(spot.sigma_x * spot.sigma_y, 1e-6)),
        eccentricity=1.0 - min(spot.sigma_x, spot.sigma_y) / max(spot.sigma_x, spot.sigma_y, 1e-6),
        local_background=0.0,
        fit_residual=0.0,
        saturation_flag=spot.saturation_flag,
        edge_or_crop_flag=spot.edge_or_crop_flag,
        detection_confidence=1.0,
    )


def oracle_for_pair(clean: np.ndarray, spots: Sequence[SyntheticSpotTruth], pair: SyntheticPairTruth, valid_mask: np.ndarray) -> dict[str, Any]:
    estimates = [spot_estimate_from_truth_v3(spot, i) for i, spot in enumerate(spots)]
    masks = pair_masks(clean.shape, estimates[pair.spot_i], estimates[pair.spot_j])
    corridor = masks.corridor_mask & valid_mask
    peak_i = float(np.percentile(clean[masks.seed_i_mask], 98.0))
    peak_j = float(np.percentile(clean[masks.seed_j_mask], 98.0))
    saddle, path = independent_maximin_saddle(clean, masks.seed_i_mask, masks.seed_j_mask, corridor)
    background = 0.0
    denom = min(peak_i, peak_j) - background
    adhesion = float(np.clip((saddle - background) / (denom + 1e-6), 0.0, 1.0)) if math.isfinite(saddle) and denom > 0 else float("nan")
    return {
        "pair_id": pair.pair_id,
        "row_id": pair.row_id,
        "site_i": pair.site_i,
        "site_j": pair.site_j,
        "nominal_bridge_control": pair.true_bridge_strength,
        "oracle_peak_i": peak_i,
        "oracle_peak_j": peak_j,
        "oracle_saddle": saddle,
        "oracle_background": background,
        "oracle_visual_adhesion_clean": adhesion,
        "oracle_path_length": len(path),
        "spacing": math.hypot(spots[pair.spot_j].center_x - spots[pair.spot_i].center_x, spots[pair.spot_j].center_y - spots[pair.spot_i].center_y),
        "spot_width_i": math.sqrt(spots[pair.spot_i].sigma_x * spots[pair.spot_i].sigma_y),
        "spot_width_j": math.sqrt(spots[pair.spot_j].sigma_x * spots[pair.spot_j].sigma_y),
        "peak_amplitude_ratio": min(spots[pair.spot_i].amplitude, spots[pair.spot_j].amplitude) / max(spots[pair.spot_i].amplitude, spots[pair.spot_j].amplitude),
        "profile_family": spots[pair.spot_i].profile_family,
    }


def oracle_rows_for_render(clean: np.ndarray, spots: Sequence[SyntheticSpotTruth], pairs: Sequence[SyntheticPairTruth], valid_mask: np.ndarray) -> list[dict[str, Any]]:
    return [oracle_for_pair(clean, spots, pair, valid_mask) for pair in pairs if pair.valid_expected]


def median_oracle_for_nominal(template: SemanticTemplate, nominal: float) -> float:
    render = render_semantic_template(template, target=float("nan"), nominal_bridge_control=nominal, image_id=f"{template.template_id}_probe")
    return float(np.median([row["oracle_visual_adhesion_clean"] for row in render.oracle_rows]))


def solve_nominal_for_target(template: SemanticTemplate, target: float, *, tol: float = 0.01, max_iter: int = 24) -> dict[str, Any]:
    lo, hi = 0.0, 1.6
    amin = median_oracle_for_nominal(template, lo)
    amax = median_oracle_for_nominal(template, hi)
    if target < amin - tol or target > amax + tol:
        return {
            "target_visual_adhesion": target,
            "solved_nominal_bridge_amplitude": float("nan"),
            "achieved_oracle_visual_adhesion": float("nan"),
            "solver_iterations": 0,
            "solver_status": "unattainable",
            "attainable_min": amin,
            "attainable_max": amax,
        }
    mid = 0.0
    achieved = amin
    for iteration in range(1, max_iter + 1):
        mid = (lo + hi) / 2.0
        achieved = median_oracle_for_nominal(template, mid)
        if abs(achieved - target) <= tol:
            break
        if achieved < target:
            lo = mid
        else:
            hi = mid
    return {
        "target_visual_adhesion": target,
        "solved_nominal_bridge_amplitude": mid,
        "achieved_oracle_visual_adhesion": achieved,
        "solver_iterations": iteration,
        "solver_status": "converged" if abs(achieved - target) <= tol else "max_iter",
        "attainable_min": amin,
        "attainable_max": amax,
    }


def calibrated_renders(split: str, *, template_count: int = 8) -> tuple[SemanticRender, ...]:
    renders: list[SemanticRender] = []
    for template in make_semantic_templates(split, count=template_count):
        for target in TARGET_VISUAL_ADHESION_GRID:
            solved = solve_nominal_for_target(template, target)
            if solved["solver_status"] == "unattainable":
                continue
            render = render_semantic_template(
                template,
                target=target,
                nominal_bridge_control=float(solved["solved_nominal_bridge_amplitude"]),
                image_id=f"{template.template_id}_target_{int(round(target * 100)):02d}",
            )
            solver_row = dict(render.solver_row)
            solver_row.update(solved)
            renders.append(
                SemanticRender(
                    **{**render.__dict__, "solver_row": solver_row}
                )
            )
    return tuple(renders)


def finite(values: Sequence[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def spearman(x: Sequence[Any], y: Sequence[Any]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return float("nan")
    return float(stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic)


def metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    target = finite([row.get("target_visual_adhesion") for row in rows])
    oracle = finite([row.get("oracle_visual_adhesion_clean") for row in rows])
    estimated = finite([row.get("estimated_adhesion_observed") for row in rows])
    errors = [abs(float(row["estimated_adhesion_observed"]) - float(row["oracle_visual_adhesion_clean"])) for row in rows if math.isfinite(float(row.get("estimated_adhesion_observed", float("nan")))) and math.isfinite(float(row.get("oracle_visual_adhesion_clean", float("nan"))))]
    labels = []
    scores = []
    for row in rows:
        oracle_value = float(row.get("oracle_visual_adhesion_clean", float("nan")))
        estimate = float(row.get("estimated_adhesion_observed", float("nan")))
        if not math.isfinite(oracle_value) or not math.isfinite(estimate):
            continue
        if oracle_value >= 0.50:
            labels.append(1)
            scores.append(estimate)
        elif oracle_value <= 0.10:
            labels.append(0)
            scores.append(estimate)
    return {
        "target_vs_oracle_spearman": spearman(target, oracle),
        "estimated_vs_oracle_spearman": spearman([row.get("oracle_visual_adhesion_clean") for row in rows], [row.get("estimated_adhesion_observed") for row in rows]),
        "estimated_vs_target_spearman": spearman([row.get("target_visual_adhesion") for row in rows], [row.get("estimated_adhesion_observed") for row in rows]),
        "mae": float(np.mean(errors)) if errors else float("nan"),
        "median_absolute_error": float(np.median(errors)) if errors else float("nan"),
        "p90_absolute_error": float(np.percentile(errors, 90)) if errors else float("nan"),
        "connected_vs_isolated_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan"),
        "false_connected_rate": sum(score >= 0.50 for label, score in zip(labels, scores) if label == 0) / max(sum(label == 0 for label in labels), 1),
    }
