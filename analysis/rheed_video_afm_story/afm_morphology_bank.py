from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .afm_dataset import load_unit_shapes
from .common import display_path, repo_path, save_parquet, write_csv
from .rq_disentanglement import project_unit_rq_np


def build_morphology_bank(manifest: pd.DataFrame, assignments: pd.DataFrame, latents: np.ndarray, recons: np.ndarray, config: dict[str, Any], resolution: int) -> pd.DataFrame:
    output_root = repo_path(config["output_root"])
    arrays_root = output_root / "morphology_bank_arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    shapes = load_unit_shapes(manifest, resolution)
    assign = assignments.set_index("sample_id")
    rows = []
    for sid, g in manifest.groupby("sample_id"):
        reps = g[g["representative_for_sample"].astype(bool)]
        row = reps.iloc[0] if len(reps) else g.iloc[0]
        idx = int(row.name)
        true_shape = project_unit_rq_np(shapes[idx])
        decoded = project_unit_rq_np(recons[idx])
        residual = true_shape - decoded
        base = f"{row['sample_id']}__{row['afm_file_id']}"
        physical_path = arrays_root / f"{base}_physical_centered.npy"
        unit_path = arrays_root / f"{base}_unit_shape.npy"
        latent_path = arrays_root / f"{base}_latent.npy"
        decoded_path = arrays_root / f"{base}_decoded_unit_shape.npy"
        residual_path = arrays_root / f"{base}_high_frequency_residual.npy"
        centered_source_paths = json.loads(row["centered_map_paths"])
        centered = np.load(repo_path(centered_source_paths[str(resolution)])).astype(np.float32)
        np.save(physical_path, centered)
        np.save(unit_path, true_shape.astype(np.float32))
        np.save(latent_path, latents[idx].astype(np.float32))
        np.save(decoded_path, decoded.astype(np.float32))
        np.save(residual_path, residual.astype(np.float32))
        residual_fft = np.abs(np.fft.fft2(residual)) ** 2
        flat = np.sort(residual_fft.ravel())
        bank_row = {
            "sample_id": row["sample_id"],
            "growth_run_id": row["growth_run_id"],
            "source_afm": row["source_afm_file"],
            "physical_map_path": display_path(physical_path),
            "unit_shape_map_path": display_path(unit_path),
            "latent_path": display_path(latent_path),
            "rq_nm": row["rq_nm"],
            "ra_nm": row["ra_nm"],
            "psd_low_fraction": row["unit_psd_low_fraction"],
            "psd_mid_fraction": row["unit_psd_mid_fraction"],
            "psd_high_fraction": row["unit_psd_high_fraction"],
            "correlation_length_nm": row["unit_autocorr_length_nm"],
            "anisotropy": row["unit_anisotropy_ratio"],
            "height_skewness": row["height_skewness"],
            "height_kurtosis": row["height_kurtosis"],
            "prototype_id": assign.loc[str(sid), "dominant_prototype"],
            "prototype_purity": float(assign.loc[str(sid), "prototype_purity"]),
            "mixed_morphology": bool(assign.loc[str(sid), "mixed_morphology"]),
            "decoder_reconstruction_path": display_path(decoded_path),
            "high_frequency_residual_path": display_path(residual_path),
            "residual_rms_amplitude": float(np.sqrt(np.mean(residual**2))),
            "residual_psd_total": float(residual_fft.sum()),
            "residual_low_band_energy": float(flat[: len(flat) // 3].sum()),
            "residual_mid_band_energy": float(flat[len(flat) // 3 : 2 * len(flat) // 3].sum()),
            "residual_high_band_energy": float(flat[2 * len(flat) // 3 :].sum()),
        }
        rows.append(bank_row)
    bank = pd.DataFrame(rows).sort_values("sample_id")
    write_csv(bank, output_root / "afm_morphology_bank.csv")
    save_parquet(bank, output_root / "afm_morphology_bank.parquet")
    return bank
