from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_nanoletters_m17_figures.py"


def _load_figure_module():
    spec = importlib.util.spec_from_file_location("make_nanoletters_m17_figures", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_mapping_is_complete_unique_and_two_digit() -> None:
    module = _load_figure_module()
    public = [public_id for public_id, _ in module.ANONYMIZED_IDS]
    internal = [internal_id for _, internal_id in module.ANONYMIZED_IDS]
    assert public == [f"{index:02d}" for index in range(1, 28)]
    assert len(internal) == len(set(internal)) == 27
    assert module.PUBLIC_BY_INTERNAL["N6342"] == "23"


def test_displayed_sources_are_checksum_locked_and_inference_safe() -> None:
    module = _load_figure_module()
    module._verify_sources()
    assert {sample.public_id for sample in module.SAMPLES} == {"04", "20", "23"}
    for sample in module.SAMPLES:
        frames, indices = module._load_clip(sample)
        generated, ensemble, predicted_sq, _ = module._load_generated(sample)
        measured = module._load_measured(sample)
        assert frames.shape == (16, 224, 224)
        assert indices.shape == (16,)
        assert generated.shape == (128, 128)
        assert ensemble.shape == (4, 128, 128)
        assert measured.shape == (256, 256)
        assert np.isclose(module._sq(generated), predicted_sq, rtol=2e-5)


def test_growth_level_outer_folds_have_no_held_growth_overlap() -> None:
    module = _load_figure_module()
    audit = pd.read_csv(module.M17_REPORT / "fold_integrity_audit.csv")
    assert len(audit) == 27
    assert set(audit["held_growth_run_id"].astype(str)) == set(module.PUBLIC_BY_INTERNAL)
    assert audit["fit_growth_count"].eq(26).all()
    assert audit["condition_inner_growth_count"].eq(26).all()
    assert audit["spectral_train_growth_count"].eq(26).all()
    assert audit["island_train_growth_count"].eq(26).all()
    assert not audit["held_overlap_with_fit"].astype(bool).any()
    assert "6081" not in set(audit["held_growth_run_id"].astype(str))
    for row in audit.itertuples(index=False):
        assert str(row.held_growth_run_id) not in str(row.fit_growth_run_ids)
        assert str(row.held_growth_run_id) not in str(row.spectral_train_growth_run_ids)
        assert str(row.held_growth_run_id) not in str(row.island_train_growth_run_ids)


def test_selected_table_matches_frozen_metrics() -> None:
    module = _load_figure_module()
    selected = pd.read_csv(module.REPORT_ROOT / "selected_case_metrics.csv")
    frozen = module._selected_metrics().set_index("growth_run_id")
    assert set(selected["public_sample_id"].astype(str).str.zfill(2)) == {"04", "20", "23"}
    for row in selected.itertuples(index=False):
        source = frozen.loc[str(row.internal_growth_run_id)]
        assert np.isclose(row.measured_sq_nm, source["sq_nm"])
        assert np.isclose(row.predicted_sq_nm, source["generated_rq_nm"])
        assert np.isclose(row.joint_confidence_index, source["joint_confidence_index"])


def test_outputs_meet_size_resolution_and_public_label_requirements() -> None:
    module = _load_figure_module()
    for stem in (
        "Figure_1_AutoRHEED_overview",
        "Figure_2_model_and_validation",
        "Figure_3_selected_results",
    ):
        for suffix in (".pdf", ".png", ".tiff"):
            path = module.FIGURE_ROOT / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 10_000
        for suffix in (".png", ".tiff"):
            with Image.open(module.FIGURE_ROOT / f"{stem}{suffix}") as image:
                assert image.width == 4200
                assert image.height >= 2670
                dpi = image.info.get("dpi")
                assert dpi is not None
                assert np.isclose(dpi[0], 600, atol=0.01)
                assert np.isclose(dpi[1], 600, atol=0.01)

    captions = (module.REPORT_ROOT / "captions.md").read_text(encoding="utf-8")
    for internal_id in module.PUBLIC_BY_INTERNAL:
        assert internal_id not in captions
    assert "Sample 23" in captions
    assert "Sample 04" in captions
    assert "Sample 20" in captions
