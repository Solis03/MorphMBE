# Full-28 line-3 metrology and generative AFM rerun

## Cohort and data organization

- Final cohort: 28 independent growth groups (original 23 + N6342, N6358, N6382, N6389, N6390).
- Explicit exclusions: 6043, 6055, and N6324. N6324 is present only in the raw-source exclusion audit and never enters AFM targets, RHEED embeddings, folds, fitting, prediction, or generation.
- The five accepted extra AFM samples are decoded again from `data/AFM-extra-five`; each 2 × 2 µm ZSensor map is divided into four non-overlapping 1 × 1 µm subfields, then flattened independently with a third-order polynomial per fast-scan line.
- Raw RHEED videos are read from `data/compressedfile`. The frozen V5 key-frame selector and frozen V8 complete-lattice ROI are transferred without AFM-target tuning.
- `data/extra_five_consolidated_v1` is the canonical derived root. Earlier extra-five derived folders are retained for safety but marked historical and are not used.
- For backward compatibility, some machine-readable tables retain the legacy
  target key `Rq_nm`; in this experiment that field contains the audited
  areal RMS height **Sq** computed from the complete 1 × 1 µm height map. All
  user-facing figures and claims call the quantity Sq.

## Strict evaluation design

Every reported point is an outer leave-one-growth-out prediction: 27 complete growth groups are fitted and all AFM subfields from the held growth remain excluded. M15b predicts Sq and FSMI from causal eight-frame R3D-18 features; the fixed M10 and M12a renderers each generate four 128 × 128 AFM height fields per held growth. The generated result is not a retrieved AFM patch.

## Scalar results

| cohort | target | n | MAE (nm) | Pearson r | Spearman ρ | confidence–error ρ |
|---|---:|---:|---:|---:|---:|---:|
| all 28 | Sq | 28 | 1.284 | 0.661 | 0.506 | -0.529 |
| all 28 | FSMI | 28 | 1.134 | 0.661 | 0.499 | -0.362 |
| extra five | Sq | 5 | 0.585 | -0.146 | 0.100 | 0.000 |
| extra five | FSMI | 5 | 0.536 | -0.112 | -0.300 | 0.000 |

Across all 28 groups, Sq and FSMI retain statistically significant positive linear and rank relationships. The extra-five MAE is numerically lower because all five occupy a narrow low-roughness interval; their within-five ordering is not learned reliably. This is reported as a batch-generalization limitation, not hidden by the lower MAE.

## Generated AFM results

- Four generated draws exist for every held growth at 128 × 128 resolution.
- Full-28 M12a texture-gate pass fraction: 0.786; median sharpness ratio: 0.724; median island-feature MAE: 1.772 z.
- The fixed M10 renderer is the stronger image-metric comparator on this expanded cohort: texture-gate pass 0.964, median sharpness ratio 0.821, and median island-feature MAE 1.421 z. Both M10 and M12a maps are preserved; the live UI retains frozen M12a behavior for version continuity.
- Maximum exact equality to a training AFM: 0.0.
- Generated-map Sq metadata and strict M15b scalar predictions match for 28/28 growths; retrieval and measured-patch-at-inference flags are false for 28/28.

The generated images contain terrace/island boundaries and non-flat height texture. They are plausible morphology samples conditioned on RHEED, not pixel-aligned reconstructions of a unique AFM field of view. High-Sq growths remain amplitude-compressed and the extra-five generated topology is coarser than some measured fine-island fields.

## What worked and what did not

- Worked: the frozen automatic key-frame/ROI transfer found complete visible spot lattices for all five new videos without using AFM labels.
- Worked: M15b retained significant all-28 Sq and FSMI relationships and improved same-cohort physics-only MAE by 0.406 nm and 0.398 nm, respectively.
- Worked: both true generators produced distinct, non-flat AFM-like height fields for 28/28 held growths, with no training-patch equality or retrieval.
- Mixed result: M10 is sharper and closer to held AFM island statistics than M12a on the expanded cohort; M12a remains the live/frozen renderer for continuity.
- Failed generalization test: adding the extra batch did not improve the original-23 subset. Relative to the prior 23-only M15b run (Sq/FSMI MAE 1.090/0.980 nm), the same original 23 under expanded-cohort fitting gives 1.436/1.265 nm.
- Failed fine ordering: the five new low-roughness samples have negative within-batch Pearson correlations despite low absolute errors.

## Confidence interpretation

The scalar M15b Sq confidence remains error-related over all 28 samples (negative confidence–absolute-error rank correlation), while the FSMI relationship is weaker. Confidence is a cross-fitted relative risk index, not a calibrated probability. It must not be interpreted as “percent correct.”

## Key outputs

- Integration figures: `reports/extra_five_integration/20260729_line3_full28_v1/figures/`
- Scalar predictions: `reports/rheed_auto_input_robustness/20260729_m15b_line3_full28_extra5_v1/`
- Generated AFM arrays and metrics: `reports/rheed_m15b_end_to_end_generation/20260729_m15b_m12a_line3_auto_full28_extra5_v1/full28_loo/`
- Full generated-image atlas: the `Fig1*_full28_loo_atlas` files in the end-to-end figure directory.
- Dedicated extra-five panel: `Fig8_extra_five_generated_afm`.
- Fixed M10 versus M12a extra-five comparison: `Fig9_extra_five_renderer_comparison`.
- Default live UI deployment: `configs/rheed_realtime_ui.json`, backed by the additive full-28 v5 bundle. The previous full-23 v4 config remains at `configs/rheed_realtime_ui_line3_full23_v4.json`.
- Verified UI screenshot: `outputs/rheed_realtime_ui/full28_line3_v5_ui_6056.png`.

## Reproduction commands

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_afm --config configs/extra_five_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_rheed --config configs/extra_five_line3_full28_v1.json --device mps
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_perturbations --config configs/extra_five_line3_full28_v1.json --device cpu
PYTHONPATH=src:. .venv/bin/python -m analysis.afm_metrology_repair.build_descriptors --config configs/rheed_video_afm_story_phase3a_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_auto_input_robustness.run --config configs/rheed_auto_input_robustness_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json --device auto
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.summarize --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.verify_integrity --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
```

## Verification

- The changed-component regression suite passes 29/29 tests.
- The independent integrity audit verifies 31/31 extra-AFM SHA-256
  hashes, all five selected RHEED-video SHA-256 hashes, all 28 RHEED
  inventory size/mtime records, 28 leakage-free outer folds, and all 28
  generated-map files.
- All 24 delivered full-28 PDF figures are valid single-page PDFs.
  Rasterized inspection of the overview and dedicated extra-five
  comparisons found no clipped or overlapping labels.
- The broader historical `tests/` collection reports 366 passes, 24
  failures, and 6 errors. The non-passing cases are outside this change:
  they require missing historical paper-freeze manifests, missing
  peak/saddle checkpoints and a human-review checkpoint, or an optional
  Parquet engine (`pyarrow`/`fastparquet`). They are recorded rather than
  silently treated as passes.

## Claim boundary

This is strict retrospective leave-one-growth-out evaluation, not a prospective untouched test. The M12a family was developed on earlier partitions. The five extra samples expand acquisition coverage but do not by themselves establish robust within-batch ranking because n=5 and the measured Sq range is narrow.
