# Legacy NN retrieval versus M17b HOO atlas

## Result

The requested historical model is the frozen 28-sample held-one-out A3 nearest-neighbor retrieval run:

- freeze: `publication_freeze/prospective_retrained_6358_6382_single_frame_v1`
- method: `A3_full_cohort_rq_conditioned_held_one_out`
- source commit: `6973e08753a6008a2b7480b7400efdf29d0b1469` (`add atlas for all`, 2026-07-23)
- frozen result table: `predictions/held_one_out_afm_28/retrieval_results.csv`
- frozen reference: `figures/main/Figure3_held_one_out_afm_prediction_atlas.pdf`

This identification was checked against the supplied screenshot. In particular, target 6022 retrieves source group 6057 / scan `GdSb_N6057_ctr_020`, and target 6029 retrieves source group 6028 / scan `N6028_500_nm_006`. A later Phase7B A3 atlas was ruled out because its target-to-source mappings differ (for example, it maps 6022 to 6085).

The current comparison result is the full27 held-one-out M17b morphology generator:

- package: `MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804`
- generative method: `M17b_topology_sparse_peak_terrace`
- HOO Sq predictor: `M16_endpoint_streak_dual_resolution`
- result commit: `6a42ab2`
- desktop package commit: `99bb75b3ed22d385367eb6622f7f05ddbc6a754e`

The one-page PDF contains the 27-sample intersection. The legacy-only target 6081 is absent because the current audited full27 cohort explicitly excludes it as erroneous.

## Panel definition

Each target shows:

1. the current frozen RHEED keyframe, with the model ROI overlaid when the raw frame is present;
2. the current audited representative measured AFM, labeled with its scan ID, the displayed array's directly computed Sq, and the sample-level median target Sq;
3. the legacy held-one-out retrieved AFM morphology, labeled with rendered output Sq, bank source group, bank scan ID, and source Sq;
4. draw 1 of 4 from the current M17b held-one-out generator, labeled with the HOO predicted Sq.

Each AFM panel is independently contrast-scaled to its 1st-99th percentile so that morphology remains visible. Height amplitude is stated by the Sq label and should not be inferred from panel color. The legacy files called areal RMS height `Rq`; the atlas corrects the display name to `Sq` without altering any frozen numeric value. N6358 is explicitly marked as clipped because its legacy raw prediction was -0.274953 nm and the frozen renderer clipped it to 0.001 nm.

## Integrity checks

- Legacy: 28/28 targets are absent from their fold AFM banks; the comparison contains no self-retrievals.
- Current: 27/27 targets are absent from their outer training folds; every fold fits 26 growth groups.
- Current: all 27 generated-map files certify `retrieval_at_inference=false` and `measured_afm_patch_used_at_inference=false`.
- Measured AFM: all 27 representative line-3 arrays are present, finite, hashed in the row manifest, and used only for comparison display.
- Reconstructed legacy map Sq agrees with the frozen rendered Sq to a worst-case absolute error of `1.65085e-06` nm.
- Current rendered map Sq agrees with the stored HOO prediction to a worst-case absolute error of `9.32417e-06` nm.
- PDF QA: one page, 2880 x 2988 pt (40 x 41.5 in); all 27 target labels, 27 measured-AFM labels, 27 legacy labels, and 27 current labels are extractable; overview and high-resolution top/middle/bottom crops were visually inspected.
- No raw RHEED or AFM file was modified. The reproducible PDF, row manifest, and provenance record are in `output/pdf`; temporary QA renders are not part of the deliverable.

## Reproduction

```bash
/Users/ziyi/Desktop/MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804/.venv/bin/python \
  scripts/build_retrieval_vs_m17_atlas.py \
  --package-root /Users/ziyi/Desktop/MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804 \
  --legacy-data-root /Users/ziyi/Desktop/LAB/code \
  --output-pdf output/pdf/NN_retrieval_vs_M17b_HOO_intersection_atlas.pdf \
  --output-manifest output/pdf/NN_retrieval_vs_M17b_HOO_intersection_manifest.csv \
  --output-provenance output/pdf/NN_retrieval_vs_M17b_HOO_intersection_provenance.json
```

The final PDF SHA-256 is `112c91e63a5bfc4a3873782ead5380fab7e64c0e42995bc8814c54450082180d`.
