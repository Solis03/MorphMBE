# Checkpoint 2B1: Real Development Diagnostics

## Annotation Validation
- Validation passed: `True`
- Annotation hashes: `{'all_sample_qc_completed.csv': '511db34e025340bd4647a565da3d4af25fa65a26100e8dc83374bd4d56b6ce4f', 'development_sample_review_completed.csv': 'bc441b771cb086ac74fe663b1e80bacf8b5ae42e9057e8a90f90f78219e4fb5a', 'development_pair_review_completed.csv': '662fae9e5de5421ee22e072ebe38e730433fae5b392d6783732031d5d11671fd'}`
- Row counts: `{'all_sample_qc_completed.csv': 25, 'development_sample_review_completed.csv': 10, 'development_pair_review_completed.csv': 23}`
- Development pair labels: `{'isolated': 0, 'partial': 7, 'connected': 11, 'unusable': 5}`
- Isolated labels present: `0`
- Concept-label coverage: `insufficient`

## Boundary Confirmation
- Split counts: `{'development_review': 10, 'blind_validation': 10, 'reserve': 5}`
- Tuning, candidate adapters, diagnostic selection, and supplemental sampling used only development anonymous IDs.
- Blind and reserve QC rows were checked only for presence and complete schema.
- Blind/reserve label values were not printed, summarized, or passed to tuning functions.
- `unblind_key.csv` was not opened.

## Safety Hashes
- Removelist SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` remains excluded by the canonical removelist.
- Stage-review SHA256: `862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e`
- No AFM arrays, AFM images, descriptor tables, Rq values, previous Rq predictions, Rq-sorted figures, or target tables were accessed.

## Development QC Statistics
- `lattice_indexing=fail`: `10/10`
- `background_correction=fail`: `5/10`
- `background_correction=partial`: `5/10`
- `overall_measurement=fail`: `6/10`
- `overall_measurement=partial`: `4/10`

## Real/Synthetic Domain Mismatch
- Synthetic rows often contained several periodic spots, while the development real images frequently show one or two visible spots per horizontal level.
- Long-row lattice-period fitting is therefore a poor structural assumption for many reviewed real images.

## Front-End Failure Taxonomy
- `adhesion_unreliable_due_to_background`: `10/10`
- `invalid_endpoint`: `10/10`
- `invalid_pair_used`: `10/10`
- `missed_spot`: `10/10`
- `missing_site_false_positive`: `10/10`
- `pair_crosses_missing_site`: `10/10`
- `pair_missed`: `10/10`
- `same_row_split`: `10/10`
- `spacing_estimate_failure`: `10/10`
- `two_column_structure_not_modeled`: `10/10`

## Background Failure Taxonomy
- `edge_crop_contamination`: `10/10`
- `global_halo_not_removed`: `10/10`
- `both_offset_corridors_contaminated`: `8/10`
- `contour_or_ringing`: `6/10`
- `real_bridge_removed_as_background`: `5/10`
- `one_offset_corridor_contaminated`: `0/10`
- `quantization_banding`: `0/10`

## Candidate Adapters
- Variant 0: frozen synthetic pipeline.
- Variant 1: two-column / y-level pair proposal on linear grayscale local maxima.
- Variant 2: Variant 1 proposals with pair-local low-order background modeling.
- Candidate output rows by variant: `{'variant0': 10, 'variant1': 10, 'variant2': 10}`
- No adapter was selected or frozen in this run.

## Human Review Files
- `reports/rheed_peak_saddle/real_development/development_adapter_comparison.html`
- `annotations/rheed_peak_saddle/real_review/development_adapter_comparison_template.csv`
- `reports/rheed_peak_saddle/real_development/development_pair_review_round2.html`
- `annotations/rheed_peak_saddle/real_review/development_pair_review_round2_template.csv`
- Supplemental round-2 pair count: `27`
- Failure contact sheets generated: `21`

## Stop Confirmation
- No pair concept model was fit.
- No ordinal or multinomial calibrator was fit.
- No connected/partial/isolated probability was calibrated.
- No pair-label accuracy was reported.
- No blind validation evaluation occurred.
- No reserve labels were used.
- No roughness model was trained.

## Status
HUMAN ROUND-2 ANNOTATION REQUIRED
