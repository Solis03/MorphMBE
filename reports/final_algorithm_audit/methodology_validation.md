| name | passed | detail |
| --- | --- | --- |
| ensemble_member_count_is_5 | True | 5 |
| embedding_dim_is_1536 | True | Phase2A temporal aggregate |
| descriptor_dim_is_11 | True | rq_nm,ra_nm,robust_height_range_nm,psd_low_fraction,psd_mid_fraction,psd_high_fraction,psd_slope,correlation_length_nm,anisotropy,height_skewness,height_kurtosis |
| strict_candidate_count_is_22 | True | [22] |
| deployment_bank_is_23_groups_116_scans | True | 23 groups / 116 scans |
| phase7b_a3_heldout_source_zero | True | Phase7B method_audit A3 |
| overall | True | Strict OOF fold coefficients are not serialized in the freeze; strict Rq predictions are recovered from Phase6A OOF artifact, while full-cohort member coefficients are serialized.; DINO backbone internals are not serialized as tensors in the freeze; patch/token shapes are recovered from the dinov2_vits14 identifier and input size, while final cached feature values are serialized.; Freeze unseen inference script is a technical smoke implementation and does not perform real DINO extraction or full strict A3 descriptor ranking. |
