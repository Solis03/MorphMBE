# Benchmark v1 Data Registry

The master registry has 29 records:
23 `historical_development`, 4 `prospective_pilot_seen`, 1
`prospective_pending_truth`, and 1 `afm_only_unmatched`.

The paired supervised cohort has 27 samples:
the 23 historical samples plus the four seen prospective pilot samples. The
historical development cohort is the only cohort eligible for model development
and primary nested CV.

Historical targets come from the frozen target table and use
`T4_second_order_trimmed_mean` in nm. Prospective pilot AFM truth comes from the
frozen prospective AFM table as `true_rq_nm_median_second_order`; this is not the
same aggregation as T4, so pilot aggregate metrics are exploratory and not
directly confirmatory.

`N6390` remains present as `prospective_pending_truth`: it has RHEED and a frozen
legacy prediction record but no AFM truth, and it is excluded from all metrics.

`N6324` remains present as `afm_only_unmatched`: it has AFM truth but no matched
RHEED prediction sample in the frozen prospective prediction package, and it is
excluded from supervised metrics.

The canonical removelist is retained at
`publication_freeze/rheed_afm_single_frame_v1_2026-07-18/data_snapshot/removelist.txt` with hash
`8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`. Removed samples are not silently
repaired or reintroduced into the historical development cohort.

AFM scans are grouped by sample. No AFM scan is treated as an independent growth
sample. RHEED grouping is by growth run, and each historical growth run appears
once as an outer held-out group.
