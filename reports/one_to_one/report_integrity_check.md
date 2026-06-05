# Descriptor Report Integrity Check

## Findings

- `reports/one_to_one/1um/descriptor_data/joined_dataset.csv` and `reports/one_to_one/all_size_representative/descriptor_data/joined_dataset.csv` are byte-identical.
- `metrics_summary.json` and `test_predictions.csv` are also byte-identical across those two report folders.
- Their modification timestamps differ, which suggests one output directory was regenerated or copied later, but from the same underlying data payload.
- This is inconsistent with the true manifest sizes: `manifest_1um_one_to_one.csv` has 37 rows, while `manifest_all_size_representative_one_to_one.csv` has 40 rows.
- The existing descriptor comparison report therefore should not be used to compare `1um` versus `all_size_representative` until the descriptor baseline is rerun with verified manifests.

## Evidence

- `joined_dataset.csv`: identical contents, same file size (1701560 bytes).
- `metrics_summary.json`: identical contents.
- `test_predictions.csv`: identical contents.
- `1um` descriptor timestamps are earlier than `all_size_representative`, consistent with reuse or regeneration from the same source rows rather than an independently joined dataset.
