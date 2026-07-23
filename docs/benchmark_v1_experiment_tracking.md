# Benchmark v1 Experiment Tracking

Future runs write full outputs under `outputs/benchmark_v1/runs/<run_id>/` and
compact finalized records under `reports/benchmark_v1/run_records/`.

Run IDs are deterministic from the experiment ID, config hash, Git commit,
protocol hash, and split hash. Existing run directories must not be overwritten;
a dry-run fails if the intended run ID already exists.

Every run manifest records protocol, registry, split, config, environment, input
source hashes, train/validation/test sample IDs, runtime, status, and failure
reason. Failed runs still write a manifest so missing outputs are auditable.

Protocol hashes identify the frozen scientific rules. Registry hashes identify
the sample table. Config hashes identify model-side settings. Environment hashes
identify the Python package state and hardware context.
