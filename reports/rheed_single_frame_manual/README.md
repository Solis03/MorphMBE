# Single-frame Manual RHEED Experiment

Reproduce full run:

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml
```

Smoke run:

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml --smoke
```

Primary outputs are under `outputs/rheed_single_frame_manual/`; figures and the static report are under `reports/rheed_single_frame_manual/`.