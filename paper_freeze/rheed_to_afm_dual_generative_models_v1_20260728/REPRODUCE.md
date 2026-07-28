# Reproduction

## M12a-Strict15

```bash
PYTHONPATH=. .venv/bin/python   -m analysis.rheed_to_afm_functional_morphology.run   --config configs/rheed_to_afm_functional_morphology_m12.json
```

## M14i-Full23 target head and generator

```bash
PYTHONPATH=. .venv/bin/python   -m analysis.rheed_to_afm_ood_robust.run   --config configs/rheed_to_afm_ood_robust_v3_final.json

PYTHONPATH=. .venv/bin/python   -m analysis.rheed_to_afm_full_cohort_loo.run   --config configs/rheed_to_afm_ood_robust_generation.json   --mode full

PYTHONPATH=. .venv/bin/python   -m analysis.rheed_to_afm_full_cohort_loo.visualization   --config configs/rheed_to_afm_ood_robust_generation.json
```

For exact reconstruction, check out the source commit recorded for the model
in `00_FREEZE_CONTROL/MODEL_NAME_REGISTRY.csv`. Do not regenerate publication
numbers from a later mutable worktree without first validating every artifact
hash.
