# Method Audit

All rows are fixed-method strict OOF paths; held-out AFM is used only for display and retrospective metrics.

| family | method_id | single_method_only | all_from_heldout_rheed_strict_prediction | uses_heldout_true_afm_for_method_seed_or_source_selection | uses_heldout_true_descriptors | max_heldout_source_contribution | source_sample_never_equals_heldout | retrieval_based | synthesis_based | q10_q50_q90_amplitude_only | deployable_for_unseen | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieval | A3 | True | True | False | False | 0.0 | True | True | False | True | True | Frozen deployment recommendation: descriptor-conditioned representative AFM retrieval. |
| quilting | VB2 | True | True | False | False | 0.0 | True | False | True | True | True | Exploratory deployable candidate; not current frozen deployment recommendation. |
| residual | C1 | True | True | False | False | 0.0 | True | False | True | True | True | Exploratory synthesis candidate; not current frozen deployment recommendation. |
| iaaft | D4 | True | True | False | False | 0.0 | True | False | True | False | True | Exploratory spectral synthesis candidate; not current frozen deployment recommendation. |
| texture | E2 | True | True | False | False | 0.0 | True | False | True | True | True | Exploratory texture optimization candidate; not current frozen deployment recommendation. |
| vq | F1 | True | True | False | False | 0.0 | True | False | True | True | True | Exploratory VQ candidate; not current frozen deployment recommendation. |
| diffusion | G4 | True | True | False | False | 0.0 | True | False | True | False | True | Exploratory residual diffusion fallback candidate; not current frozen deployment recommendation. |
