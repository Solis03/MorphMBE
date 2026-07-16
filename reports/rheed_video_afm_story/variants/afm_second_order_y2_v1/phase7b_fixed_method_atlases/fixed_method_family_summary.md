# Fixed-Method Family Summary

- Best fixed strict method by median visual composite: retrieval / A3 (0.211454).
- Current frozen deployment recommendation: retrieval / A3 (0.211454).
- Mixed-method atlas differs because it selects the best visual output separately for each sample; this rerun fixes one method per family across all 23 held-out samples.
- Lower visual composite is better.

| family | method_id | N | median_visual_composite | mean_visual_composite | median_psd_distance | median_histogram_wasserstein | median_corr_length_relative_error | strict_identity_pass | max_heldout_source_contribution | deployable_for_unseen | recommended_for_unseen | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieval | A3 | 23 | 0.2114539805176648 | 0.21738103967524475 | 0.3939389836376274 | 0.0807791181908659 | 0.2222222222222221 | True | 0.0 | True | True | Frozen deployment recommendation: descriptor-conditioned representative AFM retrieval. |
| texture | E2 | 23 | 0.2389929195079523 | 0.244722404116517 | 0.4860405818658604 | 0.0718243471491833 | 0.2222222222222221 | True | 0.0 | True | False | Exploratory texture optimization candidate; not current frozen deployment recommendation. |
| quilting | VB2 | 23 | 0.2491132394206745 | 0.2889590093651782 | 0.4370907246547087 | 0.1048031177149054 | 0.125 | True | 0.0 | True | False | Exploratory deployable candidate; not current frozen deployment recommendation. |
| iaaft | D4 | 23 | 0.2852205336837455 | 0.2899560232019887 | 0.3940528991279524 | 0.0807791182600734 | 0.2222222222222221 | True | 0.0 | True | False | Exploratory spectral synthesis candidate; not current frozen deployment recommendation. |
| residual | C1 | 23 | 0.2960113946122025 | 0.31611628694284183 | 0.7183036399284303 | 0.1094171180186167 | 0.2499999999999999 | True | 0.0 | True | False | Exploratory synthesis candidate; not current frozen deployment recommendation. |
| vq | F1 | 23 | 0.3431883140940885 | 0.33920508676709327 | 0.8847647728832282 | 0.1002641490495234 | 0.1428571428571428 | True | 0.0 | True | False | Exploratory VQ candidate; not current frozen deployment recommendation. |
| diffusion | G4 | 23 | 0.3554912495547412 | 0.33393353953563093 | 0.5836122989632045 | 0.1200975521179509 | 0.3333333333333334 | True | 0.0 | True | False | Exploratory residual diffusion fallback candidate; not current frozen deployment recommendation. |
