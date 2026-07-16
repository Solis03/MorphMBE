Draw a detailed supplementary architecture diagram with exact tensor dimensions.

Show the DINO input tensor [1,3,224,224], ViT-S/14 patch tokens [1,256,384], token sequence [1,257,384], CLS/frame embedding [1,384], and Phase2A temporal aggregate [1,1536]. Show five ridge model inputs [1,1536], five scalar member outputs, median aggregation in nm Rq space, and the strict q10/q50/q90 definitions.

For the visual lane, show the 11 descriptors in order: rq_nm, ra_nm, robust_height_range_nm, psd_low_fraction, psd_mid_fraction, psd_high_fraction, psd_slope, correlation_length_nm, anisotropy, height_skewness, height_kurtosis. Show strict A3 ranking over 22 candidate representative groups per held-out sample, selected source AFM [256,256], unit-Rq morphology [256,256], and three rescaled maps [3,256,256].

Include a deployment caveat panel: the full-cohort visual bank contains 23 groups and 116 scans, but the current frozen unseen script should be treated as a technical smoke script because it uses deterministic placeholder embeddings and Rq-nearest scan selection rather than the full strict A3 descriptor route.
