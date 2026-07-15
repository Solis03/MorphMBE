# AFM Descriptor Definitions

All descriptors are computed from plane-corrected physical height arrays in nm, never from rendered PNGs.

- Rq: root mean square of mean-centered height.
- Ra: mean absolute mean-centered height.
- robust height range: p99 - p01.
- p95-p5: 95th minus 5th height percentile.
- peak-to-valley: max - min, reported as an auxiliary outlier-sensitive descriptor.
- skewness/kurtosis: scipy moment descriptors on centered heights.
- gradient metrics: finite differences using scan-size-derived pixel spacing.
- radial PSD: FFT power averaged in radial frequency bins with DC excluded.
- PSD band fractions: low/mid/high thirds of radial PSD power normalized by total radial power.
- PSD slope: linear fit of log(power) vs log(radial frequency).
- autocorrelation length: first radial autocorrelation crossing below exp(-1), in nm.
- directional correlation lengths: same crossing along x and y center lines.
- anisotropy ratio: max directional length divided by min directional length.
- gradient orientation entropy: weighted entropy of gradient orientation modulo pi.
- Unit-shape descriptors: same morphology descriptors after mean-centering and unit-Rq normalization; amplitude descriptors become dimensionless.

No island/grain segmentation is used in Phase 3A because this repository does not yet contain a validated segmentation method.
