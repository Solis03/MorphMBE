# Height Normalization V4 Diagnosis

Rows analyzed: `168`
Per-image normalized rate: `1.000`
Network min/max median: `-1` / `1`
Network Rq median: `0.4087`
Physical Rq median: `3.719`
Scale bounds: `{'scale_low': 2.273653482957607, 'scale_high': 37.01710417872107, 'scale_median': 7.755423650503959, 'scale_mean': 9.76421854057945, 'scale_std': 6.775475217974828, 'scale_min': 1.6204627854954554, 'scale_max': 49.422011386505346}`

The standardized network inputs are effectively per-image normalized to [-1, 1]. Absolute height-scale descriptors such as Rq, Ra, and robust range cannot be recovered directly from decoder output without an external physical height-scale calibration step.
