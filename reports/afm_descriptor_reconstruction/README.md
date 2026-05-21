# AFM Descriptor Reconstruction Reports

These reports summarize compact descriptors extracted from plane-corrected ZSensor AFM height maps. The input arrays are already plane-corrected; no additional fitted-plane subtraction is applied during descriptor extraction.

This is a small-data descriptor baseline. The goal is to test how much morphology information can be retained in compact physical features before moving to heavier image reconstruction models.

The descriptor table contains height statistics, roughness metrics, slope features, simple frequency features, spatial correlation estimates, and threshold-mask component measurements. Network input arrays are separately robust-clipped and normalized to [-1, 1].
