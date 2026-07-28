"""Functional-morphology RHEED-to-AFM generation.

This package extends the frozen M10 island generator with:

* a small-data, physics-selected RHEED amplitude head;
* an absolute, multiscale AFM morphology index;
* amplitude-conditioned island topology;
* signed-distance-field (SDF) island contour rendering; and
* strictly cross-fitted confidence diagnostics.
"""

from .metrics import FSMI_COMPONENTS, extract_surface_metrics

__all__ = ["FSMI_COMPONENTS", "extract_surface_metrics"]
