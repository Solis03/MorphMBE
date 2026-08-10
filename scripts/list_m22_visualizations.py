#!/usr/bin/env python3
"""Print the ordered M22 standalone visual-result inventory."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = (
    ROOT
    / "reports/rheed_m22_dense_mid/20260809_m22_paired_comparison"
    / "figures/gwyddion_individual_height_atlas_M17_vs_M22_dual"
)
PAGES = {
    1: "6101, N6342, N6358, N6382, 6084",
    2: "6072, 6078, 6022, 6048, 6082",
    3: "N6390, N6389, 6033, 6085, 6029",
    4: "6090, 6056, 6070, 6062, 6047",
    5: "6080, 6057, 6094, 6028, 6063",
    6: "6095, 6099",
}


def main() -> None:
    print("M22 Gwyddion atlas: RHEED | measured AFM | M17 | M22 inclusive | M22 exclusion")
    for page, groups in PAGES.items():
        path = FIGURE_ROOT / f"Atlas_{page:02d}_of_06.png"
        state = "OK" if path.is_file() else "MISSING"
        print(f"[{state}] page {page}/6 ({groups})\n  {path}")
    for label, path in (
        (
            "intermediate focus (Sq 3.5-6.0 nm)",
            FIGURE_ROOT / "Focus_true_Sq_3p5_to_6p0_M17_vs_M22_dual.png",
        ),
        (
            "measured versus predicted Sq",
            FIGURE_ROOT / "M22_Sq_measured_vs_predicted_ordered.png",
        ),
        (
            "6063 UI command-line smoke",
            ROOT
            / "reproduced_outputs/model_smoke_6063"
            / "rheed_to_generated_afm_panel.png",
        ),
        (
            "6063 real UI offscreen launch",
            ROOT / "reproduced_outputs/ui_offscreen_6063/ui_offscreen.png",
        ),
    ):
        path = path.resolve()
        state = "OK" if path.is_file() else "MISSING"
        print(f"[{state}] {label}\n  {path}")


if __name__ == "__main__":
    main()
