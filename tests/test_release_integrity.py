from pathlib import Path

from scripts.validate_release import validate


def test_release_integrity() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate(root / "configs/morphmbe_m22_realtime.json")

    assert result["status"] == "PASS"
    assert result["training_growths"] == 27
    assert result["held_growth_overlap"] is False
    assert result["retrieval_at_inference"] is False
