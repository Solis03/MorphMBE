"""Small schema definitions and a minimal validator for benchmark JSON records."""

from __future__ import annotations

from typing import Any


def schema_errors(schema: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in payload:
            errors.append(f"missing required key: {key}")
    properties = schema.get("properties", {})
    for key, spec in properties.items():
        if key not in payload:
            continue
        expected = spec.get("type")
        if expected and not _type_matches(payload[key], expected):
            errors.append(f"{key}: expected {expected}, got {type(payload[key]).__name__}")
    return errors


def _type_matches(value: Any, expected: str | list[str]) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    for option in options:
        if option == "string" and isinstance(value, str):
            return True
        if option == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if option == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if option == "boolean" and isinstance(value, bool):
            return True
        if option == "array" and isinstance(value, list):
            return True
        if option == "object" and isinstance(value, dict):
            return True
        if option == "null" and value is None:
            return True
    return False

