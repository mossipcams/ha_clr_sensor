from __future__ import annotations

import json
from pathlib import Path


def test_strings_contains_wizard_and_options_labels() -> None:
    path = Path("custom_components/calibrated_logistic_regression/strings.json")
    strings = json.loads(path.read_text())

    config_steps = strings["config"]["step"]
    assert "user" in config_steps
    assert "features" in config_steps
    assert "feature_types" in config_steps
    assert "mappings" in config_steps
    assert "model" in config_steps
    assert "preview" in config_steps

    errors = strings["config"]["error"]
    assert "invalid_feature_types" in errors
    assert "missing_categorical_mappings" in errors
    assert "coefficient_mismatch" in errors

    options_steps = strings["options"]["step"]
    assert "features" in options_steps
    assert "mappings" in options_steps
    assert "calibration" in options_steps
    assert "diagnostics" in options_steps
