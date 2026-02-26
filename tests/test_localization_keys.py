from __future__ import annotations

import json
from pathlib import Path


def test_strings_contains_wizard_and_options_labels() -> None:
    path = Path("custom_components/calibrated_logistic_regression/strings.json")
    strings = json.loads(path.read_text())

    config_steps = strings["config"]["step"]
    assert "user" in config_steps
    assert "features" in config_steps
    assert "states" in config_steps
    assert "preview" in config_steps

    errors = strings["config"]["error"]
    assert "required" in errors

    options_steps = strings["options"]["step"]
    assert "features" in options_steps
    assert "states" in options_steps
    assert "model" in options_steps
    assert "feature_source" in options_steps
    assert "decision" in options_steps
    assert "diagnostics" in options_steps

    init_menu_options = options_steps["init"]["menu_options"]
    assert "model" in init_menu_options
    assert "feature_source" in init_menu_options
    assert "decision" in init_menu_options
    assert "features" in init_menu_options
    assert "diagnostics" in init_menu_options


def test_features_step_description_mentions_inline_states() -> None:
    path = Path("custom_components/calibrated_logistic_regression/strings.json")
    strings = json.loads(path.read_text())

    config_features_description = strings["config"]["step"]["features"]["description"]
    options_features_description = strings["options"]["step"]["features"]["description"]

    assert "state" in config_features_description.casefold()
    assert "state" in options_features_description.casefold()
