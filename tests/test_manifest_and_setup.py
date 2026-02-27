from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_manifest_contains_required_fields() -> None:
    manifest_path = Path("custom_components/mindml/manifest.json")
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["domain"] == "mindml"
    assert manifest["name"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])
    assert manifest["config_flow"] is True


def test_integration_module_exports_setup_functions() -> None:
    integration = importlib.import_module(
        "custom_components.mindml"
    )

    assert hasattr(integration, "async_setup")
    assert hasattr(integration, "async_setup_entry")
    assert hasattr(integration, "async_unload_entry")
