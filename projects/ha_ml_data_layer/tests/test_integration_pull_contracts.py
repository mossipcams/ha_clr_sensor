from __future__ import annotations

from pathlib import Path

from appdaemon_ml.contracts import validate_pull_contracts
from appdaemon_ml.db import connect, ensure_schema


def test_integration_pull_contracts_are_versioned_and_stable(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        contract_info = validate_pull_contracts(conn)
        assert contract_info["contract_version"] == "1"
        assert "vw_lightgbm_latest_model_artifact" in contract_info["views"]
        assert "vw_bocpd_latest_state" in contract_info["views"]
    finally:
        conn.close()
