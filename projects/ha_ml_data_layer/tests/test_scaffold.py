"""Scaffold checks for the AppDaemon ML data layer package."""

from importlib import import_module


def test_appdaemon_ml_package_scaffold_imports() -> None:
    modules = [
        "appdaemon_ml",
        "appdaemon_ml.app",
        "appdaemon_ml.config",
        "appdaemon_ml.db",
        "appdaemon_ml.ingest",
        "appdaemon_ml.features",
        "appdaemon_ml.labels",
        "appdaemon_ml.lightgbm_train",
        "appdaemon_ml.bocpd_train",
        "appdaemon_ml.contracts",
    ]

    for name in modules:
        assert import_module(name)
