from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_homeassistant_stubs() -> None:
    """Install lightweight Home Assistant stubs for unit tests."""
    if "homeassistant" in sys.modules:
        return

    homeassistant = types.ModuleType("homeassistant")
    config_entries = types.ModuleType("homeassistant.config_entries")
    data_entry_flow = types.ModuleType("homeassistant.data_entry_flow")
    core = types.ModuleType("homeassistant.core")
    components = types.ModuleType("homeassistant.components")
    sensor_component = types.ModuleType("homeassistant.components.sensor")
    helpers = types.ModuleType("homeassistant.helpers")
    selector = types.ModuleType("homeassistant.helpers.selector")
    entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")
    event_helpers = types.ModuleType("homeassistant.helpers.event")
    restore_state = types.ModuleType("homeassistant.helpers.restore_state")

    class ConfigFlow:
        @classmethod
        def __init_subclass__(cls, **kwargs):
            return super().__init_subclass__()

        def async_show_form(self, *, step_id, data_schema=None, errors=None, description_placeholders=None):
            return {
                "type": "form",
                "step_id": step_id,
                "data_schema": data_schema,
                "errors": errors or {},
                "description_placeholders": description_placeholders or {},
            }

        def async_abort(self, *, reason):
            return {"type": "abort", "reason": reason}

        def async_create_entry(self, *, title, data):
            return {"type": "create_entry", "title": title, "data": data}

    class OptionsFlow:
        def async_show_form(self, *, step_id, data_schema=None, errors=None, description_placeholders=None):
            return {
                "type": "form",
                "step_id": step_id,
                "data_schema": data_schema,
                "errors": errors or {},
                "description_placeholders": description_placeholders or {},
            }

        def async_show_menu(self, *, step_id, menu_options):
            return {"type": "menu", "step_id": step_id, "menu_options": list(menu_options)}

        def async_create_entry(self, *, title, data):
            return {"type": "create_entry", "title": title, "data": data}

    class State:
        def __init__(self, entity_id: str, state: str) -> None:
            self.entity_id = entity_id
            self.state = state

    class SensorEntity:
        @property
        def available(self) -> bool:
            return True

        async def async_added_to_hass(self) -> None:
            return None

        def async_on_remove(self, remove_callback):
            return None

        def async_write_ha_state(self) -> None:
            return None

    class SensorStateClass:
        MEASUREMENT = "measurement"

    class RestoreEntity:
        async def async_get_last_state(self):
            return None

    class SelectSelectorMode:
        DROPDOWN = "dropdown"

    class SelectOptionDict:
        def __init__(self, value, label):
            self.value = value
            self.label = label

    class SelectSelectorConfig:
        def __init__(self, options, mode):
            self.options = options
            self.mode = mode

    class SelectSelector:
        def __init__(self, config):
            self.config = config

        def __call__(self, value):
            return value

    class EntitySelectorConfig:
        def __init__(self, multiple=False):
            self.multiple = multiple

    class EntitySelector:
        def __init__(self, config):
            self.config = config

        def __call__(self, value):
            return value

    config_entries.ConfigFlow = ConfigFlow
    config_entries.OptionsFlow = OptionsFlow
    config_entries.ConfigEntry = object
    data_entry_flow.FlowResult = dict
    core.HomeAssistant = object
    core.Event = object
    core.State = State
    core.callback = lambda fn: fn
    sensor_component.SensorEntity = SensorEntity
    sensor_component.SensorStateClass = SensorStateClass
    restore_state.RestoreEntity = RestoreEntity
    selector.SelectSelectorMode = SelectSelectorMode
    selector.SelectOptionDict = SelectOptionDict
    selector.SelectSelectorConfig = SelectSelectorConfig
    selector.SelectSelector = SelectSelector
    selector.EntitySelectorConfig = EntitySelectorConfig
    selector.EntitySelector = EntitySelector
    entity_platform.AddEntitiesCallback = object
    event_helpers.async_track_state_change_event = lambda hass, entities, cb: lambda: None
    helpers.selector = selector

    sys.modules["homeassistant"] = homeassistant
    sys.modules["homeassistant.config_entries"] = config_entries
    sys.modules["homeassistant.data_entry_flow"] = data_entry_flow
    sys.modules["homeassistant.core"] = core
    sys.modules["homeassistant.components"] = components
    sys.modules["homeassistant.components.sensor"] = sensor_component
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.selector"] = selector
    sys.modules["homeassistant.helpers.entity_platform"] = entity_platform
    sys.modules["homeassistant.helpers.event"] = event_helpers
    sys.modules["homeassistant.helpers.restore_state"] = restore_state


_install_homeassistant_stubs()


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """Provide an event loop for environments that do not auto-create one."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    return None


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    return None
