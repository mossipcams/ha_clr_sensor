# Calibrated Logistic Regression Sensor (Home Assistant)

Custom integration that exposes a calibrated logistic regression output as a Home Assistant sensor.

## Features

- One sensor per config entry
- Logistic regression from source entity states
- Optional categorical encoding for non-numeric states (`state_mappings`)
- Platt-style calibration (`calibration_slope`, `calibration_intercept`)
- Availability handling when required features are missing/unmapped/non-numeric
- Diagnostic attributes (`raw_probability`, `linear_score`, feature values, missing features)

## Installation

1. Copy `custom_components/calibrated_logistic_regression` into your HA `custom_components` folder.
2. Restart Home Assistant.
3. Add integration from **Settings -> Devices & Services -> Add Integration**.

## Configuration Inputs

- `name`: Sensor name
- `intercept`: Model intercept
- `coefficients`: JSON map of feature entity IDs to weights
- `required_features`: Comma-separated list of feature entity IDs
- `state_mappings`: Optional JSON map for categorical states
- `calibration_slope`: Calibration slope (default `1.0`)
- `calibration_intercept`: Calibration intercept (default `0.0`)

`state_mappings` example:

```json
{
  "binary_sensor.back_door": {"on": 1, "off": 0},
  "climate.living_room_hvac_action": {"heating": 1, "idle": 0, "off": 0}
}
```

## Sensor Behavior

- State is calibrated probability in percent (`0-100`).
- Numeric source states are used directly.
- Non-numeric source states are encoded through `state_mappings` when present.
- Sensor becomes unavailable if any required feature is missing or cannot be converted.
- Updates when any required feature state changes.

## Best-Practice Notes

- Keep coefficients and required features aligned.
- Prefer explicit mappings for categorical states so behavior is deterministic.
- Use stable source entities and avoid frequently changing text labels.
- Start with `calibration_slope=1.0` and `calibration_intercept=0.0` if calibration is unknown.
