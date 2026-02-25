# Calibrated Logistic Regression Sensor (Home Assistant)

Custom integration that exposes a calibrated logistic regression output as a Home Assistant sensor.

## UX Overhaul Highlights

- Multi-step setup wizard (instead of one raw JSON screen)
- Guided post-setup management menu in options flow
- Built-in categorical state mapping for non-numeric entities
- Explainability attributes for runtime transparency and debugging

## Setup Wizard Steps

1. **Name & Goal**: define sensor name and what probability represents.
2. **Features**: list entity IDs used as features.
3. **Feature Types**: set each feature to `numeric` or `categorical`.
4. **Mappings**: map categorical states to numbers.
5. **Model**: provide intercept, coefficients, and calibration values.
6. **Preview**: confirm before saving.

## Management After Setup

Open the integration options and choose:

- **Features**: edit required feature entity IDs
- **Mappings**: adjust categorical state-to-number mappings
- **Calibration**: tune slope/intercept
- **Diagnostics**: see feature coverage context

## Configuration Inputs (Stored)

- `name`: Sensor name
- `goal`: Human-readable probability purpose
- `required_features`: Feature entity IDs
- `feature_types`: Feature typing map (`numeric` / `categorical`)
- `state_mappings`: Optional map for categorical states
- `intercept`: Model intercept
- `coefficients`: Feature weight map
- `calibration_slope`: Calibration slope (default `1.0`)
- `calibration_intercept`: Calibration intercept (default `0.0`)

`state_mappings` example:

```json
{
  "binary_sensor.back_door": {"on": 1, "off": 0},
  "climate.living_room_hvac_action": {"heating": 1, "idle": 0, "off": 0}
}
```

## Runtime Explainability Attributes

- `raw_probability`
- `linear_score`
- `feature_values`
- `feature_contributions`
- `mapped_state_values`
- `missing_features`
- `unavailable_reason`
- `last_computed_at`

## Sensor Behavior

- State is calibrated probability in percent (`0-100`).
- Numeric source states are used directly.
- Categorical states use `state_mappings`.
- Sensor becomes unavailable if required features are missing/unmapped.
- Sensor updates whenever required feature entities change state.
