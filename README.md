# Calibrated Logistic Regression Sensor (Home Assistant)

Custom integration that exposes a calibrated logistic regression output as a Home Assistant sensor.

## UX Overhaul Highlights

- Multi-step setup wizard (instead of one raw JSON screen)
- Guided post-setup management menu in options flow
- Built-in categorical state mapping for non-numeric entities
- Explainability attributes for runtime transparency and debugging

## Setup Wizard Steps

1. **Name & Goal**: define sensor name and what probability represents.
2. **Features**: pick feature entities with the Home Assistant entity selector.
3. **Mappings (if needed)**: review or override inferred mappings for categorical features.
4. **Model**: provide intercept, coefficients, and calibration values.
5. **Preview**: confirm before saving.

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
- `feature_types`: Auto-inferred feature typing map (`numeric` / `categorical`)
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

## Architecture (Refactor)

The runtime pipeline is now split into three layers:

- `model_provider`: loads coefficients/intercept from manual config or ML artifact contract view.
- `feature_provider`: builds the feature vector from either live HA states or ML latest-feature snapshot view.
- `inference`: pure probability/decision math (logistic + calibration + threshold).

This separation keeps the sensor entity focused on orchestration and makes it easier to evolve with `ha_ml_data_layer` contract changes.

### ML Contract Mode

When configured with:

- `ml_db_path`
- `ml_artifact_view` (default: `vw_clr_latest_model_artifact`)
- `ml_feature_source = ml_snapshot`
- `ml_feature_view` (default: `vw_latest_feature_snapshot`)

the sensor can pull both model artifacts and runtime features from the ML data-layer SQLite contract views.

## Setup Guidance

- Goal options are selectable in the wizard: `risk`, `event_probability`, `success_probability`.
- Feature selection uses a native entity picker to avoid typing mistakes.
- Feature types are inferred automatically from current states.
- Common categorical mappings (`on/off`, `home/away`, `open/closed`) are inferred automatically.
- Mapping and coefficient steps include inline JSON examples in the UI.

## Sensor Behavior

- State is calibrated probability in percent (`0-100`).
- Numeric source states are used directly.
- Categorical states use `state_mappings`.
- Sensor becomes unavailable if required features are missing/unmapped.
- Sensor updates whenever required feature entities change state.
