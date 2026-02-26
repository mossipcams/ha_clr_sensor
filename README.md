# HA-MindML Sensor (Home Assistant)

Custom integration that exposes a LightGBM-based probability sensor in Home Assistant.

## Core Runtime Model

Prediction always combines:

- **Model artifact** from ML DB (`model_provider`)
- **Runtime feature values** from `feature_provider`

Formula at runtime:

`probability_now = model.predict(features_now)`

So this is not "model or states". It is model + feature source.

## Feature Sources

- `hass_state`: live HA states at scoring time (real-time updates)
- `ml_snapshot`: latest feature snapshot from ML DB view

## Setup

Wizard collects:

1. Name and goal
2. ML DB path + artifact view
3. Runtime feature source
4. Required features
5. State mappings/threshold
6. Preview and confirm

## Options (Entity Settings)

- `Model`
- `Feature Source`
- `Decision`
- `Features`
- `Mappings`
- `Diagnostics`

## Key Stored Fields

- `name`
- `goal`
- `required_features`
- `feature_types`
- `state_mappings`
- `threshold`
- `ml_db_path`
- `ml_artifact_view`
- `ml_feature_source`
- `ml_feature_view`

## Explainability Attributes

- `raw_probability`
- `linear_score`
- `feature_values`
- `feature_contributions`
- `missing_features`
- `model_source`
- `feature_source`
- `decision`
