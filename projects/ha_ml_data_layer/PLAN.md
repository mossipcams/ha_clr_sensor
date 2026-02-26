# AppDaemon ML Data Layer Plan

## Scope and Ownership

- SQLite is the single database for the ML data layer.
- AppDaemon owns end-to-end data processing: ingest, feature engineering, label capture, CLR training, and BOCPD training/state updates.
- Custom integrations are pull-only consumers. They read from SQLite contracts and use the data/artifacts at runtime.

## System Goals

1. Collect live Home Assistant/AppDaemon events into a durable local store.
2. Produce stable engineered features for both CLR and BOCPD consumers.
3. Capture user labels for supervised CLR training.
4. Train CLR and BOCPD inside AppDaemon and persist model artifacts/state.
5. Expose versioned SQLite contracts for integrations to pull safely.

## SQLite Data Model (v1)

Core tables:

- `raw_events`
- `features`
- `labels`
- `clr_training_runs`
- `clr_model_artifacts`
- `bocpd_training_runs`
- `bocpd_model_state`
- `metadata`

Recommended contract views:

- `vw_clr_training_dataset`
- `vw_clr_latest_model_artifact`
- `vw_bocpd_feature_stream`
- `vw_bocpd_latest_state`
- `vw_latest_feature_snapshot`

Metadata keys:

- `schema_version`
- `feature_set_version`
- `contract_version`
- `last_ingest_at`
- `last_feature_compute_at`
- `last_clr_train_at`
- `last_bocpd_train_at`

## Build Plan (TDD Sequence)

### Task 1: Scaffold data layer project

- Test to write: `tests/test_scaffold.py`
- Implement:
  - `projects/ha_ml_data_layer/appdaemon_ml/` package
  - modules: `app.py`, `config.py`, `db.py`, `ingest.py`, `features.py`, `labels.py`, `clr_train.py`, `bocpd_train.py`, `contracts.py`
  - `tests/` folder
- Verify:
  - fail: scaffold test fails due to missing modules
  - pass: scaffold test passes after minimal files exist

### Task 2: SQLite bootstrap and migrations

- Test to write: `tests/test_schema_v1.py`
- Implement:
  - `ensure_schema(target_version=1)`
  - migration metadata writes (`schema_version`, migration status)
  - SQLite pragmas: WAL, synchronous mode, busy timeout
- Verify:
  - fail: schema test fails with missing tables/indexes
  - pass: schema test passes with v1 tables and metadata

### Task 3: Raw event ingest pipeline

- Test to write: `tests/test_ingest_raw_events.py`
- Implement:
  - AppDaemon callback ingestion
  - `record_raw_event()` write path
  - dedupe key strategy and UTC normalization
- Verify:
  - fail: no rows/dedupe behavior incorrect
  - pass: expected rows inserted and dedupe confirmed

### Task 4: Feature engineering pipeline

- Test to write: `tests/test_feature_engineering.py`
- Implement:
  - deterministic windowed feature computation
  - write feature rows with `window_start`, `window_end`, `feature_set_version`, `computed_at`
- Verify:
  - fail: fixture feature values mismatch
  - pass: computed feature values match expected fixtures

### Task 5: Label capture from dashboard helpers

- Test to write: `tests/test_label_capture.py`
- Implement:
  - read `input_datetime.sleep_start` and `input_datetime.sleep_end`
  - normalize to absolute UTC bounds plus local date/timezone context
  - write into `labels`
- Verify:
  - fail: invalid date handling/cross-midnight mismatch
  - pass: labels stored correctly including overnight cases

### Task 6: Anti-leakage pairing rules

- Test to write: `tests/test_label_feature_pairing_rules.py`
- Implement:
  - strict temporal join rules ensuring `feature.window_end <= label_end`
  - duplicate/overlap guards where required
- Verify:
  - fail: leakage rows included
  - pass: only valid feature-label pairs remain

### Task 7: CLR training in AppDaemon

- Test to write: `tests/test_clr_training_job.py`
- Implement:
  - build CLR training matrix from SQLite contracts
  - apply training gates (minimum labeled rows / days)
  - fit logistic regression and persist model artifact + run record
- Verify:
  - fail: training gate/artifact assertions fail
  - pass: training run recorded and artifact fields persisted

### Task 8: BOCPD training/state update in AppDaemon

- Test to write: `tests/test_bocpd_training_state_job.py`
- Implement:
  - build BOCPD input stream from features
  - run BOCPD state updates/training cycle
  - persist latest state artifact + run record
- Verify:
  - fail: expected state progression not persisted
  - pass: state and run metadata are stored correctly

### Task 9: CLR consumer contracts (pull-only)

- Test to write: `tests/test_view_clr_latest_artifact.py`
- Implement:
  - `vw_clr_training_dataset`
  - `vw_clr_latest_model_artifact`
- Verify:
  - fail: missing columns/order/shape
  - pass: view schemas and sample rows match contract

### Task 10: BOCPD consumer contracts (pull-only)

- Test to write: `tests/test_view_bocpd_latest_state.py`
- Implement:
  - `vw_bocpd_feature_stream`
  - `vw_bocpd_latest_state`
- Verify:
  - fail: state/stream contract mismatch
  - pass: integrations can query stable BOCPD inputs and latest state

### Task 11: Integration pull contract stability

- Test to write: `tests/test_integration_pull_contracts.py`
- Implement:
  - explicit contract versioning via `metadata.contract_version`
  - stable field naming/types for CLR and BOCPD pull paths
- Verify:
  - fail: version/type invariants break
  - pass: contracts are stable and versioned

### Task 12: Retention and maintenance jobs

- Test to write: `tests/test_retention_maintenance.py`
- Implement:
  - trim policy for `raw_events` and `features`
  - preserve labels and model artifact continuity
- Verify:
  - fail: contracts break after trim
  - pass: data trimmed and views remain valid

### Task 13: Diagnostics and readiness metrics

- Test to write: `tests/test_diagnostics.py`
- Implement:
  - readiness counters and timestamps
  - degraded-state markers for failed ingest/training cycles
- Verify:
  - fail: metrics missing/inaccurate
  - pass: diagnostics report expected values under fixtures

### Task 14: End-to-end flow

- Test to write: `tests/test_e2e_data_layer_flow.py`
- Implement:
  - wire app lifecycle (`initialize`, callbacks, schedules)
  - execute full nightly pipeline (ingest -> features -> labels -> CLR train -> BOCPD update)
- Verify:
  - fail: one or more pipeline stages not producing contract outputs
  - pass: full e2e test and full suite pass

## Operational Notes

- Use one SQLite DB file (for example `ha_ml_data_layer.db`) for all ML-layer data and artifacts.
- Keep writes in AppDaemon jobs short and transactional.
- Maintain deterministic feature definitions; bump `feature_set_version` on feature changes.
- Integrations must never mutate ML-layer tables directly; they only read from contract views/tables.
