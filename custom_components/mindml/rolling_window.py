"""Rolling window feature computation for real-time inference."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta


class RollingWindowTracker:

    def __init__(
        self,
        *,
        window_hours: float = 7.0,
        feature_states: dict[str, str] | None = None,
    ) -> None:
        self._window_hours = window_hours
        self._feature_states: dict[str, str] = dict(feature_states) if feature_states else {}
        self._events: deque[tuple[datetime, str, str]] = deque()

    @property
    def event_count(self) -> int:
        return len(self._events)

    def record_event(self, entity_id: str, state: str) -> None:
        if self._feature_states:
            expected_state = self._feature_states.get(entity_id)
            if expected_state is None or expected_state != state:
                return
        self._events.append((datetime.now(UTC), entity_id, state))

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=self._window_hours)
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def compute_features(self, required_features: list[str]) -> dict[str, float]:
        self._prune(datetime.now(UTC))
        event_count = len(self._events)
        on_count = sum(1 for _, _, state in self._events if state == "on")
        on_ratio = (on_count / event_count) if event_count else 0.0

        return {
            "event_count": float(event_count),
            "on_ratio": float(on_ratio),
        }
