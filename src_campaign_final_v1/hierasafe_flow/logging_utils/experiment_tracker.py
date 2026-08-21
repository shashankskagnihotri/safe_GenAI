from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

from hierasafe_flow.utils.io import ensure_dir, write_json, write_yaml


@dataclass
class ExperimentTracker:
    output_dir: Path
    config: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time)

    @classmethod
    def create(cls, output_dir: str | Path, config: dict[str, Any]) -> "ExperimentTracker":
        path = ensure_dir(output_dir)
        tracker = cls(output_dir=path, config=config)
        write_yaml(path / "resolved_config.yaml", config)
        return tracker

    def log_event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append({"time": time(), "name": name, "payload": payload or {}})
        self.flush()

    def flush(self) -> None:
        write_json(
            self.output_dir / "events.json",
            {
                "started_at": self.started_at,
                "events": self.events,
            },
        )

