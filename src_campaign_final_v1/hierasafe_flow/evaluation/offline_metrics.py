from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class OfflineMetric(Protocol):
    name: str

    def evaluate(self, output_path: Path, metadata: dict[str, Any]) -> dict[str, float]:
        ...


@dataclass
class MetricResult:
    metric_name: str
    values: dict[str, float]


def run_offline_metrics(
    output_path: str | Path,
    metadata: dict[str, Any],
    metrics: list[OfflineMetric],
) -> list[MetricResult]:
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Output path for offline evaluation does not exist: {path}")
    results = []
    for metric in metrics:
        results.append(MetricResult(metric_name=metric.name, values=metric.evaluate(path, metadata)))
    return results


def no_generation_loop_safety_model_notice() -> str:
    return (
        "Offline metrics may use external tools only after generation has completed. "
        "The core HieraSafe-Flow generation loop uses only frozen generator vector fields."
    )

