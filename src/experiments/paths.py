from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    model_dir: Path
    result_dir: Path
    log_dir: Path
    evaluation_dir: Path
    validation_dir: Path


def get_experiment_paths(
    experiment_name: str,
    seed: int,
) -> ExperimentPaths:
    model_dir = Path(
        "models"
    ) / experiment_name / f"seed_{seed}"

    result_dir = Path(
        "results"
    ) / experiment_name / f"seed_{seed}"

    return ExperimentPaths(
        model_dir=model_dir,
        result_dir=result_dir,
        log_dir=result_dir / "logs",
        evaluation_dir=result_dir / "evaluation",
        validation_dir=result_dir / "validation",
    )