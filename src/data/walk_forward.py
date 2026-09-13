from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train: pd.DataFrame
    context: pd.DataFrame
    evaluation: pd.DataFrame

    @property
    def evaluation_environment_data(
        self,
    ) -> pd.DataFrame:
        """
        Data passed to the evaluation environment.

        Context rows are included only to construct the
        first observation window. Performance measurement
        begins at the first evaluation row.
        """
        return pd.concat(
            [
                self.context,
                self.evaluation,
            ]
        ).copy()

    @property
    def evaluation_start_index(self) -> int:
        """
        Index of the first scored evaluation observation
        inside evaluation_environment_data.
        """
        return len(self.context)

    @property
    def evaluation_episode_length(self) -> int:
        """
        Number of transitions from the first evaluation
        observation through the final evaluation row.
        """
        return len(self.evaluation) - 1


def _to_utc_timestamp(
    value: str | pd.Timestamp,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def generate_expanding_annual_folds(
    data: pd.DataFrame,
    *,
    first_evaluation_year: int,
    final_evaluation_end: str | pd.Timestamp,
    window_size: int,
    training_start: str | pd.Timestamp | None = None,
) -> list[WalkForwardFold]:
    """
    Create annual expanding-window walk-forward folds.

    Each fold contains:

    - all historical training data before the evaluation year,
    - window_size - 1 context rows immediately before evaluation,
    - the evaluation period itself.

    Context rows belong to historical training data and are
    provided only so the first evaluation observation has a
    complete causal lookback window.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Cannot generate folds from empty data."
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "data must use a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index must be sorted "
            "chronologically."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "Data index contains duplicate timestamps."
        )

    if window_size < 2:
        raise ValueError(
            "window_size must be at least 2."
        )

    final_end = _to_utc_timestamp(
        final_evaluation_end
    )

    if training_start is None:
        train_start = data.index.min()
    else:
        train_start = _to_utc_timestamp(
            training_start
        )

    first_evaluation_start = pd.Timestamp(
        year=first_evaluation_year,
        month=1,
        day=1,
        tz="UTC",
    )

    if train_start >= first_evaluation_start:
        raise ValueError(
            "training_start must be earlier than "
            "the first evaluation period."
        )

    if final_end < first_evaluation_start:
        raise ValueError(
            "final_evaluation_end must not precede "
            "the first evaluation period."
        )

    context_size = window_size - 1

    folds: list[WalkForwardFold] = []

    for year in range(
        first_evaluation_year,
        final_end.year + 1,
    ):
        evaluation_start = pd.Timestamp(
            year=year,
            month=1,
            day=1,
            tz="UTC",
        )

        calendar_year_end = pd.Timestamp(
            year=year,
            month=12,
            day=31,
            tz="UTC",
        )

        evaluation_end = min(
            calendar_year_end,
            final_end,
        )

        train = data.loc[
            (data.index >= train_start)
            & (data.index < evaluation_start)
        ].copy()

        evaluation = data.loc[
            (data.index >= evaluation_start)
            & (data.index <= evaluation_end)
        ].copy()

        if len(train) < context_size:
            raise ValueError(
                f"Fold {year} contains only "
                f"{len(train)} training rows, but "
                f"{context_size} context rows are "
                "required."
            )

        if evaluation.empty:
            raise ValueError(
                f"Fold {year} contains no "
                "evaluation observations."
            )

        context = train.tail(
            context_size
        ).copy()

        if (
            train.index.max()
            >= evaluation.index.min()
        ):
            raise RuntimeError(
                f"Training and evaluation overlap "
                f"in fold {year}."
            )

        folds.append(
            WalkForwardFold(
                name=str(year),
                train=train,
                context=context,
                evaluation=evaluation,
            )
        )

    return folds