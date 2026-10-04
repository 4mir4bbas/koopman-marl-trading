from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CalibrationSplit:
    train: pd.DataFrame
    context: pd.DataFrame
    validation: pd.DataFrame
    name: str = "single"

    @property
    def validation_environment_data(
        self,
    ) -> pd.DataFrame:
        return pd.concat(
            [
                self.context,
                self.validation,
            ]
        ).copy()

    @property
    def validation_start_index(self) -> int:
        return len(self.context)

    @property
    def validation_episode_length(self) -> int:
        return len(self.validation) - 1


def _utc_timestamp(
    value: str | pd.Timestamp,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def create_calibration_split(
    data: pd.DataFrame,
    *,
    window_size: int,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    name: str = "single",
) -> CalibrationSplit:
    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "data must use a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index must be sorted chronologically."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "Data index contains duplicate timestamps."
        )

    if window_size < 2:
        raise ValueError(
            "window_size must be at least 2."
        )

    train_start = _utc_timestamp(
        train_start
    )
    train_end = _utc_timestamp(
        train_end
    )
    validation_start = _utc_timestamp(
        validation_start
    )
    validation_end = _utc_timestamp(
        validation_end
    )

    if train_end >= validation_start:
        raise ValueError(
            "Training must end before validation begins."
        )

    train = data.loc[
        (data.index >= train_start)
        & (data.index <= train_end)
    ].copy()

    validation = data.loc[
        (data.index >= validation_start)
        & (data.index <= validation_end)
    ].copy()

    if train.empty:
        raise ValueError(
            "Calibration training data is empty."
        )

    if validation.empty:
        raise ValueError(
            "Calibration validation data is empty."
        )

    context_size = window_size - 1

    if len(train) < context_size:
        raise ValueError(
            "Training data does not contain enough "
            "rows for validation context."
        )

    context = train.tail(
        context_size
    ).copy()

    return CalibrationSplit(
        train=train,
        context=context,
        validation=validation,
        name=name,
    )


def generate_pre_2018_calibration_folds(
    data: pd.DataFrame,
    *,
    window_size: int = 30,
) -> list[CalibrationSplit]:
    """
    Generate expanding pre-2018 calibration folds.

    These folds are used only for baseline calibration
    and PPO-v2 model selection. No 2018+ outer
    walk-forward data is included.
    """

    specifications = [
        {
            "name": "C1",
            "train_start": "2015-01-01",
            "train_end": "2016-06-30",
            "validation_start": "2016-07-01",
            "validation_end": "2016-12-31",
        },
        {
            "name": "C2",
            "train_start": "2015-01-01",
            "train_end": "2016-12-31",
            "validation_start": "2017-01-01",
            "validation_end": "2017-06-30",
        },
        {
            "name": "C3",
            "train_start": "2015-01-01",
            "train_end": "2017-06-30",
            "validation_start": "2017-07-01",
            "validation_end": "2017-12-31",
        },
    ]

    folds: list[CalibrationSplit] = []

    for specification in specifications:
        folds.append(
            create_calibration_split(
                data=data,
                window_size=window_size,
                train_start=(
                    specification["train_start"]
                ),
                train_end=(
                    specification["train_end"]
                ),
                validation_start=(
                    specification["validation_start"]
                ),
                validation_end=(
                    specification["validation_end"]
                ),
                name=specification["name"],
            )
        )

    return folds