from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    minimum_split_size: int = 32,
) -> TimeSeriesSplit:
    """
    Split time-series data chronologically without shuffling.

    The test ratio is inferred as:

        1 - train_ratio - validation_ratio

    Parameters
    ----------
    data:
        Time-indexed market data sorted chronologically.
    train_ratio:
        Fraction assigned to the training split.
    validation_ratio:
        Fraction assigned to the validation split.
    minimum_split_size:
        Minimum number of rows required in each split.

    Returns
    -------
    TimeSeriesSplit
        Independent train, validation, and test DataFrames.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Cannot split an empty dataset.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index must be sorted in chronological order."
        )

    if data.index.has_duplicates:
        raise ValueError("Data index contains duplicate timestamps.")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    if minimum_split_size < 1:
        raise ValueError("minimum_split_size must be positive.")

    number_of_rows = len(data)

    train_end = int(number_of_rows * train_ratio)
    validation_end = train_end + int(
        number_of_rows * validation_ratio
    )

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split_name, split_data in splits.items():
        if len(split_data) < minimum_split_size:
            raise ValueError(
                f"{split_name} split contains only "
                f"{len(split_data)} rows; at least "
                f"{minimum_split_size} rows are required."
            )

    if train.index.max() >= validation.index.min():
        raise RuntimeError(
            "Training and validation timestamps overlap."
        )

    if validation.index.max() >= test.index.min():
        raise RuntimeError(
            "Validation and test timestamps overlap."
        )

    return TimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
    )