from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_walk_forward_ppo import (
    select_fold,
)
from src.data.walk_forward import (
    generate_expanding_annual_folds,
)


def create_market() -> pd.DataFrame:
    index = pd.date_range(
        start="2015-01-01",
        end="2024-11-05",
        freq="D",
        tz="UTC",
    )

    close = np.linspace(
        100.0,
        500.0,
        len(index),
    )

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(
                len(index),
                1_000.0,
            ),
        },
        index=index,
    )


def test_selects_requested_walk_forward_fold():
    folds = generate_expanding_annual_folds(
        data=create_market(),
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=30,
        training_start="2015-01-01",
    )

    fold = select_fold(
        folds,
        2018,
    )

    assert fold.name == "2018"

    assert (
        fold.train.index.max()
        == pd.Timestamp(
            "2017-12-31",
            tz="UTC",
        )
    )

    assert (
        fold.evaluation.index.min()
        == pd.Timestamp(
            "2018-01-01",
            tz="UTC",
        )
    )


def test_rejects_unknown_fold():
    folds = generate_expanding_annual_folds(
        data=create_market(),
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=30,
        training_start="2015-01-01",
    )

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        select_fold(
            folds,
            2017,
        )