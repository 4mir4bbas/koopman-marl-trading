from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.walk_forward import (
    generate_expanding_annual_folds,
)


def create_market() -> pd.DataFrame:
    index = pd.date_range(
        start="2015-01-01",
        end="2026-08-04",
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


def create_folds():
    return generate_expanding_annual_folds(
        create_market(),
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=30,
        training_start="2015-01-01",
    )


def test_walk_forward_has_expected_folds():
    folds = create_folds()

    assert [
        fold.name
        for fold in folds
    ] == [
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
    ]

    first = folds[0]

    assert first.train.index.min() == pd.Timestamp(
        "2015-01-01",
        tz="UTC",
    )

    assert first.train.index.max() == pd.Timestamp(
        "2017-12-31",
        tz="UTC",
    )

    assert (
        first.evaluation.index.min()
        == pd.Timestamp(
            "2018-01-01",
            tz="UTC",
        )
    )

    assert (
        first.evaluation.index.max()
        == pd.Timestamp(
            "2018-12-31",
            tz="UTC",
        )
    )

    last = folds[-1]

    assert last.train.index.max() == pd.Timestamp(
        "2023-12-31",
        tz="UTC",
    )

    assert (
        last.evaluation.index.min()
        == pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        )
    )

    assert (
        last.evaluation.index.max()
        == pd.Timestamp(
            "2024-11-05",
            tz="UTC",
        )
    )


def test_fold_context_is_exactly_observation_history():
    folds = create_folds()

    for fold in folds:
        assert len(fold.context) == 29

        pd.testing.assert_frame_equal(
            fold.context,
            fold.train.tail(29),
        )

        assert (
            fold.context.index.max()
            < fold.evaluation.index.min()
        )

        environment_data = (
            fold.evaluation_environment_data
        )

        assert (
            fold.evaluation_start_index
            == 29
        )

        assert (
            environment_data.index[
                fold.evaluation_start_index
            ]
            == fold.evaluation.index.min()
        )

        assert (
            fold.evaluation_episode_length
            == len(fold.evaluation) - 1
        )


def test_evaluation_periods_do_not_overlap():
    folds = create_folds()

    for previous, current in zip(
        folds,
        folds[1:],
    ):
        assert (
            previous.evaluation.index.max()
            < current.evaluation.index.min()
        )


def test_final_test_period_is_never_used():
    folds = create_folds()

    final_test_start = pd.Timestamp(
        "2024-11-06",
        tz="UTC",
    )

    for fold in folds:
        assert (
            fold.train.index.max()
            < final_test_start
        )

        assert (
            fold.context.index.max()
            < final_test_start
        )

        assert (
            fold.evaluation.index.max()
            < final_test_start
        )


def test_rejects_unsorted_data():
    data = create_market().sort_index(
        ascending=False
    )

    with pytest.raises(
        ValueError,
        match="chronologically",
    ):
        generate_expanding_annual_folds(
            data,
            first_evaluation_year=2018,
            final_evaluation_end="2024-11-05",
            window_size=30,
            training_start="2015-01-01",
        )