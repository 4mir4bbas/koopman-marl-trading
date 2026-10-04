from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.calibration import (
    create_calibration_split,
)


from src.data.calibration import (
    create_calibration_split,
    generate_pre_2018_calibration_folds,
)


def create_market() -> pd.DataFrame:
    index = pd.date_range(
        start="2015-01-01",
        end="2018-12-31",
        freq="D",
        tz="UTC",
    )

    close = np.linspace(
        100.0,
        200.0,
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


def create_split():
    return create_calibration_split(
        create_market(),
        window_size=30,
        train_start="2015-01-01",
        train_end="2016-12-31",
        validation_start="2017-01-01",
        validation_end="2017-12-31",
    )


def test_calibration_boundaries():
    split = create_split()

    assert (
        split.train.index.min()
        == pd.Timestamp(
            "2015-01-01",
            tz="UTC",
        )
    )

    assert (
        split.train.index.max()
        == pd.Timestamp(
            "2016-12-31",
            tz="UTC",
        )
    )

    assert (
        split.validation.index.min()
        == pd.Timestamp(
            "2017-01-01",
            tz="UTC",
        )
    )

    assert (
        split.validation.index.max()
        == pd.Timestamp(
            "2017-12-31",
            tz="UTC",
        )
    )


def test_calibration_context_has_29_rows():
    split = create_split()

    assert len(split.context) == 29

    pd.testing.assert_frame_equal(
        split.context,
        split.train.tail(29),
    )

    assert (
        split.validation_start_index
        == 29
    )


def test_calibration_environment_starts_on_validation():
    split = create_split()

    environment_data = (
        split.validation_environment_data
    )

    assert (
        environment_data.index[
            split.validation_start_index
        ]
        == pd.Timestamp(
            "2017-01-01",
            tz="UTC",
        )
    )

    assert (
        split.validation_episode_length
        == len(split.validation) - 1
    )


def test_calibration_rejects_overlap():
    with pytest.raises(
        ValueError,
        match="Training must end",
    ):
        create_calibration_split(
            create_market(),
            window_size=30,
            train_start="2015-01-01",
            train_end="2017-06-01",
            validation_start="2017-01-01",
            validation_end="2017-12-31",
        )


def test_pre_2018_calibration_fold_boundaries():
    folds = generate_pre_2018_calibration_folds(
        create_market(),
        window_size=30,
    )

    assert [
        fold.name
        for fold in folds
    ] == [
        "C1",
        "C2",
        "C3",
    ]

    expected = {
        "C1": (
            "2015-01-01",
            "2016-06-30",
            "2016-07-01",
            "2016-12-31",
        ),
        "C2": (
            "2015-01-01",
            "2016-12-31",
            "2017-01-01",
            "2017-06-30",
        ),
        "C3": (
            "2015-01-01",
            "2017-06-30",
            "2017-07-01",
            "2017-12-31",
        ),
    }

    for fold in folds:
        (
            train_start,
            train_end,
            validation_start,
            validation_end,
        ) = expected[fold.name]

        assert fold.train.index.min() == pd.Timestamp(
            train_start,
            tz="UTC",
        )

        assert fold.train.index.max() == pd.Timestamp(
            train_end,
            tz="UTC",
        )

        assert (
            fold.validation.index.min()
            == pd.Timestamp(
                validation_start,
                tz="UTC",
            )
        )

        assert (
            fold.validation.index.max()
            == pd.Timestamp(
                validation_end,
                tz="UTC",
            )
        )


def test_pre_2018_folds_have_exact_context():
    folds = generate_pre_2018_calibration_folds(
        create_market(),
        window_size=30,
    )

    for fold in folds:
        assert len(fold.context) == 29

        pd.testing.assert_frame_equal(
            fold.context,
            fold.train.tail(29),
        )

        assert (
            fold.context.index.max()
            < fold.validation.index.min()
        )

        assert (
            fold.validation_environment_data.index[
                fold.validation_start_index
            ]
            == fold.validation.index.min()
        )


def test_calibration_folds_never_reach_2018():
    folds = generate_pre_2018_calibration_folds(
        create_market(),
        window_size=30,
    )

    cutoff = pd.Timestamp(
        "2018-01-01",
        tz="UTC",
    )

    for fold in folds:
        assert fold.train.index.max() < cutoff
        assert fold.context.index.max() < cutoff
        assert fold.validation.index.max() < cutoff


def test_calibration_validation_periods_do_not_overlap():
    folds = generate_pre_2018_calibration_folds(
        create_market(),
        window_size=30,
    )

    for previous, current in zip(
        folds,
        folds[1:],
    ):
        assert (
            previous.validation.index.max()
            < current.validation.index.min()
        )