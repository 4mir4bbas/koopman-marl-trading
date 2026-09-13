from __future__ import annotations

from src.data.loader import load_ohlcv
from src.data.walk_forward import (
    generate_expanding_annual_folds,
)


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    folds = generate_expanding_annual_folds(
        data=data,
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=30,
        training_start="2015-01-01",
    )

    print("Walk-forward folds")
    print("------------------")

    for fold in folds:
        print(
            f"{fold.name} | "
            f"train: {len(fold.train):>4} rows "
            f"({fold.train.index.min().date()} -> "
            f"{fold.train.index.max().date()}) | "
            f"context: {len(fold.context):>2} rows | "
            f"eval: {len(fold.evaluation):>3} rows "
            f"({fold.evaluation.index.min().date()} -> "
            f"{fold.evaluation.index.max().date()})"
        )

    print()
    print(
        "Final walk-forward evaluation ends at:",
        folds[-1].evaluation.index.max(),
    )

    print(
        "Data continues until:",
        data.index.max(),
    )

    print(
        "Rows after walk-forward development period:",
        int(
            (
                data.index
                > folds[-1].evaluation.index.max()
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()