from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import download_ohlcv, save_ohlcv


DEFAULT_OUTPUT = Path("data/raw/btc_usd_1d.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and save Bitcoin OHLCV data."
    )

    parser.add_argument(
        "--symbol",
        default="BTC-USD",
        help="Yahoo Finance ticker symbol.",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Starting date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Exclusive ending date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    data = download_ohlcv(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )

    output_path = save_ohlcv(data, args.output)

    print(f"Downloaded rows: {len(data):,}")
    print(f"First timestamp: {data.index.min()}")
    print(f"Last timestamp:  {data.index.max()}")
    print(f"Saved to: {output_path.resolve()}")
    print()
    print(data.head())
    print()
    print(data.tail())


if __name__ == "__main__":
    main()
