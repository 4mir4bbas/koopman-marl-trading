# Date: 2026-08-03

## Completed
- Created project structure
- Implemented first trading environment

## Learned
- Gymnasium API
- RL environment design

## Next
- Implement data loader

# Date 2026-08-04

Milestone: Initial Bitcoin data pipeline

Decisions:
- Python 3.11 and venv are used instead of Conda.
- BTC-USD is selected as the first baseline asset.
- Daily OHLCV data is used for initial environment validation.
- Yahoo Finance is used only for the first reproducible baseline.
- Intraday Binance data will be introduced in a later experiment.
- Raw datasets are excluded from Git.

Implemented:
- OHLCV downloader
- Structural and financial data validation
- CSV persistence and loading

# Date: 2026-08-05

Milestone: Initial long-only Bitcoin trading environment

Environment formulation:
- Market: BTC-USD
- Frequency: Daily
- Position space: Cash or long Bitcoin
- Action space: Hold, Buy, Sell
- Observation: 30-day causally normalized OHLCV window
- Portfolio state: Cash ratio, Bitcoin ratio, position
- Initial balance: 10,000 USD
- Transaction cost: 0.1%
- Reward: Logarithmic portfolio return

Implementation decisions:
- No short selling or leverage in the first baseline.
- Buy and sell actions use the full available position.
- The environment follows the Gymnasium terminated/truncated API.
- Stable-Baselines3 check_env is used for interface validation.
- A deterministic synthetic-market test validates portfolio accounting.

Known limitations:
- Trades are executed at the current candle close.
- Slippage, liquidity, spread, and market impact are not yet modeled.
- Episodes currently cover the complete dataset.
- The environment is intended as infrastructure, not a scientific
  contribution.
