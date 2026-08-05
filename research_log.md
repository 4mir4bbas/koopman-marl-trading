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


# Date: 2026-08-05

Milestone: Chronological data split and non-RL baselines

Data split:
- Training: 70%
- Validation: 15%
- Test: 15%
- Splits are chronological and never shuffled.
- Test data is reserved for final model evaluation.
- Initial baseline comparisons use validation data.

Implemented baselines:
- Cash-only policy
- Buy-and-hold policy
- Random policy over 30 independent seeds

Performance metrics:
- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Trade count
- Transaction costs

Methodological decisions:
- Crypto returns are annualized using 365 periods per year.
- The initial risk-free rate is assumed to be zero.
- Random-policy performance is reported as a distribution rather
  than a single run.
- Dataset exhaustion is modeled as truncation rather than natural
  termination.
- Small tabular experiment outputs are version controlled.
- Large checkpoints and binary artifacts remain excluded from Git.

Research note:
The current work establishes evaluation infrastructure and does not
constitute a scientific contribution. PPO must outperform simple
financial baselines out of sample and after transaction costs.