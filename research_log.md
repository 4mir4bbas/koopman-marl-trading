# 2026-08-03

## Completed
- Created project structure
- Implemented first trading environment

## Learned
- Gymnasium API
- RL environment design

## Next
- Implement data loader

# 2026-08-04

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
