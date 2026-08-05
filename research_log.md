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


# Date: 2026-08-05

Milestone: First reproducible PPO baseline

Training protocol:
- PPO is trained exclusively on the chronological training split.
- Periodic evaluation is performed on the validation split.
- The test split remains untouched.
- The best model is selected using validation episode reward.
- Deterministic evaluation is used.
- CPU execution is explicitly configured.
- Global random seed is set to 42.

Initial PPO configuration:
- Total timesteps: 100,000
- Learning rate: 0.0003
- Rollout length: 1,024
- Batch size: 64
- Epochs per update: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.20
- Policy network: [128, 128]
- Value network: [128, 128]

Reproducibility:
- Experiment settings are stored as JSON.
- TensorBoard training logs are generated.
- Periodic checkpoints are generated locally.
- Large model artifacts are excluded from Git.
- Validation metrics and equity curves are retained.

Known limitations:
- Training repeatedly uses one deterministic historical trajectory.
- Validation contains one deterministic episode.
- Best-model selection is based on total validation reward rather
  than a risk-adjusted financial metric.
- Results from one random seed are not statistically meaningful.
- Execution still occurs at the current candle close.

Next methodological step:
Introduce random-start fixed-length episodes before extensive PPO
tuning or comparison.


# Date: 2026-08-05

Milestone: Random-start fixed-length training episodes

Environment changes:
- Training episodes use random historical starting indices.
- Training episode length is fixed at 365 transitions.
- Validation uses a deterministic fixed start.
- Validation covers the complete validation split.
- Episodes cannot cross the boundary of their assigned data split.
- Dataset or episode exhaustion is reported as truncation.
- Portfolio depletion is treated as natural termination.
- Reset options can override the episode starting index.

Reproducibility:
- Random starts use Gymnasium's internally seeded NumPy generator.
- Identical reset seeds produce identical episode starts.
- Different seeds can produce different historical samples.
- Validation start is independent of the random seed.

Methodological motivation:
The previous implementation repeatedly exposed PPO to one complete
historical trajectory. Random fixed-length training episodes reduce
memorization of a single time path and expose the policy to multiple
historical subperiods.

Known limitations:
- Historical subperiods are sampled uniformly by start index.
- Market regimes are not explicitly detected or balanced.
- Highly persistent regimes may dominate the episode distribution.
- Validation still consists of one deterministic historical path.
- Current-close execution bias remains unresolved.

Research direction:
A future regime-aware episode sampler could balance or deliberately
stress the policy across trend, volatility, liquidity, and structural
break regimes. This should first be developed as an evaluation and
sampling mechanism before being claimed as an algorithmic contribution.