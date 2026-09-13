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

# Date: 2026-08-12

Milestone: Causal next-open execution model

Execution protocol:
- The observation includes OHLCV information through candle t.
- The agent makes its decision after candle t is fully observed.
- The selected action is executed at Open(t+1).
- Portfolio valuation after execution uses Close(t+1).
- Reward measures the log portfolio return from Close(t) to
  Close(t+1).

Observation changes:
- The rolling observation window now includes the current candle.
- For a window size of 30, the first valid decision index is 29.
- No future market information is included in an observation.

Bias prevention:
- The agent cannot capture an overnight price gap that occurs before
  its order is executed.
- A dedicated synthetic gap test verifies causal execution timing.
- Previous PPO models trained under same-close execution are treated
  as methodologically obsolete.

Current execution assumptions:
- Trades execute exactly at the next candle open.
- Transaction costs are modeled.
- Slippage is not yet modeled.
- Bid-ask spread is not yet modeled.
- Market impact and liquidity constraints are not yet modeled.

Research significance:
This change does not constitute an algorithmic contribution. It
strengthens the validity of the experimental framework by removing
same-candle execution bias and establishing a causal relationship
between observations, actions, executions, and rewards.

# Date: 2026-08-12

Milestone: Multi-seed PPO experiment management and validation

Experiment design:
- PPO baseline is evaluated across multiple independent random seeds.
- Initial experiment seeds: 42, 123, and 2026.
- Each seed has an independent training run, best validation model,
  configuration file, metrics, and equity curve.
- Experiment outputs are organized by experiment name and seed.
- The held-out test split remains completely untouched.

Experiment management:
- Experiment-specific model and result directories were introduced.
- Training was refactored into a reusable train_single_seed function.
- PPO evaluation accepts the seed as a command-line argument.
- Aggregate statistics are generated across seeds.
- Small tabular experiment results are retained in Git.
- Model checkpoints, TensorBoard logs, and binary evaluation artifacts
  remain excluded from Git.

Validation results:
- Mean total return: 104.50%
- Total return standard deviation: 33.50%
- Mean annualized return: 53.75%
- Mean Sharpe ratio: 1.296
- Sharpe ratio standard deviation: 0.278
- Mean Sortino ratio: 1.799
- Mean maximum drawdown: -30.90%
- Mean Calmar ratio: 1.879
- Mean trade count: 128.3
- Mean cumulative transaction cost: 1,995.86 USD

Baseline comparison:
- Buy-and-hold validation return: 212.73%
- Buy-and-hold Sharpe ratio: 1.678
- Buy-and-hold maximum drawdown: -26.18%
- Random-policy mean return: 39.28%
- Random-policy mean Sharpe ratio: 0.705

Interpretation:
- PPO consistently outperforms the random-policy baseline.
- PPO does not outperform buy-and-hold on the current validation period.
- Performance varies across seeds but does not appear extremely unstable.
- High trade count and cumulative transaction costs indicate that the
  current PPO policy may be overtrading.
- The current validation period strongly favors long Bitcoin exposure,
  so a single validation trajectory is insufficient for robust conclusions.

Research conclusion:
The PPO baseline demonstrates non-random learning behavior but provides
no evidence yet that reinforcement learning is superior to a simple
buy-and-hold strategy. More robust temporal evaluation is required
before introducing representation-learning or Koopman components.

Next step:
Implement walk-forward validation and analyze performance across
different market regimes.

# Date: 2026-08-28
## Strong Rule-Based Baselines and Metric Validation

### Objective

Strengthen the experimental baseline suite before introducing Koopman-based representations or multi-agent reinforcement learning.

The goal is to ensure that future RL models are compared not only against trivial strategies such as cash, random actions, and buy-and-hold, but also against simple rule-based trading strategies that can exploit market trends.

### Changes

* Corrected the Sortino ratio implementation to use downside deviation based on the root mean squared negative deviation from the target return.
* Added automated pytest infrastructure.
* Added unit tests for:

  * financial performance metrics,
  * causal trade execution,
  * environment accounting,
  * deterministic and random episode sampling.
* Replaced the previous manually executed environment validation scripts with automated pytest tests.
* Added two rule-based financial baselines:

  * 10/30-day moving-average crossover,
  * 20-day time-series momentum.
* Restricted rule-based strategies to historical information available within the RL agent's observation window.
* Preserved causal execution: signals are generated using information available through Close(t), while trades execute at Open(t+1).

### Validation Results

Evaluation period:

* Validation split: 2023-02-11 to 2024-11-05.
* Initial portfolio value: $10,000.
* Transaction cost: 0.1%.

Results:

| Strategy           | Total Return | Sharpe | Sortino | Maximum Drawdown | Trades |
| ------------------ | -----------: | -----: | ------: | ---------------: | -----: |
| Buy and Hold       |      212.73% | 1.6780 |  2.7190 |          -26.18% |      1 |
| 20-day Momentum    |      101.10% | 1.3484 |  2.2035 |          -34.95% |     57 |
| MA 10/30 Crossover |       70.12% | 1.0430 |  1.6166 |          -36.84% |     21 |
| Cash               |        0.00% |    N/A |     N/A |            0.00% |      0 |

The 30-run random baseline produced a mean total return of 39.28%, a mean Sharpe ratio of 0.705, and a mean maximum drawdown of -29.92%.

### Interpretation

Buy-and-hold strongly outperformed both rule-based strategies during the current validation period.

This period contains a strong upward Bitcoin market trend, so the result should not be interpreted as evidence that buy-and-hold is universally superior. Instead, it demonstrates that evaluation on a single market trajectory can strongly favor strategies whose exposure matches the dominant regime.

The 20-day momentum baseline produced substantially higher turnover than the moving-average strategy, executing 57 trades and incurring approximately $1,035 in transaction costs.

The existing PPO baseline has previously achieved approximately 104.5% mean validation return across three seeds, placing its raw-return performance close to the 20-day momentum baseline and substantially below buy-and-hold. This suggests that the current PPO agent has not yet demonstrated behavior clearly superior to a simple trend-following rule.

These findings reinforce the need for multi-window or walk-forward validation before evaluating more complex models.

### Methodological Decision

The MA 10/30 and 20-day momentum parameters will not be tuned against the current validation interval.

Repeatedly searching for better rule parameters on the same validation trajectory would introduce validation-set overfitting and weaken the fairness of future comparisons.

The next experimental milestone is therefore to construct a multi-window / walk-forward evaluation protocol that measures strategy performance across different market conditions before introducing Koopman or MARL components.

# Date: 2026-09-13
## Walk-Forward Evaluation Protocol v1

### Objective

Replace the single-path validation setup with a multi-window chronological evaluation protocol before introducing Koopman-based representations or MARL.

The previous validation period covered only one historical trajectory and was strongly influenced by the dominant Bitcoin market regime during that interval. A single validation trajectory is insufficient for assessing robustness under financial non-stationarity.

The final test period remains untouched.

### Development and Final Test Separation

The dataset is now conceptually divided into:

* Development period: 2015-01-01 to 2024-11-05
* Final untouched test: 2024-11-06 to 2026-08-04

The final test contains 636 observations and is excluded from all walk-forward folds.

It must not be used for:

* hyperparameter tuning,
* model selection,
* architecture decisions,
* baseline parameter selection,
* feature selection.

### Walk-Forward Design

An expanding-window evaluation protocol was selected.

The first evaluation year is 2018, providing approximately three full years of historical data before the first evaluation window.

Each subsequent fold adds the previous evaluation year to the available historical training data.

The folds are:

| Fold | Training Period          | Evaluation Period        |
| ---- | ------------------------ | ------------------------ |
| 2018 | 2015-01-01 to 2017-12-31 | 2018-01-01 to 2018-12-31 |
| 2019 | 2015-01-01 to 2018-12-31 | 2019-01-01 to 2019-12-31 |
| 2020 | 2015-01-01 to 2019-12-31 | 2020-01-01 to 2020-12-31 |
| 2021 | 2015-01-01 to 2020-12-31 | 2021-01-01 to 2021-12-31 |
| 2022 | 2015-01-01 to 2021-12-31 | 2022-01-01 to 2022-12-31 |
| 2023 | 2015-01-01 to 2022-12-31 | 2023-01-01 to 2023-12-31 |
| 2024 | 2015-01-01 to 2023-12-31 | 2024-01-01 to 2024-11-05 |

Expanding rather than rolling training windows were selected for the initial protocol in order to avoid introducing an additional arbitrary training-window-length hyperparameter.

Rolling-window training may later be studied explicitly as an adaptation mechanism for non-stationary environments.

### Observation Context

The trading environment uses a 30-observation state window.

For each evaluation fold, the 29 immediately preceding historical rows are therefore supplied as context.

For example:

29 historical context rows + first evaluation row = 30-row observation.

These context rows are not part of the scored evaluation period. They exist only to construct the first causal observation without discarding approximately one month from each evaluation fold.

This preserves the causal information structure:

information through Close(t)
→ action decision
→ execution at Open(t+1).

### PPO Model-Selection Decision

Outer walk-forward evaluation windows must not be used repeatedly for checkpoint selection.

The current PPO implementation uses validation-based checkpoint selection through an EvalCallback. That procedure must not be applied directly to outer walk-forward evaluation folds, because repeatedly selecting the best checkpoint using an outer fold would invalidate its out-of-sample interpretation.

For the first PPO walk-forward baseline, the intended protocol is:

historical fold training data
→ fixed training budget
→ final PPO model
→ one evaluation on the corresponding outer fold.

Existing PPO hyperparameters will initially remain frozen.

Any future hyperparameter tuning should use an inner training/validation procedure rather than the outer evaluation fold.

### Implementation

Added a walk-forward fold generator that explicitly separates:

* training data,
* historical observation context,
* evaluation data.

The generator enforces chronological ordering and prevents overlap between training and evaluation data.

Each fold exposes the evaluation environment data as:

context + evaluation.

The first scored evaluation observation is explicitly identified after the context rows.

### Automated Validation

Five automated tests were added for the walk-forward infrastructure.

The tests verify that:

* exactly seven expected folds are produced,
* fold boundaries match the intended calendar periods,
* each fold contains exactly 29 context rows,
* context rows are the final historical observations before evaluation,
* evaluation periods do not overlap,
* the final test period is absent from all folds,
* unsorted chronological data is rejected.

After these additions, the complete automated test suite contains:

18 passing tests.

### Real-Data Sanity Check

The walk-forward generator was executed on the actual BTC-USD daily dataset.

Observed folds:

* 2018: 1096 training rows, 29 context rows, 365 evaluation rows
* 2019: 1461 training rows, 29 context rows, 365 evaluation rows
* 2020: 1826 training rows, 29 context rows, 366 evaluation rows
* 2021: 2192 training rows, 29 context rows, 365 evaluation rows
* 2022: 2557 training rows, 29 context rows, 365 evaluation rows
* 2023: 2922 training rows, 29 context rows, 365 evaluation rows
* 2024: 3287 training rows, 29 context rows, 310 evaluation rows

The final walk-forward evaluation observation is:

2024-11-05.

The dataset continues through:

2026-08-04.

There are 636 observations after the walk-forward development period, confirming that the final test period remains isolated.

### Next Step

Run the fixed rule-based strategies independently on all seven walk-forward evaluation folds.

The initial walk-forward benchmark suite will include:

* Cash
* Buy and Hold
* MA 10/30 crossover
* 20-day momentum
* Random policy

No baseline parameters will be tuned separately for individual folds.

The goal is to measure how strongly strategy performance varies across historical market conditions before introducing PPO walk-forward training.


# Date: 2026-09-13
## Walk-Forward Rule-Based Baseline Evaluation

### Objective

Evaluate the fixed financial baselines across multiple historical market periods using the newly introduced expanding-window walk-forward protocol.

The purpose of this experiment was to determine whether conclusions obtained from the previous single validation trajectory were robust across substantially different Bitcoin market conditions.

No strategy parameters were tuned separately for individual folds.

The following parameters remained frozen:

* Moving-average crossover: 10/30 days
* Momentum: 20-day lookback
* Observation window: 30 days
* Transaction cost: 0.1%
* Initial portfolio value: $10,000

The final test period beginning on 2024-11-06 remained untouched.

### Implementation

A reusable evaluation/backtesting module was introduced so that standard validation and walk-forward evaluation use the same episode execution logic.

The common evaluation runner handles:

* deterministic environment creation,
* fixed evaluation start indices,
* fixed evaluation episode lengths,
* portfolio value collection,
* performance metric calculation.

The previous single-window baseline evaluation script was refactored to reuse this common implementation.

A walk-forward baseline evaluation script was added for the seven development folds from 2018 through 2024.

For every fold, the following strategies were evaluated:

* Cash
* Buy and Hold
* MA 10/30 crossover
* 20-day Momentum
* Random policy

The random strategy was evaluated using 30 seeds per fold.

Random-run results were preserved separately from per-fold averages so that market periods, rather than individual random seeds, remain the primary unit of across-fold comparison.

### Automated Tests

Two additional automated tests were added for the common backtest runner.

The complete project test suite now contains 20 passing tests.

The new tests verify that:

* evaluation begins and ends at the requested episode boundaries,
* a buy-and-hold strategy produces positive returns on a deterministic increasing synthetic market.

### Walk-Forward Results

Per-fold total returns were:

| Year | Buy and Hold | MA 10/30 | Momentum 20 | Random Mean |
| ---- | -----------: | -------: | ----------: | ----------: |
| 2018 |      -72.56% |  -38.67% |     -32.59% |     -52.38% |
| 2019 |       86.70% |   71.36% |     117.62% |      27.69% |
| 2020 |      302.26% |  382.82% |     349.42% |      78.26% |
| 2021 |       57.47% |   17.85% |      66.75% |       9.56% |
| 2022 |      -65.33% |  -45.47% |     -47.96% |     -40.58% |
| 2023 |      153.96% |   75.80% |     119.58% |      45.12% |
| 2024 |       56.81% |    6.11% |       3.29% |      16.20% |

Across-fold results were:

| Strategy     | Mean Return | Median Return | Mean Sharpe | Worst Maximum Drawdown |
| ------------ | ----------: | ------------: | ----------: | ---------------------: |
| Buy and Hold |      74.19% |        57.47% |       0.804 |                -81.53% |
| MA 10/30     |      67.11% |        17.85% |       0.707 |                -50.45% |
| Momentum 20  |      82.30% |        66.75% |       0.902 |                -49.41% |
| Random       |      11.98% |        16.20% |       0.254 |                -64.33% |

The 20-day momentum strategy outperformed Buy and Hold by total return in five of the seven evaluation folds.

The MA 10/30 strategy outperformed Buy and Hold in three of seven folds.

### Interpretation

The walk-forward results substantially change the interpretation obtained from the previous single validation period.

During the previous validation trajectory, Buy and Hold strongly outperformed the rule-based strategies. However, across multiple historical periods, this superiority was not stable.

The 20-day momentum strategy currently provides the strongest simple rule-based benchmark.

It achieved:

* the highest mean total return,
* the highest median total return,
* the highest mean Sharpe ratio,
* a substantially smaller worst drawdown than Buy and Hold,
* higher total return than Buy and Hold in five of seven folds.

The results also demonstrate strong regime dependence.

During major declining periods such as 2018 and 2022, trend-following strategies reduced losses substantially relative to continuous market exposure.

During some persistent upward periods, particularly 2023 and 2024, the same trend-following logic lost a substantial amount of upside relative to Buy and Hold.

This indicates that a single fixed trading behavior is not consistently dominant across market conditions.

The result provides empirical motivation for studying adaptation under changing market dynamics, but it does not by itself demonstrate that any specific regime-detection or Koopman-based mechanism is appropriate.

### Methodological Note

The 2024 fold ends on 2024-11-05 and is therefore shorter than the complete calendar-year folds.

Consequently, raw total-return averages should not be treated as the sole across-fold performance measure.

Future comparisons should emphasize multiple complementary statistics including:

* annualized return,
* Sharpe ratio,
* maximum drawdown,
* per-fold benchmark-relative return,
* fold win rate.

### Baseline Decision

The 20-day momentum strategy should now be treated as a primary financial baseline for future RL experiments.

A future RL model should not be considered compelling merely because it beats a random policy or produces positive returns.

It should demonstrate meaningful performance relative to both:

* Buy and Hold,
* the fixed 20-day Momentum strategy,

across multiple walk-forward periods.

### Next Step

Extend the walk-forward framework to PPO.

PPO should be trained separately for every outer fold using only historical data available before that fold.

The outer evaluation period must not be used for checkpoint selection.

Initial PPO walk-forward development runs will use the existing frozen PPO configuration and a small number of seeds before scaling to the larger multi-seed experiment.
