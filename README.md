# Bregma RL Portfolio

A sophisticated reinforcement learning framework for portfolio optimization that combines multiple RL algorithms with advanced feature engineering to maximize risk-adjusted returns in financial markets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

Bregma RL Portfolio uses deep reinforcement learning to optimize multi-asset portfolios by learning optimal asset allocation strategies that adapt to changing market conditions. The system employs an ensemble of RL algorithms (DQN, DDQN, A2C) with classifier-based selection to handle various market regimes.

## Key Features

- **Multiple RL Algorithms**: DQN, DDQN, A2C with extensible architecture
- **Ensemble Learning**: Classifier-based agent selection adapts to different market conditions
- **Advanced Feature Engineering**: Technical indicators, market regime detection, reversal signals
- **Risk Management**: Penalties for drawdowns and excessive trading
- **Performance Metrics**: Sharpe, Calmar, Max Drawdown, and more
- **Parallelized Processing**: Efficient handling of large datasets
- **Visualization Tools**: Comprehensive performance analysis and reporting

## Project Structure

```
.
├── config.yaml                  # Configuration file
├── main.py                      # Main entry point
├── src/
│   ├── agents/                  # RL agents
│   │   ├── base_agent.py        # Base agent class
│   │   ├── dqn.py               # Deep Q-Network
│   │   ├── ddqn.py              # Double Deep Q-Network
│   │   └── a2c.py               # Advantage Actor-Critic
│   ├── classifiers/             # Classifiers for ensemble methods
│   │   └── ensemble.py          # Classifier ensemble
│   ├── data/                    # Data handling
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── environment.py       # RL environment
│   └── utils/                   # Utilities
│       ├── config.py            # Config loading
│       └── replay_buffer.py     # Experience replay buffer
├── data/                        # Data directory
│   └── model68_hype_2024_bt_df4web.csv  # Historical stock data
├── checkpoints/                 # Model checkpoints
├── results/                     # Results and visualizations
└── logs/                        # Log files
```

## Installation

Choose the installation method that best suits your operating system and preferences.

### Linux

**Option 1: Using the setup script (recommended)**
```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual setup**
```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

**Option 1: Using the setup script (recommended)**
```cmd
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
setup_windows.bat
```

**Option 2: Manual setup**
```powershell
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS

#### Intel-based Macs

**Option 1: Using the setup script (recommended)**
```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual setup**
```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Apple Silicon (M1/M2/M3)

```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio
chmod +x setup_arm64.sh
./setup_arm64.sh
```

### Docker

We provide a Docker setup for easy deployment across any platform.

**Option 1: Using Docker Compose (recommended)**

```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio

# For CPU-only
docker-compose up bregma-rl-cpu

# For GPU support
docker-compose up bregma-rl
```

**Option 2: Using Docker directly**

```bash
git clone https://github.com/yourusername/bregma-rl-portfolio.git
cd bregma-rl-portfolio

# Build the Docker image
docker build -t bregma-rl-portfolio .

# Run with CPU
docker run -it --rm -v $(pwd):/app bregma-rl-portfolio

# Run with GPU (recommended for faster training)
docker run -it --rm --gpus all -v $(pwd):/app bregma-rl-portfolio
```

**Note**: Docker GPU support requires [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) on the host machine.

## Usage

### Basic Training

```bash
python main.py --config config.yaml
```

### Algorithm Selection

```bash
python main.py --algo DQN  # Choose from DQN, DDQN, A2C
```

### Dry Run (Debug Mode)

```bash
python main.py --dry-run
```

### Custom Episodes

```bash
python main.py --episodes 500
```

## Configuration

The `config.yaml` file controls all aspects of the system:

- Data paths and preprocessing settings
- Training parameters (learning rate, batch size, etc.)
- Algorithm-specific settings
- Ensemble and classifier options
- Checkpoint and evaluation settings

## Data Format

The system expects historical stock data with the following columns:
- Time (datetime)
- symbol (unique ticker)
- price (positive, non-zero)
- return_vs_spy (forward 7-day returns relative to S&P 500, used as target)
- smile (market sentiment indicator)
- btr (proprietary technical indicator)
- weight (portfolio weight, sums to ~1.0 per day)
- sector (stock sector category)
- industry (stock industry category)

## Feature Engineering

The system applies extensive feature engineering to create a rich set of features. All feature calculations are in `src/data/preprocessor.py` unless otherwise noted:

### Base Features
- **Log returns** (price changes) - Line 224: `stock_data['log_return'] = np.log(stock_data['price'] / stock_data['price'].shift(1))`
- **Standardized features** (z-score) - Lines 228-248: `stock_data[f'{col}_std'] = (stock_data[col] - mean) / std`
- **Lagged returns** (up to 10 days back) - Lines 250-251: `stock_data[f'log_return_lag{lag}'] = stock_data['log_return_std'].shift(lag)`
- **Moving averages** (5-day, 20-day) - Lines 255-257:
  ```python
  stock_data['MA5_return'] = stock_data['log_return_std'].rolling(window=5).mean()
  stock_data['MA20_return'] = stock_data['log_return_std'].rolling(window=20).mean()
  ```
- **MACD indicators** - Line 257: `stock_data['MACD_return'] = stock_data['MA5_return'] - stock_data['MA20_return']`

### Technical Indicators
All in `_calculate_technical_indicators` method (Lines 119-168):
- **RSI** (Relative Strength Index) - Lines 126-134
- **Bollinger Bands** (position, width) - Lines 138-146
- **Stochastic Oscillator** (%K, %D) - Lines 148-153
- **Rate of Change** (5, 10, 20 days) - Lines 155-157: `stock_data[f'ROC_{period}'] = price.pct_change(periods=period) * 100`
- **Volatility measures** - Lines 160-161:
  ```python
  stock_data['volatility_5d'] = returns.rolling(window=5).std()
  stock_data['volatility_20d'] = returns.rolling(window=20).std()
  ```

### Market Context
- **Sector and industry encoding** - Lines 316-329 in `one_hot_encode_categories` method
- **Market regime detection** (bull, bear, neutral) - Lines 170-216 in `_detect_market_regime` method
- **Reversal signals** - Lines 254-259:
  ```python
  stock_data[f'reversal_{lookback}d'] = -np.log(stock_data['price'] / stock_data['price'].shift(lookback))
  ```

## Optimization Targets

The system optimizes for multiple objectives:

### Primary Objective
- Maximize risk-adjusted portfolio returns

### Reward Function Components (in `src/data/environment.py`)
- **Portfolio return** (main driver) - Line 231: `base_reward = ((new_portfolio_value - old_portfolio_value) / old_portfolio_value)`
- **Transaction cost penalty** - Line 232: `transaction_cost_penalty = (total_transaction_cost / old_portfolio_value)`
- **Trading activity penalty** - Lines 235-236:
  ```python
  trading_activity = np.sum(np.abs(action))
  trading_penalty = self.trading_penalty_factor * trading_activity
  ```
- **Drawdown penalty** - Lines 240-245:
  ```python
  peak_value = np.max(self.portfolio_value_history)
  current_drawdown = max(0, (peak_value - new_portfolio_value) / peak_value)
  drawdown_penalty = self.drawdown_penalty_factor * current_drawdown
  ```
- **Diversity bonus** - Lines 248-249: `diversity_bonus = 0.001 * min(holdings_count / self.min_assets_held, 1.0)`
- **Combined reward** - Line 252: `reward = base_reward - transaction_cost_penalty - trading_penalty - drawdown_penalty + diversity_bonus`

### Performance Metrics (in `get_portfolio_metrics` method, Lines 275-298)
- **Sharpe ratio** - Line 280: `sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)`
- **Maximum drawdown** - Lines 282-286
- **Calmar ratio** - Lines 289-290: `calmar_ratio = annual_return / max_drawdown`
- **Annual return** - Line 289: `annual_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1`

### Target Variables (in `src/data/preprocessor.py`)
- **return_vs_spy_target** - Lines 240-248: Forward 7-day returns relative to SPY (standardized)
- This target is used for evaluation but NOT as an input feature to avoid lookahead bias

## Reinforcement Learning Components

### State (in `src/data/environment.py`)
- **Feature vector for each stock** (excluding lookahead features) - Lines 147-150:
  ```python
  feature_columns = [col for col in self.current_data.columns 
                    if col not in ['Time', 'symbol', 'price', 'sector', 'industry', 
                                   'return_vs_spy', 'return_vs_spy_std'] and not col.startswith('MA') and not col.startswith('MACD_return_vs_spy')]
  ```
- **Current holdings for each stock** - Line 156: `holdings_vector = self.holdings.reshape(-1, 1)`
- **Available cash** - Line 162: `state = np.append(flattened_features, self.cash / self.initial_cash)`

### Action Space (in `__init__` method)
- **Vector of allocation percentages** (-1 to 1) for each stock - Lines 40-45:
  ```python
  self.action_space = spaces.Box(
      low=-1.0,
      high=1.0,
      shape=(self.n_symbols,),
      dtype=np.float32
  )
  ```
  Where:
  - -1 = sell all
  - 0 = hold
  - 0.5 = buy with 50% of remaining cash
  - etc.

### Action Processing (in `step` method)
- **Sell orders** - Lines 191-202: Process negative action values as sells
- **Buy orders** - Lines 204-216: Process positive action values as buys

### Reward
- Portfolio return minus penalties (see Optimization Targets section above)

## Performance Metrics

- Sharpe Ratio: Risk-adjusted return
- Calmar Ratio: Return relative to maximum drawdown
- Maximum Drawdown: Largest peak-to-trough decline
- Cumulative Return: Total portfolio growth

## Results

Training results are saved to the `results/` directory:
- Metrics CSV files
- Reward plots
- Performance visualizations

## License

This project is licensed under the MIT License.