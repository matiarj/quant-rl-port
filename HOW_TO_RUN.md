# How to Run the Portfolio Optimization Analysis

This document provides instructions on how to run the full portfolio optimization analysis with reinforcement learning.

## Quick Start

To run the full analysis with the default settings:

```bash
# Make sure your conda environment is activated
conda activate bregma_rl

# Run the full analysis
./run_full_analysis.py
```

The analysis will:
1. Load data from `data/model68_hype_2024_bt_df4web.csv`
2. Preprocess the data with the specified time ranges
3. Train the RL model(s) with the ensemble approach
4. Evaluate the model(s) on validation and test data
5. Generate a summary of the results

## Visualization

After running the analysis, you can generate detailed visualizations of the backtest results:

```bash
./visualize_backtest.py
```

This will create:
- Performance plots in `results/backtest/`
- A summary report with key metrics in `results/backtest/backtest_summary.txt`

## Configuration

You can modify the following configuration parameters:

### Time Periods

The analysis uses the following time periods:
- Training: Start of data to mid-2016
- Validation: Start of 2017 to mid-2020
- Test: 2021 to end of data

To change these periods, edit the `split_data` method in `src/data/preprocessor.py`.

### Model Parameters

The model parameters are configured in `config.yaml`:
- RL algorithms to use
- Network architecture
- Training hyperparameters
- Ensemble settings

Edit this file to customize the analysis.

## Running on Smaller Datasets

To run a dry-run on a smaller dataset for testing:

```bash
python main.py --dry-run
```

## Running Single Algorithms

To run with just one algorithm instead of the ensemble:

```bash
python main.py --algo DQN
```

Replace `DQN` with any of: `DDQN`, `A2C`.

## Expected Runtime

The full analysis may take several hours to complete, depending on:
- Size of the dataset
- Number of algorithms used
- Ensemble settings
- Hardware specifications

Progress will be logged to the console and to log files in the `logs/` directory.

## Results

All results are saved in:
- `checkpoints/run_<timestamp>/` - Model checkpoints and metrics
- `results/backtest/` - Performance visualizations and summary report
- `logs/` - Detailed logs of the analysis process