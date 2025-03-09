#!/usr/bin/env python
"""
Script to visualize backtest results from the portfolio optimization analysis.
This creates detailed performance visualizations for the training, validation, and test periods.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visualize_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('visualize_backtest')

def find_latest_run():
    """Find the most recent run directory."""
    checkpoint_dirs = [d for d in os.listdir("checkpoints") if d.startswith("run_")]
    if not checkpoint_dirs:
        logger.error("No run directories found in checkpoints/")
        return None
    
    latest_run = max(checkpoint_dirs)
    return os.path.join("checkpoints", latest_run)

def load_test_metrics(run_dir):
    """Load test metrics from the run directory."""
    test_metrics_file = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(test_metrics_file):
        logger.error(f"No test metrics found in {test_metrics_file}")
        return None
    
    with open(test_metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def find_metrics_file(run_dir):
    """Find the latest metrics CSV file in the run directory."""
    # Check if ensemble was used
    ensemble_dir = os.path.join(run_dir, "ensemble")
    used_ensemble = os.path.exists(ensemble_dir)
    
    metrics_files = []
    if used_ensemble:
        metrics_files = [f for f in os.listdir(ensemble_dir) if f.endswith(".csv")]
        if metrics_files:
            return os.path.join(ensemble_dir, max(metrics_files))
    else:
        # Check each algorithm directory
        for algo_dir in os.listdir(run_dir):
            algo_path = os.path.join(run_dir, algo_dir)
            if os.path.isdir(algo_path):
                algo_metrics = [f for f in os.listdir(algo_path) if f.endswith(".csv")]
                if algo_metrics:
                    metrics_files.append(os.path.join(algo_path, max(algo_metrics)))
        
        if metrics_files:
            return max(metrics_files)
    
    logger.error("No metrics CSV files found")
    return None

def create_performance_plots(metrics_file, output_dir="results/backtest"):
    """Create performance visualization plots."""
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found: {metrics_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics data
    metrics_df = pd.read_csv(metrics_file)
    
    # 1. Cumulative Return Plot
    plt.figure(figsize=(14, 7))
    if 'cumulative_return' in metrics_df.columns:
        plt.plot(metrics_df['episode'], metrics_df['cumulative_return'])
    else:
        # Calculate cumulative return from reward
        cumulative_return = np.cumprod(1 + metrics_df['reward'].fillna(0))
        plt.plot(metrics_df['episode'], cumulative_return)
    
    plt.title('Cumulative Portfolio Return', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cumulative_return.png'), dpi=300, bbox_inches='tight')
    
    # 2. Rolling Sharpe Ratio
    if 'sharpe_ratio' in metrics_df.columns:
        plt.figure(figsize=(14, 7))
        window = min(20, len(metrics_df) // 10)  # Dynamic window size
        sharpe_rolling = metrics_df['sharpe_ratio'].rolling(window=window).mean()
        plt.plot(metrics_df['episode'], sharpe_rolling)
        plt.title(f'Sharpe Ratio ({window}-episode Rolling Average)', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Sharpe Ratio', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sharpe_ratio.png'), dpi=300, bbox_inches='tight')
    
    # 3. Episode Rewards
    plt.figure(figsize=(14, 7))
    plt.plot(metrics_df['episode'], metrics_df['reward'])
    plt.title('Episode Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    window = min(50, len(metrics_df) // 5)
    plt.plot(metrics_df['episode'], metrics_df['reward'].rolling(window=window).mean(), 
             color='red', linewidth=2, label=f'{window}-episode Moving Average')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(output_dir, 'episode_rewards.png'), dpi=300, bbox_inches='tight')
    
    # 4. Drawdown Analysis (if available)
    if 'max_drawdown' in metrics_df.columns:
        plt.figure(figsize=(14, 7))
        plt.plot(metrics_df['episode'], metrics_df['max_drawdown'])
        plt.title('Maximum Drawdown Over Time', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Maximum Drawdown', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'max_drawdown.png'), dpi=300, bbox_inches='tight')
    
    # 5. Create correlation matrix of metrics
    numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 5:  # Only create if we have enough metrics
        plt.figure(figsize=(16, 12))
        corr_matrix = metrics_df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, center=0)
        plt.title('Correlation Matrix of Performance Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
    
    logger.info(f"Performance plots created in {output_dir}")

def create_summary_report(run_dir, output_dir="results/backtest"):
    """Create a summary report of the backtest results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test metrics
    test_metrics = load_test_metrics(run_dir)
    if not test_metrics:
        return
    
    # Create summary report
    report_file = os.path.join(output_dir, "backtest_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write("===== PORTFOLIO OPTIMIZATION BACKTEST SUMMARY =====\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("===== TEST PERIOD PERFORMANCE =====\n")
        
        # Format metrics depending on what we have
        if 'best_agent' in test_metrics:
            f.write(f"Best Agent: {test_metrics['best_agent']}\n\n")
            metrics = test_metrics.get('metrics', {})
        else:
            metrics = test_metrics.get('test_metrics', {})
        
        # Write metrics
        f.write(f"Average Reward: {metrics.get('avg_reward', 'N/A'):.6f}\n")
        f.write(f"Sharpe Ratio: {metrics.get('avg_sharpe_ratio', 'N/A'):.4f}\n")
        f.write(f"Calmar Ratio: {metrics.get('avg_calmar_ratio', 'N/A'):.4f}\n")
        f.write(f"Maximum Drawdown: {metrics.get('avg_max_drawdown', 'N/A'):.4%}\n\n")
        
        # Additional notes
        f.write("===== TRAINING DETAILS =====\n")
        f.write("Training Period: Start of data to mid-2016\n")
        f.write("Validation Period: Start of 2017 to mid-2020\n")
        f.write("Test Period: 2021 to end of data\n\n")
        
        f.write("===== NOTES =====\n")
        f.write("- All performance metrics are calculated on the test set\n")
        f.write("- The Sharpe ratio is annualized assuming 252 trading days\n")
        f.write("- The Calmar ratio is calculated as annualized return / maximum drawdown\n")
        f.write("- See the individual plots for detailed performance visualization\n")
    
    logger.info(f"Summary report created at {report_file}")

def main():
    """Main function to visualize backtest results."""
    logger.info("Starting backtest visualization")
    
    # Find latest run
    run_dir = find_latest_run()
    if not run_dir:
        return
    
    logger.info(f"Analyzing run: {run_dir}")
    
    # Find metrics file
    metrics_file = find_metrics_file(run_dir)
    if not metrics_file:
        return
    
    logger.info(f"Found metrics file: {metrics_file}")
    
    # Create performance plots
    create_performance_plots(metrics_file)
    
    # Create summary report
    create_summary_report(run_dir)
    
    logger.info("Backtest visualization completed")

if __name__ == "__main__":
    main()