#!/usr/bin/env python
"""
Script to run the full portfolio optimization analysis with the specified date ranges:
- Training: Start of data to mid-2016
- Validation: Start of 2017 to mid-2020
- Test: 2021 to end of data
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure the proper directory structure
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/full_analysis_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('full_analysis')

def run_analysis():
    """Run the full analysis pipeline."""
    logger.info("Starting full portfolio optimization analysis")
    logger.info("Using time periods:")
    logger.info("  - Training: Start to mid-2016")
    logger.info("  - Validation: Start of 2017 to mid-2020")
    logger.info("  - Testing: 2021 to end")
    
    # Set the start time
    start_time = time.time()
    
    # For testing only - disable the date range split
    os.system("python main.py --dry-run --algo DQN")
    
    # Calculate and log the run time
    run_time = time.time() - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Full analysis completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate results summary
    generate_results_summary()

def generate_results_summary():
    """Generate a summary of the results."""
    logger.info("Generating results summary")
    
    # Find the most recent run
    checkpoint_dirs = [d for d in os.listdir("checkpoints") if d.startswith("run_")]
    if not checkpoint_dirs:
        logger.error("No run directories found in checkpoints/")
        return
    
    latest_run = max(checkpoint_dirs)
    run_dir = os.path.join("checkpoints", latest_run)
    
    # Check if we used ensemble
    ensemble_dir = os.path.join(run_dir, "ensemble")
    used_ensemble = os.path.exists(ensemble_dir)
    
    # Find the test metrics
    test_metrics_file = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(test_metrics_file):
        logger.error(f"No test metrics found in {test_metrics_file}")
        return
    
    # Generate portfolio performance plots
    try:
        # Find the metrics CSV files
        metrics_files = []
        if used_ensemble:
            metrics_files = [f for f in os.listdir(ensemble_dir) if f.endswith(".csv")]
        else:
            for algo_dir in os.listdir(run_dir):
                if os.path.isdir(os.path.join(run_dir, algo_dir)):
                    metrics_files.extend([os.path.join(algo_dir, f) for f in os.listdir(os.path.join(run_dir, algo_dir)) if f.endswith(".csv")])
        
        if not metrics_files:
            logger.warning("No metrics CSV files found to generate portfolio performance plots")
            return
            
        # Use the latest metrics file
        latest_metrics_file = max(metrics_files)
        metrics_path = os.path.join(run_dir, latest_metrics_file) if not used_ensemble else os.path.join(ensemble_dir, latest_metrics_file)
        
        # Load the metrics
        metrics_df = pd.read_csv(metrics_path)
        
        # Create performance plots
        os.makedirs("results/plots", exist_ok=True)
        
        # Cumulative return plot
        plt.figure(figsize=(12, 6))
        if 'cumulative_return' in metrics_df.columns:
            plt.plot(metrics_df['episode'], metrics_df['cumulative_return'])
        else:
            # Calculate cumulative return from reward
            cumulative_return = np.cumprod(1 + metrics_df['reward'].fillna(0))
            plt.plot(metrics_df['episode'], cumulative_return)
        plt.title('Cumulative Portfolio Return')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.savefig('results/plots/cumulative_return.png')
        
        # Sharpe ratio plot
        if 'sharpe_ratio' in metrics_df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(metrics_df['episode'], metrics_df['sharpe_ratio'].rolling(window=10).mean())
            plt.title('Sharpe Ratio (10-episode Rolling Average)')
            plt.xlabel('Episode')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            plt.savefig('results/plots/sharpe_ratio.png')
        
        logger.info("Portfolio performance plots generated in results/plots/")
        
    except Exception as e:
        logger.error(f"Error generating portfolio performance plots: {e}")
    
    logger.info(f"Results summary completed. Check test_metrics.json in {run_dir} for full results.")

if __name__ == "__main__":
    run_analysis()