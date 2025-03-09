import yaml
import argparse
import os

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization with RL')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--dry-run', action='store_true', help='Run in debug mode with small dataset')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation on test set')
    parser.add_argument('--algo', type=str, help='Override algorithm from config')
    parser.add_argument('--episodes', type=int, help='Override number of episodes')
    return parser.parse_args()

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['checkpoints', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)