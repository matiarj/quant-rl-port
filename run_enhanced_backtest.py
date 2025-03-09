#!/usr/bin/env python3
import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import torch

from src.data.preprocessor import DataPreprocessor
from src.data.environment import PortfolioEnv
from src.agents.dqn import DQN
from src.agents.ddqn import DDQN
from src.agents.a2c import A2C
from src.agents.grpo import GRPO
from src.classifiers.ensemble import ClassifierEnsemble

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/full_analysis_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('enhanced_backtest')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_portfolio_optimization(config: Dict[str, Any], dry_run: bool = False, custom_algo: str = None) -> Dict[str, Any]:
    """Run the full portfolio optimization analysis with reinforcement learning."""
    
    # Step 1: Load and preprocess data
    data_path = config['data']['file_path']
    logger.info(f"Starting portfolio optimization with data from {data_path}")
    
    preprocessor = DataPreprocessor(config)
    raw_data = preprocessor.load_data(data_path, dry_run=dry_run)
    train_df, val_df, test_df = preprocessor.preprocess(raw_data, dry_run=dry_run)
    
    # Step 2: Set up environments
    train_env = PortfolioEnv(train_df, config, train=True)
    val_env = PortfolioEnv(val_df, config, train=False)
    test_env = PortfolioEnv(test_df, config, train=False)
    
    # Get state and action dimensions
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    # Step 3: Set up the agent(s)
    run_ensemble = config['rl']['use_ensemble'] and not custom_algo
    
    # Determine which algorithms to use
    if custom_algo:
        algorithms = [custom_algo]
    else:
        algorithms = config['rl']['algorithms']
    
    logger.info(f"Using algorithm(s): {algorithms}")
    
    # Create run directory for checkpoints
    timestamp = int(time.time())
    run_dir = os.path.join(config['checkpoint']['path'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize agents
    agents = {}
    for algo in algorithms:
        if algo == 'DQN':
            agents[algo] = DQN(state_dim, action_dim, config)
        elif algo == 'DDQN':
            agents[algo] = DDQN(state_dim, action_dim, config)
        elif algo == 'A2C':
            agents[algo] = A2C(state_dim, action_dim, config)
        elif algo == 'GRPO':
            agents[algo] = GRPO(state_dim, action_dim, config)
        else:
            logger.error(f"Unknown algorithm: {algo}")
            continue
    
    # Initialize classifier ensemble if needed
    if run_ensemble:
        logger.info("Using classifier ensemble")
        ensemble_dir = os.path.join(run_dir, 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        
        classifier = ClassifierEnsemble(
            config=config,
            agent_names=algorithms
        )
    
    # Step 4: Train the agent(s)
    num_episodes = config['rl']['num_episodes']
    save_interval = config['checkpoint']['save_interval']
    
    logger.info(f"Starting training for {num_episodes} episodes")
    
    best_val_reward = -np.inf
    best_agent = None
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # Training logic differs based on whether we're using ensemble or single agent
        if run_ensemble:
            # Train all agents
            for algo, agent in agents.items():
                state = train_env.reset()
                done = False
                episode_reward = 0.0
                episode_losses = []
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, _ = train_env.step(action)
                    # Handle both memory-based and on-policy agents
                    if hasattr(agent, 'memory_add') and callable(agent.memory_add):
                        agent.memory_add(state, action, reward, next_state, done)
                    else:
                        agent.memory.add(state, action, reward, next_state, done)
                    
                    # Train the agent
                    losses = agent.train_step()
                    if 'loss' in losses:
                        episode_losses.append(losses['loss'])
                    
                    state = next_state
                    episode_reward += reward
                
                # Update the classifier with this agent's state-action-reward data
                if config['rl']['use_classifier']:
                    classifier.add_training_data(algo, state, reward)
                
                # Log progress
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                logger.info(f"Episode {episode}/{num_episodes} - Agent {algo} - Reward: {episode_reward:.6f}, Avg Loss: {avg_loss:.6f}")
            
            # Every K episodes, train the classifier on accumulated data
            if config['rl']['use_classifier'] and episode % config['classifier']['training_interval'] == 0:
                classifier.train()
                
                # Save the classifiers
                classifier_path = os.path.join(ensemble_dir, f'classifiers_{int(time.time())}.joblib')
                classifier.save(classifier_path)
                
            # Save agent models at intervals
            if episode % save_interval == 0:
                for algo, agent in agents.items():
                    algo_dir = os.path.join(ensemble_dir, algo)
                    os.makedirs(algo_dir, exist_ok=True)
                    agent.save_model(algo_dir)
                
                # Run ensemble evaluation
                ensemble_val_metrics = evaluate_ensemble(val_env, agents, classifier, config)
                
                # Log ensemble performance
                logger.info(f"Episode {episode}/{num_episodes} - Ensemble Validation - Reward: {ensemble_val_metrics['avg_reward']:.6f}, "
                          f"Sharpe: {ensemble_val_metrics['avg_sharpe_ratio']:.6f}")
                
                # Save ensemble metrics
                metrics_path = os.path.join(ensemble_dir, f'ensemble_metrics_episode_{episode}.csv')
                pd.DataFrame([ensemble_val_metrics]).to_csv(metrics_path, index=False)
                
                # Plot reward progress
                plot_rewards(ensemble_val_metrics['rewards'], os.path.join(ensemble_dir, f'ensemble_rewards_episode_{episode}.png'))
                
                # Plot agent usage
                if config['rl']['use_classifier']:
                    plot_agent_usage(classifier.agent_usage, os.path.join(ensemble_dir, f'agent_usage_episode_{episode}.png'))
                
                # Check if this is the best model so far
                if ensemble_val_metrics['avg_reward'] > best_val_reward:
                    best_val_reward = ensemble_val_metrics['avg_reward']
                    best_agent = 'ensemble'
        
        else:
            # Single agent training
            algo = algorithms[0]
            agent = agents[algo]
            
            state = train_env.reset()
            done = False
            episode_reward = 0.0
            episode_losses = []
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = train_env.step(action)
                # Handle both memory-based and on-policy agents
                if hasattr(agent, 'memory_add') and callable(agent.memory_add):
                    agent.memory_add(state, action, reward, next_state, done)
                else:
                    agent.memory.add(state, action, reward, next_state, done)
                
                # Train the agent
                losses = agent.train_step()
                if 'loss' in losses:
                    episode_losses.append(losses['loss'])
                
                state = next_state
                episode_reward += reward
            
            # Log progress
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            logger.info(f"Episode {episode}/{num_episodes} - Agent {algo} - "
                      f"Reward: {episode_reward:.6f}, Avg Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.4f}")
            
            # Save model and evaluate at intervals
            if episode % save_interval == 0:
                # Create agent directory
                algo_dir = os.path.join(run_dir, algo)
                os.makedirs(algo_dir, exist_ok=True)
                
                # Save model
                agent.save_model(algo_dir)
                
                # Evaluate on validation set
                val_metrics = evaluate_agent(val_env, agent, config)
                
                # Log validation performance
                logger.info(f"Episode {episode}/{num_episodes} - Validation - "
                          f"Reward: {val_metrics['avg_reward']:.6f}, Sharpe: {val_metrics['avg_sharpe_ratio']:.6f}")
                
                # Save metrics
                metrics_path = os.path.join(algo_dir, f'metrics_episode_{episode}.csv')
                pd.DataFrame([val_metrics]).to_csv(metrics_path, index=False)
                
                # Plot reward progress
                plot_rewards(val_metrics['rewards'], os.path.join(algo_dir, f'rewards_episode_{episode}.png'))
                
                # Check if this is the best model so far
                if val_metrics['avg_reward'] > best_val_reward:
                    best_val_reward = val_metrics['avg_reward']
                    best_agent = algo
    
    # Step 5: Final evaluation
    logger.info(f"Training complete. Best agent: {best_agent}")
    
    if run_ensemble:
        # Run final validation metrics
        val_metrics = evaluate_ensemble(val_env, agents, classifier, config)
        
        # Run test metrics
        test_metrics = evaluate_ensemble(test_env, agents, classifier, config)
        
        # Save test metrics
        metrics_result = {
            'best_agent': best_agent,
            'metrics': test_metrics
        }
        
    else:
        algo = algorithms[0]
        agent = agents[algo]
        
        # Run final validation metrics
        val_metrics = evaluate_agent(val_env, agent, config)
        
        # Run test metrics
        test_metrics = evaluate_agent(test_env, agent, config)
        
        # Save test metrics
        metrics_result = {
            'agent': algo,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    # Save test metrics to file
    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_result, f, indent=2)
    
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Analysis complete. Results saved to {run_dir}")
    
    return metrics_result

def evaluate_agent(env: PortfolioEnv, agent: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single agent."""
    state = env.reset()
    done = False
    rewards = []
    
    while not done:
        # Get action without exploration
        action = agent.act(state, explore=False)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = next_state
    
    # Calculate portfolio metrics
    metrics = env.get_portfolio_metrics()
    
    # Aggregate and return metrics
    return {
        'avg_reward': np.mean(rewards),
        'rewards': rewards,
        'avg_sharpe_ratio': metrics['sharpe_ratio'],
        'avg_calmar_ratio': metrics['calmar_ratio'],
        'avg_max_drawdown': metrics['max_drawdown'],
        'final_return': metrics['final_return'],
        'annual_return': metrics['annual_return']
    }

def evaluate_ensemble(env: PortfolioEnv, agents: Dict[str, Any], classifier: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ensemble of agents using the classifier."""
    state = env.reset()
    done = False
    rewards = []
    agent_selections = {}
    
    for algo in agents.keys():
        agent_selections[algo] = 0
    
    while not done:
        # Select best agent based on state
        if config['rl']['use_classifier']:
            try:
                selected_algo = classifier.select_agent(state)
            except Exception as e:
                # Fallback if classifier fails
                selected_algo = list(agents.keys())[0]  # Default to first agent
                logger.warning(f"Classifier failed to select agent: {e}. Using {selected_algo} instead.")
        else:
            # Simple round-robin if classifier is disabled
            selected_algo = list(agents.keys())[len(rewards) % len(agents)]
        
        # Track agent usage
        agent_selections[selected_algo] += 1
        
        # Get action from selected agent without exploration
        action = agents[selected_algo].act(state, explore=False)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = next_state
    
    # Calculate portfolio metrics
    metrics = env.get_portfolio_metrics()
    
    # Store agent usage in the classifier
    if config['rl']['use_classifier']:
        classifier.agent_usage = agent_selections
    
    # Aggregate and return metrics
    return {
        'avg_reward': np.mean(rewards),
        'rewards': rewards,
        'avg_sharpe_ratio': metrics['sharpe_ratio'],
        'avg_calmar_ratio': metrics['calmar_ratio'],
        'avg_max_drawdown': metrics['max_drawdown'],
        'agent_selections': agent_selections
    }

def plot_rewards(rewards: List[float], save_path: str) -> None:
    """Plot reward history and save to file."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(save_path)
    plt.close()

def plot_agent_usage(agent_usage: Dict[str, int], save_path: str) -> None:
    """Plot agent usage and save to file."""
    plt.figure(figsize=(8, 6))
    algos = list(agent_usage.keys())
    usage = list(agent_usage.values())
    plt.bar(algos, usage)
    plt.title('Agent Usage Count')
    plt.xlabel('Algorithm')
    plt.ylabel('Usage Count')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the enhanced portfolio optimization backtest.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Run with minimal data for testing')
    parser.add_argument('--algo', type=str, help='Specify a single algorithm to use (DQN, DDQN, A2C, GRPO)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the analysis
    run_portfolio_optimization(config, dry_run=args.dry_run, custom_algo=args.algo)