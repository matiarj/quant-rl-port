import os
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import argparse
import time
import json
from tqdm import tqdm

from src.utils.config import load_config, parse_args, setup_directories
from src.data.preprocessor import DataPreprocessor
from src.data.environment import PortfolioEnv
from src.agents.dqn import DQN
from src.agents.ddqn import DDQN
from src.agents.a2c import A2C
from src.classifiers.ensemble import ClassifierEnsemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

def train_agent(env: PortfolioEnv, agent: Any, config: Dict[str, Any], 
                num_episodes: int, checkpoint_dir: str) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Train a single agent.
    
    Args:
        env: Environment to train in
        agent: Agent to train
        config: Configuration dictionary
        num_episodes: Number of episodes to train for
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Tuple of:
        - episode_rewards: List of total rewards for each episode
        - metrics: List of dictionaries containing metrics for each episode
    """
    logger.info(f"Starting training for {num_episodes} episodes")
    
    # Track results
    episode_rewards = []
    metrics = []
    
    # Save interval
    save_interval = config['checkpoint']['save_interval']
    
    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            if hasattr(agent, 'memory'):
                agent.memory.push(state, action, reward, next_state, done)
            elif hasattr(agent, 'remember'):
                agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss_info = agent.train_step()
            episode_loss += loss_info.get('loss', 0)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Calculate moving average
        window_size = min(10, len(episode_rewards))
        moving_avg = sum(episode_rewards[-window_size:]) / window_size
        
        # Collect metrics
        episode_metrics = {
            'episode': episode,
            'reward': episode_reward,
            'moving_avg_reward': moving_avg,
            'steps': episode_steps,
            'avg_loss': episode_loss / max(1, episode_steps),
            **loss_info,
            **env.get_portfolio_metrics()
        }
        metrics.append(episode_metrics)
        
        # Log progress
        if episode % 10 == 0 or episode == 1:
            logger.info(f"Episode {episode}: Reward={episode_reward:.4f}, Moving Avg={moving_avg:.4f}, "
                       f"Sharpe={env.get_portfolio_metrics()['sharpe_ratio']:.4f}")
        
        # Save checkpoint
        if episode % save_interval == 0 or episode == num_episodes:
            agent.save_model(checkpoint_dir)
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{checkpoint_dir}/metrics_episode_{episode}.csv", index=False)
            
            # Plot rewards
            plt.figure(figsize=(12, 6))
            plt.plot(metrics_df['episode'], metrics_df['reward'], label='Reward')
            plt.plot(metrics_df['episode'], metrics_df['moving_avg_reward'], label='Moving Avg Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{checkpoint_dir}/rewards_episode_{episode}.png")
            plt.close()
    
    logger.info(f"Training completed. Final moving average reward: {moving_avg:.4f}")
    return episode_rewards, metrics


def train_ensemble(env: PortfolioEnv, agents: Dict[str, Any], classifier: ClassifierEnsemble,
                  config: Dict[str, Any], num_episodes: int, checkpoint_dir: str) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Train an ensemble of agents with classifier selection.
    
    Args:
        env: Environment to train in
        agents: Dictionary mapping agent names to agent objects
        classifier: Classifier ensemble
        config: Configuration dictionary
        num_episodes: Number of episodes to train for
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Tuple of:
        - episode_rewards: List of total rewards for each episode
        - metrics: List of dictionaries containing metrics for each episode
    """
    logger.info(f"Starting ensemble training for {num_episodes} episodes")
    
    # Track results
    episode_rewards = []
    metrics = []
    
    # Save interval
    save_interval = config['checkpoint']['save_interval']
    
    # Classifier training interval
    classifier_interval = config['classifier']['training_interval']
    
    # Training data for classifier
    classifier_states = []
    classifier_best_agents = []
    
    # Map agent names to indices
    agent_indices = {name: i for i, name in enumerate(agents.keys())}
    
    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Ensemble Training"):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Track agent performance in this episode
        agent_rewards = {name: 0 for name in agents.keys()}
        agent_steps = {name: 0 for name in agents.keys()}
        agent_losses = {name: 0 for name in agents.keys()}
        
        # Episode loop
        while not done:
            # Get predictions from classifier
            agent_idx, variance, exploit = classifier.predict(state)
            agent_name = list(agents.keys())[agent_idx]
            
            # Select the agent
            selected_agent = agents[agent_name]
            
            # Select action based on exploitation/exploration decision
            if exploit:
                # Use the selected agent's action
                action = selected_agent.act(state, explore=False)
            else:
                # Be risk-averse (take minimal action)
                action = np.zeros(env.action_space.shape[0])
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience in all agents
            for name, agent in agents.items():
                if hasattr(agent, 'memory'):
                    agent.memory.push(state, action, reward, next_state, done)
                elif hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)
                
                # Train each agent
                loss_info = agent.train_step()
                agent_losses[name] += loss_info.get('loss', 0)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Track rewards for the selected agent
            agent_rewards[agent_name] += reward
            agent_steps[agent_name] += 1
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Calculate moving average
        window_size = min(10, len(episode_rewards))
        moving_avg = sum(episode_rewards[-window_size:]) / window_size
        
        # Find best agent for this episode
        best_agent = max(agent_rewards.items(), key=lambda x: x[1])[0]
        best_agent_idx = agent_indices[best_agent]
        
        # Store training data for classifier
        classifier_states.append(env.reset())  # Use initial state for classifier training
        classifier_best_agents.append(best_agent_idx)
        
        # Train classifier every N episodes
        classifier_metrics = {}
        if episode % classifier_interval == 0 and len(classifier_states) > 0:
            logger.info(f"Training classifier with {len(classifier_states)} samples")
            classifier_metrics = classifier.train(
                np.array(classifier_states),
                np.array(classifier_best_agents)
            )
            
            # Save classifier
            classifier.save(checkpoint_dir)
            
            # Clear training data
            classifier_states = []
            classifier_best_agents = []
        
        # Collect metrics
        episode_metrics = {
            'episode': episode,
            'reward': episode_reward,
            'moving_avg_reward': moving_avg,
            'steps': episode_steps,
            'best_agent': best_agent,
            **{f"{name}_reward": reward for name, reward in agent_rewards.items()},
            **{f"{name}_avg_loss": loss / max(1, agent_steps[name]) for name, loss in agent_losses.items()},
            **classifier.get_stats(),
            **classifier_metrics,
            **env.get_portfolio_metrics()
        }
        metrics.append(episode_metrics)
        
        # Log progress
        if episode % 10 == 0 or episode == 1:
            logger.info(f"Episode {episode}: Reward={episode_reward:.4f}, Moving Avg={moving_avg:.4f}, "
                       f"Best Agent={best_agent}, Exploit Ratio={classifier.exploit_decisions/max(1, classifier.predict_calls):.2f}")
        
        # Save checkpoint
        if episode % save_interval == 0 or episode == num_episodes:
            # Save all agents
            for name, agent in agents.items():
                agent.save_model(os.path.join(checkpoint_dir, name))
            
            # Save classifier
            classifier.save(checkpoint_dir)
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{checkpoint_dir}/ensemble_metrics_episode_{episode}.csv", index=False)
            
            # Plot rewards
            plt.figure(figsize=(12, 6))
            plt.plot(metrics_df['episode'], metrics_df['reward'], label='Reward')
            plt.plot(metrics_df['episode'], metrics_df['moving_avg_reward'], label='Moving Avg Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Ensemble Training Rewards')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{checkpoint_dir}/ensemble_rewards_episode_{episode}.png")
            plt.close()
            
            # Plot agent usage
            plt.figure(figsize=(12, 6))
            agent_usage = {}
            for i, name in enumerate(agents.keys()):
                agent_usage[name] = (np.array(classifier_best_agents) == i).sum()
            plt.bar(agent_usage.keys(), agent_usage.values())
            plt.xlabel('Agent')
            plt.ylabel('Count')
            plt.title('Agent Usage')
            plt.savefig(f"{checkpoint_dir}/agent_usage_episode_{episode}.png")
            plt.close()
    
    logger.info(f"Ensemble training completed. Final moving average reward: {moving_avg:.4f}")
    return episode_rewards, metrics


def evaluate(env: PortfolioEnv, agent: Any, num_episodes: int = 1) -> Dict[str, float]:
    """
    Evaluate an agent.
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate for
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating agent for {num_episodes} episodes")
    
    # Track results
    episode_rewards = []
    sharpe_ratios = []
    calmar_ratios = []
    max_drawdowns = []
    
    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Select action (no exploration)
            action = agent.act(state, explore=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Get portfolio metrics
        metrics = env.get_portfolio_metrics()
        sharpe_ratios.append(metrics['sharpe_ratio'])
        calmar_ratios.append(metrics['calmar_ratio'])
        max_drawdowns.append(metrics['max_drawdown'])
        
        logger.info(f"Evaluation Episode {episode}: Reward={episode_reward:.4f}, "
                   f"Sharpe={metrics['sharpe_ratio']:.4f}, Calmar={metrics['calmar_ratio']:.4f}")
    
    # Calculate overall metrics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios)
    avg_calmar = sum(calmar_ratios) / len(calmar_ratios)
    avg_max_drawdown = sum(max_drawdowns) / len(max_drawdowns)
    
    logger.info(f"Evaluation completed. Average reward: {avg_reward:.4f}")
    logger.info(f"Average Sharpe: {avg_sharpe:.4f}, Average Calmar: {avg_calmar:.4f}, "
               f"Average Max Drawdown: {avg_max_drawdown:.4f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_sharpe_ratio': avg_sharpe,
        'avg_calmar_ratio': avg_calmar,
        'avg_max_drawdown': avg_max_drawdown
    }


def create_agent(name: str, state_dim: int, action_dim: int, config: Dict[str, Any]) -> Any:
    """
    Create an agent based on the algorithm name.
    
    Args:
        name: Algorithm name
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Configuration dictionary
        
    Returns:
        Agent object
    """
    if name == "DQN":
        return DQN(state_dim, action_dim, config)
    elif name == "DDQN":
        return DDQN(state_dim, action_dim, config)
    elif name == "A2C":
        return A2C(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.algo:
        config['rl']['algorithms'] = [args.algo]
    if args.episodes:
        config['rl']['num_episodes'] = args.episodes
    
    # Setup directories
    setup_directories()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create timestamp for this run
    timestamp = int(time.time())
    run_dir = os.path.join(config['checkpoint']['path'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data
    data = preprocessor.load_data(config['data']['file_path'], args.dry_run)
    train_data, val_data, test_data = preprocessor.preprocess(data, args.dry_run)
    
    logger.info(f"Preprocessed data shapes: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")
    
    # Create environment
    train_env = PortfolioEnv(train_data, config, train=True)
    val_env = PortfolioEnv(val_data, config, train=False)
    test_env = PortfolioEnv(test_data, config, train=False)
    
    # Get state and action dimensions
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Select algorithms to use
    algorithms = config['rl']['algorithms']
    logger.info(f"Selected algorithms: {algorithms}")
    
    # Initialize agents
    if config['rl']['use_ensemble']:
        # Create multiple agents for ensemble
        agents = {}
        for algo in algorithms:
            agents[algo] = create_agent(algo, state_dim, action_dim, config)
        
        # Create classifier ensemble
        classifier = ClassifierEnsemble(config, list(agents.keys()))
        
        # Train ensemble
        ensemble_dir = os.path.join(run_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        train_ensemble(
            train_env,
            agents,
            classifier,
            config,
            config['rl']['num_episodes'],
            ensemble_dir
        )
        
        # Evaluate ensemble on validation set
        best_val_reward = -float('inf')
        best_agent = None
        
        for name, agent in agents.items():
            val_metrics = evaluate(val_env, agent)
            if val_metrics['avg_reward'] > best_val_reward:
                best_val_reward = val_metrics['avg_reward']
                best_agent = name
        
        logger.info(f"Best agent on validation set: {best_agent} with reward {best_val_reward:.4f}")
        
        # Test on test set
        test_metrics = evaluate(test_env, agents[best_agent], num_episodes=5)
        
        # Save test metrics
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'best_agent': best_agent,
                'metrics': test_metrics
            }, f, indent=2)
        
    else:
        # Use a single agent
        algo = algorithms[0]
        agent = create_agent(algo, state_dim, action_dim, config)
        
        # Train agent
        algo_dir = os.path.join(run_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)
        
        train_agent(
            train_env,
            agent,
            config,
            config['rl']['num_episodes'],
            algo_dir
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(val_env, agent)
        
        # Evaluate on test set
        test_metrics = evaluate(test_env, agent, num_episodes=5)
        
        # Save test metrics
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'agent': algo,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }, f, indent=2)


if __name__ == "__main__":
    main()