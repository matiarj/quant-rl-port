import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """Portfolio optimization environment for reinforcement learning."""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], train: bool = True):
        super().__init__()
        
        self.logger = self._setup_logger()
        self.config = config
        self.data = data
        self.train = train
        
        # Extract unique dates and symbols
        self.dates = sorted(data['Time'].unique())
        self.symbols = sorted(data['symbol'].unique())
        self.n_symbols = len(self.symbols)
        
        # Set initial parameters
        self.initial_cash = config['trading']['initial_cash']
        self.transaction_cost = config['trading']['transaction_cost']
        
        # Penalty factors
        self.trading_penalty_factor = config['trading'].get('trading_penalty_factor', 0.0)
        self.drawdown_penalty_factor = config['trading'].get('drawdown_penalty_factor', 0.0)
        
        # Risk management parameters
        self.max_allocation_per_asset = 0.2  # Maximum 20% in any single asset
        self.min_assets_held = 5  # Encourage diversification
        
        # Initialize state
        self.reset()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_symbols,),
            dtype=np.float32
        )
        
        # Determine state dimension based on features
        # We'll do this by checking the shape of get_state()
        initial_state = self.get_state()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_state.shape,
            dtype=np.float32
        )
        
        self.logger.info(f"Environment initialized with {self.n_symbols} symbols and {len(self.dates)} trading days")
        self.logger.info(f"Observation space: {self.observation_space}")
        self.logger.info(f"Action space: {self.action_space}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger."""
        logger = logging.getLogger('PortfolioEnv')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/portfolio_env.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def reset(self) -> np.ndarray:
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.current_date = self.dates[0]
        
        # Portfolio state
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_symbols)
        self.portfolio_value_history = [self.initial_cash]
        self.portfolio_return_history = [0.0]
        
        # Get initial state
        self._update_state()
        return self.get_state()
    
    def _update_state(self) -> None:
        """Update the current state based on the current date."""
        self.current_date = self.dates[self.current_step]
        self.current_data = self.data[self.data['Time'] == self.current_date]
        
        # Ensure we have data for all symbols on this date
        if len(self.current_data) != self.n_symbols:
            missing_symbols = set(self.symbols) - set(self.current_data['symbol'])
            self.logger.warning(f"Missing data for {len(missing_symbols)} symbols on {self.current_date}")
            
            # Create dummy data for missing symbols (copy from last available data)
            for symbol in missing_symbols:
                last_data = self.data[(self.data['symbol'] == symbol) & 
                                     (self.data['Time'] < self.current_date)].iloc[-1].copy()
                last_data['Time'] = self.current_date
                self.current_data = pd.concat([self.current_data, pd.DataFrame([last_data])])
        
        # Ensure data is sorted by symbol for consistent indexing
        self.current_data = self.current_data.sort_values('symbol').reset_index(drop=True)
        
        # Update portfolio value based on current prices
        self._update_portfolio_value()
    
    def _update_portfolio_value(self) -> None:
        """Update the portfolio value based on current holdings and prices."""
        current_prices = self.current_data['price'].values
        holdings_value = np.sum(self.holdings * current_prices)
        self.portfolio_value = self.cash + holdings_value
        
        # Calculate return since last step
        if len(self.portfolio_value_history) > 0:
            last_value = self.portfolio_value_history[-1]
            if last_value > 0:
                current_return = (self.portfolio_value - last_value) / last_value
                self.portfolio_return_history.append(current_return)
            else:
                self.portfolio_return_history.append(0.0)
        
        self.portfolio_value_history.append(self.portfolio_value)
    
    def get_state(self) -> np.ndarray:
        """
        Construct the state representation.
        
        State includes:
        1. Feature vector for each stock (excluding lookahead features)
        2. Current holdings for each stock
        3. Available cash
        """
        # Get feature vectors for each stock, excluding returns_vs_spy which has lookahead bias
        feature_columns = [col for col in self.current_data.columns 
                          if col not in ['Time', 'symbol', 'price', 'sector', 'industry', 
                                         'return_vs_spy', 'return_vs_spy_std'] and not col.startswith('MA') and not col.startswith('MACD_return_vs_spy')]
        
        # Extract features as a 2D array: n_symbols x n_features
        stock_features = self.current_data[feature_columns].values
        
        # Reshape holdings to match feature dimensions
        holdings_vector = self.holdings.reshape(-1, 1)
        
        # Combine stock features and holdings
        combined_features = np.hstack([stock_features, holdings_vector])
        
        # Flatten the features and add cash as a scalar
        flattened_features = combined_features.flatten()
        state = np.append(flattened_features, self.cash / self.initial_cash)  # Normalize cash
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action in the environment.
        
        Args:
            action: Array of allocation percentages (-1 to 1) for each stock
                   -1 = sell all, 0 = hold, 0.5 = buy with 50% of remaining cash, etc.
        
        Returns:
            next_state: New state after action
            reward: Reward for the action
            done: Whether the episode is over
            info: Additional information
        """
        # Clip action to ensure it's within bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate portfolio value before action
        old_portfolio_value = self.portfolio_value
        
        # Execute trades sequentially
        current_prices = self.current_data['price'].values
        total_transaction_cost = 0.0
        
        # Process sell orders first (negative actions)
        for i, (a, price) in enumerate(zip(action, current_prices)):
            if a < 0:  # Sell
                sell_ratio = abs(a)
                shares_to_sell = self.holdings[i] * sell_ratio
                sell_value = shares_to_sell * price
                transaction_fee = sell_value * self.transaction_cost
                
                # Update holdings and cash
                self.holdings[i] -= shares_to_sell
                self.cash += sell_value - transaction_fee
                total_transaction_cost += transaction_fee
        
        # Then process buy orders (positive actions)
        for i, (a, price) in enumerate(zip(action, current_prices)):
            if a > 0:  # Buy
                buy_ratio = a
                cash_to_spend = self.cash * buy_ratio
                # Ensure we don't spend more than available cash
                cash_to_spend = min(cash_to_spend, self.cash)
                shares_to_buy = cash_to_spend / price
                transaction_fee = cash_to_spend * self.transaction_cost
                
                # Update holdings and cash
                self.holdings[i] += shares_to_buy
                self.cash -= (cash_to_spend + transaction_fee)
                total_transaction_cost += transaction_fee
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.dates)
        
        if not done:
            # Update state for the next day
            self._update_state()
            next_state = self.get_state()
            
            # Calculate reward with penalties
            new_portfolio_value = self.portfolio_value
            
            # Base reward: portfolio return minus transaction costs
            base_reward = ((new_portfolio_value - old_portfolio_value) / old_portfolio_value)
            transaction_cost_penalty = (total_transaction_cost / old_portfolio_value)
            
            # Trading activity penalty (penalize excessive trading)
            trading_activity = np.sum(np.abs(action))  # Sum of absolute action values
            trading_penalty = self.trading_penalty_factor * trading_activity
            
            # Drawdown penalty
            # Calculate drawdown from peak
            if len(self.portfolio_value_history) > 1:
                peak_value = np.max(self.portfolio_value_history)
                current_drawdown = max(0, (peak_value - new_portfolio_value) / peak_value)
                drawdown_penalty = self.drawdown_penalty_factor * current_drawdown
            else:
                drawdown_penalty = 0.0
                
            # Diversity bonus (encourage holding multiple assets)
            holdings_count = np.sum(self.holdings > 0)
            diversity_bonus = 0.001 * min(holdings_count / self.min_assets_held, 1.0)
            
            # Calculate final reward
            reward = base_reward - transaction_cost_penalty - trading_penalty - drawdown_penalty + diversity_bonus
        else:
            # If episode is done, return last state and calculate final reward
            next_state = self.get_state()
            reward = 0.0  # No reward for the terminal state
        
        # Additional info for logging
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'transaction_cost': total_transaction_cost,
            'date': self.current_date if not done else self.dates[-1],
            'base_reward': base_reward if not done else 0.0,
            'transaction_penalty': transaction_cost_penalty if not done else 0.0,
            'trading_penalty': trading_penalty if not done else 0.0,
            'drawdown_penalty': drawdown_penalty if not done else 0.0,
            'diversity_bonus': diversity_bonus if not done else 0.0,
            'active_positions': np.sum(self.holdings > 0)
        }
        
        return next_state, reward, done, info
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate and return portfolio performance metrics."""
        returns = np.array(self.portfolio_return_history[1:])  # Skip the first 0 return
        
        # Sharpe Ratio (assuming daily returns, annualized)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calmar Ratio (annualized return / maximum drawdown)
        annual_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1 if len(returns) > 0 and cumulative_returns[-1] > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'final_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
            'annual_return': annual_return
        }
    
    def render(self, mode='human'):
        """Render the environment."""
        print(f"Date: {self.current_date}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Holdings:")
        for i, symbol in enumerate(self.symbols):
            print(f"  {symbol}: {self.holdings[i]:.4f} shares (${self.holdings[i] * self.current_data.iloc[i]['price']:.2f})")
        print("-------------------------------")