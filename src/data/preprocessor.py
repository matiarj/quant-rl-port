import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import logging
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from scipy.stats import zscore

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.num_lags = config['preprocessing']['num_lags']
        self.reversal_lookbacks = config['preprocessing']['reversal_lookbacks']
        self.correlation_threshold = config['preprocessing']['correlation_threshold']
        self.train_val_test_split = config['data']['train_val_test_split']
        self.add_technical_indicators = config['preprocessing'].get('add_technical_indicators', False)
        self.market_regime_detection = config['preprocessing'].get('market_regime_detection', False)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger."""
        logger = logging.getLogger('DataPreprocessor')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/data_preprocessor.log')
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
    
    def load_data(self, file_path: str, dry_run: bool = False) -> pd.DataFrame:
        """Load data from CSV file."""
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert Time to datetime
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Sort by Time and symbol
            df = df.sort_values(['Time', 'symbol'])
            
            if dry_run:
                # For dry run, select a moderately sized subset of data
                symbols = df['symbol'].unique()[:10]  # First 10 symbols (increased from 5)
                start_date = df['Time'].min()
                end_date = start_date + pd.Timedelta(days=120)  # 120 days of data (increased from 60)
                
                df = df[(df['symbol'].isin(symbols)) & 
                        (df['Time'] >= start_date) & 
                        (df['Time'] <= end_date)]
                
                self.logger.info(f"Enhanced dry run mode: Using {len(symbols)} symbols and data from "
                                f"{start_date.date()} to {end_date.date()}")
            
            self.logger.info(f"Data loaded with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate data before preprocessing."""
        self.logger.info("Validating data...")
        
        try:
            # Check for NaNs (except in return_vs_spy which can have NaN on first day)
            nan_counts = df.isna().sum()
            if (nan_counts - nan_counts.get('return_vs_spy', 0) > 0).any():
                self.logger.error(f"Data contains NaN values: {nan_counts}")
                raise ValueError("Data contains NaN values")
            
            # Check for non-positive prices
            if (df['price'] <= 0).any():
                invalid_prices = df[df['price'] <= 0]
                self.logger.error(f"Data contains non-positive prices: {invalid_prices.shape[0]} rows")
                self.logger.error(f"Sample: {invalid_prices.head()}")
                raise ValueError("Data contains non-positive prices")
            
            # Check weight sum per day
            daily_weight_sum = df.groupby('Time')['weight'].sum()
            weight_deviation = daily_weight_sum.apply(lambda x: abs(x - 1.0))
            if (weight_deviation > 0.05).any():  # Allow small deviation due to floating point
                bad_days = weight_deviation[weight_deviation > 0.05]
                self.logger.warning(f"Weight sum deviates from 1.0 by more than 5% on {len(bad_days)} days")
                self.logger.warning(f"Sample days: {bad_days.head()}")
            
            # Check for duplicate symbols on same day
            dupes = df.duplicated(subset=['Time', 'symbol'], keep=False)
            if dupes.any():
                duplicate_rows = df[dupes]
                self.logger.error(f"Data contains duplicate symbol entries for the same day: {duplicate_rows.shape[0]} rows")
                self.logger.error(f"Sample: {duplicate_rows.head()}")
                raise ValueError("Data contains duplicate symbol entries for the same day")
                
            self.logger.info("Data validation passed")
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            raise
    
    def _calculate_technical_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a stock."""
        symbol = stock_data['symbol'].iloc[0]
        price = stock_data['price']
        returns = stock_data['log_return_std']
        
        # RSI (Relative Strength Index)
        delta = stock_data['log_return']
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        stock_data['RSI'].fillna(50, inplace=True)  # Fill NaN with neutral value
        
        # Bollinger Bands
        stock_data['BB_middle'] = price.rolling(window=20).mean()
        stock_data['BB_std'] = price.rolling(window=20).std()
        stock_data['BB_upper'] = stock_data['BB_middle'] + (stock_data['BB_std'] * 2)
        stock_data['BB_lower'] = stock_data['BB_middle'] - (stock_data['BB_std'] * 2)
        stock_data['BB_width'] = (stock_data['BB_upper'] - stock_data['BB_lower']) / stock_data['BB_middle']
        
        # BB position (where price is within the bands, normalized to -1 to 1)
        stock_data['BB_position'] = (price - stock_data['BB_middle']) / (stock_data['BB_std'] + 1e-10)
        
        # Stochastic Oscillator
        window = 14
        stock_data['lowest_low'] = price.rolling(window=window).min()
        stock_data['highest_high'] = price.rolling(window=window).max()
        stock_data['%K'] = 100 * ((price - stock_data['lowest_low']) / 
                                  (stock_data['highest_high'] - stock_data['lowest_low'] + 1e-10))
        stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            stock_data[f'ROC_{period}'] = price.pct_change(periods=period) * 100
        
        # Standard Deviation of Returns
        stock_data['volatility_5d'] = returns.rolling(window=5).std()
        stock_data['volatility_20d'] = returns.rolling(window=20).std()
        
        # Fill NaN values
        tech_columns = ['RSI', 'BB_width', 'BB_position', '%K', '%D', 
                        'ROC_5', 'ROC_10', 'ROC_20', 'volatility_5d', 'volatility_20d']
        stock_data[tech_columns] = stock_data[tech_columns].fillna(0)
        
        return stock_data
        
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using clustering on market features."""
        self.logger.info("Detecting market regimes...")
        
        # Group data by date to get market-wide metrics
        market_data = df.groupby('Time').agg({
            'log_return_std': ['mean', 'std'],
            'volatility_20d': 'mean',
            'RSI': 'mean',
            'BB_width': 'mean'
        })
        
        market_data.columns = ['avg_return', 'return_dispersion', 'avg_volatility', 'avg_RSI', 'avg_BB_width']
        
        # Standardize features for clustering
        regime_features = zscore(market_data.fillna(0))
        
        # Apply K-means clustering to identify regimes (3 regimes: bull, bear, neutral/choppy)
        kmeans = KMeans(n_clusters=3, random_state=42)
        market_data['regime'] = kmeans.fit_predict(regime_features)
        
        # Map regimes to interpretable labels based on average returns
        regime_returns = market_data.groupby('regime')['avg_return'].mean()
        
        # Label regimes as bull (highest returns), bear (lowest returns), and neutral (middle)
        regime_mapping = {
            regime_returns.idxmax(): 0,  # Bull market (0)
            regime_returns.idxmin(): 1,  # Bear market (1)
            3 - regime_returns.idxmax() - regime_returns.idxmin(): 2  # Neutral/choppy market (2)
        }
        
        market_data['market_regime'] = market_data['regime'].map(regime_mapping)
        
        # Create dummy variables for the regimes
        regime_dummies = pd.get_dummies(market_data['market_regime'], prefix='regime')
        market_data = pd.concat([market_data, regime_dummies], axis=1)
        
        # Create interaction features
        # For example: Is it a high volatility bull market? Low volatility bear market?
        market_data['bull_volatility'] = market_data['regime_0'] * market_data['avg_volatility']
        market_data['bear_volatility'] = market_data['regime_1'] * market_data['avg_volatility']
        
        # Map the regimes back to individual stock data
        df = df.join(market_data[['market_regime', 'regime_0', 'regime_1', 'regime_2', 
                                 'bull_volatility', 'bear_volatility']], on='Time')
        
        return df
    
    def _process_single_stock(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Process a single stock's data."""
        symbol = stock_data['symbol'].iloc[0]
        
        try:
            # Calculate log returns
            stock_data['log_return'] = np.log(stock_data['price'] / stock_data['price'].shift(1))
            stock_data['log_return'].fillna(0, inplace=True)  # First day's return is 0
            
            # Standardize numerical features
            # Note: return_vs_spy represents forward 7-day returns relative to SPY (lookahead)
            # It should be treated as a target variable rather than a feature
            for col in ['log_return', 'smile', 'btr', 'weight']:
                if col in stock_data.columns:
                    mean = stock_data[col].mean()
                    std = stock_data[col].std()
                    if std != 0:  # Avoid division by zero
                        stock_data[f'{col}_std'] = (stock_data[col] - mean) / std
                    else:
                        stock_data[f'{col}_std'] = 0
                        self.logger.warning(f"Zero std for {col} in {symbol}. Setting {col}_std to 0.")
                        
            # Store return_vs_spy as a target variable (not a feature)
            if 'return_vs_spy' in stock_data.columns:
                mean = stock_data['return_vs_spy'].mean()
                std = stock_data['return_vs_spy'].std()
                if std != 0:
                    stock_data['return_vs_spy_target'] = (stock_data['return_vs_spy'] - mean) / std
                else:
                    stock_data['return_vs_spy_target'] = 0
                    self.logger.warning(f"Zero std for return_vs_spy in {symbol}. Setting return_vs_spy_target to 0.")
            
            # Calculate lagged returns
            for lag in range(1, self.num_lags + 1):
                stock_data[f'log_return_lag{lag}'] = stock_data['log_return_std'].shift(lag)
            
            # Calculate moving averages
            stock_data['MA5_return'] = stock_data['log_return_std'].rolling(window=5).mean()
            stock_data['MA20_return'] = stock_data['log_return_std'].rolling(window=20).mean()
            stock_data['MACD_return'] = stock_data['MA5_return'] - stock_data['MA20_return']
            
            # Do the same for other features (excluding return_vs_spy)
            for feature in ['smile_std', 'btr_std', 'weight_std']:
                if feature in stock_data.columns:
                    stock_data[f'MA5_{feature}'] = stock_data[feature].rolling(window=5).mean()
                    stock_data[f'MA20_{feature}'] = stock_data[feature].rolling(window=20).mean()
                    stock_data[f'MACD_{feature}'] = stock_data[f'MA5_{feature}'] - stock_data[f'MA20_{feature}']
            
            # Create moving averages for the target (for evaluation purposes only)
            if 'return_vs_spy_target' in stock_data.columns:
                stock_data['MA5_return_vs_spy_target'] = stock_data['return_vs_spy_target'].rolling(window=5).mean()
                stock_data['MA20_return_vs_spy_target'] = stock_data['return_vs_spy_target'].rolling(window=20).mean()
            
            # Calculate reversal signals
            for lookback in self.reversal_lookbacks:
                if len(stock_data) > lookback:
                    stock_data[f'reversal_{lookback}d'] = -np.log(
                        stock_data['price'] / stock_data['price'].shift(lookback)
                    )
            
            # Add technical indicators if configured
            if self.add_technical_indicators:
                stock_data = self._calculate_technical_indicators(stock_data)
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error processing stock {symbol}: {e}")
            raise
    
    def process_data_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all stocks in parallel."""
        self.logger.info("Processing data in parallel...")
        
        # Group data by symbol
        grouped = df.groupby('symbol')
        
        # Process each stock in parallel
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            results = pool.map(self._process_single_stock, [group for _, group in grouped])
        
        # Combine processed data
        processed_df = pd.concat(results)
        
        # Sort by Time and symbol
        processed_df = processed_df.sort_values(['Time', 'symbol'])
        
        # Check for NaN/inf after processing
        nan_cols = processed_df.columns[processed_df.isna().any()].tolist()
        
        # Modified inf check to handle mixed types
        inf_cols = []
        for col in processed_df.columns:
            try:
                if processed_df[col].apply(lambda x: np.isinf(x) if isinstance(x, (int, float)) else False).any():
                    inf_cols.append(col)
            except:
                # Skip columns that can't be checked for inf
                pass
        
        if nan_cols:
            self.logger.warning(f"NaN values found in columns after processing: {nan_cols}")
            self.logger.warning(f"NaN counts: {processed_df[nan_cols].isna().sum()}")
            # Fill NaNs with 0 for now
            processed_df[nan_cols] = processed_df[nan_cols].fillna(0)
        
        if inf_cols:
            self.logger.warning(f"Inf values found in columns after processing: {inf_cols}")
            # Replace inf with large values
            for col in inf_cols:
                processed_df[col] = processed_df[col].replace([np.inf, -np.inf], [1e6, -1e6])
        
        self.logger.info(f"Data processing complete. Resulting shape: {processed_df.shape}")
        return processed_df
    
    def one_hot_encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode sector and industry."""
        self.logger.info("One-hot encoding categorical features...")
        
        # One-hot encode sector
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        # One-hot encode industry
        industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
        df = pd.concat([df, industry_dummies], axis=1)
        
        self.logger.info(f"One-hot encoding added {len(sector_dummies.columns) + len(industry_dummies.columns)} new columns")
        return df
    
    def analyze_correlation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """Analyze feature correlations."""
        self.logger.info("Analyzing feature correlations...")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Save correlation matrix visualization
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/correlation_matrix.png')
        plt.close()
        
        # Find highly correlated features
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with correlation above threshold
        for col in upper.columns:
            for idx, value in upper[col].items():
                if abs(value) > self.correlation_threshold:
                    high_corr_pairs.append((idx, col, value))
        
        # Log and save high correlation pairs
        if high_corr_pairs:
            self.logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs:")
            with open('results/high_correlation_pairs.txt', 'w') as f:
                for feat1, feat2, corr in high_corr_pairs:
                    line = f"{feat1} and {feat2}: {corr:.4f}"
                    self.logger.info(line)
                    f.write(line + '\n')
        else:
            self.logger.info("No highly correlated feature pairs found.")
        
        return corr_matrix, high_corr_pairs
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation, and test sets."""
        self.logger.info("Splitting data into train/validation/test sets...")
        
        # Get the available date range
        min_date = df['Time'].min()
        max_date = df['Time'].max()
        
        self.logger.info(f"Available data range: {min_date} to {max_date}")

        # Check if we're in dry run mode by looking at the date range
        is_dry_run = (max_date - min_date).days < 180  # Less than 6 months of data
        
        if is_dry_run:
            # For dry run, use simple percentage-based split
            self.logger.info("Using percentage-based split for dry run mode")
            
            # Sort dates and calculate split points
            dates = sorted(df['Time'].unique())
            train_size = int(len(dates) * 0.6)
            val_size = int(len(dates) * 0.2)
            
            train_end_idx = train_size
            val_end_idx = train_size + val_size
            
            # Split dates
            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]
            test_dates = dates[val_end_idx:]
            
            # Split data
            train_df = df[df['Time'].isin(train_dates)]
            val_df = df[df['Time'].isin(val_dates)]
            test_df = df[df['Time'].isin(test_dates)]
            
        else:
            # For full analysis, use the specified date ranges
            self.logger.info("Using specified date ranges for full analysis")
            
            # Define date ranges for specific train/val/test periods
            train_start = pd.Timestamp('2006-01-01')  # Start of data
            train_end = pd.Timestamp('2016-06-30')    # Mid 2016
            
            val_start = pd.Timestamp('2017-01-01')    # Start of 2017
            val_end = pd.Timestamp('2020-06-30')      # Mid 2020
            
            test_start = pd.Timestamp('2021-01-01')   # Start of 2021
            test_end = pd.Timestamp('2025-12-31')     # End of data (far future to include all)
            
            # Split data based on dates
            train_df = df[(df['Time'] >= train_start) & (df['Time'] <= train_end)]
            val_df = df[(df['Time'] >= val_start) & (df['Time'] <= val_end)]
            test_df = df[(df['Time'] >= test_start) & (df['Time'] <= test_end)]
        
        # Get actual date ranges
        train_dates = sorted(train_df['Time'].unique())
        val_dates = sorted(val_df['Time'].unique())
        test_dates = sorted(test_df['Time'].unique())
        
        self.logger.info(f"Train set: {train_df.shape} from {train_dates[0] if len(train_dates) > 0 else 'N/A'} to {train_dates[-1] if len(train_dates) > 0 else 'N/A'}")
        self.logger.info(f"Validation set: {val_df.shape} from {val_dates[0] if len(val_dates) > 0 else 'N/A'} to {val_dates[-1] if len(val_dates) > 0 else 'N/A'}")
        self.logger.info(f"Test set: {test_df.shape} from {test_dates[0] if len(test_dates) > 0 else 'N/A'} to {test_dates[-1] if len(test_dates) > 0 else 'N/A'}")
        
        return train_df, val_df, test_df
    
    def preprocess(self, df: pd.DataFrame, dry_run: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the full preprocessing pipeline."""
        self.logger.info("Starting preprocessing pipeline...")
        
        try:
            # Validate data
            self.validate_data(df)
            
            # Process data
            processed_df = self.process_data_parallel(df)
            
            # Apply market regime detection if configured
            if self.market_regime_detection and self.add_technical_indicators:
                self.logger.info("Applying market regime detection...")
                processed_df = self._detect_market_regime(processed_df)
            
            # One-hot encode categorical features
            processed_df = self.one_hot_encode_categories(processed_df)
            
            # Analyze correlation
            _, high_corr_pairs = self.analyze_correlation(processed_df)
            
            # Split data
            train_df, val_df, test_df = self.split_data(processed_df)
            
            self.logger.info("Preprocessing complete!")
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            raise