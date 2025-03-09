import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import pickle
import time
import joblib
from tqdm import tqdm
import platform
import importlib.util
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

# Helper functions for parallel processing - defined at module level
def train_single_classifier(args):
    """Train a single classifier in parallel.
    
    Args:
        args: Tuple of (name, classifier, states, labels)
        
    Returns:
        Dictionary with training results
    """
    name, clf, states, labels = args
    result = {
        'name': name,
        'success': False,
        'metrics': {}
    }
    
    try:
        # Fit classifier
        clf.fit(states, labels)
        result['success'] = True
        
        # Use scikit-learn's built-in parallel cross-validation
        n_folds = min(5, len(states))
        cv_scores = cross_val_score(
            clf, states, labels, 
            cv=n_folds, 
            n_jobs=-1  # Use all available cores
        )
        result['metrics'][f'{name}_cv_score'] = cv_scores.mean()
        
        # Make predictions
        predictions = clf.predict(states)
        acc = accuracy_score(labels, predictions)
        result['metrics'][f'{name}_accuracy'] = acc
        
        # Store the trained classifier
        result['classifier'] = clf
        
        # Log success
        result['log_message'] = f"Trained {name}: CV score={cv_scores.mean():.4f}, Accuracy={acc:.4f}"
        
    except Exception as e:
        result['error'] = str(e)
        result['log_message'] = f"Error training {name}: {e}"
    
    return result

def analyze_feature_importance(args):
    """Analyze feature importance for a classifier.
    
    Args:
        args: Tuple of (name, classifier)
        
    Returns:
        Dictionary with feature importance results
    """
    name, clf = args
    result = {'name': name, 'success': False}
    
    try:
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            top_indices = np.argsort(importances)[-10:]
            result['top_indices'] = top_indices
            result['importance_values'] = importances[top_indices]
            result['success'] = True
        else:
            result['error'] = f"{name} trained but feature_importances_ not available"
    except Exception as e:
        result['error'] = str(e)
        
    return result

# Setup logger
logger = logging.getLogger('ClassifierEnsemble')

# Determine if we're on Apple Silicon and get the Python architecture
is_apple_silicon = platform.processor() == 'arm' or platform.machine() == 'arm64'
python_arch = platform.machine()
logger.info(f"System processor: {platform.processor()}")
logger.info(f"Python architecture: {python_arch}")

# Check for XGBoost without actually importing it
XGBOOST_AVAILABLE = False
XGBClassifier = None  # Will be set to the actual class or a mock

# Use importlib.util.find_spec to check if xgboost is available without importing it
if importlib.util.find_spec("xgboost") is not None:
    logger.info("XGBoost package found in environment")
    try:
        # Try to import using a safer approach that won't fail if the library can't be loaded
        # This will only attempt to load XGBoost and its dependencies when we explicitly use it
        def get_xgboost():
            try:
                import xgboost as xgb
                return xgb.XGBClassifier
            except:
                return None
        
        # We'll set it to a function that returns the actual class or None
        XGBClassifier = get_xgboost
        XGBOOST_AVAILABLE = True
        logger.info("XGBoost import function prepared successfully")
    except Exception as e:
        logger.warning(f"Error preparing XGBoost import: {e}")
else:
    logger.warning("XGBoost package not found in environment")

# Define a CustomXGBClassifier that gracefully falls back to RandomForest if needed
class CustomXGBClassifier:
    """
    A wrapper around XGBoost that falls back to RandomForest if XGBoost fails
    due to architecture or library issues.
    """
    def __init__(self, **kwargs):
        self.params = kwargs
        self.classifier = None
        self.using_fallback = False
        
        # Try to use XGBoost if available
        if XGBOOST_AVAILABLE:
            try:
                # Add parallelization settings if not already provided
                if 'nthread' not in kwargs:
                    # Use number of cores minus 1, but at least 1
                    n_threads = max(1, mp.cpu_count() - 1)
                    kwargs['nthread'] = n_threads
                
                # Ensure we have the right parameters for XGBoost
                if 'use_label_encoder' not in kwargs:
                    kwargs['use_label_encoder'] = False
                
                # Get the XGBClassifier class (or None if it fails)
                XGBClassifierClass = XGBClassifier()
                
                if XGBClassifierClass is not None:
                    self.classifier = XGBClassifierClass(**kwargs)
                    logger.info(f"Using XGBoost classifier with {kwargs.get('nthread')} threads")
                else:
                    logger.warning("XGBoost not available, falling back to RandomForest")
                    self.using_fallback = True
            except Exception as e:
                logger.warning(f"Error initializing XGBClassifier: {e}")
                self.using_fallback = True
        else:
            self.using_fallback = True
            
        # Fall back to RandomForest if needed
        if self.using_fallback:
            # Convert XGBoost parameters to optimized RandomForest equivalents
            rf_params = {
                'n_estimators': kwargs.get('n_estimators', 200),
                'max_depth': kwargs.get('max_depth', 6),
                'min_samples_split': 3,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': kwargs.get('random_state', 42),
                'n_jobs': max(1, mp.cpu_count() - 1)  # Add parallel processing
            }
            self.classifier = RandomForestClassifier(**rf_params)
            logger.info(f"Using RandomForest as fallback classifier with {rf_params['n_jobs']} parallel jobs")
            
    def fit(self, X, y):
        return self.classifier.fit(X, y)
        
    def predict(self, X):
        return self.classifier.predict(X)
        
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
    def get_params(self, deep=True):
        return self.params
    
    def __getattr__(self, name):
        # Forward all other attribute access to the underlying classifier
        return getattr(self.classifier, name)
    
    def __str__(self):
        if self.using_fallback:
            return f"CustomXGBClassifier(using_fallback=RandomForest)"
        else:
            return f"CustomXGBClassifier(using_xgboost=True)"

# Import check is already done above

class ClassifierEnsemble:
    """Classifier ensemble for agent selection."""
    
    def __init__(self, config: Dict[str, Any], agent_names: List[str]):
        """
        Initialize ClassifierEnsemble.
        
        Args:
            config: Configuration dictionary
            agent_names: Names of the agents in the ensemble
        """
        self.config = config
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        self.logger = self._setup_logger()
        
        # Initialize classifiers
        self.classifiers = self._initialize_classifiers()
        
        # Settings for when to use the ensemble
        # Lower variance threshold to be more aggressive about using the ensemble predictions
        # Analysis shows the DQN was overused (16 vs 1 and 1)
        default_threshold = 0.1  # Lower than the typical 0.2 value
        self.variance_threshold = config['classifier'].get('variance_threshold', default_threshold)
        
        # Training data storage
        self.training_states = []
        self.training_rewards = []
        self.training_agent_names = []
        
        # Track agent usage
        self.agent_usage = {name: 0 for name in agent_names}
        
        # Statistics
        self.train_samples = 0
        self.predict_calls = 0
        self.exploit_decisions = 0
        self.risk_averse_decisions = 0
        
        # Feature importance analysis
        self.feature_importance_analysis = config['classifier'].get('feature_importance_analysis', False)
        
        self.logger.info(f"Initialized ClassifierEnsemble with {len(self.classifiers)} classifiers")
        self.logger.info(f"Agent names: {agent_names}")
        self.logger.info(f"Variance threshold: {self.variance_threshold}")
        
        if 'xgboost' in self.classifiers:
            self.logger.info("XGBoost is enabled and ready to use")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger."""
        logger = logging.getLogger('ClassifierEnsemble')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/classifier_ensemble.log')
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
    
    def _initialize_classifiers(self) -> Dict[str, Any]:
        """Initialize classifiers."""
        classifiers = {}
        
        # Get classifier types from config
        clf_types = self.config['classifier'].get('types', ['SVM', 'DecisionTree', 'RandomForest', 'LogisticRegression', 'XGBoost'])
        
        # Support Vector Machine
        if 'SVM' in clf_types:
            classifiers['svm_linear'] = SVC(kernel='linear', probability=True, random_state=42)
            classifiers['svm_rbf'] = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Decision Tree
        if 'DecisionTree' in clf_types:
            classifiers['decision_tree'] = DecisionTreeClassifier(
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            )
        
        # Random Forest with parallelization and optimized parameters
        if 'RandomForest' in clf_types:
            # Use number of cores minus 1, but at least 1
            n_jobs = max(1, mp.cpu_count() - 1)
            classifiers['random_forest'] = RandomForestClassifier(
                n_estimators=200,           # Increased from 100 for better accuracy
                max_depth=12,               # Increased from 10 for more complex patterns
                min_samples_split=3,        # Decreased from 5 to capture more detailed patterns
                min_samples_leaf=2,         # Added to prevent overfitting
                max_features='sqrt',        # Use sqrt of features for each tree (common practice)
                bootstrap=True,             # Use bootstrap samples
                class_weight='balanced',    # Handle potential class imbalance
                random_state=42,
                n_jobs=n_jobs              # Parallelize across multiple cores
            )
            self.logger.info(f"Initialized RandomForest with {n_jobs} parallel jobs and optimized parameters")
        
        # Logistic Regression
        if 'LogisticRegression' in clf_types:
            classifiers['logistic_regression'] = LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=42
            )
            
                                # XGBoost with automatic fallback if needed
        if 'XGBoost' in clf_types:
            try:
                # Get the number of cores for parallelization
                n_threads = max(1, mp.cpu_count() - 1)
                
                # Use our CustomXGBClassifier with optimized parameters for financial data
                classifiers['xgboost'] = CustomXGBClassifier(
                    n_estimators=300,            # More trees for better accuracy
                    max_depth=6,                 # Slightly deeper but not too deep to avoid overfitting
                    learning_rate=0.05,          # Lower learning rate for more stability
                    min_child_weight=3,          # Higher to prevent overfitting on noisy financial data
                    subsample=0.8,               # Use 80% of data for each tree to reduce overfitting
                    colsample_bytree=0.8,        # Use 80% of features for each tree
                    gamma=0.1,                   # Minimum loss reduction for partition
                    reg_alpha=0.1,               # L1 regularization
                    reg_lambda=1.0,              # L2 regularization
                    random_state=42,
                    use_label_encoder=False,     # Avoid warning in newer XGBoost versions
                    eval_metric='mlogloss',      # Required for multi-class problems
                    nthread=n_threads,           # Parallelize XGBoost
                    tree_method='auto',          # Allow XGBoost to choose best method
                    objective='multi:softprob'   # Multi-class probability output
                )
                
                if hasattr(classifiers['xgboost'], 'using_fallback') and classifiers['xgboost'].using_fallback:
                    self.logger.info("Added XGBoost classifier (using RandomForest fallback)")
                else:
                    self.logger.info("Successfully added native XGBoost classifier")
            except Exception as e:
                self.logger.error(f"Error initializing XGBoost: {e}")
                self.logger.warning("Continuing without XGBoost")
        
        # If no classifiers specified, add default ones
        if not classifiers:
            classifiers['svm_rbf'] = SVC(kernel='rbf', probability=True, random_state=42)
            classifiers['decision_tree'] = DecisionTreeClassifier(random_state=42)
            
        return classifiers
    
    def train(self) -> Dict[str, float]:
        """
        Train the classifiers on accumulated data.
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        # Check if we have enough data
        if len(self.training_states) == 0:
            self.logger.warning("No training data available")
            return metrics
        
        # Convert training data to numpy arrays
        states = np.array(self.training_states)
        
        # Find best agent for each state based on reward
        best_agents = []
        
        # Group by unique state index and find best agent for each
        state_dict = {}
        for i, (state, agent_name, reward) in enumerate(zip(self.training_states, self.training_agent_names, self.training_rewards)):
            state_key = i  # Use index since state vectors may not be hashable
            
            if state_key not in state_dict or reward > state_dict[state_key][1]:
                state_dict[state_key] = (agent_name, reward)
                
        # Convert agent names to indices
        agent_to_idx = {name: i for i, name in enumerate(self.agent_names)}
        best_agent_names = [state_dict[i][0] for i in range(len(state_dict))]
        best_agent_indices = [agent_to_idx[name] for name in best_agent_names]
        
        # Apply feature selection if enabled
        if self.feature_importance_analysis and 'random_forest' in self.classifiers:
            # Use Random Forest for feature selection
            try:
                # First ensure the classifier is properly trained
                rf = self.classifiers['random_forest']
                
                # The model must be trained first - this will be done in the training loop below
                # Skip feature importance calculation here, and do it after training
            except Exception as e:
                self.logger.error(f"Error in feature importance analysis: {e}")
        
        # Apply variance thresholding to remove low-variance features
        try:
            selector = VarianceThreshold(threshold=0.01)  # Very low threshold just to remove constant features
            states_filtered = selector.fit_transform(states)
            
            # Only use filtered states if they retain enough features
            if states_filtered.shape[1] > states.shape[1] / 2:
                states = states_filtered
                self.logger.info(f"Applied variance thresholding: {states.shape[1]} features retained out of {states.shape[1]}")
        except Exception as e:
            self.logger.error(f"Error in variance thresholding: {e}")
        
        self.train_samples += len(states)
        
        # Log start of training
        n_classifiers = len(self.classifiers)
        self.logger.info(f"Starting training of {n_classifiers} classifiers on {len(states)} samples")
        
        # Determine number of parallel workers
        n_workers = min(max(1, mp.cpu_count() - 1), n_classifiers)
        self.logger.info(f"Using {n_workers} parallel workers for classifier training")
        
        # Prepare arguments for parallel training
        train_args = [(name, clf, states, best_agent_indices) for name, clf in self.classifiers.items()]
        
        # Use ProcessPoolExecutor for parallel training
        results = []
        with tqdm(total=n_classifiers, desc="Training classifiers", unit="classifier") as pbar:
            # Use ProcessPoolExecutor since this is a CPU-bound task
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(train_single_classifier, args) for args in train_args]
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    if 'log_message' in result:
                        pbar.set_description(f"Training: {result['name']}")
                        if result['success']:
                            self.logger.info(result['log_message'])
                        else:
                            self.logger.error(result['log_message'])
        
        # Update classifiers and metrics
        for result in results:
            if result['success']:
                name = result['name']
                # Update metrics
                metrics.update(result['metrics'])
                # Update classifier (in case it was modified during training)
                if 'classifier' in result:
                    self.classifiers[name] = result['classifier']
            else:
                name = result['name']
                metrics[f'{name}_error'] = result.get('error', 'Unknown error')
                
        # Process feature importances and XGBoost performance after all classifiers are trained
        # First check XGBoost performance
        if 'xgboost' in self.classifiers and 'xgboost' in metrics and f"xgboost_error" not in metrics and XGBOOST_AVAILABLE:
            try:
                cv_score = metrics.get('xgboost_cv_score', 0)
                accuracy = metrics.get('xgboost_accuracy', 0)
                self._check_xgboost_performance(cv_score, accuracy)
            except Exception as e:
                self.logger.error(f"Error checking XGBoost performance: {e}")
        
        # Only analyze tree-based models with feature importance capability
        if self.feature_importance_analysis:
            models_to_analyze = [
                (name, clf) for name, clf in self.classifiers.items() 
                if name in ['random_forest', 'xgboost'] 
                and (f"{name}_error" not in metrics)
            ]
            
            if models_to_analyze:
                self.logger.info(f"Analyzing feature importance for {len(models_to_analyze)} models in parallel")
                
                # Use ThreadPoolExecutor as this is less computationally intensive
                with tqdm(total=len(models_to_analyze), desc="Feature importance analysis", unit="model") as pbar:
                    with ThreadPoolExecutor(max_workers=len(models_to_analyze)) as executor:
                        futures = [executor.submit(analyze_feature_importance, item) for item in models_to_analyze]
                        
                        for future in as_completed(futures):
                            result = future.result()
                            name = result['name']
                            
                            # Update progress bar
                            pbar.update(1)
                            pbar.set_description(f"Analyzing: {name}")
                            
                            if result['success']:
                                self.logger.info(f"Top 10 most important features for {name}: {result['top_indices']}")
                                self.logger.info(f"Importance values: {result['importance_values']}")
                            else:
                                self.logger.error(f"Error in feature importance analysis for {name}: {result.get('error', 'Unknown error')}")
        
        # Clear training data after successful training
        self.training_states = []
        self.training_rewards = []
        self.training_agent_names = []
        
        return metrics
    
    def add_training_data(self, agent_name: str, state: np.ndarray, reward: float) -> None:
        """
        Add training data from an agent's experience.
        
        Args:
            agent_name: Name of the agent
            state: State vector
            reward: Reward received
        """
        # Store the data for future training
        self.training_states.append(state)
        self.training_rewards.append(reward)
        self.training_agent_names.append(agent_name)
        
        # Log
        self.logger.debug(f"Added training data for agent {agent_name} with reward {reward:.6f}")
    
    def select_agent(self, state: np.ndarray) -> str:
        """
        Select the best agent for the given state.
        
        Args:
            state: State vector
            
        Returns:
            Name of the selected agent
        """
        agent_idx, _, _ = self.predict(state)
        selected_agent = self.agent_names[agent_idx]
        
        # Track usage
        self.agent_usage[selected_agent] += 1
        
        return selected_agent
    
    def predict(self, state: np.ndarray) -> Tuple[int, float, bool]:
        """
        Predict the best agent for the given state.
        
        Args:
            state: State vector
            
        Returns:
            Tuple of:
            - best_agent_idx: Index of the best agent
            - variance: Variance of predictions across classifiers
            - exploit: Whether to exploit (True) or be risk-averse (False)
        """
        self.predict_calls += 1
        
        # Ensure state is 2D (required by scikit-learn)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each classifier
        for name, clf in self.classifiers.items():
            try:
                pred = clf.predict(state)[0]
                predictions[name] = pred
                
                # Get probability distribution over classes
                if hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba(state)[0]
                    probabilities[name] = probs
                    
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
                continue
        
        # If no predictions were made, return a random agent and indicate risk-averse
        if not predictions:
            self.risk_averse_decisions += 1
            return np.random.randint(self.num_agents), 1.0, False
        
        # Calculate voting result
        vote_counts = np.zeros(self.num_agents)
        for pred in predictions.values():
            vote_counts[pred] += 1
        
        # Calculate variance of predictions
        if probabilities:
            # If we have probabilities, calculate the variance of the probability distributions
            prob_matrix = np.zeros((len(probabilities), self.num_agents))
            for i, probs in enumerate(probabilities.values()):
                prob_matrix[i] = probs
            
            # Variance of predictions across classifiers
            variance = np.mean(np.var(prob_matrix, axis=0))
        else:
            # If no probabilities, use the variance of the vote distribution
            variance = np.var(vote_counts / len(predictions))
        
        # Determine whether to exploit or be risk-averse
        exploit = variance < self.variance_threshold
        
        # Select best agent
        best_agent_idx = np.argmax(vote_counts)
        
        # Update statistics
        if exploit:
            self.exploit_decisions += 1
        else:
            self.risk_averse_decisions += 1
        
        self.logger.debug(f"Predicted agent {best_agent_idx} with variance {variance:.4f}, exploit={exploit}")
        
        return best_agent_idx, variance, exploit
    
    def _check_xgboost_performance(self, cv_score: float, accuracy: float) -> None:
        """
        Check if XGBoost is performing well. If not, log a warning.
        
        Args:
            cv_score: Cross-validation score
            accuracy: Accuracy on training data
        """
        # Define thresholds for good performance
        CV_THRESHOLD = 0.5
        ACC_THRESHOLD = 0.5
        
        if cv_score < CV_THRESHOLD or accuracy < ACC_THRESHOLD:
            self.logger.warning(
                f"XGBoost performance is below threshold: CV={cv_score:.4f}, Accuracy={accuracy:.4f}. "
                f"Consider tuning hyperparameters or checking for data issues."
            )
        else:
            self.logger.info(f"XGBoost performance is good: CV={cv_score:.4f}, Accuracy={accuracy:.4f}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ensemble."""
        return {
            "train_samples": self.train_samples,
            "predict_calls": self.predict_calls,
            "exploit_decisions": self.exploit_decisions,
            "risk_averse_decisions": self.risk_averse_decisions,
            "exploit_ratio": self.exploit_decisions / max(1, self.predict_calls),
            "variance_threshold": self.variance_threshold
        }
    
    def save(self, path: str) -> None:
        """
        Save classifiers to disk.
        
        Args:
            path: Directory to save the classifiers
        """
        os.makedirs(path, exist_ok=True)
        timestamp = int(time.time())
        filepath = os.path.join(path, f'classifiers_{timestamp}.joblib')
        
        try:
            joblib.dump({
                'classifiers': self.classifiers,
                'agent_names': self.agent_names,
                'stats': self.get_stats()
            }, filepath)
            self.logger.info(f"Classifiers saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving classifiers: {e}")
    
    def load(self, path: str) -> None:
        """
        Load classifiers from disk.
        
        Args:
            path: Path to the saved classifiers
        """
        if not os.path.exists(path):
            self.logger.error(f"Classifier path {path} does not exist")
            return
        
        try:
            data = joblib.load(path)
            self.classifiers = data['classifiers']
            
            # Check if agent names match
            if set(data['agent_names']) != set(self.agent_names):
                self.logger.warning(f"Loaded classifiers were trained for different agents: {data['agent_names']}")
            
            # Restore statistics
            stats = data.get('stats', {})
            self.train_samples = stats.get('train_samples', 0)
            self.predict_calls = stats.get('predict_calls', 0)
            self.exploit_decisions = stats.get('exploit_decisions', 0)
            self.risk_averse_decisions = stats.get('risk_averse_decisions', 0)
            
            self.logger.info(f"Classifiers loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading classifiers: {e}")