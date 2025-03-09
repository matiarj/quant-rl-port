#!/usr/bin/env python
"""
Script to update ensemble.py to work optimally with native arm64 environment.
"""

import os
import sys
import re
import platform
import shutil
from datetime import datetime

# Check if running in the right environment
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if not conda_prefix or 'bregma_rl_arm64' not in conda_prefix:
    print("⚠️  Warning: This script should ideally be run from the bregma_rl_arm64 environment")
    print(f"Current environment: {conda_prefix}")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# Path to ensemble.py
ensemble_path = '/Users/matiarjafari/bregma/ql_port/src/classifiers/ensemble.py'

# Check if file exists
if not os.path.exists(ensemble_path):
    print(f"Error: Could not find {ensemble_path}")
    sys.exit(1)

# Create backup
backup_path = f"{ensemble_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
shutil.copy2(ensemble_path, backup_path)
print(f"Created backup at {backup_path}")

# Read original file
with open(ensemble_path, 'r') as f:
    content = f.read()

# Updated imports section for optimal XGBoost support
new_imports = """import numpy as np
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

# Setup logger
logger = logging.getLogger('ClassifierEnsemble')

# Determine if we're on Apple Silicon and get the Python architecture
is_apple_silicon = platform.processor() == 'arm' or platform.machine() == 'arm64'
python_arch = platform.machine()
logger.info(f"System processor: {platform.processor()}")
logger.info(f"Python architecture: {python_arch}")

# Check for XGBoost
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    # Test if XGBoost actually works by creating a classifier
    try:
        test_clf = XGBClassifier()
        XGBOOST_AVAILABLE = True
        logger.info(f"✅ XGBoost {xgb.__version__} imported successfully")
    except Exception as e:
        logger.warning(f"⚠️ XGBoost imported but could not be initialized: {e}")
except ImportError:
    logger.warning("⚠️ XGBoost not available. Using RandomForest as fallback.")

# Define a CustomXGBClassifier that gracefully falls back to RandomForest if needed
class CustomXGBClassifier:
    \"\"\"
    A wrapper around XGBoost that falls back to RandomForest if XGBoost fails
    due to architecture or library issues.
    \"\"\"
    def __init__(self, **kwargs):
        self.params = kwargs
        self.classifier = None
        self.using_fallback = False
        
        # Try to use XGBoost if available
        if XGBOOST_AVAILABLE:
            try:
                self.classifier = XGBClassifier(**kwargs)
                logger.info("Using XGBoost classifier")
            except Exception as e:
                logger.warning(f"Error initializing XGBClassifier: {e}")
                self.using_fallback = True
        else:
            self.using_fallback = True
            
        # Fall back to RandomForest if needed
        if self.using_fallback:
            # Convert XGBoost parameters to RandomForest equivalents
            rf_params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', 5),
                'random_state': kwargs.get('random_state', 42)
            }
            self.classifier = RandomForestClassifier(**rf_params)
            logger.info("Using RandomForest as fallback classifier")
            
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
"""

# Updated XGBoost initialization code for _initialize_classifiers method
xgboost_init = """        # XGBoost with automatic fallback if needed
        if 'XGBoost' in clf_types:
            try:
                # Use our CustomXGBClassifier for robust behavior
                classifiers['xgboost'] = CustomXGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,  # Avoid warning in newer XGBoost versions
                    eval_metric='mlogloss'    # Required for multi-class problems
                )
                
                if hasattr(classifiers['xgboost'], 'using_fallback') and classifiers['xgboost'].using_fallback:
                    self.logger.info("Added XGBoost classifier (using RandomForest fallback)")
                else:
                    self.logger.info("Successfully added native XGBoost classifier")
            except Exception as e:
                self.logger.error(f"Error initializing XGBoost: {e}")
                self.logger.warning("Continuing without XGBoost")"""

# Replace the imports section
content = re.sub(r'import numpy as np.*?XGBOOST_AVAILABLE = False', new_imports, content, flags=re.DOTALL)

# Replace the XGBoost initialization part
content = re.sub(r'# XGBoost.*?self\.logger\.warning\("Continuing without XGBoost"\)', xgboost_init, content, flags=re.DOTALL)

# Write the updated file
with open(ensemble_path, 'w') as f:
    f.write(content)

print(f"Updated {ensemble_path} with optimal XGBoost support")
print("This version will work in both x86_64 and arm64 environments")
print(f"The original file was backed up to {backup_path}")

print("\nWhat to do next:")
print("1. Create and activate the arm64 environment: ./setup_arm64.sh")
print("2. Run a test to make sure XGBoost works: python -c \"import xgboost; print('XGBoost works!')\"")
print("3. Run your application with the new environment: conda run -n bregma_rl_arm64 python ./run_enhanced_backtest.py --dry-run")