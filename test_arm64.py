import platform
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import os

print(f"Python version: {platform.python_version()}")
print(f"Python architecture: {platform.machine()}")
print(f"System architecture: {platform.uname().machine}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"XGBoost version: {xgb.__version__}")

# Check if PyTorch can use MPS (Metal Performance Shaders)
print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")

# Try to create an XGBoost classifier
try:
    clf = xgb.XGBClassifier()
    print("✅ Successfully created XGBClassifier")
except Exception as e:
    print(f"❌ Error creating XGBClassifier: {e}")

print("\nEnvironment variables:")
for var in ['CONDA_PREFIX', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")
