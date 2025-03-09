#!/bin/bash
# Script to create a native arm64 conda environment for bregma_rl

echo "Setting up native arm64 environment for bregma_rl"
echo "================================================="

# Check if we're running on arm64 Mac
if [[ "$(uname)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This script should be run on an arm64 Mac (M1/M2/M3)."
    echo "Make sure you're not running in Rosetta 2 mode."
    exit 1
fi

# Create a new conda environment for arm64
echo "Creating new conda environment: bregma_rl_arm64..."
conda create -n bregma_rl_arm64 python=3.9 -y

# Activate the new environment
echo "Activating bregma_rl_arm64 environment..."
eval "$(conda shell.bash hook)"
conda activate bregma_rl_arm64

# Install core dependencies from conda-forge (optimized for arm64)
echo "Installing core dependencies from conda-forge..."
conda install -c conda-forge xgboost -y
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn -y
conda install -c conda-forge pyyaml tqdm joblib -y

# Install PyTorch (optimized for arm64)
echo "Installing PyTorch (optimized for Apple Silicon)..."
conda install -c pytorch pytorch -y

# Install remaining dependencies with pip
echo "Installing remaining dependencies with pip..."
pip install gym==0.26.2
pip install sympy==1.13.3 
pip install cloudpickle==3.1.1
pip install networkx==3.2.1
pip install typing-extensions==4.12.2

# Verify installations
echo "Verifying XGBoost installation..."
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"

echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}')"

echo "Creating a test script to verify architecture..."
cat > test_arm64.py << 'EOL'
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
EOL

echo "Running test script..."
python test_arm64.py

echo
echo "Setup complete! Your new native arm64 environment is ready."
echo "To use this environment in the future, run: conda activate bregma_rl_arm64"
echo 
echo "NOTE: This environment is optimized for Apple Silicon and should provide better performance."
echo "You may now need to update any environment-specific paths in your code."