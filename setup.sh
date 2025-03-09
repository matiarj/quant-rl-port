#!/bin/bash
# Setup script for Linux and Intel-based macOS

echo "Setting up Bregma RL Portfolio environment"
echo "=========================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Please install venv module."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

# Verify installation
echo "Verifying installation..."
python -c "import torch; import numpy; import pandas; import xgboost; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__); print('XGBoost:', xgboost.__version__)"
if [ $? -ne 0 ]; then
    echo "Verification failed. Some packages may not be installed correctly."
    exit 1
fi

echo
echo "Setup complete! Your environment is ready to use."
echo "To activate this environment in the future, run: source venv/bin/activate"
echo
echo "To run the basic training, use: python main.py --config config.yaml"
echo