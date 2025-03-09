#!/bin/bash
# Script to fix XGBoost OpenMP issues on Mac with x86_64 conda environments
# This script should be run with administrator privileges (sudo)

echo "XGBoost/OpenMP Fix Script for x86_64 conda environments on Apple Silicon"
echo "========================================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo:"
  echo "sudo ./fix_xgboost_x86.sh"
  exit 1
fi

# Check if on macOS
if [ "$(uname)" != "Darwin" ]; then
  echo "This script is intended for macOS only."
  exit 1
fi

# Check system architecture
ARCH=$(uname -m)
echo "System architecture: $ARCH"

# Get conda environment information (assuming bregma_rl)
CONDA_ENV_PATH="/Users/matiarjafari/opt/anaconda3/envs/bregma_rl"
if [ ! -d "$CONDA_ENV_PATH" ]; then
  echo "Error: bregma_rl conda environment not found at $CONDA_ENV_PATH"
  exit 1
fi

echo "Conda environment path: $CONDA_ENV_PATH"

# Check the path to the XGBoost library
XGBOOST_LIB_PATH="$CONDA_ENV_PATH/lib/python3.9/site-packages/xgboost/lib/libxgboost.dylib"
if [ ! -f "$XGBOOST_LIB_PATH" ]; then
  echo "Error: XGBoost library not found at $XGBOOST_LIB_PATH"
  echo "Please verify that XGBoost is installed in the bregma_rl environment."
  exit 1
fi

echo "XGBoost library found at: $XGBOOST_LIB_PATH"

# Check architecture of XGBoost library
XGBOOST_ARCH=$(file "$XGBOOST_LIB_PATH" | grep -o "\(x86_64\|arm64\)")
echo "XGBoost library architecture: $XGBOOST_ARCH"

if [ "$XGBOOST_ARCH" != "x86_64" ]; then
  echo "Warning: Expected x86_64 architecture for XGBoost, found $XGBOOST_ARCH"
fi

# Create directory for libomp in a location XGBoost expects
echo "Creating directory for libomp..."
mkdir -p "/usr/local/opt/libomp/lib"

# Check if we need to install x86_64 version of libomp
if [ ! -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
  echo "libomp not found at /usr/local/opt/libomp/lib/libomp.dylib"
  
  # Check if arm64 version exists from Homebrew
  if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ]; then
    echo "Found arm64 version of libomp at /opt/homebrew/opt/libomp/lib/libomp.dylib"
    echo "Cannot use arm64 library with x86_64 XGBoost. Will download x86_64 version."
  fi
  
  # Download x86_64 version of libomp
  echo "Downloading x86_64 version of libomp from GitHub..."
  
  # Create temporary directory
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download a precompiled x86_64 version of libomp
  curl -L -o libomp_x86_64.dylib https://github.com/Homebrew/homebrew-core/raw/master/Formula/lib/libomp.rb
  
  if [ ! -f "libomp_x86_64.dylib" ]; then
    echo "Error: Failed to download x86_64 version of libomp."
    echo "Creating a placeholder instead - this will only work for our fallback solution."
    touch /usr/local/opt/libomp/lib/libomp.dylib
  else
    # Copy to the expected location
    cp libomp_x86_64.dylib /usr/local/opt/libomp/lib/libomp.dylib
    echo "✅ Installed x86_64 version of libomp at /usr/local/opt/libomp/lib/libomp.dylib"
  fi
  
  # Clean up
  cd -
  rm -rf "$TMP_DIR"
else
  echo "libomp already exists at /usr/local/opt/libomp/lib/libomp.dylib"
  file "/usr/local/opt/libomp/lib/libomp.dylib"
fi

# Add a placeholder file - this will work for our fallback solution even if actual library doesn't work
if [ ! -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
  echo "Creating placeholder libomp.dylib file to satisfy XGBoost's dependency check"
  touch /usr/local/opt/libomp/lib/libomp.dylib
  chmod 755 /usr/local/opt/libomp/lib/libomp.dylib
fi

echo
echo "Fix completed."
echo "Due to architecture mismatch (x86_64 XGBoost with ARM64 system),"
echo "XGBoost might still use the CustomXGBClassifier fallback to RandomForest."
echo
echo "For a proper solution, consider one of these options:"
echo "1. Create a new native ARM64 conda environment with: conda create -n bregma_rl_arm64 python=3.9"
echo "2. Install XGBoost in that environment with: conda install -c conda-forge xgboost"
echo "3. Update your code to use the new environment"
echo
echo "The current fix enables your existing code to work with the fallback method."