#!/bin/bash
# Script to fix XGBoost OpenMP issues on macOS
# This script should be run with administrator privileges (sudo)

echo "XGBoost/OpenMP Fix Script"
echo "========================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo:"
  echo "sudo ./fix_xgboost.sh"
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

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
  echo "Homebrew not found. Please install Homebrew first."
  echo "Visit https://brew.sh for installation instructions."
  exit 1
fi

# Install libomp if not already installed
if ! brew list libomp &> /dev/null; then
  echo "Installing libomp via Homebrew..."
  brew install libomp
else
  echo "libomp is already installed."
fi

# Get libomp location
LIBOMP_PATH=$(brew --prefix libomp)
LIBOMP_LIB_PATH="$LIBOMP_PATH/lib"
LIBOMP_DYLIB="$LIBOMP_LIB_PATH/libomp.dylib"

echo "libomp location: $LIBOMP_DYLIB"

# Check if libomp.dylib exists
if [ ! -f "$LIBOMP_DYLIB" ]; then
  echo "Error: libomp.dylib not found at $LIBOMP_DYLIB"
  exit 1
fi

# Create symlinks to standard locations
echo "Creating symlinks for libomp.dylib..."

# Create /usr/local/opt/libomp/lib if it doesn't exist
mkdir -p /usr/local/opt/libomp/lib

# Create symbolic link
ln -sf "$LIBOMP_DYLIB" /usr/local/opt/libomp/lib/libomp.dylib
echo "Created symlink: /usr/local/opt/libomp/lib/libomp.dylib -> $LIBOMP_DYLIB"

# Get conda location and environment
CONDA_EXE=$(which conda)
if [ -z "$CONDA_EXE" ]; then
  echo "Warning: conda not found in PATH."
else
  echo "Found conda at: $CONDA_EXE"
  
  # Get active conda environment
  CONDA_ENV=$(conda info --envs | grep "\*" | awk '{print $1}')
  echo "Active conda environment: $CONDA_ENV"
  
  # Offer to reinstall XGBoost
  echo
  echo "Would you like to reinstall XGBoost in the bregma_rl environment? (y/n)"
  read -r answer
  if [ "$answer" = "y" ]; then
    echo "Reinstalling XGBoost..."
    conda install -c conda-forge xgboost -n bregma_rl -y
  fi
fi

echo
echo "Fix completed. Please try running the XGBoost code again."
echo "If problems persist, consider using the CustomXGBClassifier fallback approach."
