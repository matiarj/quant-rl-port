#!/usr/bin/env python
"""
XGBoost Debugging Script for bregma_rl environment

This script analyzes and diagnoses issues with XGBoost installation
in the bregma_rl conda environment, with a focus on fixing OpenMP/libomp
architecture issues on macOS systems.
"""

import os
import sys
import platform
import subprocess
import shutil
import ctypes
from ctypes.util import find_library
import importlib.util

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")

def run_command(cmd, capture_output=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            shell=True, 
            check=True
        )
        return result.stdout if capture_output else "Command executed successfully"
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return None

def check_environment():
    """Check the current Python environment."""
    print("Checking Python environment...")
    
    # Python version and executable
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    conda_prefix = os.environ.get('CONDA_PREFIX', 'Not in a conda environment')
    print(f"Conda prefix: {conda_prefix}")
    
    # Check if we're in bregma_rl
    env_name = os.path.basename(conda_prefix) if conda_prefix != 'Not in a conda environment' else None
    if env_name != 'bregma_rl':
        print("\n⚠️  WARNING: Not running in bregma_rl environment!")
        print("Please run this script with: conda run -n bregma_rl python debug_xgboost.py")
    else:
        print(f"✅ Running in correct environment: {env_name}")
    
    # System architecture
    print(f"System architecture: {platform.machine()}")
    print(f"System: {platform.system()} {platform.release()}")
    
    # CPU information on macOS
    if platform.system() == 'Darwin':
        cpu_info = run_command("sysctl -n machdep.cpu.brand_string")
        print(f"CPU: {cpu_info.strip() if cpu_info else 'Unknown'}")
        
        # Check if we're on Apple Silicon (M1/M2/M3)
        if platform.machine() == 'arm64':
            print("✅ Running on Apple Silicon (M1/M2/M3)")
            print("⚠️  XGBoost might need special handling for ARM architecture")
            
            # Check if running under Rosetta
            rosetta = run_command("sysctl -n sysctl.proc_translated 2>/dev/null || echo 0")
            if rosetta and rosetta.strip() == '1':
                print("✅ Running under Rosetta 2 translation")
            else:
                print("❌ Not running under Rosetta 2 translation")

def check_libomp():
    """Check the OpenMP/libomp installation."""
    print("Checking OpenMP/libomp installation...")
    
    # Check Homebrew installation
    brew_path = shutil.which('brew')
    if not brew_path:
        print("❌ Homebrew not found. It's recommended for installing libomp.")
        brew_installed = False
    else:
        print(f"✅ Homebrew found: {brew_path}")
        brew_installed = True
    
    # Check libomp installation via Homebrew
    if brew_installed:
        libomp_check = run_command("brew list | grep -q libomp && echo installed || echo not installed")
        if libomp_check and "installed" in libomp_check:
            print("✅ libomp is installed via Homebrew")
            
            # Get libomp details
            libomp_info = run_command("brew info libomp")
            print(f"libomp info: {libomp_info.strip() if libomp_info else 'Unknown'}")
            
            # Check libomp location
            libomp_path = run_command("brew --prefix libomp")
            if libomp_path:
                libomp_lib_path = os.path.join(libomp_path.strip(), "lib")
                print(f"libomp library path: {libomp_lib_path}")
                
                # Check if libomp.dylib exists
                libomp_dylib = os.path.join(libomp_lib_path, "libomp.dylib")
                if os.path.exists(libomp_dylib):
                    print(f"✅ Found libomp.dylib: {libomp_dylib}")
                    
                    # Check architecture of libomp.dylib
                    file_info = run_command(f"file {libomp_dylib}")
                    print(f"libomp.dylib file info: {file_info.strip() if file_info else 'Unknown'}")
                    
                    if file_info and "arm64" in file_info:
                        print("⚠️  libomp.dylib is for ARM64 architecture")
                    elif file_info and "x86_64" in file_info:
                        print("⚠️  libomp.dylib is for x86_64 architecture")
                else:
                    print(f"❌ libomp.dylib not found at {libomp_dylib}")
        else:
            print("❌ libomp is not installed via Homebrew")
            print("   You can install it with: brew install libomp")

    # Check for libomp in other common locations
    common_libomp_paths = [
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/lib/libomp.dylib",
        "/usr/lib/libomp.dylib"
    ]
    
    for path in common_libomp_paths:
        if os.path.exists(path):
            print(f"Found libomp at: {path}")
            # Check architecture
            file_info = run_command(f"file {path}")
            print(f"  File info: {file_info.strip() if file_info else 'Unknown'}")
    
    # Check the dynamic library path variables
    print("\nChecking dynamic library paths:")
    for var in ['LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def check_xgboost_installation():
    """Check the XGBoost installation."""
    print("Checking XGBoost installation...")
    
    # Check if XGBoost is installed
    xgboost_spec = importlib.util.find_spec("xgboost")
    if xgboost_spec is None:
        print("❌ XGBoost is not installed in the current environment")
        print("   You can install it with: conda install -c conda-forge xgboost")
        return
    
    print(f"✅ XGBoost package found: {xgboost_spec.origin}")
    
    # Try to import XGBoost and get version
    try:
        import xgboost as xgb
        print(f"✅ XGBoost imported successfully")
        print(f"   Version: {xgb.__version__}")
        
        # Check XGBoost library path
        if hasattr(xgb, '_LIB'):
            print(f"XGBoost library path: {xgb._LIB._name}")
        elif hasattr(xgb, 'core') and hasattr(xgb.core, '_LIB'):
            print(f"XGBoost library path: {xgb.core._LIB._name}")
        else:
            # Try to find the actual library
            xgb_lib_path = os.path.join(os.path.dirname(xgb.__file__), 'lib')
            if os.path.exists(xgb_lib_path):
                print(f"XGBoost library directory: {xgb_lib_path}")
                libs = [f for f in os.listdir(xgb_lib_path) if 'libxgboost' in f]
                for lib in libs:
                    full_path = os.path.join(xgb_lib_path, lib)
                    print(f"Found XGBoost library: {full_path}")
                    file_info = run_command(f"file {full_path}")
                    print(f"  File info: {file_info.strip() if file_info else 'Unknown'}")
            
        # Try to create an XGBoost classifier as a deeper test
        try:
            clf = xgb.XGBClassifier()
            print("✅ Successfully created XGBClassifier instance")
        except Exception as e:
            print(f"❌ Error creating XGBClassifier: {e}")
            print(f"   This suggests runtime/library issues")
            
    except ImportError as e:
        print(f"❌ Error importing XGBoost: {e}")
    except Exception as e:
        print(f"❌ Unexpected error with XGBoost: {e}")

def suggest_fixes():
    """Suggest potential fixes based on the diagnostics."""
    print("Suggesting potential fixes...")
    
    is_macos = platform.system() == 'Darwin'
    is_arm64 = platform.machine() == 'arm64'
    
    if is_macos:
        if is_arm64:
            print("\n1. For Apple Silicon (M1/M2/M3) Macs:")
            print("   - Make sure you're using a conda environment created for ARM64")
            print("   - Try installing XGBoost from conda-forge: conda install -c conda-forge xgboost")
            print("   - If still having issues, consider using Rosetta 2 by running:")
            print("     arch -x86_64 zsh")
            print("     conda activate bregma_rl")
            
        print("\n2. Install libomp with Homebrew:")
        print("   brew install libomp")
        
        print("\n3. Set the OpenMP library path environment variables:")
        print("   export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
        print("   export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_FALLBACK_LIBRARY_PATH")
        
        print("\n4. Create symlinks to help XGBoost find libomp:")
        print("   sudo mkdir -p /usr/local/opt/libomp/lib")
        print("   sudo ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib")
        
        print("\n5. Reinstall XGBoost with the correct architecture:")
        print("   conda install -c conda-forge xgboost")
        
        print("\nNOTE: For options requiring sudo, you'll need administrator privileges.")
    
    print("\n6. Use the CustomXGBClassifier fallback solution (current approach):")
    print("   This approach gracefully falls back to RandomForest when XGBoost can't be initialized")

def create_fix_script():
    """Create a shell script that can fix common issues."""
    fix_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_xgboost.sh")
    
    script_content = """#!/bin/bash
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
  CONDA_ENV=$(conda info --envs | grep "\\*" | awk '{print $1}')
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
"""
    
    with open(fix_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(fix_script_path, 0o755)
    
    print(f"\nCreated fix script: {fix_script_path}")
    print("You can run it with sudo permissions to fix common issues:")
    print(f"sudo {fix_script_path}")

def main():
    """Main function to run all checks."""
    print("\n🔍 XGBoost Debugging Tool for bregma_rl environment 🔍\n")
    
    print_separator()
    check_environment()
    
    print_separator()
    check_libomp()
    
    print_separator()
    check_xgboost_installation()
    
    print_separator()
    suggest_fixes()
    
    print_separator()
    create_fix_script()
    
    print_separator()
    print("Debugging complete. Review the output above for information and suggestions.")
    print("If you need to fix XGBoost with sudo privileges, run the generated fix script.")

if __name__ == "__main__":
    main()