#!/usr/bin/env python
"""
Fix XGBoost Script

This script provides options to fix XGBoost compatibility issues on Mac,
particularly related to the libomp library and architecture mismatch.
"""

import os
import sys
import platform
import subprocess
import logging
import argparse
from typing import Dict, List, Any, Optional
import glob
import shutil

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("xgboost_fix")

# Define path constants
HOMEBREW_ARM_LIBOMP = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
HOMEBREW_X86_LIBOMP = "/usr/local/opt/libomp/lib/libomp.dylib"

def check_sudo() -> bool:
    """Check if we have sudo access."""
    if os.geteuid() == 0:
        return True
        
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    info = {
        "system": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "is_conda": "conda" in sys.executable or "Anaconda" in sys.version,
    }
    
    if info["system"] == "Darwin":
        info["is_mac"] = True
        info["is_arm_mac"] = info["architecture"] == "arm64"
        
        # Check if running under Rosetta
        try:
            result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], 
                                  capture_output=True, text=True, check=False)
            info["under_rosetta"] = result.stdout.strip() == "1"
        except:
            info["under_rosetta"] = "Unknown"
    else:
        info["is_mac"] = False
        info["is_arm_mac"] = False
        info["under_rosetta"] = False
    
    return info

def find_libomp_paths() -> List[str]:
    """Find all libomp.dylib paths on the system."""
    paths = []
    
    # Common locations
    common_paths = [
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/Cellar/libomp/*/lib/libomp.dylib",
        "/usr/local/Cellar/libomp/*/lib/libomp.dylib"
    ]
    
    # Expand glob patterns
    for path in common_paths:
        if '*' in path:
            paths.extend(glob.glob(path))
        elif os.path.exists(path):
            paths.append(path)
    
    return paths

def get_file_architecture(path: str) -> Optional[str]:
    """Get the architecture of a binary file."""
    if not os.path.exists(path):
        return None
        
    try:
        output = subprocess.run(["file", path], capture_output=True, text=True, check=True)
        if "x86_64" in output.stdout:
            return "x86_64"
        elif "arm64" in output.stdout:
            return "arm64"
        else:
            return "unknown"
    except:
        return None

def create_symlink(source: str, target: str, use_sudo: bool = False) -> bool:
    """Create a symbolic link."""
    if not os.path.exists(source):
        logger.error(f"Source file does not exist: {source}")
        return False
        
    # Make sure the target directory exists
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        try:
            if use_sudo:
                subprocess.run(["sudo", "mkdir", "-p", target_dir], check=True)
            else:
                os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {target_dir}: {e}")
            return False
    
    # Create the symlink
    try:
        if os.path.exists(target):
            if os.path.islink(target):
                if use_sudo:
                    subprocess.run(["sudo", "rm", target], check=True)
                else:
                    os.remove(target)
            else:
                logger.error(f"Target exists and is not a symlink: {target}")
                return False
                
        if use_sudo:
            subprocess.run(["sudo", "ln", "-s", source, target], check=True)
        else:
            os.symlink(source, target)
            
        logger.info(f"Created symlink: {source} -> {target}")
        return True
    except Exception as e:
        logger.error(f"Failed to create symlink: {e}")
        return False

def fix_conda_xgboost(system_info: Dict[str, Any], force: bool = False) -> bool:
    """Fix XGBoost in the current conda environment."""
    if not system_info["is_conda"]:
        logger.warning("Not running in a conda environment, skipping conda-specific fixes")
        return False
        
    try:
        # Get conda path
        conda_path = None
        if "CONDA_EXE" in os.environ:
            conda_path = os.environ["CONDA_EXE"]
        else:
            python_path = sys.executable
            conda_path = python_path.replace("bin/python", "bin/conda")
            
        if not conda_path or not os.path.exists(conda_path):
            logger.error(f"Could not find conda executable: {conda_path}")
            return False
            
        # Get current conda env
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "base")
        logger.info(f"Installing XGBoost in conda environment: {env_name}")
        
        # Install XGBoost from conda-forge
        channel_arg = "-c conda-forge"
        if system_info["is_arm_mac"]:
            logger.info("Detected ARM Mac, installing arm64-compatible XGBoost")
            
        # Run conda install
        cmd = [conda_path, "install", "-y", channel_arg, "xgboost"]
        if force:
            cmd.append("--force-reinstall")
            
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        logger.info("XGBoost successfully installed via conda")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install XGBoost via conda: {e}")
        return False

def create_libomp_symlinks(libomp_paths: List[str], use_sudo: bool) -> bool:
    """Create necessary symlinks for libomp."""
    if not libomp_paths:
        logger.error("No libomp libraries found")
        return False
        
    success = False
    
    # Find arm64 and x86_64 libraries
    arm_path = None
    x86_path = None
    
    for path in libomp_paths:
        arch = get_file_architecture(path)
        if arch == "arm64":
            arm_path = path
        elif arch == "x86_64":
            x86_path = path
    
    # Create appropriate symlinks
    if arm_path and not os.path.exists(HOMEBREW_ARM_LIBOMP):
        create_symlink(arm_path, HOMEBREW_ARM_LIBOMP, use_sudo)
        success = True
        
    if x86_path and not os.path.exists(HOMEBREW_X86_LIBOMP):
        create_symlink(x86_path, HOMEBREW_X86_LIBOMP, use_sudo)
        success = True
        
    # If we have an ARM library but need an x86 one (or vice versa)
    if arm_path and not x86_path and not os.path.exists(HOMEBREW_X86_LIBOMP):
        logger.warning("Creating symlink from ARM64 to x86_64 (may not work due to architecture mismatch)")
        create_symlink(arm_path, HOMEBREW_X86_LIBOMP, use_sudo)
        success = True
        
    if x86_path and not arm_path and not os.path.exists(HOMEBREW_ARM_LIBOMP):
        logger.warning("Creating symlink from x86_64 to ARM64 (may not work due to architecture mismatch)")
        create_symlink(x86_path, HOMEBREW_ARM_LIBOMP, use_sudo)
        success = True
    
    return success

def install_libomp() -> bool:
    """Install libomp using brew."""
    try:
        logger.info("Installing libomp with Homebrew")
        result = subprocess.run(["brew", "install", "libomp"], check=True)
        logger.info("Successfully installed libomp")
        return True
    except Exception as e:
        logger.error(f"Failed to install libomp: {e}")
        return False

def create_xgboost_wrapper(env_setup: Dict[str, str]) -> bool:
    """Create a wrapper script for XGBoost."""
    # Determine location to create the wrapper
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        logger.error("Cannot create wrapper: CONDA_PREFIX not set")
        return False
        
    wrapper_dir = os.path.join(conda_prefix, "etc", "conda", "activate.d")
    os.makedirs(wrapper_dir, exist_ok=True)
    
    wrapper_path = os.path.join(wrapper_dir, "xgboost_env_vars.sh")
    
    # Create wrapper script content
    content = "#!/bin/bash\n\n# XGBoost environment variables for libomp\n\n"
    
    for key, value in env_setup.items():
        content += f"export {key}=\"{value}\"\n"
    
    # Write the wrapper script
    try:
        with open(wrapper_path, "w") as f:
            f.write(content)
        
        # Make executable
        os.chmod(wrapper_path, 0o755)
        logger.info(f"Created XGBoost wrapper script: {wrapper_path}")
        
        # Also apply these variables to current environment
        for key, value in env_setup.items():
            os.environ[key] = value
            
        return True
    except Exception as e:
        logger.error(f"Failed to create wrapper script: {e}")
        return False

def main() -> None:
    """Main function to fix XGBoost issues."""
    parser = argparse.ArgumentParser(description="Fix XGBoost compatibility issues on Mac.")
    parser.add_argument("--fix-conda", action="store_true", help="Fix XGBoost in conda environment")
    parser.add_argument("--create-symlinks", action="store_true", help="Create libomp symlinks")
    parser.add_argument("--install-libomp", action="store_true", help="Install libomp with Homebrew")
    parser.add_argument("--create-wrapper", action="store_true", help="Create XGBoost wrapper script")
    parser.add_argument("--all", action="store_true", help="Apply all fixes")
    parser.add_argument("--force", action="store_true", help="Force reinstall/recreation of files")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for operations that require it")
    
    args = parser.parse_args()
    
    # Check if any action was specified
    if not any([args.fix_conda, args.create_symlinks, args.install_libomp, 
                args.create_wrapper, args.all]):
        parser.print_help()
        return
    
    # Get system information
    system_info = get_system_info()
    logger.info(f"System: {system_info['system']}, Architecture: {system_info['architecture']}")
    logger.info(f"Python: {system_info['python_version']}, Path: {system_info['python_path']}")
    
    if not system_info["is_mac"]:
        logger.warning("This script is designed to fix XGBoost issues on Mac")
    
    # Check sudo if needed
    if args.sudo:
        if not check_sudo():
            logger.error("Sudo access required but not available")
            return
    
    # Find existing libomp paths
    libomp_paths = find_libomp_paths()
    if libomp_paths:
        logger.info(f"Found libomp libraries: {libomp_paths}")
    else:
        logger.warning("No libomp libraries found")
    
    # Apply fixes based on arguments
    if args.all or args.install_libomp:
        if not libomp_paths:
            install_libomp()
            # Refresh paths
            libomp_paths = find_libomp_paths()
    
    if args.all or args.create_symlinks:
        create_libomp_symlinks(libomp_paths, args.sudo)
    
    if args.all or args.fix_conda:
        fix_conda_xgboost(system_info, args.force)
    
    if args.all or args.create_wrapper:
        # Create environment setup
        env_setup = {}
        
        # Find the appropriate libomp path
        arm_path = None
        x86_path = None
        
        for path in libomp_paths:
            arch = get_file_architecture(path)
            if arch == "arm64":
                arm_path = path
            elif arch == "x86_64":
                x86_path = path
        
        # Set appropriate environment variables
        if system_info["is_arm_mac"]:
            if arm_path:
                lib_dir = os.path.dirname(arm_path)
                env_setup["DYLD_LIBRARY_PATH"] = f"{lib_dir}:$DYLD_LIBRARY_PATH"
                env_setup["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_dir}:$DYLD_FALLBACK_LIBRARY_PATH"
        else:
            if x86_path:
                lib_dir = os.path.dirname(x86_path)
                env_setup["DYLD_LIBRARY_PATH"] = f"{lib_dir}:$DYLD_LIBRARY_PATH"
                env_setup["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_dir}:$DYLD_FALLBACK_LIBRARY_PATH"
        
        create_xgboost_wrapper(env_setup)
    
    logger.info("Fix attempts completed")

if __name__ == "__main__":
    main()