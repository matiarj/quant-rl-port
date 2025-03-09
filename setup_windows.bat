@echo off
echo Setting up Bregma RL Portfolio environment for Windows
echo ====================================================

:: Check Python version
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9+ and try again.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. Please install venv module.
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    exit /b 1
)

:: Verify installation
echo Verifying installation...
python -c "import torch; import numpy; import pandas; import xgboost; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__); print('XGBoost:', xgboost.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo Verification failed. Some packages may not be installed correctly.
    exit /b 1
)

echo.
echo Setup complete! Your environment is ready to use.
echo To activate this environment in the future, run: venv\Scripts\activate
echo.
echo To run the basic training, use: python main.py --config config.yaml
echo.