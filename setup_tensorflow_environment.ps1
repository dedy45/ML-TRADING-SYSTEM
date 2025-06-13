# TENSORFLOW ENVIRONMENT SETUP SCRIPT
# Run this AFTER Anaconda is installed and working

Write-Host "🤖 SETTING UP TENSORFLOW ENVIRONMENT" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Step 1: Verify Anaconda installation
Write-Host "🔍 Step 1: Verifying Anaconda installation..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version
    Write-Host "✅ Anaconda/Conda found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Conda not found! Please ensure Anaconda is properly installed." -ForegroundColor Red
    Write-Host "Try closing and reopening PowerShell, or restart your computer." -ForegroundColor Yellow
    exit 1
}

# Step 2: Create TensorFlow environment with Python 3.11
Write-Host "🐍 Step 2: Creating TensorFlow environment with Python 3.11..." -ForegroundColor Yellow
conda create -n tensorflow_trading python=3.11 -y

# Step 3: Activate environment
Write-Host "🔄 Step 3: Activating tensorflow_trading environment..." -ForegroundColor Yellow
conda activate tensorflow_trading

# Step 4: Install core packages
Write-Host "📦 Step 4: Installing core ML packages..." -ForegroundColor Yellow

# Install TensorFlow and core ML libraries
conda install -c conda-forge tensorflow=2.15 -y
conda install -c conda-forge scikit-learn=1.3 -y
conda install -c conda-forge pandas=2.1 -y
conda install -c conda-forge numpy=1.24 -y

# Install additional ML packages
conda install -c conda-forge matplotlib seaborn plotly -y
conda install -c conda-forge jupyter notebook -y

# Install MLflow
pip install mlflow==2.8.1

# Install trading-specific packages
pip install TA-Lib
pip install yfinance
pip install python-dotenv

# Step 5: Verify TensorFlow installation
Write-Host "✅ Step 5: Verifying TensorFlow installation..." -ForegroundColor Yellow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"

# Step 6: Create environment activation script
Write-Host "📝 Step 6: Creating environment activation script..." -ForegroundColor Yellow
$activateScript = @"
# TENSORFLOW TRADING ENVIRONMENT ACTIVATION
# Run this script to activate your TensorFlow environment

Write-Host "🚀 Activating TensorFlow Trading Environment..." -ForegroundColor Green
conda activate tensorflow_trading

Write-Host "📊 Environment Information:" -ForegroundColor Yellow
python --version
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import numpy as np; print(f'Numpy: {np.__version__}')"

Write-Host "✅ Ready for TensorFlow trading analysis!" -ForegroundColor Green
Write-Host "Navigate to: E:\aiml\MLFLOW\" -ForegroundColor Cyan
Write-Host "Run: python enhanced_fibonacci_tensorflow.py" -ForegroundColor Cyan
"@

$activateScript | Out-File -FilePath "E:\aiml\MLFLOW\activate_tensorflow_env.ps1"

Write-Host "✅ TENSORFLOW ENVIRONMENT SETUP COMPLETED!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "📋 ENVIRONMENT READY FOR:" -ForegroundColor Yellow
Write-Host "• TensorFlow 2.15 Deep Learning" -ForegroundColor White
Write-Host "• Scikit-learn Machine Learning" -ForegroundColor White
Write-Host "• MLflow Experiment Tracking" -ForegroundColor White
Write-Host "• Advanced Fibonacci Analysis" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "🚀 TO START TRADING ANALYSIS:" -ForegroundColor Cyan
Write-Host "1. Run: conda activate tensorflow_trading" -ForegroundColor White
Write-Host "2. Navigate to: E:\aiml\MLFLOW\" -ForegroundColor White
Write-Host "3. Run: python enhanced_fibonacci_tensorflow.py" -ForegroundColor White
