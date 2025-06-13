# Enhanced Deep Learning Fibonacci Setup and Troubleshooting
# PowerShell script to setup and validate the enhanced deep learning fibonacci module

Write-Host "🚀 ENHANCED DEEP LEARNING FIBONACCI SETUP" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Function to check if command exists
function Test-CommandExists {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to run Python command with timeout
function Invoke-PythonWithTimeout {
    param(
        [string]$Command,
        [int]$TimeoutSeconds = 300
    )
    
    Write-Host "⏱️ Running with $TimeoutSeconds second timeout: $Command" -ForegroundColor Yellow
    
    try {
        $job = Start-Job -ScriptBlock {
            param($cmd)
            python -c $cmd
        } -ArgumentList $Command
        
        if (Wait-Job $job -Timeout $TimeoutSeconds) {
            $result = Receive-Job $job
            Remove-Job $job
            return $result
        } else {
            Write-Host "⏰ Command timed out after $TimeoutSeconds seconds" -ForegroundColor Red
            Stop-Job $job
            Remove-Job $job
            return $null
        }
    } catch {
        Write-Host "❌ Error running command: $_" -ForegroundColor Red
        return $null
    }
}

# Step 1: Environment Validation
Write-Host "`n🔍 STEP 1: ENVIRONMENT VALIDATION" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

# Check Python
if (Test-CommandExists "python") {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check pip
if (Test-CommandExists "pip") {
    Write-Host "✅ pip available" -ForegroundColor Green
} else {
    Write-Host "❌ pip not found" -ForegroundColor Red
    exit 1
}

# Check conda (optional)
if (Test-CommandExists "conda") {
    Write-Host "✅ Anaconda/Miniconda available" -ForegroundColor Green
    $useAnaconda = $true
} else {
    Write-Host "⚠️ Anaconda not found - using pip instead" -ForegroundColor Yellow
    $useAnaconda = $false
}

# Step 2: Directory Structure Setup
Write-Host "`n📁 STEP 2: DIRECTORY STRUCTURE SETUP" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

$requiredDirs = @(
    "deep_learning_fibonacci",
    "config",
    "utils", 
    "data",
    "core",
    "models",
    "logs",
    "notebooks"
)

foreach ($dir in $requiredDirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "📁 Directory exists: $dir" -ForegroundColor Cyan
    }
}

# Step 3: Dependencies Installation
Write-Host "`n📦 STEP 3: DEPENDENCIES INSTALLATION" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

$corePackages = @(
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "scikit-learn>=1.0.0",
    "joblib>=1.1.0",
    "mlflow>=2.0.0"
)

$optionalPackages = @(
    "tensorflow>=2.10.0",
    "keras>=2.10.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "jupyter>=1.0.0"
)

Write-Host "Installing core packages..." -ForegroundColor Yellow
foreach ($package in $corePackages) {
    Write-Host "📦 Installing $package" -ForegroundColor Cyan
    try {
        if ($useAnaconda) {
            conda install $package.Split('>=')[0] -y 2>&1 | Out-Null
        } else {
            pip install $package --quiet
        }
        Write-Host "✅ Installed $package" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Failed to install $package" -ForegroundColor Yellow
    }
}

Write-Host "`nInstalling optional packages..." -ForegroundColor Yellow
foreach ($package in $optionalPackages) {
    Write-Host "📦 Installing $package (optional)" -ForegroundColor Cyan
    try {
        if ($useAnaconda) {
            conda install $package.Split('>=')[0] -y 2>&1 | Out-Null
        } else {
            pip install $package --quiet
        }
        Write-Host "✅ Installed $package" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Optional package $package failed to install" -ForegroundColor Yellow
    }
}

# Step 4: Module Validation
Write-Host "`n🧪 STEP 4: MODULE VALIDATION" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

# Test basic imports
$testImports = @"
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print('✅ Core dependencies working')
"@

$importResult = Invoke-PythonWithTimeout -Command $testImports -TimeoutSeconds 30
if ($importResult -like "*✅*") {
    Write-Host "✅ Core dependencies validated" -ForegroundColor Green
} else {
    Write-Host "❌ Core dependency validation failed" -ForegroundColor Red
}

# Test MLflow
$testMLflow = @"
try:
    import mlflow
    print('✅ MLflow available')
except ImportError:
    print('⚠️ MLflow not available')
"@

$mlflowResult = Invoke-PythonWithTimeout -Command $testMLflow -TimeoutSeconds 15
Write-Host $mlflowResult

# Test TensorFlow (optional)
$testTensorFlow = @"
try:
    import tensorflow as tf
    print('✅ TensorFlow available')
except ImportError:
    print('⚠️ TensorFlow not available (optional)')
"@

$tfResult = Invoke-PythonWithTimeout -Command $testTensorFlow -TimeoutSeconds 20
Write-Host $tfResult

# Step 5: Configuration Files Check
Write-Host "`n⚙️ STEP 5: CONFIGURATION FILES CHECK" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

$configFiles = @(
    "config\config.py",
    "utils\timeout_utils.py",
    "utils\logging_utils.py",
    "data\data_processor.py",
    "core\model_trainer.py",
    "enhanced_fibonacci_analyzer.py"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Missing: $file" -ForegroundColor Yellow
    }
}

# Step 6: Deep Learning Module Test
Write-Host "`n🧠 STEP 6: DEEP LEARNING MODULE TEST" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

if (Test-Path "test_deep_learning_fibonacci.py") {
    Write-Host "🧪 Running deep learning module test..." -ForegroundColor Yellow
    $testResult = Invoke-PythonWithTimeout -Command "exec(open('test_deep_learning_fibonacci.py').read())" -TimeoutSeconds 60
    
    if ($testResult -like "*PASS*") {
        Write-Host "✅ Deep learning module test passed" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Deep learning module test had issues" -ForegroundColor Yellow
        Write-Host $testResult
    }
} else {
    Write-Host "⚠️ Test script not found" -ForegroundColor Yellow
}

# Step 7: Quick Integration Test
Write-Host "`n🔗 STEP 7: QUICK INTEGRATION TEST" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Green

$quickTest = @"
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

try:
    from deep_learning_fibonacci.tensorflow_fibonacci_predictor import FibonacciDeepLearningPredictor
    predictor = FibonacciDeepLearningPredictor()
    print('✅ Integration test passed')
except Exception as e:
    print(f'⚠️ Integration test failed: {e}')
"@

Write-Host "🧪 Running quick integration test..." -ForegroundColor Yellow
$integrationResult = Invoke-PythonWithTimeout -Command $quickTest -TimeoutSeconds 30

if ($integrationResult -like "*✅*") {
    Write-Host "✅ Integration test successful" -ForegroundColor Green
} else {
    Write-Host "⚠️ Integration test failed" -ForegroundColor Yellow
    Write-Host $integrationResult
}

# Step 8: Final Summary and Recommendations
Write-Host "`n📋 STEP 8: SETUP SUMMARY AND RECOMMENDATIONS" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

Write-Host "`n🎯 SETUP COMPLETE!" -ForegroundColor Cyan
Write-Host "`n📊 SYSTEM STATUS:" -ForegroundColor White

# Check overall status
$coreWorking = $importResult -like "*✅*"
$mlflowWorking = $mlflowResult -like "*✅*"
$integrationWorking = $integrationResult -like "*✅*"

if ($coreWorking -and $integrationWorking) {
    Write-Host "🟢 System Status: READY" -ForegroundColor Green
    Write-Host "`n🚀 NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "1. Run: python run_integrated_deep_learning_fibonacci.py" -ForegroundColor White
    Write-Host "2. Check logs/ directory for detailed output" -ForegroundColor White
    Write-Host "3. Review models/ directory for trained models" -ForegroundColor White
    Write-Host "4. Use saved models in your EA MQL5 integration" -ForegroundColor White
} elseif ($coreWorking) {
    Write-Host "🟡 System Status: PARTIAL - Basic functionality available" -ForegroundColor Yellow
    Write-Host "`n🔧 RECOMMENDATIONS:" -ForegroundColor Cyan
    Write-Host "1. Install missing dependencies manually" -ForegroundColor White
    Write-Host "2. Use fallback mode for basic functionality" -ForegroundColor White
    Write-Host "3. Consider using Anaconda environment" -ForegroundColor White
} else {
    Write-Host "🔴 System Status: NEEDS ATTENTION" -ForegroundColor Red
    Write-Host "`n🛠️ TROUBLESHOOTING:" -ForegroundColor Cyan
    Write-Host "1. Check Python installation and version" -ForegroundColor White
    Write-Host "2. Reinstall core dependencies manually" -ForegroundColor White
    Write-Host "3. Check for conflicting package versions" -ForegroundColor White
}

Write-Host "`n📞 SUPPORT:" -ForegroundColor Cyan
Write-Host "• Check MLFLOW logs/ directory for detailed error information" -ForegroundColor White
Write-Host "• Ensure data files are in dataBT/ directory" -ForegroundColor White
Write-Host "• Run with smaller datasets first for testing" -ForegroundColor White

Write-Host "`n✅ ENHANCED DEEP LEARNING FIBONACCI SETUP COMPLETED" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
