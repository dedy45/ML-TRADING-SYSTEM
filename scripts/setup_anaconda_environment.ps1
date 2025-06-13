# Setup script untuk Anaconda environment
# Automated environment setup and validation

param(
    [switch]$CreateEnv,
    [switch]$ActivateEnv,
    [switch]$InstallDeps,
    [switch]$Validate,
    [switch]$All
)

Write-Host "🐍 ANACONDA ENVIRONMENT SETUP FOR FIBONACCI DEEP LEARNING" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green

# Function to check if conda is available
function Test-CondaAvailable {
    try {
        $condaVersion = conda --version 2>$null
        if ($condaVersion) {
            Write-Host "✅ Conda detected: $condaVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "❌ Conda not found in PATH" -ForegroundColor Red
        return $false
    }
    return $false
}

# Function to create conda environment
function New-FibonacciEnvironment {
    Write-Host "🔧 Creating Fibonacci ML environment..." -ForegroundColor Yellow
    
    $envFile = "config\environment.yml"
    if (-not (Test-Path $envFile)) {
        Write-Host "❌ Environment file not found: $envFile" -ForegroundColor Red
        return $false
    }
    
    try {
        Write-Host "Creating environment from $envFile..." -ForegroundColor Yellow
        conda env create -f $envFile --force
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Environment created successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Failed to create environment" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Error creating environment: $_" -ForegroundColor Red
        return $false
    }
}

# Function to activate environment
function Enable-FibonacciEnvironment {
    Write-Host "🔄 Activating fibonacci-ml environment..." -ForegroundColor Yellow
    
    try {
        # Check if environment exists
        $envList = conda env list 2>$null
        if ($envList -notmatch "fibonacci-ml") {
            Write-Host "❌ Environment 'fibonacci-ml' not found" -ForegroundColor Red
            Write-Host "💡 Run with -CreateEnv to create the environment first" -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "✅ Environment found, please run:" -ForegroundColor Green
        Write-Host "conda activate fibonacci-ml" -ForegroundColor Cyan
        return $true
    }
    catch {
        Write-Host "❌ Error checking environment: $_" -ForegroundColor Red
        return $false
    }
}

# Function to install additional dependencies
function Install-AdditionalDependencies {
    Write-Host "📦 Installing additional dependencies..." -ForegroundColor Yellow
    
    # Check if in correct environment
    $currentEnv = $env:CONDA_DEFAULT_ENV
    if ($currentEnv -ne "fibonacci-ml") {
        Write-Host "⚠️  Not in fibonacci-ml environment (current: $currentEnv)" -ForegroundColor Yellow
        Write-Host "💡 Please activate environment first: conda activate fibonacci-ml" -ForegroundColor Yellow
        return $false
    }
    
    try {
        # Install via pip (packages not available in conda)
        Write-Host "Installing asyncio-timeout..." -ForegroundColor Yellow
        pip install asyncio-timeout async-timeout retrying

        # Install optional packages
        Write-Host "Installing optional packages..." -ForegroundColor Yellow
        pip install psutil memory-profiler line-profiler

        Write-Host "✅ Additional dependencies installed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Error installing dependencies: $_" -ForegroundColor Red
        return $false
    }
}

# Function to validate installation
function Test-FibonacciEnvironment {
    Write-Host "🔍 Validating installation..." -ForegroundColor Yellow
    
    $validation = @()
    
    # Test Python
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion) {
            Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
            $validation += $true
        } else {
            Write-Host "❌ Python not found" -ForegroundColor Red
            $validation += $false
        }
    }
    catch {
        Write-Host "❌ Python test failed" -ForegroundColor Red
        $validation += $false
    }
    
    # Test core libraries
    $libraries = @(
        "numpy", "pandas", "scikit-learn", "matplotlib", 
        "seaborn", "mlflow", "joblib", "scipy"
    )
    
    foreach ($lib in $libraries) {
        try {
            $result = python -c "import $lib; print(f'$lib: {$lib.__version__}')" 2>$null
            if ($result) {
                Write-Host "✅ $result" -ForegroundColor Green
                $validation += $true
            } else {
                Write-Host "❌ $lib: Import failed" -ForegroundColor Red
                $validation += $false
            }
        }
        catch {
            Write-Host "❌ $lib: Test failed" -ForegroundColor Red
            $validation += $false
        }
    }
    
    # Test TensorFlow (optional)
    try {
        $tfResult = python -c "import tensorflow as tf; print(f'tensorflow: {tf.__version__}')" 2>$null
        if ($tfResult) {
            Write-Host "✅ $tfResult (optional)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  TensorFlow not available (optional)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "⚠️  TensorFlow test failed (optional)" -ForegroundColor Yellow
    }
    
    # Calculate success rate
    $successCount = ($validation | Where-Object { $_ -eq $true }).Count
    $totalCount = $validation.Count
    $successRate = [math]::Round(($successCount / $totalCount) * 100, 1)
    
    Write-Host "`n📊 VALIDATION RESULTS" -ForegroundColor Cyan
    Write-Host "Success Rate: $successRate% ($successCount/$totalCount)" -ForegroundColor Cyan
    
    if ($successRate -ge 90) {
        Write-Host "🎉 Environment validation successful!" -ForegroundColor Green
        return $true
    } elseif ($successRate -ge 70) {
        Write-Host "⚠️  Environment partially validated" -ForegroundColor Yellow
        return $true
    } else {
        Write-Host "❌ Environment validation failed" -ForegroundColor Red
        return $false
    }
}

# Function to test the analyzer
function Test-FibonacciAnalyzer {
    Write-Host "🧪 Testing Fibonacci analyzer..." -ForegroundColor Yellow
    
    try {
        # Run quick validation
        $testResult = python -c "
import sys
sys.path.append('.')
from config.config import config
print('✅ Configuration loaded')
print(f'Data path: {config.data.data_path}')
print(f'Target win rate: {config.trading.target_win_rate:.1%}')
print('✅ Quick test passed')
" 2>$null

        if ($testResult) {
            Write-Host $testResult -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Analyzer test failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Error testing analyzer: $_" -ForegroundColor Red
        return $false
    }
}

# Function to show environment info
function Show-EnvironmentInfo {
    Write-Host "`n📋 ENVIRONMENT INFORMATION" -ForegroundColor Cyan
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor White
    Write-Host "PowerShell version: $($PSVersionTable.PSVersion)" -ForegroundColor White
    
    if (Test-CondaAvailable) {
        $condaInfo = conda info --envs 2>$null
        Write-Host "`nConda environments:" -ForegroundColor White
        Write-Host $condaInfo -ForegroundColor Gray
    }
    
    Write-Host "`nCurrent environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor White
}

# Main execution
Write-Host "`n🚀 Starting environment setup..." -ForegroundColor Green

# Check if conda is available
if (-not (Test-CondaAvailable)) {
    Write-Host "❌ Anaconda/Miniconda not found!" -ForegroundColor Red
    Write-Host "💡 Please install Anaconda or Miniconda first:" -ForegroundColor Yellow
    Write-Host "   https://www.anaconda.com/products/distribution" -ForegroundColor Yellow
    exit 1
}

# Show environment info
Show-EnvironmentInfo

# Execute based on parameters
if ($All) {
    Write-Host "`n🔄 Running complete setup..." -ForegroundColor Green
    
    $success = $true
    $success = $success -and (New-FibonacciEnvironment)
    $success = $success -and (Enable-FibonacciEnvironment)
    
    if ($success) {
        Write-Host "`n✅ Environment setup completed!" -ForegroundColor Green
        Write-Host "🔄 Please run the following commands:" -ForegroundColor Yellow
        Write-Host "1. conda activate fibonacci-ml" -ForegroundColor Cyan
        Write-Host "2. python enhanced_fibonacci_analyzer.py" -ForegroundColor Cyan
    } else {
        Write-Host "`n❌ Setup failed!" -ForegroundColor Red
        exit 1
    }
}
elseif ($CreateEnv) {
    New-FibonacciEnvironment
}
elseif ($ActivateEnv) {
    Enable-FibonacciEnvironment
}
elseif ($InstallDeps) {
    Install-AdditionalDependencies
}
elseif ($Validate) {
    Test-FibonacciEnvironment
    Test-FibonacciAnalyzer
}
else {
    Write-Host "`n💡 USAGE:" -ForegroundColor Yellow
    Write-Host ".\setup_anaconda_environment.ps1 -All          # Complete setup" -ForegroundColor White
    Write-Host ".\setup_anaconda_environment.ps1 -CreateEnv    # Create environment only" -ForegroundColor White
    Write-Host ".\setup_anaconda_environment.ps1 -ActivateEnv  # Activate environment" -ForegroundColor White
    Write-Host ".\setup_anaconda_environment.ps1 -InstallDeps  # Install additional deps" -ForegroundColor White
    Write-Host ".\setup_anaconda_environment.ps1 -Validate     # Validate installation" -ForegroundColor White
    
    Write-Host "`n🔧 RECOMMENDED WORKFLOW:" -ForegroundColor Cyan
    Write-Host "1. .\setup_anaconda_environment.ps1 -All" -ForegroundColor White
    Write-Host "2. conda activate fibonacci-ml" -ForegroundColor White
    Write-Host "3. .\setup_anaconda_environment.ps1 -Validate" -ForegroundColor White
    Write-Host "4. python enhanced_fibonacci_analyzer.py" -ForegroundColor White
}

Write-Host "`n🏁 Setup script completed!" -ForegroundColor Green
