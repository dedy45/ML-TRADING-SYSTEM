# Activate the fibonacci-ml-simple conda environment
# Usage: .\activate_environment.ps1

Write-Host "Activating fibonacci-ml-simple environment..." -ForegroundColor Green

# Method 1: Direct activation (works in most cases)
try {
    conda activate fibonacci-ml-simple
    Write-Host "Environment activated successfully!" -ForegroundColor Green
    Write-Host "You can now run your Python scripts with full ML capabilities." -ForegroundColor Cyan
} catch {
    Write-Host "Direct activation failed. Using alternative method..." -ForegroundColor Yellow
    
    # Method 2: Direct python execution
    Write-Host "Environment location: C:\Users\dedy\anaconda3\envs\fibonacci-ml-simple" -ForegroundColor Cyan
    Write-Host "To run Python scripts, use:" -ForegroundColor Cyan
    Write-Host "C:\Users\dedy\anaconda3\envs\fibonacci-ml-simple\python.exe your_script.py" -ForegroundColor White
}

# Test the environment
Write-Host "`nTesting environment..." -ForegroundColor Green
& "C:\Users\dedy\anaconda3\envs\fibonacci-ml-simple\python.exe" -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} ready!')"
