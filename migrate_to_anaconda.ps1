# COMPREHENSIVE PYTHON 3.13.4 TO ANACONDA MIGRATION SCRIPT
# This script will uninstall Python 3.13.4 and install Anaconda with TensorFlow-ready environment

Write-Host "🚀 STARTING PYTHON ENVIRONMENT MIGRATION" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Step 1: Backup PATH before changes
Write-Host "📋 Step 1: Backing up current PATH..." -ForegroundColor Yellow
$originalPath = $env:PATH
$originalPath | Out-File -FilePath "E:\aiml\MLFLOW\original_path_backup.txt"
Write-Host "✅ PATH backed up to original_path_backup.txt" -ForegroundColor Green

# Step 2: Remove Python from PATH
Write-Host "🧹 Step 2: Removing Python 3.13.4 from PATH..." -ForegroundColor Yellow
$pathArray = $env:PATH -split ';'
$cleanedPath = $pathArray | Where-Object { $_ -notlike "*Python313*" -and $_ -notlike "*python313*" }
$env:PATH = $cleanedPath -join ';'

# Update system PATH permanently
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
$systemPath = (Get-ItemProperty -Path $regPath -Name PATH).PATH
$systemPathArray = $systemPath -split ';'
$cleanedSystemPath = $systemPathArray | Where-Object { $_ -notlike "*Python313*" -and $_ -notlike "*python313*" }
$newSystemPath = $cleanedSystemPath -join ';'
Set-ItemProperty -Path $regPath -Name PATH -Value $newSystemPath

Write-Host "✅ Python paths removed from PATH" -ForegroundColor Green

# Step 3: Uninstall Python 3.13.4 components
Write-Host "🗑️ Step 3: Uninstalling Python 3.13.4 components..." -ForegroundColor Yellow

$pythonProducts = Get-WmiObject -Class Win32_Product | Where-Object Name -like "*Python 3.13.4*"
foreach ($product in $pythonProducts) {
    Write-Host "Uninstalling: $($product.Name)" -ForegroundColor Cyan
    try {
        $product.Uninstall() | Out-Null
        Write-Host "✅ Successfully uninstalled: $($product.Name)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Error uninstalling $($product.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Step 4: Remove Python directory if it still exists
Write-Host "📁 Step 4: Cleaning up Python directory..." -ForegroundColor Yellow
if (Test-Path "C:\Python313") {
    try {
        Remove-Item -Path "C:\Python313" -Recurse -Force
        Write-Host "✅ Removed C:\Python313 directory" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not remove C:\Python313: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "You may need to manually delete this folder after restart" -ForegroundColor Yellow
    }
}

# Step 5: Download and Install Anaconda
Write-Host "⬇️ Step 5: Downloading Anaconda..." -ForegroundColor Yellow
$anacondaUrl = "https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Windows-x86_64.exe"
$anacondaInstaller = "E:\aiml\MLFLOW\Anaconda3-installer.exe"

Write-Host "Downloading from: $anacondaUrl" -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $anacondaUrl -OutFile $anacondaInstaller -UseBasicParsing
    Write-Host "✅ Anaconda installer downloaded" -ForegroundColor Green
} catch {
    Write-Host "❌ Error downloading Anaconda: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please manually download from: https://www.anaconda.com/download" -ForegroundColor Yellow
    exit 1
}

Write-Host "🚀 Step 6: Installing Anaconda (silent install)..." -ForegroundColor Yellow
Write-Host "This will take several minutes..." -ForegroundColor Cyan

try {
    # Silent install with default options
    Start-Process -FilePath $anacondaInstaller -ArgumentList "/S", "/AddToPath=1" -Wait
    Write-Host "✅ Anaconda installation completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Error during Anaconda installation: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please run the installer manually: $anacondaInstaller" -ForegroundColor Yellow
}

Write-Host "🔄 Step 7: Refreshing environment..." -ForegroundColor Yellow
# Refresh environment variables
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")

Write-Host "✅ MIGRATION COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Close this PowerShell window" -ForegroundColor White
Write-Host "2. Open a NEW PowerShell window as Administrator" -ForegroundColor White
Write-Host "3. Run: conda --version (to verify installation)" -ForegroundColor White
Write-Host "4. Run the TensorFlow environment setup script" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "⚠️ IMPORTANT: Restart your computer if conda is not recognized" -ForegroundColor Red
