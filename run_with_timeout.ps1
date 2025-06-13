# PowerShell script with timeout protection
param(
    [string]$ScriptName = "basic_csv_test.py",
    [int]$TimeoutSeconds = 10
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "TIMEOUT PROTECTED PYTHON EXECUTION" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Script: $ScriptName" -ForegroundColor Yellow
Write-Host "Timeout: $TimeoutSeconds seconds" -ForegroundColor Yellow
Write-Host ""

# Start the Python process
$process = Start-Process python -ArgumentList $ScriptName -PassThru -NoNewWindow -RedirectStandardOutput "output.txt" -RedirectStandardError "error.txt"

# Wait for completion or timeout
$completed = $process.WaitForExit($TimeoutSeconds * 1000)

if ($completed) {
    Write-Host "[SUCCESS] Script completed within timeout" -ForegroundColor Green
    Write-Host "Exit code: $($process.ExitCode)" -ForegroundColor Cyan
    
    # Show output
    if (Test-Path "output.txt") {
        Write-Host "`n--- OUTPUT ---" -ForegroundColor Yellow
        Get-Content "output.txt"
    }
    
    if (Test-Path "error.txt") {
        $errorContent = Get-Content "error.txt"
        if ($errorContent) {
            Write-Host "`n--- ERRORS ---" -ForegroundColor Red
            Write-Host $errorContent -ForegroundColor Red
        }
    }
} else {
    Write-Host "[TIMEOUT] Script exceeded $TimeoutSeconds seconds - KILLING PROCESS" -ForegroundColor Red
    
    # Kill the process
    try {
        $process.Kill()
        $process.WaitForExit()
        Write-Host "[KILLED] Process terminated successfully" -ForegroundColor Yellow
    } catch {
        Write-Host "[ERROR] Failed to kill process: $_" -ForegroundColor Red
    }
}

# Cleanup
if (Test-Path "output.txt") { Remove-Item "output.txt" -Force }
if (Test-Path "error.txt") { Remove-Item "error.txt" -Force }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "EXECUTION COMPLETED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
