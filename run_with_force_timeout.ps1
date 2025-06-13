# PowerShell script to run Python with timeout and force kill
param(
    [string]$ScriptName = "basic_csv_test.py",
    [int]$TimeoutSeconds = 10
)

Write-Host "Running Python script with timeout: $TimeoutSeconds seconds"
Write-Host "Script: $ScriptName"
Write-Host "=" * 50

# Start Python process
$process = Start-Process python -ArgumentList $ScriptName -PassThru -NoNewWindow -RedirectStandardOutput "temp_output.txt" -RedirectStandardError "temp_error.txt"

# Wait for completion or timeout
$completed = $process.WaitForExit($TimeoutSeconds * 1000)

if ($completed) {
    Write-Host "[SUCCESS] Script completed normally in $($process.ExitTime - $process.StartTime) seconds"
    
    # Show output
    if (Test-Path "temp_output.txt") {
        Write-Host "`n[OUTPUT]"
        Get-Content "temp_output.txt"
    }
    
    # Show errors if any
    if (Test-Path "temp_error.txt" -and (Get-Item "temp_error.txt").Length -gt 0) {
        Write-Host "`n[ERRORS]"
        Get-Content "temp_error.txt"
    }
} else {
    Write-Host "[TIMEOUT] Script hung for more than $TimeoutSeconds seconds - FORCE KILLING"
    
    # Force kill the process
    $process.Kill()
    $process.WaitForExit()
    
    Write-Host "[KILLED] Process terminated forcefully"
    
    # Try to show partial output
    if (Test-Path "temp_output.txt") {
        Write-Host "`n[PARTIAL OUTPUT]"
        Get-Content "temp_output.txt"
    }
}

# Cleanup
Remove-Item "temp_output.txt" -ErrorAction SilentlyContinue
Remove-Item "temp_error.txt" -ErrorAction SilentlyContinue

Write-Host "`n" + "=" * 50
Write-Host "Test completed"
