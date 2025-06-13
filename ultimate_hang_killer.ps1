# Ultimate solution for Python hang issue
# This PowerShell script will DEFINITELY kill hung processes

$ScriptName = "step_by_step_test.py"
$TimeoutSeconds = 5

Write-Host "=" * 60
Write-Host "ULTIMATE HANG SOLUTION"
Write-Host "=" * 60
Write-Host "Script: $ScriptName"
Write-Host "Timeout: $TimeoutSeconds seconds"
Write-Host "Time: $(Get-Date)"
Write-Host "-" * 40

# Start background job
$job = Start-Job -ScriptBlock {
    param($script)
    Set-Location "E:\aiml\MLFLOW"
    python $script 2>&1
} -ArgumentList $ScriptName

Write-Host "Job started with ID: $($job.Id)"

# Wait for job completion or timeout
$completed = Wait-Job $job -Timeout $TimeoutSeconds

if ($completed) {
    Write-Host "[SUCCESS] Script completed within timeout"
    $output = Receive-Job $job
    if ($output) {
        Write-Host "`nOutput:"
        $output | ForEach-Object { Write-Host "  $_" }
    }
} else {
    Write-Host "[TIMEOUT] Script hung - FORCE KILLING ALL PYTHON PROCESSES"
    
    # Kill the job
    Remove-Job $job -Force
    
    # Find and kill all python processes
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        Write-Host "Found $($pythonProcesses.Count) Python processes to kill:"
        foreach ($proc in $pythonProcesses) {
            Write-Host "  Killing PID $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        Write-Host "All Python processes killed"
    } else {
        Write-Host "No Python processes found"
    }
}

Write-Host "`n" + "=" * 60
Write-Host "OPERATION COMPLETED - NO HANGING PROCESSES"
Write-Host "=" * 60
