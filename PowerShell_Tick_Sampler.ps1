# PowerShell Tick Data Processor - No Python Hang Issues
# Samples 1% of 2.7GB tick data safely

param(
    [string]$InputFile = "datatickxau\2025.6.11XAUUSD_dukascopy_TICK_UTC-TICK-Forex_245.csv",
    [string]$OutputFile = "tick_sample_1_percent.csv",
    [int]$SampleRate = 100  # Take every 100th line (1% sample)
)

Write-Host "=== POWERSHELL TICK DATA SAMPLER ===" -ForegroundColor Green
Write-Host "Input: $InputFile" -ForegroundColor Cyan
Write-Host "Output: $OutputFile" -ForegroundColor Cyan
Write-Host "Sample Rate: 1 in $SampleRate ($(100/$SampleRate)%)" -ForegroundColor Cyan

# Check if input file exists
if (-not (Test-Path $InputFile)) {
    Write-Host "ERROR: Input file not found: $InputFile" -ForegroundColor Red
    exit 1
}

# Get file size
$fileSize = (Get-Item $InputFile).Length
$fileSizeMB = [math]::Round($fileSize / 1MB, 1)
$expectedOutputMB = [math]::Round($fileSizeMB / $SampleRate, 1)

Write-Host "Input file size: $fileSizeMB MB" -ForegroundColor Yellow
Write-Host "Expected output size: $expectedOutputMB MB" -ForegroundColor Yellow

$startTime = Get-Date
Write-Host "Started: $($startTime.ToString('HH:mm:ss'))" -ForegroundColor Green

try {
    # Read and process file in chunks to avoid memory issues
    $lineNumber = 0
    $sampledLines = 0
    
    # Get header first
    $header = Get-Content $InputFile -First 1
    
    Write-Host "Header: $header" -ForegroundColor Cyan
    
    # Initialize output file with header
    $header | Out-File $OutputFile -Encoding UTF8
    
    # Process file in batches
    $batchSize = 100000  # Process 100K lines at a time
    $currentBatch = 0
    
    Write-Host "Processing in batches of $batchSize lines..." -ForegroundColor Yellow
    
    # Use .NET StreamReader for efficient large file processing
    $reader = [System.IO.StreamReader]::new($InputFile)
    $writer = [System.IO.StreamWriter]::new($OutputFile, $true)  # Append mode
    
    # Skip header in reader
    $null = $reader.ReadLine()
    $lineNumber = 1
    
    while (($line = $reader.ReadLine()) -ne $null) {
        $lineNumber++
        
        # Sample every nth line
        if ($lineNumber % $SampleRate -eq 0) {
            $writer.WriteLine($line)
            $sampledLines++
        }
        
        # Progress indicator
        if ($lineNumber % 1000000 -eq 0) {
            $currentTime = Get-Date
            $elapsed = ($currentTime - $startTime).TotalMinutes
            Write-Host "Processed: $($lineNumber.ToString('N0')) lines | Sampled: $($sampledLines.ToString('N0')) | Elapsed: $([math]::Round($elapsed, 1)) min" -ForegroundColor Green
        }
        
        # Safety limit for testing (remove for production)
        if ($sampledLines -ge 100000) {
            Write-Host "Reached safety limit of 100K sampled lines" -ForegroundColor Yellow
            break
        }
    }
    
    $reader.Close()
    $writer.Close()
    
    $endTime = Get-Date
    $totalTime = ($endTime - $startTime).TotalMinutes
    
    # Get output file size
    $outputSize = (Get-Item $OutputFile).Length
    $outputSizeMB = [math]::Round($outputSize / 1MB, 1)
    
    Write-Host "`n=== SAMPLING COMPLETED ===" -ForegroundColor Green
    Write-Host "Total lines processed: $($lineNumber.ToString('N0'))" -ForegroundColor Cyan
    Write-Host "Lines sampled: $($sampledLines.ToString('N0'))" -ForegroundColor Cyan
    Write-Host "Sampling rate: $([math]::Round($sampledLines/$lineNumber*100, 3))%" -ForegroundColor Cyan
    Write-Host "Output file size: $outputSizeMB MB" -ForegroundColor Cyan
    Write-Host "Processing time: $([math]::Round($totalTime, 1)) minutes" -ForegroundColor Cyan
    Write-Host "Sample file: $OutputFile" -ForegroundColor Yellow
    
    # Quick analysis of sample
    Write-Host "`n=== QUICK SAMPLE ANALYSIS ===" -ForegroundColor Green
    $sampleLines = Get-Content $OutputFile -First 6
    Write-Host "First 5 sample rows:" -ForegroundColor Cyan
    for ($i = 1; $i -lt 6; $i++) {
        if ($i -lt $sampleLines.Count) {
            Write-Host "  $i`: $($sampleLines[$i])" -ForegroundColor White
        }
    }
    
    Write-Host "`nSample created successfully! Ready for ML processing." -ForegroundColor Green
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
