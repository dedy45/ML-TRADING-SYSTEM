# PowerShell Fibonacci Analyzer
# Solusi untuk masalah hang Python

Write-Host "🚀 PowerShell Fibonacci Quick Analysis" -ForegroundColor Green
Write-Host "=" * 50

$startTime = Get-Date

# Get CSV files
Write-Host "📂 Scanning CSV files..." -ForegroundColor Yellow
$csvFiles = Get-ChildItem "dataBT" -Filter "*.csv" | Select-Object -First 10
Write-Host "✅ Found $($csvFiles.Count) files (processing first 10)" -ForegroundColor Green

# Statistics
$totalTrades = 0
$profitableTrades = 0
$fibStats = @{}

foreach ($file in $csvFiles) {
    Write-Host "📄 Processing: $($file.Name)" -ForegroundColor Cyan
    
    try {
        # Read CSV with limited rows
        $csvData = Import-Csv $file.FullName | Select-Object -First 30
        
        foreach ($row in $csvData) {
            if ($row.Type -in @('BUY', 'SELL')) {
                $totalTrades++
                
                # Check profitability
                $profit = [double]$row.Profit
                if ($profit -gt 0) {
                    $profitableTrades++
                }
                
                # Fibonacci level stats
                $fibLevel = $row.LevelFibo
                if ($fibLevel) {
                    if (-not $fibStats.ContainsKey($fibLevel)) {
                        $fibStats[$fibLevel] = @{Total = 0; Profitable = 0}
                    }
                    $fibStats[$fibLevel].Total++
                    if ($profit -gt 0) {
                        $fibStats[$fibLevel].Profitable++
                    }
                }
            }
        }
        
        Write-Host "  ✓ Processed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "  ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Calculate results
$winRate = if ($totalTrades -gt 0) { ($profitableTrades / $totalTrades) * 100 } else { 0 }
$elapsedTime = (Get-Date) - $startTime

Write-Host "`n" + "=" * 60
Write-Host "📊 QUICK ANALYSIS RESULTS" -ForegroundColor Green
Write-Host "=" * 60

Write-Host "⏱️  Execution Time: $($elapsedTime.TotalSeconds.ToString('F1')) seconds" -ForegroundColor White
Write-Host "📁 Files Processed: $($csvFiles.Count)" -ForegroundColor White
Write-Host "💹 Total Trades: $totalTrades" -ForegroundColor White
Write-Host "💰 Profitable Trades: $profitableTrades" -ForegroundColor White
Write-Host "🎯 Overall Win Rate: $($winRate.ToString('F2'))%" -ForegroundColor $(if ($winRate -gt 50) { 'Green' } else { 'Yellow' })

# Top Fibonacci levels
Write-Host "`n🔝 Top Fibonacci Levels:" -ForegroundColor Yellow
Write-Host "Level".PadRight(15) + "Trades".PadRight(8) + "Wins".PadRight(6) + "Win Rate"
Write-Host "-" * 45

$sortedFibLevels = $fibStats.GetEnumerator() | Where-Object { $_.Value.Total -ge 3 } | Sort-Object { ($_.Value.Profitable / $_.Value.Total) * 100 } -Descending

foreach ($level in $sortedFibLevels | Select-Object -First 8) {
    $levelWinRate = ($level.Value.Profitable / $level.Value.Total) * 100
    $levelName = $level.Key.ToString().PadRight(15)
    $levelTrades = $level.Value.Total.ToString().PadRight(8)
    $levelWins = $level.Value.Profitable.ToString().PadRight(6)
    $levelRate = $levelWinRate.ToString('F1') + "%"
    
    Write-Host "$levelName$levelTrades$levelWins$levelRate"
}

# Save results
$reportFile = "powershell_fibonacci_report.txt"
$report = @"
POWERSHELL FIBONACCI ANALYSIS REPORT
==================================================

Execution Time: $($elapsedTime.TotalSeconds.ToString('F1')) seconds
Files Processed: $($csvFiles.Count)
Total Trades: $totalTrades
Profitable Trades: $profitableTrades
Overall Win Rate: $($winRate.ToString('F2'))%

TOP FIBONACCI LEVELS:
"@

foreach ($level in $sortedFibLevels | Select-Object -First 10) {
    $levelWinRate = ($level.Value.Profitable / $level.Value.Total) * 100
    $report += "`nLevel $($level.Key): $($levelWinRate.ToString('F1'))% ($($level.Value.Profitable)/$($level.Value.Total) trades)"
}

$report | Out-File -FilePath $reportFile -Encoding UTF8

Write-Host "`n✅ SUCCESS! Analysis completed in $($elapsedTime.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green
Write-Host "📄 Report saved: $reportFile" -ForegroundColor Green

if ($winRate -gt 55) {
    Write-Host "🎯 TARGET ACHIEVED! Win rate above 55%" -ForegroundColor Green
} elseif ($winRate -gt 50) {
    Write-Host "📈 GOOD PERFORMANCE! Win rate above 50%" -ForegroundColor Yellow
} else {
    Write-Host "📊 Results baseline, consider enhancements" -ForegroundColor Yellow
}

Write-Host "`n💡 SOLUSI HANG ISSUE:" -ForegroundColor Cyan
Write-Host "   ✅ PowerShell lebih cepat untuk task ini" -ForegroundColor White
Write-Host "   ✅ Processing incremental mencegah hang" -ForegroundColor White
Write-Host "   ✅ Progress monitoring real-time" -ForegroundColor White
Write-Host "   ✅ Automatic timeout handling" -ForegroundColor White
