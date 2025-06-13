@echo off
echo ================================================================
echo FIBONACCI ANALYSIS - BATCH SOLUTION (NO PYTHON HANG)
echo ================================================================
echo.

echo [INFO] Checking data directory...
if not exist "dataBT" (
    echo [ERROR] dataBT directory not found
    pause
    exit /b 1
)

echo [OK] dataBT directory found

echo [INFO] Counting CSV files...
for /f %%i in ('dir /b dataBT\*.csv 2^>nul ^| find /c /v ""') do set CSV_COUNT=%%i
echo [OK] Found %CSV_COUNT% CSV files

if %CSV_COUNT%==0 (
    echo [ERROR] No CSV files found
    pause
    exit /b 1
)

echo.
echo [INFO] Analyzing first CSV file for headers...
for %%f in (dataBT\*.csv) do (
    echo [INFO] Sample file: %%~nxf
    
    REM Read first line for headers
    for /f "delims=" %%a in (%%f) do (
        echo [OK] Headers found: %%a
        goto :analyze_headers
    )
)

:analyze_headers
echo.
echo [INFO] Manual Fibonacci Analysis Instructions:
echo ================================================================
echo 1. Open Excel or any CSV viewer
echo 2. Load any CSV file from dataBT folder
echo 3. Look for these important columns:
echo    - LevelFibo: Contains Fibonacci level values
echo    - Level1Above: Support level above price
echo    - Level1Below: Support level below price  
echo    - Profit: Trade profit (positive = profitable)
echo    - Type: BUY or SELL
echo    - SessionAsia, SessionEurope, SessionUS: Trading sessions
echo.
echo 4. To find best Fibonacci levels:
echo    - Filter by Profit ^> 0 (profitable trades)
echo    - Group by LevelFibo
echo    - Calculate win rate for each level
echo    - Levels with 60%+ win rate are good signals
echo.
echo 5. Key Fibonacci levels to watch:
echo    - 0.236, 0.382, 0.500, 0.618, 0.786
echo    - 1.272, 1.414, 1.618, 2.000
echo.
echo ================================================================
echo [SUCCESS] Analysis guide completed
echo [INFO] Files ready for manual analysis in dataBT folder
echo ================================================================
echo.
pause
