@echo off
echo =======================================
echo 🔢 FAST FIBONACCI ANALYZER
echo =======================================
echo.
echo Optimized Fibonacci level analysis
echo for trading data - MUCH FASTER!
echo.
echo Choose analysis type:
echo.
echo 1. Quick Test (10 files) - 30 seconds
echo 2. Medium Analysis (30 files) - 2 minutes  
echo 3. Large Analysis (50 files) - 5 minutes
echo 4. Interactive Mode (choose parameters)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto medium
if "%choice%"=="3" goto large
if "%choice%"=="4" goto interactive
goto invalid

:quick
echo.
echo 🚀 Running Quick Fibonacci Analysis...
python -c "from fast_fibonacci_analyzer import FastFibonacciAnalyzer; a=FastFibonacciAnalyzer(); a.load_sample_data(10, 500); a.generate_comprehensive_report()"
goto end

:medium
echo.
echo 📊 Running Medium Fibonacci Analysis...
python -c "from fast_fibonacci_analyzer import FastFibonacciAnalyzer; a=FastFibonacciAnalyzer(); a.load_sample_data(30, 1000); a.generate_comprehensive_report()"
goto end

:large
echo.
echo 🔍 Running Large Fibonacci Analysis...
python -c "from fast_fibonacci_analyzer import FastFibonacciAnalyzer; a=FastFibonacciAnalyzer(); a.load_sample_data(50, 2000); a.generate_comprehensive_report()"
goto end

:interactive
echo.
echo ⚙️ Starting Interactive Mode...
python run_fast_fibonacci.py
goto end

:invalid
echo.
echo ❌ Invalid choice! Please run the script again.

:end
echo.
echo ✅ Analysis completed!
echo 📊 Check the reports/ folder for detailed results
echo.
pause
