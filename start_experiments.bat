@echo off
echo =======================================
echo 🎯 TRADING ML EXPERIMENT SUITE
echo =======================================
echo.
echo Choose what you want to do:
echo.
echo 1. Start MLflow UI (for viewing results)
echo 2. Run Quick Experiment (3 files)
echo 3. Run Medium Experiment (10 files)
echo 4. Run Interactive Experiment (choose options)
echo 5. Run Simple Pipeline (basic version)
echo 6. Check System Status
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto mlflow_ui
if "%choice%"=="2" goto quick_exp
if "%choice%"=="3" goto medium_exp
if "%choice%"=="4" goto interactive_exp
if "%choice%"=="5" goto simple_pipeline
if "%choice%"=="6" goto check_status
goto invalid_choice

:mlflow_ui
echo.
echo 🚀 Starting MLflow UI...
echo Open http://127.0.0.1:5000 in your browser
echo Press Ctrl+C to stop the UI
python -m mlflow ui --port 5000
goto end

:quick_exp
echo.
echo 🚀 Running Quick Experiment (3 files)...
python -c "from advanced_ml_pipeline import AdvancedTradingPipeline; p=AdvancedTradingPipeline(); p.run_complete_pipeline(3)"
echo.
echo ✅ Done! Check results at http://127.0.0.1:5000
pause
goto end

:medium_exp
echo.
echo 🚀 Running Medium Experiment (10 files)...
python -c "from advanced_ml_pipeline import AdvancedTradingPipeline; p=AdvancedTradingPipeline(); p.run_complete_pipeline(10)"
echo.
echo ✅ Done! Check results at http://127.0.0.1:5000
pause
goto end

:interactive_exp
echo.
echo 🚀 Starting Interactive Experiment Runner...
python run_experiments.py
pause
goto end

:simple_pipeline
echo.
echo 🚀 Running Simple Pipeline...
python simple_ml_pipeline.py
pause
goto end

:check_status
echo.
echo 🔍 Checking System Status...
echo.
echo Python version:
python --version
echo.
echo Installed packages:
pip list | findstr -i "pandas numpy scikit-learn mlflow joblib"
echo.
echo Data files:
dir dataBT\*.csv | find /c ".csv"
echo CSV files found in dataBT folder
echo.
echo MLflow experiments:
if exist mlruns (
    echo MLflow tracking folder exists
    dir mlruns /s /b | find /c "\"
    echo experiment runs found
) else (
    echo MLflow tracking folder not found
)
echo.
pause
goto end

:invalid_choice
echo.
echo ❌ Invalid choice! Please enter 1-6
pause
goto end

:end
echo.
echo 👋 Goodbye!
