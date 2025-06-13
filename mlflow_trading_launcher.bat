@echo off
title MLflow Trading System - Complete Setup
color 0A

echo.
echo =========================================================================
echo                    🎯 MLFLOW TRADING SYSTEM LAUNCHER 🎯
echo =========================================================================
echo.
echo   Complete ML Trading System with MLflow Tracking
echo   Ready for Paper Trading, Live Signals, and Performance Monitoring
echo.
echo =========================================================================
echo.

:main_menu
echo 📋 MAIN MENU - Choose an option:
echo.
echo 1. 🚀 Quick Start - Run Paper Trading System
echo 2. 🔧 Setup MLflow Environment (First Time)
echo 3. 📊 Start MLflow UI Server
echo 4. 🎯 Run Integrated Trading System
echo 5. 📈 Launch Trading Dashboard
echo 6. 🧪 Test Signal Detectors
echo 7. 📄 View System Status
echo 8. 🔍 Run System Diagnostics
echo 9. 📚 View Best Practices Guide
echo 0. ❌ Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto quick_start
if "%choice%"=="2" goto setup_environment
if "%choice%"=="3" goto start_mlflow_ui
if "%choice%"=="4" goto integrated_system
if "%choice%"=="5" goto trading_dashboard
if "%choice%"=="6" goto test_signals
if "%choice%"=="7" goto system_status
if "%choice%"=="8" goto diagnostics
if "%choice%"=="9" goto best_practices
if "%choice%"=="0" goto exit
goto invalid_choice

:quick_start
echo.
echo 🚀 QUICK START - Paper Trading System
echo ======================================
echo.
echo Starting paper trading with your trained models...
echo This will run for 30 minutes with $10,000 virtual balance
echo.
echo Press Ctrl+C to stop at any time
echo.
pause
python paper_trading_system.py
echo.
echo ✅ Paper trading session completed!
echo Check logs/paper_trading.log for details
echo.
pause
goto main_menu

:setup_environment
echo.
echo 🔧 MLFLOW ENVIRONMENT SETUP
echo ============================
echo.
echo This will set up the complete MLflow environment with best practices...
echo.
pause
python mlflow_best_practices_setup.py
echo.
echo ✅ Environment setup completed!
echo.
pause
goto main_menu

:start_mlflow_ui
echo.
echo 📊 STARTING MLFLOW UI SERVER
echo =============================
echo.
echo Starting MLflow tracking server...
echo Open your browser to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.
start "MLflow UI" mlflow ui --host 127.0.0.1 --port 5000
echo.
echo MLflow UI is starting in a new window...
timeout /t 3 >nul
goto main_menu

:integrated_system
echo.
echo 🎯 INTEGRATED TRADING SYSTEM
echo =============================
echo.
echo Starting the complete integrated trading system...
echo This includes signal detection, paper trading, and MLflow tracking
echo.
pause
python integrated_trading_system.py
echo.
echo ✅ Integrated system session completed!
echo.
pause
goto main_menu

:trading_dashboard
echo.
echo 📈 LAUNCHING TRADING DASHBOARD
echo ===============================
echo.
echo Starting Streamlit trading dashboard...
echo Dashboard will open in your browser automatically
echo.
echo Press Ctrl+C in the dashboard window to stop
echo.
start "Trading Dashboard" streamlit run mlflow_trading_dashboard.py
echo.
echo Trading dashboard is starting...
timeout /t 3 >nul
goto main_menu

:test_signals
echo.
echo 🧪 TESTING SIGNAL DETECTORS
echo ============================
echo.
echo Choose a signal detector to test:
echo.
echo 1. Fibonacci Signal Detector
echo 2. Ensemble Signal Detector  
echo 3. Final Trading System (All Models)
echo 4. Quick Signal Test
echo 5. Back to Main Menu
echo.
set /p test_choice="Enter choice (1-5): "

if "%test_choice%"=="1" goto test_fibonacci
if "%test_choice%"=="2" goto test_ensemble
if "%test_choice%"=="3" goto test_final_system
if "%test_choice%"=="4" goto quick_signal_test
if "%test_choice%"=="5" goto main_menu

:test_fibonacci
echo.
echo Testing Fibonacci Signal Detector...
python fibonacci_signal_detector.py
pause
goto test_signals

:test_ensemble
echo.
echo Testing Ensemble Signal Detector...
python ensemble_signal_detector.py
pause
goto test_signals

:test_final_system
echo.
echo Testing Final Trading System...
python final_trading_system.py
pause
goto test_signals

:quick_signal_test
echo.
echo Running Quick Signal Test...
python quick_signal_test.py
pause
goto test_signals

:system_status
echo.
echo 📄 SYSTEM STATUS CHECK
echo ======================
echo.
echo Checking system components...
echo.

REM Check Python
echo 🐍 Python Version:
python --version
echo.

REM Check required packages
echo 📦 Package Status:
python -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')" 2>nul || echo "❌ Pandas: Not installed"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" 2>nul || echo "❌ NumPy: Not installed"
python -c "import sklearn; print(f'✅ Scikit-learn: {sklearn.__version__}')" 2>nul || echo "❌ Scikit-learn: Not installed"
python -c "import mlflow; print(f'✅ MLflow: {mlflow.__version__}')" 2>nul || echo "❌ MLflow: Not installed"
echo.

REM Check trained models
echo 🤖 Trained Models:
if exist "models\fibonacci_signal_detector.pkl" (echo ✅ Fibonacci Detector) else (echo ❌ Fibonacci Detector)
if exist "models\ensemble_signal_detector.pkl" (echo ✅ Ensemble Detector) else (echo ❌ Ensemble Detector)
if exist "models\fixed_signal_optimizer.pkl" (echo ✅ Signal Optimizer) else (echo ❌ Signal Optimizer)
echo.

REM Check data
echo 📊 Data Status:
if exist "dataBT" (
    for /f %%i in ('dir /b dataBT\*.csv 2^>nul ^| find /c ".csv"') do echo ✅ CSV Files: %%i found
) else (
    echo ❌ dataBT folder not found
)
echo.

REM Check logs
echo 📝 Log Files:
if exist "logs" (
    echo ✅ Logs directory exists
    dir logs\*.log /b 2>nul | find /c ".log" >nul && echo ✅ Log files present || echo ⚠️ No log files yet
) else (
    echo ❌ Logs directory not found
)
echo.

echo 🎯 System Status Check Complete
echo.
pause
goto main_menu

:diagnostics
echo.
echo 🔍 RUNNING SYSTEM DIAGNOSTICS
echo ==============================
echo.
echo Running comprehensive system diagnostics...
echo.
python system_diagnostic.py
echo.
echo ✅ Diagnostics completed!
echo.
pause
goto main_menu

:best_practices
echo.
echo 📚 MLFLOW BEST PRACTICES GUIDE
echo ===============================
echo.
echo Opening best practices guide...
echo.
if exist "MLFLOW_BEST_PRACTICES_GUIDE.md" (
    start "" "MLFLOW_BEST_PRACTICES_GUIDE.md"
    echo ✅ Guide opened in default editor
) else (
    echo ❌ Best practices guide not found
    echo Run "Setup MLflow Environment" first to generate the guide
)
echo.
pause
goto main_menu

:invalid_choice
echo.
echo ❌ Invalid choice. Please select a number from 0-9.
echo.
timeout /t 2 >nul
goto main_menu

:exit
echo.
echo 👋 Thank you for using MLflow Trading System!
echo.
echo 📋 Quick Reference:
echo   • Paper Trading: python paper_trading_system.py
echo   • MLflow UI: mlflow ui --port 5000
echo   • Dashboard: streamlit run mlflow_trading_dashboard.py
echo   • Integrated System: python integrated_trading_system.py
echo.
echo 🎯 Happy Trading! 
echo.
pause
exit /b 0
