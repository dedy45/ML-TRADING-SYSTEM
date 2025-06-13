@echo off
echo === SOLUSI ANTI-HANG: MENGGUNAKAN CMD ===
echo.

echo [1] Testing basic Python...
python -c "print('Python OK')"
if errorlevel 1 (
    echo ERROR: Python tidak berjalan
    pause
    exit /b 1
)

echo [2] Testing file access...
python -c "import os; print('Files found:', len([f for f in os.listdir('datatickxau') if f.endswith('.csv')]))"
if errorlevel 1 (
    echo ERROR: Tidak bisa akses datatickxau
    pause
    exit /b 1
)

echo [3] Testing data tick analysis without pandas...
python ultra_simple_tick_analyzer.py
if errorlevel 1 (
    echo ERROR: Analysis gagal
    pause
    exit /b 1
)

echo.
echo === SELESAI ===
pause
