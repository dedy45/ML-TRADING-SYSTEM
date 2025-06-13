@echo off
REM Batch script to run Python scripts with the fibonacci-ml-simple environment
REM Usage: run_with_environment.bat your_script.py

echo Activating fibonacci-ml-simple environment...

if "%1"=="" (
    echo Usage: run_with_environment.bat your_script.py
    echo.
    echo Testing environment...
    "C:\Users\dedy\anaconda3\envs\fibonacci-ml-simple\python.exe" -c "import tensorflow as tf; import numpy as np; import pandas as pd; print('Environment ready!'); print(f'TensorFlow: {tf.__version__}'); print(f'NumPy: {np.__version__}'); print(f'Pandas: {pd.__version__}')"
) else (
    echo Running %1 with fibonacci-ml-simple environment...
    "C:\Users\dedy\anaconda3\envs\fibonacci-ml-simple\python.exe" %1 %2 %3 %4 %5
)
