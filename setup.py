"""
Setup script untuk Trading ML Pipeline
Jalankan script ini untuk setup lengkap project
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run command dengan error handling"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} berhasil")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} gagal: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setup Trading ML Pipeline dengan MLflow")
    print("=" * 50)
    
    # 1. Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Instalasi dependencies gagal. Lanjutkan secara manual.")
    
    # 2. Create missing directories
    print("\n📁 Membuat direktori yang diperlukan...")
    directories = [
        "data/raw", "data/processed", "data/features",
        "models", "experiments", "logs", "mlruns", "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Direktori {directory} siap")
    
    # 3. Validate installation
    print("\n🔍 Validasi instalasi...")
    try:
        import pandas as pd
        import numpy as np
        import mlflow
        import xgboost
        import lightgbm
        import sklearn
        print("✅ Semua dependencies terinstall dengan baik")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # 4. Initialize MLflow
    print("\n🎯 Inisialisasi MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("trading_signal_prediction")
        print("✅ MLflow siap digunakan")
    except Exception as e:
        print(f"❌ MLflow initialization gagal: {e}")
    
    # 5. Test data loading
    print("\n📊 Testing data loading...")
    if Path("dataBT").exists():
        csv_files = list(Path("dataBT").glob("*.csv"))
        print(f"✅ Ditemukan {len(csv_files)} file backtest data")
    else:
        print("⚠️  Folder dataBT tidak ditemukan")
    
    if Path("datatickxau").exists():
        tick_files = list(Path("datatickxau").glob("*.csv"))
        print(f"✅ Ditemukan {len(tick_files)} file tick data")
    else:
        print("⚠️  Folder datatickxau tidak ditemukan")
    
    print("\n" + "=" * 50)
    print("🎉 Setup lengkap!")
    print("\n📋 Langkah selanjutnya:")
    print("1. Jalankan pipeline: python main.py")
    print("2. Buka MLflow UI: mlflow ui --port 5000")
    print("3. Akses di browser: http://localhost:5000")
    print("\n📖 Baca README.md untuk dokumentasi lengkap")
    print("=" * 50)

if __name__ == "__main__":
    main()
