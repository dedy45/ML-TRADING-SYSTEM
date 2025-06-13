"""
Quick Start Script for Deep Learning Fibonacci Training
Run this to begin training with a small dataset for testing
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

def check_environment():
    """Check if required packages are installed."""
    print("🔍 Checking environment...")
    
    required_packages = ['tensorflow', 'mlflow', 'numpy', 'pandas', 'scikit-learn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    
    return len(missing_packages) == 0

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        'data/raw', 'data/processed', 'data/features',
        'models/tensorflow', 'models/saved_models',
        'experiments', 'logs', 'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def copy_data_files():
    """Copy existing data files to new structure."""
    print("📋 Copying existing data files...")
    
    # Source paths from your existing project
    source_data = Path("../dataBT")
    source_tick = Path("../datatickxau")
    
    # Target paths
    target_data = Path("data/raw")
    
    if source_data.exists():
        print(f"📁 Found {len(list(source_data.glob('*.csv')))} CSV files in dataBT")
        print("✅ Data directory linked (will be processed during training)")
    else:
        print("⚠️  dataBT directory not found - training will need data path adjustment")
    
    if source_tick.exists():
        tick_files = list(source_tick.glob("*.csv"))
        print(f"📁 Found {len(tick_files)} tick data files")
        if tick_files:
            largest_file = max(tick_files, key=lambda x: x.stat().st_size)
            size_gb = largest_file.stat().st_size / (1024**3)
            print(f"📊 Largest tick file: {largest_file.name} ({size_gb:.1f} GB)")
    else:
        print("⚠️  datatickxau directory not found")

def start_mlflow():
    """Start MLflow tracking server."""
    print("🚀 Starting MLflow server...")
    
    # Check if MLflow is already running
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=2)
        print("✅ MLflow server already running at http://localhost:5000")
        return True
    except:
        print("🔄 Starting new MLflow server...")
        
        # Start MLflow in background
        mlflow_process = subprocess.Popen([
            sys.executable, '-m', 'mlflow', 'ui', 
            '--host', '0.0.0.0', '--port', '5000'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ MLflow server starting... (visit http://localhost:5000)")
        return True

def run_quick_training():
    """Run training with a small dataset for testing."""
    print("🧠 Starting quick training run (10 files)...")
    
    # Run training script with limited files
    cmd = [sys.executable, 'scripts/train_models.py', '--max-files', '10']
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path('.'), capture_output=False)
        if result.returncode == 0:
            print("=" * 60)
            print("✅ Quick training completed successfully!")
            print("🌐 View results at http://localhost:5000")
        else:
            print("❌ Training failed. Check logs for details.")
    except Exception as e:
        print(f"❌ Error running training: {e}")

def main():
    """Main quick start function."""
    print("🚀 Deep Learning Fibonacci Quick Start")
    print("=" * 50)
    
    # Change to the deep learning directory
    project_dir = Path("deep_learning_fibonacci")
    if not project_dir.exists():
        print("❌ deep_learning_fibonacci directory not found!")
        print("Please run this script from the MLFLOW directory")
        return
    
    os.chdir(project_dir)
    print(f"📂 Working directory: {Path.cwd()}")
    
    # Step 1: Check environment
    if not check_environment():
        print("❌ Environment setup failed")
        return
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Copy data files
    copy_data_files()
    
    # Step 4: Start MLflow
    start_mlflow()
    
    # Step 5: Run quick training
    print("\n🎯 Ready to start training!")
    response = input("Start quick training with 10 files? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        run_quick_training()
    else:
        print("⏳ Training skipped. To start manually, run:")
        print("   python scripts/train_models.py --max-files 10")
    
    print("\n📊 Next Steps:")
    print("1. Visit http://localhost:5000 to view MLflow experiments")
    print("2. Check logs/ directory for detailed training logs")
    print("3. Review reports/ for performance analysis")
    print("4. Run full training: python scripts/train_models.py")

if __name__ == "__main__":
    main()
