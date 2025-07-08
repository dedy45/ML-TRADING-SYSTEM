#!/usr/bin/env python3
"""
🎯 VALIDATION SCRIPT - SISTEM ML TRADING
========================================

Script ini untuk memvalidasi bahwa semua komponen sistem berjalan dengan baik.
Run this script untuk memastikan environment setup sudah correct.

Author: ML Trading System
Date: December 2024
"""

import sys
import os
from pathlib import Path
import traceback

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.END}")

def check_python_version():
    """Check Python version compatibility"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (>= 3.8)")
        return True
    else:
        print_error("Python version too old. Need Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    dependencies = {
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library', 
        'sklearn': 'Machine learning library',
        'mlflow': 'ML experiment tracking',
        'xgboost': 'Gradient boosting library',
        'lightgbm': 'Gradient boosting library',
        'matplotlib': 'Plotting library',
        'seaborn': 'Statistical plotting',
        'joblib': 'Model serialization'
    }
    
    all_ok = True
    
    for lib, description in dependencies.items():
        try:
            if lib == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'Unknown')
            
            print_success(f"{lib:12} v{version:10} - {description}")
        except ImportError as e:
            print_error(f"{lib:12} NOT FOUND   - {description}")
            all_ok = False
        except Exception as e:
            print_warning(f"{lib:12} ERROR: {str(e)}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if required directories exist"""
    print_header("CHECKING DIRECTORY STRUCTURE")
    
    required_dirs = [
        'dataBT',      # Data directory
        'logs',        # Log files
        'results',     # Analysis results
        'models',      # Trained models
        'reports',     # Generated reports
        'mlruns'       # MLflow experiment tracking
    ]
    
    all_ok = True
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print_success(f"Directory exists: {directory}")
        else:
            print_warning(f"Creating directory: {directory}")
            try:
                dir_path.mkdir(exist_ok=True)
                print_success(f"Created directory: {directory}")
            except Exception as e:
                print_error(f"Failed to create {directory}: {e}")
                all_ok = False
    
    return all_ok

def check_key_files():
    """Check if key files are present"""
    print_header("CHECKING KEY FILES")
    
    key_files = [
        ('simple_ml_pipeline.py', 'Basic ML pipeline'),
        ('advanced_ml_pipeline_working.py', 'Advanced ML pipeline'),
        ('QUICK_START.md', 'Quick start guide'),
        ('STEP_BY_STEP_GUIDE.md', 'Comprehensive guide'),
        ('deep_learning_fibonacci/', 'Fibonacci analysis module')
    ]
    
    all_ok = True
    
    for file_path, description in key_files:
        path = Path(file_path)
        if path.exists():
            print_success(f"{file_path:35} - {description}")
        else:
            print_error(f"{file_path:35} - {description} (MISSING)")
            all_ok = False
    
    return all_ok

def test_basic_functionality():
    """Test basic ML functionality"""
    print_header("TESTING BASIC FUNCTIONALITY")
    
    try:
        print_info("Creating sample data...")
        
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Create sample trading data
        n_samples = 100
        np.random.seed(42)
        
        data = {
            'open_price': np.random.uniform(1900, 2100, n_samples),
            'close_price': np.random.uniform(1900, 2100, n_samples),
            'volume': np.random.randint(1, 100, n_samples),
            'profit': np.random.uniform(-50, 100, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['is_profitable'] = (df['profit'] > 0).astype(int)
        
        print_success(f"Created sample dataset with {len(df)} rows")
        
        # Create simple features
        df['price_change'] = df['close_price'] - df['open_price']
        df['price_range'] = abs(df['price_change'])
        
        # Basic ML test
        features = ['open_price', 'volume', 'price_change', 'price_range']
        X = df[features]
        y = df['is_profitable']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print_success(f"ML model trained successfully")
        print_success(f"Test accuracy: {accuracy:.2%}")
        
        # Test MLflow integration
        try:
            import mlflow
            print_success("MLflow import successful")
        except Exception as e:
            print_warning(f"MLflow test failed: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_fibonacci_module():
    """Test Fibonacci deep learning module"""
    print_header("TESTING FIBONACCI MODULE")
    
    try:
        fib_dir = Path("deep_learning_fibonacci")
        if not fib_dir.exists():
            print_warning("Fibonacci module directory not found")
            return False
        
        # Check key Fibonacci files
        key_fib_files = [
            'ea_mql5_integration.py',
            'enhanced_ml_fibonacci.py',
            'fast_signal_generator.py'
        ]
        
        missing_files = []
        for file_name in key_fib_files:
            file_path = fib_dir / file_name
            if file_path.exists():
                print_success(f"Fibonacci file found: {file_name}")
            else:
                print_warning(f"Fibonacci file missing: {file_name}")
                missing_files.append(file_name)
        
        if len(missing_files) == 0:
            print_success("All key Fibonacci files present")
            return True
        else:
            print_warning(f"Missing {len(missing_files)} Fibonacci files")
            return False
            
    except Exception as e:
        print_error(f"Fibonacci module test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print_header("CREATING SAMPLE DATA")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample trading data
        n_trades = 200
        start_date = datetime.now() - timedelta(days=30)
        
        np.random.seed(42)
        
        sample_data = {
            'Symbol': ['XAUUSD'] * n_trades,
            'Timestamp': [start_date + timedelta(hours=i) for i in range(n_trades)],
            'Type': np.random.choice(['BUY', 'SELL'], n_trades),
            'OpenPrice': np.random.uniform(1950, 2050, n_trades),
            'ClosePrice': np.random.uniform(1950, 2050, n_trades),
            'Volume': np.random.randint(1, 10, n_trades),
            'Profit': np.random.uniform(-100, 150, n_trades),
            'MAE_pips': np.random.uniform(0, 50, n_trades),
            'MFE_pips': np.random.uniform(0, 80, n_trades),
            'ExitReason': np.random.choice(['SL', 'TP', 'Manual'], n_trades),
            'Session': np.random.choice(['Asia', 'Europe', 'US'], n_trades)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        sample_file = Path('dataBT/sample_trading_data.csv')
        df.to_csv(sample_file, index=False)
        
        print_success(f"Created sample data: {sample_file}")
        print_success(f"Sample contains {len(df)} trading records")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to create sample data: {e}")
        return False

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("🎯 ML TRADING SYSTEM VALIDATION")
    print("================================")
    print("Memvalidasi setup environment untuk sistem trading ML")
    print(f"{Colors.END}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("Directory Structure", check_directories),
        ("Key Files", check_key_files),
        ("Basic Functionality", test_basic_functionality),
        ("Fibonacci Module", test_fibonacci_module),
        ("Sample Data", create_sample_data)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_error(f"Check '{check_name}' failed with error: {e}")
            results[check_name] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{check_name:25} - {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Overall Result: {passed_checks}/{total_checks} checks passed{Colors.END}")
    
    if passed_checks == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL VALIDATIONS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Your ML Trading System is ready to use!{Colors.END}")
        print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
        print(f"{Colors.BLUE}1. Run: python simple_ml_pipeline.py{Colors.END}")
        print(f"{Colors.BLUE}2. Run: python advanced_ml_pipeline_working.py{Colors.END}")
        print(f"{Colors.BLUE}3. Start MLflow UI: mlflow ui --port 5000{Colors.END}")
        
    elif passed_checks >= total_checks - 2:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ MOSTLY READY - Minor issues detected{Colors.END}")
        print(f"{Colors.YELLOW}System should work but some optimizations needed{Colors.END}")
        
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SETUP INCOMPLETE{Colors.END}")
        print(f"{Colors.RED}Please fix the failed checks before proceeding{Colors.END}")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)