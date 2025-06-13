#!/usr/bin/env python3
"""
SIMPLE SIGNAL TESTER
Test sederhana untuk memastikan sistem bekerja
"""

import pandas as pd
import numpy as np
import os
import sys

def test_fibonacci_signal():
    """Test Fibonacci signal detector"""
    print("=" * 60)
    print("TESTING FIBONACCI SIGNAL DETECTOR")
    print("=" * 60)
    
    try:
        from fibonacci_signal_detector import FibonacciSignalDetector
        
        detector = FibonacciSignalDetector()
        
        # Test case
        test_data = {
            'LevelFibo': 'B_0',
            'Type': 'BUY',
            'SessionEurope': 1,
            'SessionUS': 0,
            'SessionAsia': 0,
            'OpenPrice': 2650.0,
            'TP': 2655.0,
            'SL': 2648.0
        }
        
        print(f"Testing with: {test_data}")
        
        result = detector.detect_signal(test_data)
        
        print(f"\nFibonacci Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Fibonacci Signal Detector: WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Fibonacci Signal Detector Error: {e}")
        return False

def test_ensemble_signal():
    """Test ensemble signal detector"""
    print("\n" + "=" * 60)
    print("TESTING ENSEMBLE SIGNAL DETECTOR")
    print("=" * 60)
    
    try:
        from ensemble_signal_detector import EnsembleSignalDetector
        
        detector = EnsembleSignalDetector()
        
        # Try to load existing model
        if detector.load_ensemble_model():
            print("✅ Ensemble model loaded")
            
            # Test prediction
            test_data = {
                'LevelFibo': 'B_0',
                'Type': 'BUY',
                'SessionEurope': 1,
                'SessionUS': 0,
                'SessionAsia': 0,
                'OpenPrice': 2650.0,
                'TP': 2655.0,
                'SL': 2648.0,
                'Volume': 0.1
            }
            
            print(f"Testing with: {test_data}")
            
            result = detector.predict_signal_strength(test_data)
            
            print(f"\nEnsemble Result:")
            for key, value in result.items():
                if key != 'individual_models':
                    print(f"  {key}: {value}")
            
            print("\n✅ Ensemble Signal Detector: WORKING")
            return True
        else:
            print("⚠️  No ensemble model found. Need to train first.")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble Signal Detector Error: {e}")
        return False

def test_fixed_optimizer():
    """Test fixed optimizer"""
    print("\n" + "=" * 60)
    print("TESTING FIXED ADVANCED OPTIMIZER")
    print("=" * 60)
    
    try:
        from fixed_advanced_signal_optimizer import FixedAdvancedSignalOptimizer
        
        optimizer = FixedAdvancedSignalOptimizer()
        
        # Try to load existing model
        if optimizer.load_model():
            print("✅ Fixed optimizer model loaded")
            
            # Test prediction
            test_data = {
                'LevelFibo': 'B_0',
                'Type': 'BUY',
                'SessionEurope': 1,
                'SessionUS': 0,
                'SessionAsia': 0,
                'OpenPrice': 2650.0,
                'TP': 2655.0,
                'SL': 2648.0,
                'Volume': 0.1
            }
            
            print(f"Testing with: {test_data}")
            
            result = optimizer.get_signal_strength(test_data)
            
            print(f"\nFixed Optimizer Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            print("\n✅ Fixed Advanced Optimizer: WORKING")
            return True
        else:
            print("⚠️  No optimizer model found. Need to train first.")
            return False
            
    except Exception as e:
        print(f"❌ Fixed Advanced Optimizer Error: {e}")
        return False

def test_data_availability():
    """Test data availability"""
    print("\n" + "=" * 60)
    print("CHECKING DATA AVAILABILITY")
    print("=" * 60)
    
    data_path = "dataBT"
    
    if os.path.exists(data_path):
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        print(f"✅ Data folder found: {data_path}")
        print(f"✅ CSV files found: {len(csv_files)}")
        
        if csv_files:
            # Sample first file
            sample_file = os.path.join(data_path, csv_files[0])
            try:
                df = pd.read_csv(sample_file, nrows=5)
                print(f"✅ Sample file readable: {csv_files[0]}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Sample rows: {len(df)}")
                return True
            except Exception as e:
                print(f"❌ Error reading sample file: {e}")
                return False
        else:
            print("❌ No CSV files found in dataBT folder")
            return False
    else:
        print(f"❌ Data folder not found: {data_path}")
        return False

def test_training_quick():
    """Quick training test with small data"""
    print("\n" + "=" * 60)
    print("QUICK TRAINING TEST")
    print("=" * 60)
    
    try:
        from fixed_advanced_signal_optimizer import FixedAdvancedSignalOptimizer
        
        optimizer = FixedAdvancedSignalOptimizer()
        
        print("Attempting quick training with limited data...")
        
        # Load very small amount of data
        X, y = optimizer.load_and_prepare_data(max_files=3, sample_per_file=100)
        
        if X is not None and y is not None:
            print(f"✅ Data loaded: {len(X)} samples, {len(X.columns)} features")
            
            # Quick training
            model = optimizer.train_model(X, y)
            
            if model is not None:
                print("✅ Model trained successfully")
                
                # Save model
                optimizer.save_model()
                
                # Test prediction
                test_data = {
                    'LevelFibo': 'B_0',
                    'Type': 'BUY',
                    'SessionEurope': 1,
                    'SessionUS': 0,
                    'SessionAsia': 0,
                    'OpenPrice': 2650.0,
                    'TP': 2655.0,
                    'SL': 2648.0,
                    'Volume': 0.1
                }
                
                result = optimizer.get_signal_strength(test_data)
                print(f"✅ Prediction test successful: {result}")
                
                return True
            else:
                print("❌ Model training failed")
                return False
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎯 SIGNAL SYSTEM DIAGNOSTIC TEST")
    print("=" * 80)
    
    tests = [
        ("Data Availability", test_data_availability),
        ("Fibonacci Signal Detector", test_fibonacci_signal),
        ("Ensemble Signal Detector", test_ensemble_signal),
        ("Fixed Optimizer", test_fixed_optimizer),
        ("Quick Training", test_training_quick)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    working_count = 0
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30}: {status}")
        if success:
            working_count += 1
    
    print(f"\nOverall: {working_count}/{len(tests)} tests passed")
    
    if working_count >= 3:
        print("🎉 System is mostly functional!")
    elif working_count >= 1:
        print("⚠️  System has some issues but partially working")
    else:
        print("🚨 System needs major fixes")
    
    print("\n💡 Next steps:")
    if not results.get("Data Availability", False):
        print("  1. Check dataBT folder and CSV files")
    if not results.get("Quick Training", False):
        print("  2. Fix training pipeline issues")
    if results.get("Fibonacci Signal Detector", False):
        print("  3. Fibonacci detector is working - use as fallback")
    
    print("\n🔧 To fix errors, run individual components:")
    print("  - python fibonacci_signal_detector.py")
    print("  - python fixed_advanced_signal_optimizer.py")
    print("  - python ensemble_signal_detector.py")

if __name__ == "__main__":
    main()
