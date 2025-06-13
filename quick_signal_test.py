#!/usr/bin/env python3
"""
QUICK SIGNAL TEST
Test cepat untuk semua sistem
"""

def test_all_systems():
    """Test semua sistem dengan signal test"""
    print("🚀 QUICK SIGNAL SYSTEM TEST")
    print("=" * 60)
    
    # Test data
    test_signal = {
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
    
    print(f"Test Signal: {test_signal}\n")
    
    results = {}
    
    # 1. Test Fibonacci
    try:
        from fibonacci_signal_detector import FibonacciSignalDetector
        fib = FibonacciSignalDetector()
        result = fib.detect_signal(test_signal)
        results['fibonacci'] = {
            'status': 'SUCCESS',
            'recommendation': result.get('recommendation', 'UNKNOWN'),
            'win_rate': result.get('expected_win_rate', 0),
            'strength': result.get('signal_strength', 'UNKNOWN')
        }
        print("✅ Fibonacci: SUCCESS")
    except Exception as e:
        results['fibonacci'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ Fibonacci: {e}")
    
    # 2. Test Ensemble
    try:
        from ensemble_signal_detector import EnsembleSignalDetector
        ensemble = EnsembleSignalDetector()
        if ensemble.load_ensemble_model():
            result = ensemble.predict_signal_strength(test_signal)
            results['ensemble'] = {
                'status': 'SUCCESS',
                'recommendation': result.get('recommendation', 'UNKNOWN'),
                'probability': result.get('ensemble_probability', 0),
                'strength': result.get('signal_strength', 'UNKNOWN')
            }
            print("✅ Ensemble: SUCCESS")
        else:
            results['ensemble'] = {'status': 'NO_MODEL'}
            print("⚠️  Ensemble: No model")
    except Exception as e:
        results['ensemble'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ Ensemble: {e}")
    
    # 3. Test Fixed Optimizer
    try:
        from fixed_advanced_signal_optimizer import FixedAdvancedSignalOptimizer
        optimizer = FixedAdvancedSignalOptimizer()
        if optimizer.load_model():
            result = optimizer.get_signal_strength(test_signal)
            results['optimizer'] = {
                'status': 'SUCCESS',
                'recommendation': result.get('recommendation', 'UNKNOWN'),
                'probability': result.get('win_probability', 0),
                'strength': result.get('signal_strength', 'UNKNOWN')
            }
            print("✅ Optimizer: SUCCESS")
        else:
            results['optimizer'] = {'status': 'NO_MODEL'}
            print("⚠️  Optimizer: No model")
    except Exception as e:
        results['optimizer'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ Optimizer: {e}")
    
    # Summary
    print(f"\n📊 RESULTS SUMMARY:")
    print("=" * 60)
    
    working_systems = 0
    for system, result in results.items():
        status = result['status']
        if status == 'SUCCESS':
            working_systems += 1
            rec = result.get('recommendation', 'N/A')
            strength = result.get('strength', 'N/A')
            prob = result.get('probability', result.get('win_rate', 0))
            print(f"{system.upper():<12}: ✅ {rec} ({strength}) - {prob:.1%}" if prob else f"{system.upper():<12}: ✅ {rec} ({strength})")
        elif status == 'NO_MODEL':
            print(f"{system.upper():<12}: ⚠️  Model not available")
        else:
            print(f"{system.upper():<12}: ❌ Error")
    
    print(f"\nWorking Systems: {working_systems}/3")
    
    if working_systems >= 2:
        print("🎉 System is ready for trading!")
    elif working_systems >= 1:
        print("⚠️  System partially ready")
    else:
        print("🚨 System needs fixes")
    
    return results

if __name__ == "__main__":
    test_all_systems()
