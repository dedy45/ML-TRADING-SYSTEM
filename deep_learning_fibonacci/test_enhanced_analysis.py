"""
Quick test to verify data loading and run enhanced ML analysis
"""

import sys
from pathlib import Path

# Test data path
test_path = Path("../../dataBT").resolve()
print(f"Testing data path: {test_path}")
print(f"Path exists: {test_path.exists()}")

if test_path.exists():
    csv_files = list(test_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    if csv_files:
        print("✅ Data found! Running enhanced analysis...")
        
        # Import and run the analyzer
        from enhanced_ml_fibonacci import EnhancedFibonacciAnalyzer
        
        analyzer = EnhancedFibonacciAnalyzer()
        results = analyzer.run_enhanced_analysis(max_files=20)
        
        if results:
            print(f"\n🎉 Analysis Complete!")
            print(f"Best model: {results['best_model']}")
            print(f"Win rate: {results['best_win_rate']:.1%}")
            print(f"Improvement: {results['improvement']:+.3f}")
            print(f"Target achieved: {results['target_achieved']}")
        else:
            print("❌ Analysis failed")
    else:
        print("❌ No CSV files found")
else:
    print("❌ Data directory not found")
    print("Available directories:")
    parent = Path("..").resolve()
    for item in parent.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
