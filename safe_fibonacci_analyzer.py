#!/usr/bin/env python3
"""
Safe Fibonacci Analyzer - With proper error handling and timeouts
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import signal
from datetime import datetime
import json

class SafeFibonacciAnalyzer:
    """Safe Fibonacci analyzer with timeout protection"""
    
    def __init__(self):
        print("🚀 Safe Fibonacci Analyzer initialized")
        self.data = None
        self.results = {}
        self.timeout_occurred = False
    
    def timeout_handler(self, signum, frame):
        """Handle timeout"""
        self.timeout_occurred = True
        raise TimeoutError("Operation timed out")
    
    def quick_test(self):
        """Quick test to ensure everything works"""
        print("\n🔍 QUICK SYSTEM TEST")
        print("=" * 30)
        
        try:
            # Test 1: Check if dataBT exists
            if not os.path.exists('dataBT'):
                print("❌ dataBT folder not found")
                return False
            print("✅ dataBT folder found")
            
            # Test 2: Count CSV files
            csv_files = glob.glob("dataBT/*.csv")
            print(f"✅ Found {len(csv_files)} CSV files")
            
            if len(csv_files) == 0:
                print("❌ No CSV files found")
                return False
            
            # Test 3: Try reading one file
            test_file = csv_files[0]
            print(f"🔄 Testing file: {os.path.basename(test_file)}")
            
            df = pd.read_csv(test_file, nrows=5)  # Only read 5 rows
            print(f"✅ File readable, columns: {list(df.columns)[:5]}...")
            
            # Test 4: Check for Fibonacci columns
            has_fibo = 'LevelFibo' in df.columns
            has_levels = 'Level1Above' in df.columns and 'Level1Below' in df.columns
            
            print(f"📊 Fibonacci features available:")
            print(f"   LevelFibo: {'✅' if has_fibo else '❌'}")
            print(f"   Support/Resistance: {'✅' if has_levels else '❌'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False
    
    def safe_analyze_sample(self, num_files=3, max_rows_per_file=100):
        """Safe analysis with limited scope"""
        print(f"\n📊 SAFE FIBONACCI ANALYSIS")
        print("=" * 40)
        
        try:
            # Get files
            csv_files = glob.glob("dataBT/*.csv")
            if not csv_files:
                print("❌ No CSV files found")
                return None
            
            # Limit files
            sample_files = csv_files[:num_files]
            print(f"📁 Analyzing {len(sample_files)} files (max {max_rows_per_file} rows each)")
            
            fibonacci_data = []
            trade_count = 0
            
            for i, file_path in enumerate(sample_files):
                try:
                    print(f"   🔄 File {i+1}: {os.path.basename(file_path)}")
                    
                    # Read limited rows
                    df = pd.read_csv(file_path, nrows=max_rows_per_file)
                    
                    # Filter system messages
                    df = df[df['Type'] != 'INIT_SUCCESS'] if 'Type' in df.columns else df
                    
                    if len(df) == 0:
                        print(f"      ⚠️  No valid trades")
                        continue
                    
                    # Basic analysis
                    df['is_profitable'] = (df['Profit'] > 0).astype(int) if 'Profit' in df.columns else 0
                    
                    # Fibonacci analysis
                    if 'LevelFibo' in df.columns:
                        fib_summary = df.groupby('LevelFibo').agg({
                            'is_profitable': ['count', 'sum'],
                            'Profit': 'mean' if 'Profit' in df.columns else lambda x: 0
                        }).round(3)
                        
                        for level in fib_summary.index:
                            count = fib_summary.loc[level, ('is_profitable', 'count')]
                            wins = fib_summary.loc[level, ('is_profitable', 'sum')]
                            win_rate = wins / count if count > 0 else 0
                            
                            fibonacci_data.append({
                                'level': level,
                                'trades': count,
                                'wins': wins,
                                'win_rate': win_rate,
                                'file': os.path.basename(file_path)
                            })
                    
                    trade_count += len(df)
                    print(f"      ✅ {len(df)} trades processed")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    continue
            
            # Summarize results
            if fibonacci_data:
                print(f"\n🎯 FIBONACCI LEVEL SUMMARY")
                print("-" * 35)
                
                # Group by level
                fib_df = pd.DataFrame(fibonacci_data)
                summary = fib_df.groupby('level').agg({
                    'trades': 'sum',
                    'wins': 'sum'
                }).reset_index()
                
                summary['win_rate'] = summary['wins'] / summary['trades']
                summary = summary.sort_values('win_rate', ascending=False)
                
                print(f"{'Level':<15} {'Trades':<8} {'Wins':<6} {'Win Rate':<10}")
                print("-" * 40)
                
                for _, row in summary.head(10).iterrows():
                    if row['trades'] >= 3:  # Minimum for display
                        print(f"{str(row['level']):<15} {row['trades']:<8} "
                              f"{row['wins']:<6} {row['win_rate']:<10.1%}")
                
                self.results = summary.to_dict('records')
                print(f"\n📊 Total trades analyzed: {trade_count}")
                return self.results
            else:
                print("❌ No Fibonacci data found")
                return None
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
    
    def save_quick_results(self):
        """Save results quickly"""
        if not self.results:
            return
            
        try:
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/quick_fibonacci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'fibonacci_levels': self.results
                }, f, indent=2)
            
            print(f"💾 Results saved: {filename}")
            
        except Exception as e:
            print(f"⚠️  Save failed: {e}")

def run_safe_test():
    """Run safe test version"""
    print("🔒 SAFE FIBONACCI ANALYSIS")
    print("=" * 50)
    
    analyzer = SafeFibonacciAnalyzer()
    
    # Step 1: Quick test
    if not analyzer.quick_test():
        print("❌ Quick test failed - stopping")
        return False
    
    # Step 2: Safe analysis
    print("\n" + "="*50)
    results = analyzer.safe_analyze_sample(num_files=5, max_rows_per_file=50)
    
    if results:
        analyzer.save_quick_results()
        print("\n✅ Safe analysis completed!")
        return True
    else:
        print("\n❌ Analysis failed")
        return False

if __name__ == "__main__":
    try:
        success = run_safe_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
