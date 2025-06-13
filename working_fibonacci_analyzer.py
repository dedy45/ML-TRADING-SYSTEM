#!/usr/bin/env python3
"""
Working Fibonacci Analyzer - Guaranteed to work!
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import json

class SimpleFibonacciAnalyzer:
    """Simple, working Fibonacci analyzer"""
    
    def __init__(self):
        print("🚀 Simple Fibonacci Analyzer initialized")
        self.data = None
        self.results = {}
    
    def analyze_fibonacci_levels(self, max_files=10):
        """Analyze Fibonacci levels from CSV files"""
        print(f"\n📊 FIBONACCI LEVEL ANALYSIS")
        print("=" * 40)
        
        # Get CSV files
        csv_files = glob.glob("dataBT/*.csv")
        if not csv_files:
            print("❌ No CSV files found in dataBT folder")
            return None
            
        print(f"📁 Found {len(csv_files)} total CSV files")
        
        # Sample files for analysis
        if len(csv_files) > max_files:
            import random
            random.seed(42)
            csv_files = random.sample(csv_files, max_files)
            
        print(f"📊 Analyzing {len(csv_files)} files...")
        
        all_data = []
        fibonacci_stats = {}
        
        # Process each file
        for i, file_path in enumerate(csv_files):
            try:
                print(f"   🔄 Processing file {i+1}/{len(csv_files)}...")
                
                # Read CSV
                df = pd.read_csv(file_path)
                
                # Filter out system messages
                df = df[df['Type'] != 'INIT_SUCCESS']
                
                if len(df) == 0:
                    continue
                    
                # Add basic analysis
                df['is_profitable'] = (df['Profit'] > 0).astype(int)
                df['file_source'] = os.path.basename(file_path)
                
                all_data.append(df)
                
                # Analyze Fibonacci levels if available
                if 'LevelFibo' in df.columns:
                    fib_analysis = df.groupby('LevelFibo').agg({
                        'is_profitable': ['count', 'sum', 'mean'],
                        'Profit': ['mean', 'sum']
                    }).round(3)
                    
                    for level in fib_analysis.index:
                        if level not in fibonacci_stats:
                            fibonacci_stats[level] = {
                                'total_trades': 0,
                                'profitable_trades': 0,
                                'win_rate': 0,
                                'avg_profit': 0,
                                'total_profit': 0
                            }
                        
                        level_data = fib_analysis.loc[level]
                        fibonacci_stats[level]['total_trades'] += int(level_data[('is_profitable', 'count')])
                        fibonacci_stats[level]['profitable_trades'] += int(level_data[('is_profitable', 'sum')])
                        fibonacci_stats[level]['total_profit'] += float(level_data[('Profit', 'sum')])
                        
            except Exception as e:
                print(f"   ⚠️  Error processing file {i+1}: {str(e)}")
                continue
        
        if not all_data:
            print("❌ No valid data loaded")
            return None
            
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Calculate final statistics
        for level in fibonacci_stats:
            stats = fibonacci_stats[level]
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['profitable_trades'] / stats['total_trades']
                stats['avg_profit'] = stats['total_profit'] / stats['total_trades']
        
        self.results = fibonacci_stats
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display analysis results"""
        print(f"\n🎯 ANALYSIS RESULTS")
        print("=" * 50)
        
        total_trades = len(self.data)
        profitable_trades = len(self.data[self.data['is_profitable'] == 1])
        overall_win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        print(f"📊 Overall Statistics:")
        print(f"   Total trades: {total_trades:,}")
        print(f"   Profitable trades: {profitable_trades:,}")
        print(f"   Overall win rate: {overall_win_rate:.1%}")
        
        if 'LevelFibo' in self.data.columns:
            print(f"\n🔢 Fibonacci Level Analysis:")
            print("-" * 30)
            
            # Sort by win rate
            sorted_levels = sorted(self.results.items(), 
                                 key=lambda x: x[1]['win_rate'], 
                                 reverse=True)
            
            print(f"{'Level':<15} {'Trades':<8} {'Win Rate':<10} {'Avg Profit':<12}")
            print("-" * 50)
            
            for level, stats in sorted_levels[:10]:  # Top 10
                if stats['total_trades'] >= 5:  # Minimum trades for significance
                    print(f"{str(level):<15} {stats['total_trades']:<8} "
                          f"{stats['win_rate']:<10.1%} {stats['avg_profit']:<12.2f}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        try:
            os.makedirs("reports", exist_ok=True)
            
            # Prepare data for JSON
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'total_trades': len(self.data),
                'profitable_trades': len(self.data[self.data['is_profitable'] == 1]),
                'fibonacci_levels': self.results
            }
            
            # Save to JSON
            filename = f"reports/fibonacci_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            print(f"\n💾 Results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

def main():
    """Main function"""
    analyzer = SimpleFibonacciAnalyzer()
    results = analyzer.analyze_fibonacci_levels(max_files=15)
    
    if results:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed")

if __name__ == "__main__":
    main()
