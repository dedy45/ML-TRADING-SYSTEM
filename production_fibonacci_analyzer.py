# -*- coding: utf-8 -*-
"""
PRODUCTION READY Fibonacci Analyzer
No unicode issues, optimized for performance
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from collections import defaultdict
import time

class ProductionFibonacciAnalyzer:
    """Production-ready Fibonacci analyzer with no unicode issues"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.results = {}
        self.total_files_processed = 0
        self.total_trades_analyzed = 0
        
    def analyze_fibonacci_levels(self, max_files=10, sample_size=200):
        """Analyze Fibonacci levels performance"""
        print("=" * 60)
        print("FIBONACCI LEVEL ANALYSIS")
        print("=" * 60)
        
        # Get CSV files
        csv_files = glob.glob(f"{self.data_path}/*.csv")
        if not csv_files:
            print("[ERROR] No CSV files found in", self.data_path)
            return None
            
        print(f"[INFO] Found {len(csv_files)} CSV files")
        print(f"[INFO] Processing {min(max_files, len(csv_files))} files")
        print(f"[INFO] Sample size per file: {sample_size} rows")
        print("-" * 60)
        
        # Process files
        fibonacci_stats = defaultdict(lambda: {
            'total_trades': 0, 
            'profitable_trades': 0, 
            'total_profit': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0
        })
        
        all_data = []
        
        for i, file_path in enumerate(csv_files[:max_files]):
            try:
                filename = os.path.basename(file_path)
                print(f"[{i+1:2d}/{max_files}] Processing: {filename}")
                
                # Read file with limited rows
                df = pd.read_csv(file_path, nrows=sample_size)
                
                # Filter valid trades only
                df = df[df['Type'].isin(['BUY', 'SELL'])].copy()
                
                if len(df) == 0:
                    print(f"        [SKIP] No valid trades found")
                    continue
                
                # Add profitability indicator
                df['is_profitable'] = (df['Profit'] > 0).astype(int)
                
                # Fibonacci analysis
                if 'LevelFibo' in df.columns:
                    fib_levels = df['LevelFibo'].dropna().unique()
                    
                    for level in fib_levels:
                        level_data = df[df['LevelFibo'] == level]
                        
                        total_trades = len(level_data)
                        profitable_trades = level_data['is_profitable'].sum()
                        total_profit = level_data['Profit'].sum()
                        
                        # Update statistics
                        stats = fibonacci_stats[level]
                        stats['total_trades'] += total_trades
                        stats['profitable_trades'] += profitable_trades
                        stats['total_profit'] += total_profit
                    
                    print(f"        [OK] {len(df)} trades, {len(fib_levels)} Fibonacci levels")
                else:
                    print(f"        [WARN] No LevelFibo column found")
                
                all_data.append(df)
                self.total_files_processed += 1
                self.total_trades_analyzed += len(df)
                
            except Exception as e:
                print(f"        [ERROR] Failed to process file: {str(e)}")
                continue
        
        if not all_data:
            print("[ERROR] No data successfully processed")
            return None
        
        # Calculate final statistics
        for level, stats in fibonacci_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['profitable_trades'] / stats['total_trades']) * 100
                stats['avg_profit'] = stats['total_profit'] / stats['total_trades']
        
        # Sort by win rate
        sorted_levels = sorted(
            fibonacci_stats.items(), 
            key=lambda x: x[1]['win_rate'], 
            reverse=True
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("FIBONACCI LEVEL PERFORMANCE RESULTS")
        print("=" * 80)
        
        print(f"Total Files Processed: {self.total_files_processed}")
        print(f"Total Trades Analyzed: {self.total_trades_analyzed}")
        print(f"Fibonacci Levels Found: {len(fibonacci_stats)}")
        
        print("\n" + "-" * 80)
        print(f"{'Level':<20} {'Trades':<8} {'Wins':<6} {'Win Rate':<10} {'Avg Profit':<12}")
        print("-" * 80)
        
        # Show top performing levels
        for level, stats in sorted_levels:
            if stats['total_trades'] >= 5:  # Minimum trades for significance
                print(f"{str(level):<20} {stats['total_trades']:<8} "
                      f"{stats['profitable_trades']:<6} "
                      f"{stats['win_rate']:<10.1f}% "
                      f"{stats['avg_profit']:<12.2f}")
        
        # Store results
        self.results = {
            'fibonacci_levels': dict(fibonacci_stats),
            'summary': {
                'files_processed': self.total_files_processed,
                'total_trades': self.total_trades_analyzed,
                'levels_found': len(fibonacci_stats)
            }
        }
        
        return self.results
    
    def analyze_session_performance(self, max_files=5):
        """Analyze trading session performance"""
        print("\n" + "=" * 60)
        print("TRADING SESSION ANALYSIS")
        print("=" * 60)
        
        csv_files = glob.glob(f"{self.data_path}/*.csv")[:max_files]
        
        session_stats = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0
        })
        
        for i, file_path in enumerate(csv_files):
            try:
                filename = os.path.basename(file_path)
                print(f"[{i+1}/{len(csv_files)}] Processing: {filename}")
                
                df = pd.read_csv(file_path, nrows=100)  # Smaller sample for session analysis
                df = df[df['Type'].isin(['BUY', 'SELL'])].copy()
                
                if len(df) == 0:
                    continue
                
                df['is_profitable'] = (df['Profit'] > 0).astype(int)
                
                # Analyze sessions
                sessions = ['SessionAsia', 'SessionEurope', 'SessionUS']
                for session in sessions:
                    if session in df.columns:
                        session_trades = df[df[session] == 1]
                        if len(session_trades) > 0:
                            stats = session_stats[session]
                            stats['total_trades'] += len(session_trades)
                            stats['profitable_trades'] += session_trades['is_profitable'].sum()
                            stats['total_profit'] += session_trades['Profit'].sum()
                
            except Exception as e:
                print(f"        [ERROR] {str(e)}")
                continue
        
        # Calculate win rates
        for session, stats in session_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['profitable_trades'] / stats['total_trades']) * 100
        
        # Display session results
        print("\nSESSION PERFORMANCE:")
        print("-" * 40)
        print(f"{'Session':<15} {'Trades':<8} {'Win Rate':<10} {'Total Profit':<12}")
        print("-" * 40)
        
        for session, stats in session_stats.items():
            if stats['total_trades'] > 0:
                print(f"{session:<15} {stats['total_trades']:<8} "
                      f"{stats['win_rate']:<10.1f}% "
                      f"{stats['total_profit']:<12.2f}")
    
    def save_results(self, filename="fibonacci_analysis_results.txt"):
        """Save results to text file"""
        if not self.results:
            print("[WARNING] No results to save")
            return False
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("FIBONACCI ANALYSIS RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                summary = self.results['summary']
                f.write(f"Files Processed: {summary['files_processed']}\n")
                f.write(f"Total Trades: {summary['total_trades']}\n")
                f.write(f"Fibonacci Levels: {summary['levels_found']}\n\n")
                
                f.write("FIBONACCI LEVEL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                
                # Sort levels by win rate
                fib_levels = self.results['fibonacci_levels']
                sorted_levels = sorted(
                    fib_levels.items(),
                    key=lambda x: x[1]['win_rate'],
                    reverse=True
                )
                
                for level, stats in sorted_levels:
                    if stats['total_trades'] >= 5:
                        f.write(f"Level {level}:\n")
                        f.write(f"  Trades: {stats['total_trades']}\n")
                        f.write(f"  Win Rate: {stats['win_rate']:.1f}%\n")
                        f.write(f"  Avg Profit: {stats['avg_profit']:.2f}\n\n")
            
            print(f"[SUCCESS] Results saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {str(e)}")
            return False

def main():
    """Main analysis function"""
    start_time = time.time()
    
    print("PRODUCTION FIBONACCI ANALYZER")
    print("Starting analysis...")
    
    # Create analyzer
    analyzer = ProductionFibonacciAnalyzer()
    
    # Run Fibonacci analysis
    results = analyzer.analyze_fibonacci_levels(max_files=10, sample_size=200)
    
    if results:
        # Run session analysis
        analyzer.analyze_session_performance(max_files=5)
        
        # Save results
        analyzer.save_results("production_fibonacci_results.txt")
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Total Time: {elapsed_time:.2f} seconds")
        print(f"Files Processed: {analyzer.total_files_processed}")
        print(f"Trades Analyzed: {analyzer.total_trades_analyzed}")
        print("Results saved to: production_fibonacci_results.txt")
        
    else:
        print("\n[ERROR] Analysis failed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Analysis stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
