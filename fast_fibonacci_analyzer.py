#!/usr/bin/env python3
"""
Optimized Fibonacci Level Analysis for Trading Data
Menganalisis file CSV dengan performa yang lebih baik
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import time

warnings.filterwarnings('ignore')

class FastFibonacciAnalyzer:
    """Optimized analyzer untuk level Fibonacci dan statistik trading"""
    
    def __init__(self):
        self.all_data = None
        self.fibonacci_stats = {}
        self.session_stats = {}
        self.hourly_stats = {}
        self.level_performance = {}
        
    def load_sample_data(self, data_folder="dataBT", max_files=50, max_rows_per_file=1000):
        """Load data dengan optimasi untuk performa cepat"""
        print("🚀 FAST FIBONACCI ANALYZER")
        print("=" * 60)
        print("🔄 Loading optimized sample data...")
        
        start_time = time.time()
        
        csv_files = glob.glob(f"{data_folder}/*.csv")
        total_files = len(csv_files)
        
        # Randomly sample files for diversity
        import random
        random.seed(42)
        if len(csv_files) > max_files:
            csv_files = random.sample(csv_files, max_files)
            
        print(f"📁 Found {total_files} total files")
        print(f"📊 Processing {len(csv_files)} files (max {max_rows_per_file} rows each)")
        
        dataframes = []
        file_stats = []
        total_rows = 0
        
        for i, file in enumerate(csv_files):
            try:
                # Read limited rows to speed up loading
                df = pd.read_csv(file, nrows=max_rows_per_file * 2)
                
                # Quick filtering
                df = df[df['Type'] != 'INIT_SUCCESS']
                
                if len(df) > max_rows_per_file:
                    df = df.sample(n=max_rows_per_file, random_state=42)
                
                if len(df) > 0:
                    df['source_file'] = os.path.basename(file)
                    dataframes.append(df)
                    total_rows += len(df)
                    
                    # Quick stats
                    profitable = (df['Profit'] > 0).sum()
                    file_stats.append({
                        'file': os.path.basename(file),
                        'trades': len(df),
                        'profitable': profitable,
                        'win_rate': profitable / len(df) * 100 if len(df) > 0 else 0,
                        'avg_profit': df['Profit'].mean()
                    })
                
                # Progress every 10 files
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"   📋 {i+1}/{len(csv_files)} files | {total_rows:,} rows | {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"   ❌ Error: {os.path.basename(file)}: {str(e)[:50]}...")
                continue
        
        if dataframes:
            self.all_data = pd.concat(dataframes, ignore_index=True)
            self.all_data['Timestamp'] = pd.to_datetime(self.all_data['Timestamp'])
            
            elapsed_total = time.time() - start_time
            
            print(f"\n✅ LOADING COMPLETED in {elapsed_total:.1f} seconds!")
            print(f"📊 Total trades: {len(self.all_data):,}")
            print(f"📁 Files processed: {len(dataframes)}")
            
            profitable_trades = (self.all_data['Profit'] > 0).sum()
            print(f"📈 Win rate: {profitable_trades / len(self.all_data) * 100:.1f}%")
            print(f"💰 Total profit: {self.all_data['Profit'].sum():.2f}")
            
            return file_stats
        else:
            print("❌ No data loaded!")
            return []
    
    def quick_fibonacci_analysis(self):
        """Analisis cepat level Fibonacci"""
        if self.all_data is None:
            print("❌ No data loaded!")
            return
            
        print("\n🔢 QUICK FIBONACCI ANALYSIS")
        print("=" * 40)
        
        df = self.all_data.copy()
        
        # Check available columns
        fib_columns = ['LevelFibo'] if 'LevelFibo' in df.columns else []
        if not fib_columns:
            print("❌ No LevelFibo column found!")
            return
        
        # Basic Fibonacci analysis
        fib_analysis = df.groupby('LevelFibo').agg({
            'Profit': ['count', 'sum', 'mean'],
            'Type': lambda x: (x == 'BUY').sum()
        }).round(4)
        
        fib_analysis.columns = ['Total_Trades', 'Total_Profit', 'Avg_Profit', 'BUY_Count']
        fib_analysis['Win_Rate'] = df.groupby('LevelFibo')['Profit'].apply(lambda x: (x > 0).mean() * 100).round(2)
        fib_analysis['SELL_Count'] = fib_analysis['Total_Trades'] - fib_analysis['BUY_Count']
        
        # Sort by win rate
        fib_analysis = fib_analysis.sort_values('Win_Rate', ascending=False)
        
        print("🏆 TOP FIBONACCI LEVELS BY WIN RATE:")
        print(fib_analysis.head(10).to_string())
        
        # Save results
        self.fibonacci_stats = fib_analysis.to_dict('index')
        
        # Quick statistics
        print(f"\n📊 FIBONACCI SUMMARY:")
        print(f"   🔢 Total levels analyzed: {len(fib_analysis)}")
        print(f"   🏆 Best level: {fib_analysis.index[0]} (Win Rate: {fib_analysis.iloc[0]['Win_Rate']:.1f}%)")
        print(f"   💰 Most profitable: {fib_analysis.sort_values('Total_Profit', ascending=False).index[0]}")
        print(f"   📈 Most traded: {fib_analysis.sort_values('Total_Trades', ascending=False).index[0]}")
        
        return fib_analysis
    
    def analyze_trading_sessions(self):
        """Analisis sesi trading dengan cepat"""
        if self.all_data is None:
            return
            
        print("\n🌏 TRADING SESSION ANALYSIS")
        print("=" * 40)
        
        df = self.all_data.copy()
        
        # Session columns
        session_cols = ['SessionAsia', 'SessionEurope', 'SessionUS']
        available_sessions = [col for col in session_cols if col in df.columns]
        
        if not available_sessions:
            print("❌ No session columns found!")
            return
        
        # Create session combinations
        df['Active_Sessions'] = 0
        for col in available_sessions:
            df['Active_Sessions'] += df[col]
        
        # Session analysis
        session_analysis = df.groupby('Active_Sessions').agg({
            'Profit': ['count', 'mean', lambda x: (x > 0).mean() * 100]
        }).round(4)
        
        session_analysis.columns = ['Total_Trades', 'Avg_Profit', 'Win_Rate']
        session_analysis = session_analysis.sort_values('Win_Rate', ascending=False)
        
        print("📊 SESSION OVERLAP ANALYSIS:")
        print(session_analysis.to_string())
        
        self.session_stats = session_analysis.to_dict('index')
        
        return session_analysis
    
    def analyze_hourly_patterns(self):
        """Analisis pola jam trading"""
        if self.all_data is None:
            return
            
        print("\n⏰ HOURLY TRADING PATTERNS")
        print("=" * 40)
        
        df = self.all_data.copy()
        df['Hour'] = df['Timestamp'].dt.hour
        
        hourly_analysis = df.groupby('Hour').agg({
            'Profit': ['count', 'mean', lambda x: (x > 0).mean() * 100]
        }).round(4)
        
        hourly_analysis.columns = ['Total_Trades', 'Avg_Profit', 'Win_Rate']
        
        # Find best hours
        best_hours = hourly_analysis.sort_values('Win_Rate', ascending=False).head(5)
        
        print("🏆 TOP 5 BEST TRADING HOURS:")
        print(best_hours.to_string())
        
        self.hourly_stats = hourly_analysis.to_dict('index')
        
        return hourly_analysis
    
    def generate_comprehensive_report(self):
        """Generate laporan komprehensif"""
        print("\n📋 COMPREHENSIVE FIBONACCI REPORT")
        print("=" * 60)
        
        # Run all analyses
        fib_stats = self.quick_fibonacci_analysis()
        session_stats = self.analyze_trading_sessions()
        hourly_stats = self.analyze_hourly_patterns()
        
        # Create summary
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_trades': len(self.all_data),
                'total_files': len(self.all_data['source_file'].unique()),
                'date_range': f"{self.all_data['Timestamp'].min()} to {self.all_data['Timestamp'].max()}",
                'overall_win_rate': (self.all_data['Profit'] > 0).mean() * 100,
                'total_profit': self.all_data['Profit'].sum()
            },
            'top_fibonacci_levels': fib_stats.head(10).to_dict('index') if fib_stats is not None else {},
            'session_performance': session_stats.to_dict('index') if session_stats is not None else {},
            'hourly_performance': hourly_stats.to_dict('index') if hourly_stats is not None else {}
        }
        
        # Save report
        report_file = f"reports/fibonacci_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Report saved: {report_file}")
        
        # Print key insights
        if fib_stats is not None and len(fib_stats) > 0:
            print(f"\n🎯 KEY INSIGHTS:")
            print(f"   🏆 Best Fibonacci Level: {fib_stats.index[0]} ({fib_stats.iloc[0]['Win_Rate']:.1f}% win rate)")
            print(f"   💰 Most Profitable Level: {fib_stats.sort_values('Total_Profit', ascending=False).index[0]}")
            print(f"   📊 Total Fibonacci Levels: {len(fib_stats)}")
            
            # High-performance levels (>60% win rate)
            high_perf = fib_stats[fib_stats['Win_Rate'] > 60]
            if len(high_perf) > 0:
                print(f"   🚀 High Performance Levels (>60%): {len(high_perf)} levels")
                print("      Levels:", list(high_perf.index[:5]))
        
        return report

def main():
    """Main function untuk testing cepat"""
    analyzer = FastFibonacciAnalyzer()
    
    print("🚀 Starting Fast Fibonacci Analysis...")
    
    # Load sample data (fast)
    file_stats = analyzer.load_sample_data(max_files=30, max_rows_per_file=500)
    
    if file_stats:
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the reports/ folder for detailed results")
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()
