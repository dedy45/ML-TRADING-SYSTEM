#!/usr/bin/env python3
"""
Comprehensive Fibonacci Level Analysis for Trading Data
Menganalisis semua file CSV untuk mencari level Fibonacci terbaik
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

warnings.filterwarnings('ignore')

class FibonacciAnalyzer:
    """Analyzer untuk level Fibonacci dan statistik trading"""
    
    def __init__(self):
        self.all_data = None
        self.fibonacci_stats = {}
        self.session_stats = {}
        self.hourly_stats = {}
        self.level_performance = {}
      def load_all_data(self, data_folder="dataBT", max_files=50, max_rows_per_file=2000, target_total_rows=50000):
        """Load data dengan optimasi untuk performa"""
        print("🔄 Loading data with performance optimization...")
        print("=" * 60)
        
        csv_files = glob.glob(f"{data_folder}/*.csv")
        total_files = len(csv_files)
        
        # Limit files if specified
        if max_files and max_files < len(csv_files):
            # Randomly sample files for better diversity
            import random
            random.seed(42)
            csv_files = random.sample(csv_files, max_files)
            
        print(f"📁 Found {total_files} CSV files")
        print(f"📊 Processing {len(csv_files)} files (max {max_rows_per_file} rows each)")
        print(f"🎯 Target: {target_total_rows:,} total rows")
        
        dataframes = []
        file_stats = []
        total_rows_loaded = 0
        files_processed = 0
        
        for i, file in enumerate(csv_files):
            try:
                # Check if we've reached target
                if total_rows_loaded >= target_total_rows:
                    print(f"🎯 Target reached: {total_rows_loaded:,} rows from {files_processed} files")
                    break
                
                # Read file with chunking for large files
                df = pd.read_csv(file, nrows=max_rows_per_file * 2)  # Read a bit more to sample
                
                # Remove system messages
                df = df[df['Type'] != 'INIT_SUCCESS']
                
                if len(df) > 0:
                    # Sample if file is large
                    if len(df) > max_rows_per_file:
                        df = df.sample(n=max_rows_per_file, random_state=42)
                    
                    df['source_file'] = os.path.basename(file)
                    dataframes.append(df)
                    total_rows_loaded += len(df)
                    files_processed += 1
                    
                    # Collect file statistics
                    file_stat = {
                        'file': os.path.basename(file),
                        'total_trades': len(df),
                        'profitable_trades': len(df[df['Profit'] > 0]),
                        'win_rate': len(df[df['Profit'] > 0]) / len(df) * 100 if len(df) > 0 else 0,
                        'avg_profit': df['Profit'].mean(),
                        'total_profit': df['Profit'].sum()
                    }
                    file_stats.append(file_stat)
                
                # Progress indicator - more frequent for feedback
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"   📋 Processed {i + 1}/{len(csv_files)} files | Rows: {total_rows_loaded:,}")
                    
            except Exception as e:
                print(f"   ❌ Error in file {os.path.basename(file)}: {str(e)}")
                continue
        
        if dataframes:
            self.all_data = pd.concat(dataframes, ignore_index=True)
            
            # Convert timestamp
            self.all_data['Timestamp'] = pd.to_datetime(self.all_data['Timestamp'])
            
            print(f"\n🎉 DATA LOADING COMPLETED!")
            print(f"📊 Total trades loaded: {len(self.all_data):,}")
            print(f"📁 Files processed: {len(dataframes)}")
            print(f"📈 Overall win rate: {len(self.all_data[self.all_data['Profit'] > 0]) / len(self.all_data) * 100:.2f}%")
            print(f"💰 Total profit: {self.all_data['Profit'].sum():.2f}")
            
            return file_stats
        else:
            print("❌ No data loaded!")
            return []
    
    def analyze_fibonacci_levels(self):
        """Analisis mendalam level Fibonacci"""
        if self.all_data is None:
            print("❌ No data loaded!")
            return
            
        print("\n🔢 FIBONACCI LEVEL ANALYSIS")
        print("=" * 50)
        
        df = self.all_data.copy()
        
        # Analisis LevelFibo
        if 'LevelFibo' in df.columns:
            fib_analysis = df.groupby('LevelFibo').agg({
                'Profit': ['count', 'sum', 'mean', 'std'],
                'MAE_pips': 'mean',
                'MFE_pips': 'mean',
                'TP': 'mean',
                'SL': 'mean'
            }).round(4)
            
            # Flatten column names
            fib_analysis.columns = ['_'.join(col).strip() for col in fib_analysis.columns]
            
            # Calculate win rate
            win_rates = df.groupby('LevelFibo').apply(
                lambda x: (x['Profit'] > 0).sum() / len(x) * 100
            ).round(2)
            
            fib_analysis['win_rate'] = win_rates
            
            # Calculate risk-reward ratio
            rr_ratios = df.groupby('LevelFibo').apply(
                lambda x: x['TP'].mean() / x['SL'].mean() if x['SL'].mean() > 0 else 0
            ).round(3)
            
            fib_analysis['risk_reward_ratio'] = rr_ratios
            
            # Sort by profitability
            fib_analysis = fib_analysis.sort_values('Profit_sum', ascending=False)
            
            self.fibonacci_stats = fib_analysis
            
            print("📊 TOP FIBONACCI LEVELS BY PROFITABILITY:")
            print("-" * 50)
            
            for i, (level, row) in enumerate(fib_analysis.head(10).iterrows()):
                print(f"{i+1:2d}. Level {level}:")
                print(f"    💰 Total Profit: {row['Profit_sum']:8.2f}")
                print(f"    📊 Win Rate:     {row['win_rate']:8.2f}%")
                print(f"    🎯 Avg Profit:   {row['Profit_mean']:8.4f}")
                print(f"    📈 Total Trades: {row['Profit_count']:8.0f}")
                print(f"    ⚖️  Risk/Reward:  {row['risk_reward_ratio']:8.3f}")
                print(f"    📉 MAE (pips):   {row['MAE_pips_mean']:8.2f}")
                print(f"    📈 MFE (pips):   {row['MFE_pips_mean']:8.2f}")
                print()
                
        # Analisis Level1Above dan Level1Below
        self._analyze_support_resistance_levels()
        
    def _analyze_support_resistance_levels(self):
        """Analisis level support dan resistance"""
        df = self.all_data.copy()
        
        print("📈 SUPPORT & RESISTANCE ANALYSIS")
        print("-" * 40)
        
        if 'Level1Above' in df.columns and 'Level1Below' in df.columns:
            # Calculate distance to levels
            df['distance_to_above'] = abs(df['OpenPrice'] - df['Level1Above'])
            df['distance_to_below'] = abs(df['OpenPrice'] - df['Level1Below'])
            df['nearest_level'] = np.where(
                df['distance_to_above'] < df['distance_to_below'], 
                'above', 'below'
            )
            
            # Analisis per nearest level
            level_analysis = df.groupby('nearest_level').agg({
                'Profit': ['count', 'sum', 'mean'],
                'MAE_pips': 'mean',
                'MFE_pips': 'mean'
            }).round(4)
            
            level_analysis.columns = ['_'.join(col).strip() for col in level_analysis.columns]
            
            # Win rates
            level_win_rates = df.groupby('nearest_level').apply(
                lambda x: (x['Profit'] > 0).sum() / len(x) * 100
            ).round(2)
            
            level_analysis['win_rate'] = level_win_rates
            
            print("🎯 Performance by Nearest Level:")
            for level, row in level_analysis.iterrows():
                print(f"   {level.upper()}: Win Rate {row['win_rate']:.1f}%, "
                      f"Avg Profit {row['Profit_mean']:.4f}, "
                      f"Trades {row['Profit_count']:.0f}")
    
    def analyze_trading_sessions(self):
        """Analisis performance per sesi trading"""
        if self.all_data is None:
            print("❌ No data loaded!")
            return
            
        print("\n🌍 TRADING SESSION ANALYSIS")
        print("=" * 40)
        
        df = self.all_data.copy()
        
        # Session columns
        session_cols = ['SessionAsia', 'SessionEurope', 'SessionUS']
        available_sessions = [col for col in session_cols if col in df.columns]
        
        if not available_sessions:
            print("❌ No session data available")
            return
            
        session_stats = {}
        
        for session in available_sessions:
            session_name = session.replace('Session', '')
            
            # Filter trades during this session
            session_trades = df[df[session] == 1]
            
            if len(session_trades) > 0:
                stats = {
                    'total_trades': len(session_trades),
                    'profitable_trades': len(session_trades[session_trades['Profit'] > 0]),
                    'win_rate': len(session_trades[session_trades['Profit'] > 0]) / len(session_trades) * 100,
                    'avg_profit': session_trades['Profit'].mean(),
                    'total_profit': session_trades['Profit'].sum(),
                    'avg_mae': session_trades['MAE_pips'].mean() if 'MAE_pips' in session_trades.columns else 0,
                    'avg_mfe': session_trades['MFE_pips'].mean() if 'MFE_pips' in session_trades.columns else 0
                }
                
                session_stats[session_name] = stats
                
                print(f"🌏 {session_name.upper()} Session:")
                print(f"   📊 Total Trades:     {stats['total_trades']:,}")
                print(f"   📈 Win Rate:         {stats['win_rate']:.2f}%")
                print(f"   💰 Average Profit:   {stats['avg_profit']:.4f}")
                print(f"   💵 Total Profit:     {stats['total_profit']:.2f}")
                print(f"   📉 Average MAE:      {stats['avg_mae']:.2f} pips")
                print(f"   📈 Average MFE:      {stats['avg_mfe']:.2f} pips")
                print()
        
        self.session_stats = session_stats
        
        # Multi-session analysis
        df['active_sessions'] = df[available_sessions].sum(axis=1)
        multi_session_analysis = df.groupby('active_sessions').agg({
            'Profit': ['count', 'mean', 'sum'],
        }).round(4)
        
        multi_session_analysis.columns = ['_'.join(col).strip() for col in multi_session_analysis.columns]
        
        multi_session_win_rates = df.groupby('active_sessions').apply(
            lambda x: (x['Profit'] > 0).sum() / len(x) * 100
        ).round(2)
        
        multi_session_analysis['win_rate'] = multi_session_win_rates
        
        print("🔄 MULTI-SESSION ANALYSIS:")
        print("-" * 30)
        for sessions, row in multi_session_analysis.iterrows():
            session_desc = "No sessions" if sessions == 0 else f"{sessions} session(s)"
            print(f"   {session_desc}: Win Rate {row['win_rate']:.1f}%, "
                  f"Avg Profit {row['Profit_mean']:.4f}, "
                  f"Trades {row['Profit_count']:.0f}")
    
    def analyze_hourly_patterns(self):
        """Analisis pola per jam"""
        if self.all_data is None:
            print("❌ No data loaded!")
            return
            
        print("\n⏰ HOURLY PATTERN ANALYSIS")
        print("=" * 35)
        
        df = self.all_data.copy()
        df['hour'] = df['Timestamp'].dt.hour
        
        hourly_stats = df.groupby('hour').agg({
            'Profit': ['count', 'mean', 'sum'],
            'MAE_pips': 'mean',
            'MFE_pips': 'mean'
        }).round(4)
        
        hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
        
        hourly_win_rates = df.groupby('hour').apply(
            lambda x: (x['Profit'] > 0).sum() / len(x) * 100
        ).round(2)
        
        hourly_stats['win_rate'] = hourly_win_rates
        
        # Sort by profitability
        top_hours = hourly_stats.sort_values('Profit_sum', ascending=False).head(10)
        
        print("🕐 TOP 10 MOST PROFITABLE HOURS:")
        print("-" * 35)
        
        for hour, row in top_hours.iterrows():
            print(f"   {hour:2d}:00 - Win Rate: {row['win_rate']:5.1f}%, "
                  f"Avg Profit: {row['Profit_mean']:7.4f}, "
                  f"Total: {row['Profit_sum']:8.2f}, "
                  f"Trades: {row['Profit_count']:4.0f}")
        
        self.hourly_stats = hourly_stats
    
    def analyze_risk_management(self):
        """Analisis risk management parameters"""
        if self.all_data is None:
            print("❌ No data loaded!")
            return
            
        print("\n⚖️  RISK MANAGEMENT ANALYSIS")
        print("=" * 40)
        
        df = self.all_data.copy()
        
        # TP/SL Analysis
        if 'TP' in df.columns and 'SL' in df.columns:
            df['risk_reward_ratio'] = np.where(df['SL'] > 0, df['TP'] / df['SL'], 0)
            
            # Bin risk-reward ratios
            df['rr_category'] = pd.cut(df['risk_reward_ratio'], 
                                     bins=[0, 1, 2, 3, float('inf')], 
                                     labels=['Low (0-1)', 'Medium (1-2)', 'High (2-3)', 'Very High (3+)'])
            
            rr_analysis = df.groupby('rr_category').agg({
                'Profit': ['count', 'mean', 'sum'],
                'MAE_pips': 'mean',
                'MFE_pips': 'mean'
            }).round(4)
            
            rr_analysis.columns = ['_'.join(col).strip() for col in rr_analysis.columns]
            
            rr_win_rates = df.groupby('rr_category').apply(
                lambda x: (x['Profit'] > 0).sum() / len(x) * 100
            ).round(2)
            
            rr_analysis['win_rate'] = rr_win_rates
            
            print("⚖️  RISK-REWARD RATIO ANALYSIS:")
            print("-" * 35)
            
            for category, row in rr_analysis.iterrows():
                if pd.notna(category):
                    print(f"   {category}: Win Rate {row['win_rate']:5.1f}%, "
                          f"Avg Profit {row['Profit_mean']:7.4f}, "
                          f"Trades {row['Profit_count']:4.0f}")
        
        # AutoTPSL Analysis
        if 'AutoTPSL' in df.columns:
            auto_analysis = df.groupby('AutoTPSL').agg({
                'Profit': ['count', 'mean', 'sum']
            }).round(4)
            
            auto_analysis.columns = ['_'.join(col).strip() for col in auto_analysis.columns]
            
            auto_win_rates = df.groupby('AutoTPSL').apply(
                lambda x: (x['Profit'] > 0).sum() / len(x) * 100
            ).round(2)
            
            auto_analysis['win_rate'] = auto_win_rates
            
            print(f"\n🤖 AUTO TP/SL ANALYSIS:")
            print("-" * 25)
            
            for auto_value, row in auto_analysis.iterrows():
                auto_desc = "Manual" if auto_value == 0 else "Auto"
                print(f"   {auto_desc}: Win Rate {row['win_rate']:5.1f}%, "
                      f"Avg Profit {row['Profit_mean']:7.4f}, "
                      f"Trades {row['Profit_count']:4.0f}")
    
    def generate_comprehensive_report(self, save_to_file=True):
        """Generate laporan komprehensif"""
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 45)
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_trades': len(self.all_data) if self.all_data is not None else 0,
            'fibonacci_stats': self.fibonacci_stats.to_dict() if hasattr(self.fibonacci_stats, 'to_dict') else {},
            'session_stats': self.session_stats,
            'hourly_stats': self.hourly_stats.to_dict() if hasattr(self.hourly_stats, 'to_dict') else {}
        }
        
        if save_to_file:
            # Save as JSON
            report_file = f"data/processed/fibonacci_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"💾 Report saved to: {report_file}")
            
            # Save as CSV for easy reading
            if hasattr(self.fibonacci_stats, 'to_csv'):
                csv_file = f"data/processed/fibonacci_levels_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.fibonacci_stats.to_csv(csv_file)
                print(f"📊 Fibonacci analysis saved to: {csv_file}")
        
        return report
    
    def run_complete_analysis(self, max_files=None):
        """Jalankan analisis lengkap"""
        print("🚀 STARTING COMPREHENSIVE FIBONACCI ANALYSIS")
        print("=" * 60)
        
        # Load all data
        file_stats = self.load_all_data(max_files=max_files)
        
        if self.all_data is None:
            print("❌ No data to analyze!")
            return
        
        # Run all analyses
        self.analyze_fibonacci_levels()
        self.analyze_trading_sessions() 
        self.analyze_hourly_patterns()
        self.analyze_risk_management()
        
        # Generate report
        report = self.generate_comprehensive_report()
        
        print("\n🎉 ANALYSIS COMPLETED!")
        print("=" * 30)
        print("📊 Check the generated reports in data/processed/ folder")
        print("🔍 Use the statistics to optimize your trading strategy")
        
        return report

def main():
    """Main function untuk menjalankan analisis"""
    analyzer = FibonacciAnalyzer()
    
    # Analisis dengan semua file (untuk testing gunakan max_files=50)
    print("🎯 Choose analysis scope:")
    print("1. Quick test (50 files)")
    print("2. Medium test (200 files)")
    print("3. Full analysis (ALL 544 files)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        max_files = 50
    elif choice == '2':
        max_files = 200
    else:
        max_files = None
        
    # Run analysis
    report = analyzer.run_complete_analysis(max_files=max_files)
    
    print(f"\n📋 Analysis Summary:")
    print(f"   🔢 Total trades analyzed: {report['total_trades']:,}")
    print(f"   📊 Fibonacci levels found: {len(report['fibonacci_stats'])}")
    print(f"   🌍 Trading sessions analyzed: {len(report['session_stats'])}")

if __name__ == "__main__":
    main()
