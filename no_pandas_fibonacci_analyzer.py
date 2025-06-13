#!/usr/bin/env python3
"""
NO-PANDAS Fibonacci Analyzer
Uses only built-in Python modules for maximum compatibility
"""

import csv
import glob
import os
from collections import defaultdict, Counter

class NoPandasFibonacciAnalyzer:
    """Fibonacci analyzer using only built-in Python modules"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.results = {}
        
    def read_csv_file(self, file_path, max_rows=100):
        """Read CSV file using built-in csv module"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = []
                count = 0
                
                for row in reader:
                    if count >= max_rows:
                        break
                    rows.append(row)
                    count += 1
                
                return rows
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            return []
    
    def analyze_fibonacci_levels(self, max_files=10, max_rows_per_file=50):
        """Analyze Fibonacci levels without pandas"""
        print(f"[INFO] NO-PANDAS Fibonacci Analysis")
        print(f"[INFO] Processing max {max_files} files, {max_rows_per_file} rows each")
        print("-" * 60)
        
        # Get CSV files
        csv_files = glob.glob(f"{self.data_path}/*.csv")
        if not csv_files:
            print("[ERROR] No CSV files found")
            return None
        
        print(f"[INFO] Found {len(csv_files)} CSV files")
        
        # Limit files for testing
        sample_files = csv_files[:max_files]
        
        # Statistics tracking
        fibonacci_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
        session_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
        total_trades = 0
        total_profitable = 0
        
        for i, file_path in enumerate(sample_files):
            print(f"[INFO] Processing file {i+1}/{len(sample_files)}: {os.path.basename(file_path)}")
            
            # Read CSV data
            rows = self.read_csv_file(file_path, max_rows_per_file)
            
            if not rows:
                print(f"[WARN] No data read from file")
                continue
            
            valid_trades = 0
            
            for row in rows:
                # Check if it's a valid trade
                trade_type = row.get('Type', '')
                if trade_type not in ['BUY', 'SELL']:
                    continue
                
                valid_trades += 1
                total_trades += 1
                
                # Check profitability
                try:
                    profit = float(row.get('Profit', 0))
                    is_profitable = profit > 0
                    if is_profitable:
                        total_profitable += 1
                except ValueError:
                    continue
                
                # Fibonacci level analysis
                fib_level = row.get('LevelFibo', '')
                if fib_level:
                    fibonacci_stats[fib_level]['total'] += 1
                    if is_profitable:
                        fibonacci_stats[fib_level]['profitable'] += 1
                
                # Session analysis
                sessions = ['SessionAsia', 'SessionEurope', 'SessionUS']
                for session in sessions:
                    if row.get(session, '0') == '1':
                        session_stats[session]['total'] += 1
                        if is_profitable:
                            session_stats[session]['profitable'] += 1
            
            print(f"[INFO] Processed {valid_trades} valid trades from file")
        
        if total_trades == 0:
            print("[ERROR] No valid trades found")
            return None
        
        # Calculate win rates
        overall_win_rate = (total_profitable / total_trades) * 100
        
        # Calculate Fibonacci level win rates
        fib_results = []
        for level, stats in fibonacci_stats.items():
            if stats['total'] >= 3:  # Minimum trades for reliability
                win_rate = (stats['profitable'] / stats['total']) * 100
                fib_results.append({
                    'level': level,
                    'total': stats['total'],
                    'profitable': stats['profitable'],
                    'win_rate': win_rate
                })
        
        # Sort by win rate
        fib_results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # Display results
        print("\n" + "=" * 60)
        print("[RESULTS] FIBONACCI ANALYSIS (NO-PANDAS)")
        print("=" * 60)
        
        print(f"[INFO] Overall Statistics:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Profitable Trades: {total_profitable}")
        print(f"  Overall Win Rate: {overall_win_rate:.2f}%")
        print(f"  Files Processed: {len(sample_files)}")
        
        print(f"\n[INFO] Top Fibonacci Levels by Win Rate:")
        print("-" * 50)
        print(f"{'Level':<15} {'Trades':<8} {'Wins':<6} {'Win Rate':<10}")
        print("-" * 50)
        
        for result in fib_results[:10]:  # Top 10
            print(f"{result['level']:<15} {result['total']:<8} {result['profitable']:<6} {result['win_rate']:<10.1f}%")
        
        # Session analysis
        print(f"\n[INFO] Session Performance:")
        print("-" * 30)
        for session, stats in session_stats.items():
            if stats['total'] > 0:
                session_win_rate = (stats['profitable'] / stats['total']) * 100
                print(f"  {session}: {session_win_rate:.1f}% ({stats['profitable']}/{stats['total']} trades)")
        
        # Store results
        self.results = {
            'fibonacci_results': fib_results,
            'session_stats': dict(session_stats),
            'total_trades': total_trades,
            'total_profitable': total_profitable,
            'overall_win_rate': overall_win_rate,
            'files_processed': len(sample_files)
        }
        
        return self.results
    
    def save_results(self, filename="no_pandas_fibonacci_report.txt"):
        """Save results to text file"""
        if not self.results:
            print("[WARNING] No results to save")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("NO-PANDAS FIBONACCI ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Trades: {self.results['total_trades']}\n")
                f.write(f"Profitable Trades: {self.results['total_profitable']}\n")
                f.write(f"Overall Win Rate: {self.results['overall_win_rate']:.2f}%\n")
                f.write(f"Files Processed: {self.results['files_processed']}\n\n")
                
                f.write("FIBONACCI LEVEL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                
                for result in self.results['fibonacci_results']:
                    f.write(f"Level {result['level']}: {result['win_rate']:.1f}% ")
                    f.write(f"({result['profitable']}/{result['total']} trades)\n")
                
                f.write("\nSESSION PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                
                for session, stats in self.results['session_stats'].items():
                    if stats['total'] > 0:
                        win_rate = (stats['profitable'] / stats['total']) * 100
                        f.write(f"{session}: {win_rate:.1f}% ({stats['profitable']}/{stats['total']} trades)\n")
            
            print(f"[INFO] Report saved to: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
    
    def quick_data_summary(self):
        """Quick summary of available data"""
        print("[INFO] Quick Data Summary")
        print("-" * 25)
        
        csv_files = glob.glob(f"{self.data_path}/*.csv")
        print(f"[INFO] Total CSV files: {len(csv_files)}")
        
        if csv_files:
            # Check first file structure
            sample_file = csv_files[0]
            rows = self.read_csv_file(sample_file, 5)
            
            if rows:
                print(f"[INFO] Sample file: {os.path.basename(sample_file)}")
                print(f"[INFO] Available columns: {list(rows[0].keys())}")
                
                # Check for key columns
                key_columns = ['LevelFibo', 'Level1Above', 'Level1Below', 'SessionAsia', 'SessionEurope', 'SessionUS']
                available_key_cols = [col for col in key_columns if col in rows[0]]
                print(f"[INFO] Key columns available: {available_key_cols}")

def main():
    """Main function"""
    print("=" * 60)
    print("[INFO] NO-PANDAS FIBONACCI ANALYZER")
    print("=" * 60)
    
    analyzer = NoPandasFibonacciAnalyzer()
    
    # Quick data summary
    analyzer.quick_data_summary()
    
    print("\n" + "=" * 60)
    
    # Main analysis
    results = analyzer.analyze_fibonacci_levels(max_files=10, max_rows_per_file=100)
    
    if results:
        # Save results
        analyzer.save_results()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] NO-PANDAS analysis completed successfully!")
        print("[INFO] Check 'no_pandas_fibonacci_report.txt' for detailed results")
        print("=" * 60)
    else:
        print("[ERROR] Analysis failed")

if __name__ == "__main__":
    main()
