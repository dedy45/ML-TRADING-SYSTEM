#!/usr/bin/env python3
"""
SAFE TIMEOUT Fibonacci Analyzer
Dengan timeout, monitoring, dan error handling yang lengkap
"""

import csv
import glob
import os
import time
import signal
import sys
from collections import defaultdict, Counter
from contextlib import contextmanager

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

@contextmanager
def timeout_handler(seconds):
    """Context manager untuk timeout"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operasi timeout setelah {seconds} detik")
    
    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class SafeTimeoutFibonacciAnalyzer:
    """Fibonacci analyzer dengan timeout dan monitoring"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.results = {}
        self.start_time = None
        self.processed_files = 0
        self.processed_rows = 0
        
    def log_progress(self, message):
        """Log dengan timestamp"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[{elapsed:.1f}s] {message}")
    
    def read_csv_file_safe(self, file_path, max_rows=50, timeout_seconds=30):
        """Read CSV file dengan timeout dan safety checks"""
        self.log_progress(f"Membaca file: {os.path.basename(file_path)}")
        
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                self.log_progress(f"[SKIP] File terlalu besar: {file_size/1024/1024:.1f}MB")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = []
                count = 0
                
                start_read = time.time()
                
                for row in reader:
                    # Check timeout setiap 100 rows
                    if count % 100 == 0:
                        elapsed = time.time() - start_read
                        if elapsed > timeout_seconds:
                            self.log_progress(f"[TIMEOUT] Baca file timeout setelah {elapsed:.1f}s")
                            break
                    
                    if count >= max_rows:
                        break
                    
                    rows.append(row)
                    count += 1
                    self.processed_rows += 1
                
                elapsed = time.time() - start_read
                self.log_progress(f"Berhasil baca {len(rows)} rows dalam {elapsed:.2f}s")
                return rows
                
        except Exception as e:
            self.log_progress(f"[ERROR] Gagal baca {file_path}: {e}")
            return []
    
    def analyze_fibonacci_levels_safe(self, max_files=5, max_rows_per_file=30, total_timeout=300):
        """Analyze dengan timeout keseluruhan dan monitoring"""
        self.start_time = time.time()
        
        self.log_progress("🚀 MEMULAI SAFE FIBONACCI ANALYSIS")
        self.log_progress(f"Max files: {max_files}, Max rows per file: {max_rows_per_file}")
        self.log_progress(f"Total timeout: {total_timeout} detik")
        print("-" * 60)
        
        try:
            # Get CSV files
            self.log_progress("Mencari file CSV...")
            csv_files = glob.glob(f"{self.data_path}/*.csv")
            
            if not csv_files:
                self.log_progress("[ERROR] Tidak ada file CSV ditemukan")
                return None
            
            self.log_progress(f"Ditemukan {len(csv_files)} file CSV")
            
            # Limit files untuk safety
            sample_files = csv_files[:max_files]
            self.log_progress(f"Akan proses {len(sample_files)} file")
            
            # Statistics tracking
            fibonacci_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
            session_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
            total_trades = 0
            total_profitable = 0
            
            for i, file_path in enumerate(sample_files):
                # Check total timeout
                elapsed_total = time.time() - self.start_time
                if elapsed_total > total_timeout:
                    self.log_progress(f"[TIMEOUT] Total timeout tercapai: {elapsed_total:.1f}s")
                    break
                
                self.log_progress(f"📁 File {i+1}/{len(sample_files)}: {os.path.basename(file_path)}")
                
                # Read CSV dengan timeout per file
                rows = self.read_csv_file_safe(file_path, max_rows_per_file, timeout_seconds=60)
                
                if not rows:
                    self.log_progress(f"[SKIP] Tidak ada data dari file")
                    continue
                
                valid_trades = 0
                file_start = time.time()
                
                for row_idx, row in enumerate(rows):
                    # Progress monitoring setiap 10 rows
                    if row_idx % 10 == 0 and row_idx > 0:
                        file_elapsed = time.time() - file_start
                        self.log_progress(f"  Progress: {row_idx}/{len(rows)} rows ({file_elapsed:.1f}s)")
                    
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
                    except (ValueError, TypeError):
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
                
                file_elapsed = time.time() - file_start
                self.log_progress(f"✅ File selesai: {valid_trades} trades valid dalam {file_elapsed:.1f}s")
                self.processed_files += 1
            
            if total_trades == 0:
                self.log_progress("[ERROR] Tidak ada trade valid ditemukan")
                return None
            
            # Calculate results
            overall_win_rate = (total_profitable / total_trades) * 100
            
            # Calculate Fibonacci level win rates
            fib_results = []
            for level, stats in fibonacci_stats.items():
                if stats['total'] >= 2:  # Minimum trades untuk reliability
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
            total_elapsed = time.time() - self.start_time
            print("\n" + "=" * 60)
            self.log_progress("📊 HASIL ANALISIS FIBONACCI")
            print("=" * 60)
            
            print(f"⏱️  Total waktu eksekusi: {total_elapsed:.1f} detik")
            print(f"📁 File diproses: {self.processed_files}")
            print(f"📄 Total rows diproses: {self.processed_rows}")
            print(f"💹 Total trades: {total_trades}")
            print(f"💰 Profitable trades: {total_profitable}")
            print(f"🎯 Win rate keseluruhan: {overall_win_rate:.2f}%")
            
            if fib_results:
                print(f"\n🔝 Top Fibonacci Levels:")
                print("-" * 50)
                print(f"{'Level':<15} {'Trades':<8} {'Wins':<6} {'Win Rate':<10}")
                print("-" * 50)
                
                for result in fib_results[:8]:  # Top 8
                    print(f"{result['level']:<15} {result['total']:<8} {result['profitable']:<6} {result['win_rate']:<10.1f}%")
            
            # Session analysis
            if session_stats:
                print(f"\n📅 Performa per Sesi:")
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
                'files_processed': self.processed_files,
                'execution_time': total_elapsed,
                'rows_processed': self.processed_rows
            }
            
            return self.results
            
        except KeyboardInterrupt:
            self.log_progress("[INTERRUPT] Analisis dihentikan oleh user")
            return None
        except Exception as e:
            self.log_progress(f"[ERROR] Error tidak terduga: {e}")
            return None
    
    def save_results_safe(self, filename="safe_fibonacci_report.txt"):
        """Save results dengan error handling"""
        if not self.results:
            self.log_progress("[WARNING] Tidak ada hasil untuk disimpan")
            return False
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("SAFE TIMEOUT FIBONACCI ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Execution Time: {self.results['execution_time']:.1f} seconds\n")
                f.write(f"Files Processed: {self.results['files_processed']}\n")
                f.write(f"Rows Processed: {self.results['rows_processed']}\n")
                f.write(f"Total Trades: {self.results['total_trades']}\n")
                f.write(f"Profitable Trades: {self.results['total_profitable']}\n")
                f.write(f"Overall Win Rate: {self.results['overall_win_rate']:.2f}%\n\n")
                
                f.write("FIBONACCI LEVEL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                
                for result in self.results['fibonacci_results']:
                    f.write(f"Level {result['level']}: {result['win_rate']:.1f}% ")
                    f.write(f"({result['profitable']}/{result['total']} trades)\n")
                
                if self.results['session_stats']:
                    f.write("\nSESSION PERFORMANCE:\n")
                    f.write("-" * 20 + "\n")
                    
                    for session, stats in self.results['session_stats'].items():
                        if stats['total'] > 0:
                            win_rate = (stats['profitable'] / stats['total']) * 100
                            f.write(f"{session}: {win_rate:.1f}% ({stats['profitable']}/{stats['total']} trades)\n")
            
            self.log_progress(f"✅ Report berhasil disimpan: {filename}")
            return True
            
        except Exception as e:
            self.log_progress(f"[ERROR] Gagal simpan report: {e}")
            return False
    
    def quick_data_check(self):
        """Quick check data dengan timeout"""
        self.log_progress("🔍 Quick Data Check")
        print("-" * 25)
        
        try:
            csv_files = glob.glob(f"{self.data_path}/*.csv")
            self.log_progress(f"Total file CSV: {len(csv_files)}")
            
            if csv_files:
                # Check first file
                sample_file = csv_files[0]
                file_size = os.path.getsize(sample_file)
                self.log_progress(f"Sample file: {os.path.basename(sample_file)} ({file_size/1024:.1f}KB)")
                
                # Quick read dengan limit sangat kecil
                rows = self.read_csv_file_safe(sample_file, max_rows=3, timeout_seconds=10)
                
                if rows:
                    self.log_progress(f"Kolom tersedia: {list(rows[0].keys())}")
                    
                    # Check key columns
                    key_columns = ['LevelFibo', 'Type', 'Profit', 'SessionAsia', 'SessionEurope', 'SessionUS']
                    available_key_cols = [col for col in key_columns if col in rows[0]]
                    self.log_progress(f"Key columns: {available_key_cols}")
                    
                    return True
                else:
                    self.log_progress("[ERROR] Tidak bisa baca sample file")
                    return False
            else:
                self.log_progress("[ERROR] Tidak ada file CSV")
                return False
                
        except Exception as e:
            self.log_progress(f"[ERROR] Quick check failed: {e}")
            return False

def main():
    """Main function dengan error handling lengkap"""
    print("=" * 60)
    print("🛡️  SAFE TIMEOUT FIBONACCI ANALYZER")
    print("=" * 60)
    
    analyzer = SafeTimeoutFibonacciAnalyzer()
    
    # Quick data check dulu
    if not analyzer.quick_data_check():
        print("\n[ERROR] Quick data check gagal. Periksa path data atau file CSV.")
        return
    
    print("\n" + "=" * 60)
    print("🚀 MEMULAI ANALISIS UTAMA")
    print("💡 Tekan Ctrl+C untuk membatalkan jika terlalu lama")
    print("=" * 60)
    
    try:
        # Main analysis dengan parameter conservative
        results = analyzer.analyze_fibonacci_levels_safe(
            max_files=3,           # Sangat sedikit untuk test
            max_rows_per_file=20,  # Rows sedikit per file  
            total_timeout=120      # 2 menit max
        )
        
        if results:
            # Save results
            if analyzer.save_results_safe():
                print("\n" + "=" * 60)
                print("✅ ANALISIS BERHASIL DISELESAIKAN!")
                print("📄 Cek file 'safe_fibonacci_report.txt' untuk detail")
                print(f"⏱️  Total waktu: {results['execution_time']:.1f} detik")
                print(f"🎯 Win rate: {results['overall_win_rate']:.2f}%")
                print("=" * 60)
            else:
                print("[ERROR] Gagal simpan hasil")
        else:
            print("[ERROR] Analisis gagal atau tidak ada hasil")
    
    except KeyboardInterrupt:
        print("\n[INFO] Analisis dibatalkan oleh user")
    except Exception as e:
        print(f"\n[ERROR] Error tidak terduga: {e}")

if __name__ == "__main__":
    main()
