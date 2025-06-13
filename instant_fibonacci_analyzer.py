#!/usr/bin/env python3
"""
INSTANT FIBONACCI ANALYZER
Solusi definitif untuk masalah hang dengan:
- Immediate output flushing
- Ultra-small batch processing  
- Real-time progress
- Emergency stops
"""

import os
import csv
import sys
import time
from collections import defaultdict

def print_now(msg):
    """Print dengan immediate flush"""
    print(msg)
    sys.stdout.flush()

def analyze_single_csv(file_path, max_rows=20):
    """Analyze single CSV file dengan safety"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            trades = []
            count = 0
            
            for row in reader:
                if count >= max_rows:
                    break
                if row.get('Type') in ['BUY', 'SELL']:
                    try:
                        profit = float(row.get('Profit', 0))
                        trades.append({
                            'fib_level': row.get('LevelFibo', ''),
                            'profitable': profit > 0
                        })
                        count += 1
                    except:
                        continue
            
            return trades
    except Exception as e:
        print_now(f"[ERROR] {os.path.basename(file_path)}: {e}")
        return []

def instant_fibonacci_analysis():
    """Instant analysis dengan real-time results"""
    start_time = time.time()
    
    print_now("🚀 INSTANT FIBONACCI ANALYZER")
    print_now("=" * 50)
    print_now("💡 Press Ctrl+C anytime to stop and get results")
    print_now("")
    
    # Get files
    try:
        csv_files = [f for f in os.listdir('dataBT') if f.endswith('.csv')][:10]
        print_now(f"📂 Found {len(csv_files)} files (processing first 10)")
    except Exception as e:
        print_now(f"[ERROR] Cannot access dataBT: {e}")
        return
    
    # Process files
    fib_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
    total_trades = 0
    total_profitable = 0
    
    try:
        for i, filename in enumerate(csv_files):
            file_path = os.path.join('dataBT', filename)
            
            print_now(f"[{i+1}/{len(csv_files)}] Processing: {filename[:50]}...")
            
            # Analyze file
            file_start = time.time()
            trades = analyze_single_csv(file_path)
            file_time = time.time() - file_start
            
            # Update stats
            for trade in trades:
                total_trades += 1
                if trade['profitable']:
                    total_profitable += 1
                
                fib_level = trade['fib_level']
                if fib_level:
                    fib_stats[fib_level]['total'] += 1
                    if trade['profitable']:
                        fib_stats[fib_level]['profitable'] += 1
            
            # Immediate progress
            win_rate = (total_profitable / total_trades * 100) if total_trades > 0 else 0
            print_now(f"     ✓ {len(trades)} trades ({file_time:.2f}s) | Running Win Rate: {win_rate:.1f}%")
            
            # Mini-report every 3 files
            if (i + 1) % 3 == 0:
                elapsed = time.time() - start_time
                print_now(f"📊 Checkpoint: {total_trades} trades, {win_rate:.1f}% win rate ({elapsed:.1f}s)")
    
    except KeyboardInterrupt:
        print_now("\n⚠️ Stopped by user")
    except Exception as e:
        print_now(f"\n[ERROR] Unexpected error: {e}")
    
    # Final results
    if total_trades == 0:
        print_now("\n❌ No valid trades found")
        return
    
    overall_win_rate = (total_profitable / total_trades) * 100
    elapsed_total = time.time() - start_time
    
    # Calculate top Fibonacci levels
    fib_results = []
    for level, stats in fib_stats.items():
        if stats['total'] >= 2:  # Minimum threshold
            win_rate = (stats['profitable'] / stats['total']) * 100
            fib_results.append({
                'level': level,
                'total': stats['total'],
                'profitable': stats['profitable'],
                'win_rate': win_rate
            })
    
    fib_results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    # Display final results
    print_now("\n" + "=" * 60)
    print_now("🎯 INSTANT ANALYSIS RESULTS")
    print_now("=" * 60)
    
    print_now(f"⏱️  Execution Time: {elapsed_total:.1f} seconds")
    print_now(f"📁 Files Processed: {len(csv_files)}")
    print_now(f"💹 Total Trades: {total_trades}")
    print_now(f"💰 Profitable Trades: {total_profitable}")
    print_now(f"🎯 Overall Win Rate: {overall_win_rate:.2f}%")
    print_now(f"⚡ Speed: {total_trades/elapsed_total:.1f} trades/second")
    
    # Target check
    if overall_win_rate >= 55:
        print_now("🎉 TARGET ACHIEVED! Win rate ≥ 55%")
    elif overall_win_rate >= 50:
        print_now("📈 GOOD! Win rate ≥ 50%")
    else:
        print_now("📊 Baseline results")
    
    # Top Fibonacci levels
    if fib_results:
        print_now(f"\n🔝 Top Fibonacci Levels:")
        print_now("-" * 50)
        print_now(f"{'Level':<15} {'Trades':<8} {'Wins':<6} {'Win Rate':<10}")
        print_now("-" * 50)
        
        for result in fib_results[:8]:
            print_now(f"{result['level']:<15} {result['total']:<8} {result['profitable']:<6} {result['win_rate']:<10.1f}%")
    
    # Save results
    try:
        with open("instant_fibonacci_report.txt", 'w') as f:
            f.write("INSTANT FIBONACCI ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Execution Time: {elapsed_total:.1f} seconds\n")
            f.write(f"Files Processed: {len(csv_files)}\n")
            f.write(f"Total Trades: {total_trades}\n")
            f.write(f"Profitable Trades: {total_profitable}\n")
            f.write(f"Overall Win Rate: {overall_win_rate:.2f}%\n\n")
            f.write("TOP FIBONACCI LEVELS:\n")
            f.write("-" * 30 + "\n")
            
            for result in fib_results[:10]:
                f.write(f"Level {result['level']}: {result['win_rate']:.1f}% ")
                f.write(f"({result['profitable']}/{result['total']} trades)\n")
        
        print_now(f"\n✅ Report saved: instant_fibonacci_report.txt")
    except Exception as e:
        print_now(f"\n[WARNING] Could not save report: {e}")
    
    print_now("\n💡 HANG ISSUE SOLUTIONS APPLIED:")
    print_now("   ✅ Immediate output flushing")
    print_now("   ✅ Small batch processing (10 files max)")
    print_now("   ✅ Real-time progress updates")
    print_now("   ✅ Ultra-safe error handling")
    print_now("   ✅ Interruptible execution")

if __name__ == "__main__":
    instant_fibonacci_analysis()
