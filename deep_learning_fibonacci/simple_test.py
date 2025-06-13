#!/usr/bin/env python3
"""
Minimal Enhanced Fibonacci Analyzer
Simple enhancement that definitely works with existing setup
"""

import csv
import glob
import os

def simple_enhancement():
    """Simple enhancement analysis"""
    print("🚀 Simple Enhanced Fibonacci Analysis")
    print("Target: Improve 52% to 55%+ win rate")
    print("-" * 40)
    
    # Check data availability
    data_path = "../dataBT"
    csv_files = glob.glob(f"{data_path}/*.csv")
    
    if not csv_files:
        print("❌ No CSV files found in", data_path)
        return False
    
    print(f"✅ Found {len(csv_files)} CSV files")
    
    # Process first few files
    processed_count = 0
    b0_wins = 0
    b0_total = 0
    enhanced_wins = 0
    enhanced_total = 0
    
    for i, file_path in enumerate(csv_files[:10]):  # Process first 10 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for j, row in enumerate(reader):
                    if j >= 20:  # Max 20 rows per file
                        break
                    
                    # Get key data
                    level_fibo = row.get('LevelFibo', '0')
                    session_europe = row.get('SessionEurope', '0')
                    tp = row.get('TP', '1')
                    sl = row.get('SL', '1')
                    result = row.get('Result', '0')
                    
                    try:
                        level_val = float(level_fibo)
                        europe_session = int(session_europe)
                        tp_val = float(tp)
                        sl_val = float(sl)
                        result_val = float(result)
                        
                        won = result_val > 0
                        
                        # Track B_0 level (baseline)
                        if level_val == 0.0:
                            b0_total += 1
                            if won:
                                b0_wins += 1
                        
                        # Enhanced signal: B_0 or B_-1.8 + Europe session + good TP/SL ratio
                        is_enhanced_signal = False
                        
                        if level_val in [0.0, -1.8]:  # Primary levels
                            if europe_session == 1:  # Europe session
                                if sl_val > 0 and tp_val / sl_val >= 1.8:  # Good ratio
                                    is_enhanced_signal = True
                        
                        if is_enhanced_signal:
                            enhanced_total += 1
                            if won:
                                enhanced_wins += 1
                        
                        processed_count += 1
                        
                    except (ValueError, ZeroDivisionError):
                        continue
            
            print(f"   Processed file {i+1}/10")
                
        except Exception as e:
            print(f"❌ Error with {file_path}: {e}")
            continue
    
    print(f"\n✅ Processed {processed_count} trades")
    
    # Calculate results
    if b0_total > 0:
        b0_win_rate = b0_wins / b0_total
        print(f"📊 B_0 Level Win Rate: {b0_win_rate:.1%} ({b0_wins}/{b0_total})")
    
    if enhanced_total > 0:
        enhanced_win_rate = enhanced_wins / enhanced_total
        print(f"🚀 Enhanced Signal Win Rate: {enhanced_win_rate:.1%} ({enhanced_wins}/{enhanced_total})")
        
        baseline = 0.524  # 52.4% baseline
        if enhanced_win_rate > baseline:
            improvement = enhanced_win_rate - baseline
            print(f"✅ Improvement: +{improvement:.3f} ({improvement/baseline:+.1%})")
            
            if enhanced_win_rate >= 0.55:
                print("🎯 TARGET ACHIEVED: 55%+ win rate!")
                return True
            else:
                print("📈 Progress made, continue optimization")
        else:
            print("⚠️ No improvement yet, refine strategy")
    else:
        print("⚠️ No enhanced signals found in sample")
    
    print("\n💡 Enhancement Strategy Applied:")
    print("   - Focus on B_0 and B_-1.8 levels (proven winners)")
    print("   - Prioritize Europe trading session")
    print("   - Require good TP/SL ratios (≥1.8)")
    
    return False

if __name__ == "__main__":
    success = simple_enhancement()
    if success:
        print("\n🎉 Ready for live trading!")
    else:
        print("\n🔧 Continue with deeper analysis")
