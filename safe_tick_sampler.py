#!/usr/bin/env python3
"""
SAFE TICK DATA SAMPLER - No Hang Guaranteed
Samples 1% of 2.7GB tick data = 27MB manageable size
"""

import os
import csv
from datetime import datetime

def safe_tick_sampler():
    """Sample tick data safely without loading full file"""
    
    print("SAFE TICK DATA SAMPLER")
    print("=" * 40)
    print("Starting:", datetime.now().strftime("%H:%M:%S"))
    
    # File info
    tick_folder = "datatickxau"
    csv_files = [f for f in os.listdir(tick_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("ERROR: No CSV files found")
        return False
    
    input_file = os.path.join(tick_folder, csv_files[0])
    output_file = "tick_sample_1_percent.csv"
    
    print(f"Input file: {csv_files[0]}")
    
    # File size
    file_size_mb = os.path.getsize(input_file) / (1024*1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    print(f"Sampling 1% -> Expected output: {file_size_mb/100:.1f} MB")
    print(f"Output file: {output_file}")
    
    # Safe sampling - read every 100th line
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Copy header
            header = next(reader)
            writer.writerow(header)
            print(f"Header: {header}")
            
            line_count = 0
            written_count = 0
            
            # Process in batches to avoid memory issues
            for row in reader:
                line_count += 1
                
                # Take every 100th row (1% sampling)
                if line_count % 100 == 0:
                    writer.writerow(row)
                    written_count += 1
                
                # Progress indicator
                if line_count % 1000000 == 0:  # Every 1M lines
                    print(f"  Processed: {line_count:,} lines, Written: {written_count:,}")
                
                # Safety limit for testing
                if written_count >= 100000:  # Max 100K sampled rows
                    print("  Reached safety limit (100K sampled rows)")
                    break
            
            print(f"COMPLETED!")
            print(f"  Total lines processed: {line_count:,}")
            print(f"  Sample lines written: {written_count:,}")
            print(f"  Sampling rate: {written_count/line_count*100:.2f}%")
            
            # Output file size
            output_size_mb = os.path.getsize(output_file) / (1024*1024)
            print(f"  Output file size: {output_size_mb:.1f} MB")
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nSample file created: {output_file}")
    print("Ready for ML processing!")
    
    return True

def analyze_sample():
    """Quick analysis of the sample"""
    
    sample_file = "tick_sample_1_percent.csv"
    
    if not os.path.exists(sample_file):
        print("Sample file not found")
        return
    
    print("\nQUICK SAMPLE ANALYSIS")
    print("-" * 30)
    
    try:
        with open(sample_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Read first few rows
            sample_rows = []
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                sample_rows.append(row)
            
            print(f"Sample rows: {len(sample_rows)}")
            print("First 3 rows:")
            for i, row in enumerate(sample_rows[:3]):
                print(f"  {i+1}: {row}")
                
    except Exception as e:
        print(f"Error analyzing sample: {e}")

if __name__ == "__main__":
    # Run sampling
    success = safe_tick_sampler()
    
    if success:
        # Quick analysis
        analyze_sample()
        
        print("\n" + "=" * 40)
        print("NEXT STEPS:")
        print("1. Check tick_sample_1_percent.csv")
        print("2. Convert to OHLC with tick_to_ohlc.py")
        print("3. Integrate with Fibonacci signals")
        print("4. Train enhanced ML model")
