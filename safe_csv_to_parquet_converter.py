#!/usr/bin/env python3
"""
CSV to Parquet Converter with Chunking - No Hang Solution
Converts large CSV tick data to efficient Parquet format
"""

import os
import gc
import csv
from pathlib import Path
import time

class NoHangCSVToParquetConverter:
    """Convert large CSV to Parquet without hanging using chunking"""
    
    def __init__(self, input_folder="datatickxau", output_folder="datatickxau_parquet"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.chunk_size = 100000  # Process 100K rows at a time
        
        # Create output folder
        Path(self.output_folder).mkdir(exist_ok=True)
        
    def analyze_csv_structure(self, csv_file, sample_rows=1000):
        """Analyze CSV structure without loading full file"""
        print(f"[INFO] Analyzing CSV structure: {csv_file}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # Read header
                reader = csv.reader(f)
                header = next(reader)
                
                # Sample first few rows
                sample_data = []
                for i, row in enumerate(reader):
                    if i >= sample_rows:
                        break
                    sample_data.append(row)
                
                print(f"[INFO] Columns found: {len(header)}")
                print(f"[INFO] Header: {header}")
                print(f"[INFO] Sample rows analyzed: {len(sample_data)}")
                
                # Check if it's tick data format
                has_timestamp = any('time' in col.lower() or 'date' in col.lower() for col in header)
                has_price = any('price' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower() for col in header)
                has_volume = any('volume' in col.lower() or 'size' in col.lower() for col in header)
                
                print(f"[INFO] Tick data indicators:")
                print(f"  - Has timestamp: {has_timestamp}")
                print(f"  - Has price: {has_price}")
                print(f"  - Has volume: {has_volume}")
                
                return {
                    'header': header,
                    'sample_data': sample_data[:5],  # First 5 rows
                    'is_tick_data': has_timestamp and has_price,
                    'total_columns': len(header)
                }
                
        except Exception as e:
            print(f"[ERROR] Failed to analyze CSV: {e}")
            return None
    
    def convert_csv_to_parquet_chunked(self, csv_file, max_chunks=None):
        """Convert CSV to Parquet using chunking to avoid memory issues"""
        
        start_time = time.time()
        csv_path = os.path.join(self.input_folder, csv_file)
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            return False
        
        # Get file size
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"[INFO] Processing file: {csv_file} ({file_size_mb:.1f} MB)")
        
        # Analyze structure first
        structure = self.analyze_csv_structure(csv_path)
        if not structure:
            return False
        
        try:
            # Import pandas only when needed and use chunking
            import pandas as pd
            
            parquet_file = csv_file.replace('.csv', '.parquet')
            parquet_path = os.path.join(self.output_folder, parquet_file)
            
            chunk_count = 0
            total_rows = 0
            
            print(f"[INFO] Starting chunked conversion...")
            print(f"[INFO] Chunk size: {self.chunk_size:,} rows")
            
            # Process file in chunks
            chunk_iterator = pd.read_csv(
                csv_path, 
                chunksize=self.chunk_size,
                low_memory=False
            )
            
            first_chunk = True
            
            for chunk_num, chunk in enumerate(chunk_iterator):
                
                # Stop if max_chunks specified
                if max_chunks and chunk_num >= max_chunks:
                    print(f"[INFO] Stopped at chunk {chunk_num} (max_chunks={max_chunks})")
                    break
                
                print(f"[INFO] Processing chunk {chunk_num + 1}: {len(chunk):,} rows")
                
                # Basic data cleaning
                if 'timestamp' in chunk.columns or 'Timestamp' in chunk.columns:
                    timestamp_col = 'timestamp' if 'timestamp' in chunk.columns else 'Timestamp'
                    try:
                        chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col])
                    except:
                        print(f"[WARNING] Could not convert timestamp in chunk {chunk_num + 1}")
                
                # Convert to Parquet (append mode for subsequent chunks)
                if first_chunk:
                    chunk.to_parquet(parquet_path, index=False, compression='snappy')
                    first_chunk = False
                    print(f"[INFO] Created Parquet file: {parquet_path}")
                else:
                    # For subsequent chunks, we need to append
                    # Pandas doesn't support append mode for Parquet directly
                    # So we'll create separate files and mention this
                    chunk_parquet_path = parquet_path.replace('.parquet', f'_chunk_{chunk_num}.parquet')
                    chunk.to_parquet(chunk_parquet_path, index=False, compression='snappy')
                    print(f"[INFO] Created chunk file: {chunk_parquet_path}")
                
                total_rows += len(chunk)
                chunk_count += 1
                
                # Clean memory
                del chunk
                gc.collect()
                
                # Progress update
                elapsed = time.time() - start_time
                print(f"[PROGRESS] Processed {total_rows:,} rows in {elapsed:.1f}s")
            
            # Final summary
            elapsed = time.time() - start_time
            parquet_size_mb = sum(
                os.path.getsize(os.path.join(self.output_folder, f)) / (1024 * 1024)
                for f in os.listdir(self.output_folder)
                if f.startswith(parquet_file.replace('.parquet', ''))
            )
            
            compression_ratio = (file_size_mb / parquet_size_mb) if parquet_size_mb > 0 else 0
            
            print(f"\n[SUCCESS] Conversion completed!")
            print(f"  - Original CSV: {file_size_mb:.1f} MB")
            print(f"  - Parquet files: {parquet_size_mb:.1f} MB")
            print(f"  - Compression ratio: {compression_ratio:.1f}x")
            print(f"  - Total rows: {total_rows:,}")
            print(f"  - Chunks created: {chunk_count}")
            print(f"  - Processing time: {elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return False
    
    def create_tick_features_from_parquet(self, parquet_folder="datatickxau_parquet"):
        """Create ML features from tick data stored in Parquet format"""
        print(f"\n[INFO] Creating ML features from Parquet files...")
        
        try:
            import pandas as pd
            
            parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]
            
            if not parquet_files:
                print(f"[ERROR] No Parquet files found in {parquet_folder}")
                return False
            
            print(f"[INFO] Found {len(parquet_files)} Parquet files")
            
            # Process each parquet file to create features
            all_features = []
            
            for parquet_file in parquet_files[:3]:  # Limit to first 3 files for demo
                print(f"[INFO] Processing {parquet_file}...")
                
                parquet_path = os.path.join(parquet_folder, parquet_file)
                df = pd.read_parquet(parquet_path)
                
                print(f"  - Loaded {len(df):,} rows, {len(df.columns)} columns")
                print(f"  - Columns: {list(df.columns)}")
                
                # Create basic price movement features
                if 'bid' in df.columns and 'ask' in df.columns:
                    df['spread'] = df['ask'] - df['bid']
                    df['mid_price'] = (df['bid'] + df['ask']) / 2
                
                # Time-based features
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['hour'] = df['timestamp'].dt.hour
                    df['minute'] = df['timestamp'].dt.minute
                    df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # Price movement features (using mid_price if available)
                price_col = 'mid_price' if 'mid_price' in df.columns else df.select_dtypes(include=['float64', 'int64']).columns[0]
                
                if price_col in df.columns:
                    # Rolling features (small windows to avoid memory issues)
                    df['price_ma_10'] = df[price_col].rolling(10, min_periods=1).mean()
                    df['price_std_10'] = df[price_col].rolling(10, min_periods=1).std().fillna(0)
                    df['price_change'] = df[price_col].diff().fillna(0)
                    df['price_change_pct'] = df[price_col].pct_change().fillna(0)
                
                # Sample features (take every 100th row to reduce size)
                feature_sample = df.iloc[::100].copy()
                all_features.append(feature_sample)
                
                print(f"  - Created features: {len(feature_sample)} sampled rows")
            
            # Combine all features
            if all_features:
                combined_features = pd.concat(all_features, ignore_index=True)
                
                # Save features
                features_path = "tick_data_features.parquet"
                combined_features.to_parquet(features_path, index=False, compression='snappy')
                
                print(f"\n[SUCCESS] Features created!")
                print(f"  - Total feature rows: {len(combined_features):,}")
                print(f"  - Feature columns: {len(combined_features.columns)}")
                print(f"  - Saved to: {features_path}")
                
                return True
            
        except Exception as e:
            print(f"[ERROR] Feature creation failed: {e}")
            return False
    
    def run_safe_conversion(self, max_chunks=5):
        """Run safe conversion with limited chunks to avoid hanging"""
        print("=" * 60)
        print("[INFO] SAFE CSV-TO-PARQUET CONVERTER")
        print("=" * 60)
        
        # Check input folder
        if not os.path.exists(self.input_folder):
            print(f"[ERROR] Input folder not found: {self.input_folder}")
            return False
        
        # Find CSV files
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"[ERROR] No CSV files found in {self.input_folder}")
            return False
        
        print(f"[INFO] Found {len(csv_files)} CSV files")
        
        # Process first CSV file with limited chunks
        first_csv = csv_files[0]
        print(f"[INFO] Processing first file: {first_csv}")
        
        success = self.convert_csv_to_parquet_chunked(first_csv, max_chunks=max_chunks)
        
        if success:
            print(f"\n[INFO] Conversion successful! Now creating ML features...")
            self.create_tick_features_from_parquet()
        
        return success

def main():
    """Main function - safe execution"""
    converter = NoHangCSVToParquetConverter()
    
    print("Starting safe CSV to Parquet conversion...")
    print("This will process only a small portion to avoid hanging.")
    
    success = converter.run_safe_conversion(max_chunks=3)  # Limit to 3 chunks
    
    if success:
        print("\n[SUCCESS] Conversion completed without hanging!")
        print("Next steps:")
        print("1. Check datatickxau_parquet folder for output")
        print("2. Check tick_data_features.parquet for ML-ready features")
        print("3. Integrate with your Fibonacci analysis")
    else:
        print("\n[ERROR] Conversion failed")

if __name__ == "__main__":
    main()
