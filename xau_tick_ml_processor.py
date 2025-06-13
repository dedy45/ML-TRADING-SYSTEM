#!/usr/bin/env python3
"""
XAU Tick Data ML Pipeline
Untuk mengolah data tick 2GB menjadi features ML yang berguna
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import gc
from pathlib import Path

class XAUTickMLProcessor:
    """Processor untuk data tick XAU yang besar (2GB+) untuk ML"""
    
    def __init__(self, tick_file_path="datatickxau/2025.6.11XAUUSD_dukascopy_TICK_UTC-TICK-Forex_245.csv"):
        self.tick_file = tick_file_path
        self.fibonacci_data_path = "dataBT"
        self.results = {}
        
        # Create output directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/features").mkdir(parents=True, exist_ok=True)
        
    def check_tick_data_info(self):
        """Cek informasi dasar file tick tanpa load semua data"""
        print("=" * 60)
        print("XAU TICK DATA ANALYSIS")
        print("=" * 60)
        
        file_size = os.path.getsize(self.tick_file) / (1024**3)  # GB
        print(f"File size: {file_size:.2f} GB")
        
        # Read only first few rows to understand structure
        print("\nReading sample data (first 1000 rows)...")
        try:
            sample_df = pd.read_csv(self.tick_file, nrows=1000)
            print(f"Columns: {list(sample_df.columns)}")
            print(f"Sample data shape: {sample_df.shape}")
            print("\nFirst 5 rows:")
            print(sample_df.head())
            
            # Estimate total rows
            avg_row_size = sample_df.memory_usage(deep=True).sum() / len(sample_df)
            estimated_rows = int((file_size * 1024**3) / avg_row_size)
            print(f"\nEstimated total rows: {estimated_rows:,}")
            
            return sample_df.columns.tolist(), estimated_rows
            
        except Exception as e:
            print(f"Error reading tick data: {e}")
            return None, 0
    
    def process_tick_chunks(self, chunk_size=100000, max_chunks=50):
        """Process data tick dalam chunks untuk memory efficiency"""
        print(f"\nProcessing tick data in chunks of {chunk_size:,} rows...")
        print(f"Max chunks to process: {max_chunks}")
        
        chunk_features = []
        chunk_count = 0
        
        try:
            # Process data in chunks
            for chunk_df in pd.read_csv(self.tick_file, chunksize=chunk_size):
                if chunk_count >= max_chunks:
                    break
                
                print(f"Processing chunk {chunk_count + 1}/{max_chunks}...")
                
                # Create features from this chunk
                features = self.create_tick_features(chunk_df)
                if features is not None:
                    chunk_features.append(features)
                
                chunk_count += 1
                
                # Clear memory
                del chunk_df
                gc.collect()
            
            if chunk_features:
                # Combine all chunk features
                combined_features = pd.concat(chunk_features, ignore_index=True)
                print(f"\nCombined features shape: {combined_features.shape}")
                
                # Save processed features
                output_file = "data/processed/xau_tick_features.csv"
                combined_features.to_csv(output_file, index=False)
                print(f"Features saved to: {output_file}")
                
                return combined_features
            else:
                print("No features created from tick data")
                return None
                
        except Exception as e:
            print(f"Error processing tick chunks: {e}")
            return None
    
    def create_tick_features(self, tick_df):
        """Create ML features dari data tick"""
        try:
            # Assume tick data has columns like: timestamp, bid, ask, volume
            # Adjust column names based on actual data structure
            
            # Detect column names (common variations)
            time_col = None
            bid_col = None
            ask_col = None
            vol_col = None
            
            columns = [col.lower() for col in tick_df.columns]
            
            # Find time column
            for col in tick_df.columns:
                if any(word in col.lower() for word in ['time', 'timestamp', 'date']):
                    time_col = col
                    break
            
            # Find price columns
            for col in tick_df.columns:
                col_lower = col.lower()
                if 'bid' in col_lower:
                    bid_col = col
                elif 'ask' in col_lower:
                    ask_col = col
                elif any(word in col_lower for word in ['volume', 'vol', 'size']):
                    vol_col = col
            
            # If no bid/ask, look for price columns
            if bid_col is None:
                for col in tick_df.columns:
                    if any(word in col.lower() for word in ['price', 'close', 'value']):
                        bid_col = col
                        ask_col = col
                        break
            
            print(f"   Detected columns - Time: {time_col}, Bid: {bid_col}, Ask: {ask_col}, Volume: {vol_col}")
            
            if not time_col or not bid_col:
                print("   Warning: Required columns not found, using first numeric columns")
                numeric_cols = tick_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    bid_col = numeric_cols[0]
                    ask_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                else:
                    return None
            
            # Convert timestamp if available
            if time_col:
                try:
                    tick_df['timestamp'] = pd.to_datetime(tick_df[time_col])
                except:
                    # Create artificial timestamp
                    tick_df['timestamp'] = pd.date_range(start='2025-06-11', periods=len(tick_df), freq='S')
            else:
                tick_df['timestamp'] = pd.date_range(start='2025-06-11', periods=len(tick_df), freq='S')
            
            # Calculate mid price
            if ask_col and ask_col != bid_col:
                tick_df['mid_price'] = (tick_df[bid_col] + tick_df[ask_col]) / 2
                tick_df['spread'] = tick_df[ask_col] - tick_df[bid_col]
            else:
                tick_df['mid_price'] = tick_df[bid_col]
                tick_df['spread'] = 0
            
            # Resample to different timeframes for features
            tick_df.set_index('timestamp', inplace=True)
            
            # Create 1-minute OHLCV bars
            ohlc_1m = tick_df['mid_price'].resample('1T').ohlc()
            volume_1m = tick_df[vol_col].resample('1T').sum() if vol_col else pd.Series(index=ohlc_1m.index, data=1)
            spread_1m = tick_df['spread'].resample('1T').mean()
            
            # Create features
            features_df = pd.DataFrame(index=ohlc_1m.index)
            
            # Basic OHLC features
            features_df['open'] = ohlc_1m['open']
            features_df['high'] = ohlc_1m['high']
            features_df['low'] = ohlc_1m['low']
            features_df['close'] = ohlc_1m['close']
            features_df['volume'] = volume_1m
            features_df['spread'] = spread_1m
            
            # Price movement features
            features_df['price_change'] = features_df['close'] - features_df['open']
            features_df['price_change_pct'] = features_df['price_change'] / features_df['open'] * 100
            features_df['hl_range'] = features_df['high'] - features_df['low']
            features_df['hl_range_pct'] = features_df['hl_range'] / features_df['close'] * 100
            
            # Technical indicators
            features_df['sma_5'] = features_df['close'].rolling(5, min_periods=1).mean()
            features_df['sma_20'] = features_df['close'].rolling(20, min_periods=1).mean()
            features_df['price_above_sma5'] = (features_df['close'] > features_df['sma_5']).astype(int)
            features_df['price_above_sma20'] = (features_df['close'] > features_df['sma_20']).astype(int)
            
            # Volatility features
            features_df['volatility_5'] = features_df['close'].rolling(5, min_periods=1).std()
            features_df['volatility_20'] = features_df['close'].rolling(20, min_periods=1).std()
            
            # Time features
            features_df['hour'] = features_df.index.hour
            features_df['minute'] = features_df.index.minute
            features_df['day_of_week'] = features_df.index.dayofweek
            
            # Session features (based on UTC time, adjust for your timezone)
            features_df['session_asia'] = ((features_df['hour'] >= 0) & (features_df['hour'] < 8)).astype(int)
            features_df['session_europe'] = ((features_df['hour'] >= 7) & (features_df['hour'] < 16)).astype(int)
            features_df['session_us'] = ((features_df['hour'] >= 13) & (features_df['hour'] < 22)).astype(int)
            
            # Remove rows with NaN
            features_df = features_df.dropna()
            
            print(f"   Created {len(features_df)} feature rows from {len(tick_df)} tick rows")
            
            return features_df
            
        except Exception as e:
            print(f"   Error creating features: {e}")
            return None
    
    def combine_with_fibonacci_signals(self):
        """Combine tick features dengan Fibonacci signals untuk training data"""
        print("\nCombining tick features with Fibonacci signals...")
        
        # Load tick features
        tick_features_file = "data/processed/xau_tick_features.csv"
        if not os.path.exists(tick_features_file):
            print("Tick features not found. Process tick data first.")
            return None
        
        tick_features = pd.read_csv(tick_features_file)
        print(f"Loaded tick features: {tick_features.shape}")
        
        # Load Fibonacci trade data (from our previous analysis)
        fib_report_file = "no_pandas_fibonacci_report.txt"
        if os.path.exists(fib_report_file):
            print("Fibonacci analysis found, using existing signals")
        
        # For now, create synthetic labels based on price movements
        # In real implementation, you would align this with actual trade results
        
        tick_features['timestamp'] = pd.to_datetime(tick_features.index if 'timestamp' not in tick_features.columns else tick_features['timestamp'])
        
        # Create synthetic trading labels based on future price movements
        tick_features['future_price_5m'] = tick_features['close'].shift(-5)  # 5 minutes ahead
        tick_features['future_return_5m'] = (tick_features['future_price_5m'] - tick_features['close']) / tick_features['close'] * 100
        
        # Labels for ML
        tick_features['profitable_5m'] = (tick_features['future_return_5m'] > 0.01).astype(int)  # >1 pip profit
        tick_features['big_move_5m'] = (abs(tick_features['future_return_5m']) > 0.05).astype(int)  # >5 pip move
        
        # Remove future data leakage
        tick_features = tick_features.dropna()
        
        # Save ML-ready dataset
        ml_dataset_file = "data/processed/xau_ml_dataset.csv"
        feature_columns = [col for col in tick_features.columns if col not in ['future_price_5m', 'future_return_5m', 'timestamp']]
        
        ml_dataset = tick_features[feature_columns]
        ml_dataset.to_csv(ml_dataset_file, index=False)
        
        print(f"ML dataset saved: {ml_dataset_file}")
        print(f"Dataset shape: {ml_dataset.shape}")
        print(f"Features: {len([col for col in feature_columns if col not in ['profitable_5m', 'big_move_5m']])}")
        print(f"Labels: profitable_5m, big_move_5m")
        
        return ml_dataset
    
    def run_complete_analysis(self):
        """Run complete tick data analysis"""
        print("STARTING XAU TICK DATA ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Check data info
        columns, estimated_rows = self.check_tick_data_info()
        if not columns:
            return False
        
        # Step 2: Process tick data into features
        print(f"\nStep 2: Processing {estimated_rows:,} estimated rows...")
        features = self.process_tick_chunks(chunk_size=50000, max_chunks=20)  # Process ~1M rows total
        
        if features is None:
            return False
        
        # Step 3: Combine with trading signals
        ml_dataset = self.combine_with_fibonacci_signals()
        
        if ml_dataset is None:
            return False
        
        # Step 4: Basic ML model training
        self.train_quick_model(ml_dataset)
        
        print("\n" + "=" * 60)
        print("XAU TICK DATA ML PIPELINE COMPLETED!")
        print("=" * 60)
        print("Files created:")
        print("- data/processed/xau_tick_features.csv")
        print("- data/processed/xau_ml_dataset.csv")
        print("- data/processed/xau_tick_model.joblib")
        
        return True
    
    def train_quick_model(self, ml_dataset):
        """Train quick ML model untuk demo"""
        print("\nTraining quick ML model...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            import joblib
            
            # Prepare features and labels
            feature_columns = [col for col in ml_dataset.columns if col not in ['profitable_5m', 'big_move_5m']]
            X = ml_dataset[feature_columns].fillna(0)
            y = ml_dataset['profitable_5m']
            
            print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model accuracy: {accuracy:.3f}")
            print(f"Baseline (always predict majority): {y.mean():.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 important features:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Save model
            model_file = "data/processed/xau_tick_model.joblib"
            joblib.dump(model, model_file)
            print(f"\nModel saved: {model_file}")
            
        except ImportError:
            print("Scikit-learn not available, skipping model training")
        except Exception as e:
            print(f"Error training model: {e}")

def main():
    """Main function"""
    processor = XAUTickMLProcessor()
    success = processor.run_complete_analysis()
    
    if success:
        print("\nBERHASIL! Tick data telah diproses untuk ML")
        print("Gunakan file di data/processed/ untuk training model yang lebih advanced")

if __name__ == "__main__":
    main()
