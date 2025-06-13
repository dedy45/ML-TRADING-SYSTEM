#!/usr/bin/env python3
"""
TensorFlow Deep Learning Fibonacci Signal Generator - Wrapper
Enhanced version dengan MLFLOW infrastructure integration
Menghasilkan signal prediksi trading yang akurat untuk EA MQL5
"""

import sys
import time
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import enhanced version
try:
    from enhanced_tensorflow_fibonacci_predictor import EnhancedFibonacciDeepLearningPredictor
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced predictor loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced predictor not available: {e}")
    print("⚠️ Falling back to basic implementation")
    ENHANCED_AVAILABLE = False

# Fallback imports if enhanced version not available
if not ENHANCED_AVAILABLE:
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

class FibonacciDeepLearningPredictor:
    """
    Wrapper class for backwards compatibility
    Routes to enhanced version when available
    """
    
    def __init__(self, data_path="../dataBT", model_save_path="models/"):
        if ENHANCED_AVAILABLE:
            self.predictor = EnhancedFibonacciDeepLearningPredictor(
                data_path=data_path,
                model_save_path=model_save_path
            )
            print("🧠 Using Enhanced Fibonacci Deep Learning Predictor")
        else:
            # Basic fallback implementation
            self.data_path = Path(data_path)
            self.model_save_path = Path(model_save_path)
            self.model_save_path.mkdir(exist_ok=True)
            self.ensemble_model = None
            self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
            print("🧠 Using Basic Fibonacci Deep Learning Predictor (fallback)")
    
    def load_and_prepare_data(self, max_files=50, max_rows_per_file=100):
        """Load data dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.load_and_prepare_data(max_files, max_rows_per_file)
        else:
            # Basic fallback implementation
            print(f"📂 Loading data from {self.data_path}")
            csv_files = list(self.data_path.glob("*.csv"))[:max_files]
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {self.data_path}")
            
            all_data = []
            for i, file_path in enumerate(csv_files[:5]):  # Limit to 5 files for fallback
                try:
                    df = pd.read_csv(file_path, nrows=max_rows_per_file)
                    df['file_index'] = i
                    all_data.append(df)
                except Exception as e:
                    print(f"   ❌ Error loading {file_path.name}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No data loaded")
            
            df = pd.concat(all_data, ignore_index=True)
            print(f"✅ Loaded {len(df)} records (fallback mode)")
            return df
    
    def engineer_features(self, df):
        """Feature engineering dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.engineer_features(df)
        else:
            # Basic feature engineering
            features = {}
            
            # Convert key columns to numeric
            numeric_cols = ['LevelFibo', 'TP', 'SL', 'Profit']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Basic fibonacci features
            if 'LevelFibo' in df.columns:
                features['fib_level'] = df['LevelFibo']
                features['fib_is_zero'] = (df['LevelFibo'] == 0.0).astype(int)
                features['fib_abs'] = np.abs(df['LevelFibo'])
            
            # Basic ratio features
            if 'TP' in df.columns and 'SL' in df.columns:
                tp_sl_ratio = np.where(df['SL'] != 0, df['TP'] / df['SL'], 0)
                features['tp_sl_ratio'] = tp_sl_ratio
            
            feature_df = pd.DataFrame(features).fillna(0)
            print(f"✅ Created {len(features)} basic features")
            return feature_df
    
    def create_target_variable(self, df):
        """Create target variable dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.create_target_variable(df)
        else:
            # Basic target creation
            if 'Profit' in df.columns:
                profit_values = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
                y = (profit_values > 0).astype(int)
                print(f"✅ Using Profit column: Win rate {y.mean():.1%}")
            else:
                # Random target for demo
                np.random.seed(42)
                y = np.random.binomial(1, 0.52, len(df))
                print(f"✅ Using simulated target: Win rate {y.mean():.1%}")
            
            return y
    
    def train_models(self, X, y, test_size=0.2):
        """Train models dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.train_models(X, y, test_size)
        else:
            # Basic training
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            print("🚀 Training basic model...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.ensemble_model = model
            
            results = {
                'random_forest': {
                    'model': model,
                    'accuracy': accuracy,
                    'high_conf_win_rate': accuracy,  # Simplified
                    'high_conf_signals': len(y_test)
                }
            }
            
            print(f"✅ Basic model trained: {accuracy:.1%} accuracy")
            return results
    
    def generate_trading_signal(self, market_data):
        """Generate trading signal dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.generate_trading_signal(market_data)
        else:
            # Basic signal generation
            if self.ensemble_model is None:
                raise ValueError("Model not trained")
            
            features = self.engineer_features(market_data)
            features_scaled = self.scaler.transform(features)
            
            prediction_proba = self.ensemble_model.predict_proba(features_scaled)[0, 1]
            prediction = self.ensemble_model.predict(features_scaled)[0]
            
            signal = {
                'signal_type': 'BUY' if prediction == 1 else 'HOLD',
                'confidence': float(prediction_proba),
                'timestamp': datetime.now().isoformat(),
                'model_version': 'basic_fibonacci_v1.0'
            }
            
            return signal
    
    def run_complete_analysis(self, max_files=30, max_rows_per_file=50):
        """Run complete analysis dengan compatibility wrapper"""
        if ENHANCED_AVAILABLE:
            return self.predictor.run_complete_analysis(max_files, max_rows_per_file)
        else:
            # Basic analysis
            print("🧠 BASIC FIBONACCI ANALYSIS (Fallback Mode)")
            print("=" * 50)
            
            start_time = time.time()
            
            try:
                # Load data
                df = self.load_and_prepare_data(max_files, max_rows_per_file)
                
                # Feature engineering
                X = self.engineer_features(df)
                
                # Create target
                y = self.create_target_variable(df)
                
                # Train models
                results = self.train_models(X, y)
                
                # Summary
                total_time = time.time() - start_time
                
                print("\n" + "=" * 50)
                print("🎯 BASIC ANALYSIS RESULTS")
                print("=" * 50)
                print(f"⏱️ Total time: {total_time:.1f} seconds")
                print(f"📊 Records: {len(df)}")
                print(f"🔧 Features: {X.shape[1]}")
                
                if results:
                    best_model = list(results.keys())[0]
                    best_accuracy = results[best_model]['accuracy']
                    print(f"🏆 Model accuracy: {best_accuracy:.1%}")
                    
                    if best_accuracy >= 0.55:
                        print("✅ Decent performance achieved!")
                    else:
                        print("📈 Consider using enhanced version for better results")
                
                return results
                
            except Exception as e:
                print(f"❌ Basic analysis failed: {e}")
                return None

def main():
    """Main execution function"""
    try:
        print("🚀 Starting Fibonacci Deep Learning Analysis")
        
        # Initialize predictor
        predictor = FibonacciDeepLearningPredictor()
        
        # Run analysis
        results = predictor.run_complete_analysis(
            max_files=15,        # Reduced for compatibility
            max_rows_per_file=30 # Reduced for compatibility
        )
        
        if results:
            print("\n🎉 ANALYSIS COMPLETED!")
            
            if ENHANCED_AVAILABLE:
                print("✅ Full enhanced analysis completed successfully")
            else:
                print("⚠️ Basic fallback analysis completed")
                print("📋 For full features, ensure MLFLOW infrastructure is available")
        else:
            print("\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

def main():
    """Main execution function"""
    try:
        # Initialize predictor
        predictor = FibonacciDeepLearningPredictor()
        
        # Run complete analysis
        results = predictor.run_complete_analysis(
            max_files=25,        # Process 25 files for speed
            max_rows_per_file=40 # 40 rows per file
        )
        
        if results:
            print("\n🎉 DEEP LEARNING ANALYSIS COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
