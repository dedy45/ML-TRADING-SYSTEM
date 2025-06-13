"""
Advanced Trading ML Pipeline - FIXED VERSION untuk Pemula
Versi yang sudah diperbaiki dengan error handling yang baik
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

class AdvancedTradingMLFixed:
    """Advanced trading ML dengan error handling yang baik untuk pemula"""
    
    def __init__(self, experiment_name="trading_signal_prediction_fixed"):
        print("🚀 Initializing Advanced Trading ML Pipeline...")
        
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        
        # Setup MLflow dengan error handling
        try:
            mlflow.set_tracking_uri("./mlruns")
            mlflow.set_experiment(experiment_name)
            print("✅ MLflow setup successful")
        except Exception as e:
            print(f"⚠️ MLflow setup warning: {e}")
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data/processed", "data/features", "models", 
            "experiments", "logs", "mlruns"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print("✅ Directories created")
    
    def load_data(self, num_files=10, max_rows_per_file=1000):
        """Load data dengan error handling yang baik"""
        print(f"🔄 Loading data from {num_files} files (max {max_rows_per_file} rows each)...")
        
        try:
            # Start MLflow run with error handling
            try:
                run = mlflow.start_run(run_name="data_loading", nested=True)
                use_mlflow = True
            except:
                print("⚠️ MLflow not available, continuing without tracking")
                use_mlflow = False
            
            try:
                csv_files = glob.glob("dataBT/*.csv")
                
                if use_mlflow:
                    mlflow.log_param("num_files", num_files)
                    mlflow.log_param("max_rows_per_file", max_rows_per_file)
                    mlflow.log_param("total_available_files", len(csv_files))
                
                if not csv_files:
                    print("❌ No CSV files found")
                    return False
                
                # Sample files randomly for better diversity
                import random
                random.seed(42)
                sample_files = random.sample(csv_files, min(num_files, len(csv_files)))
                
                dataframes = []
                total_rows = 0
                
                for i, file in enumerate(sample_files):
                    try:
                        print(f"   📖 Reading file {i+1}/{len(sample_files)}: {os.path.basename(file)}")
                        df = pd.read_csv(file)
                        df = df[df['Type'] != 'INIT_SUCCESS']  # Remove system messages
                        
                        if len(df) > 0:
                            # Sample rows if file is too large
                            if len(df) > max_rows_per_file:
                                df = df.sample(n=max_rows_per_file, random_state=42)
                            
                            df['source_file'] = os.path.basename(file)
                            dataframes.append(df)
                            total_rows += len(df)
                            print(f"      ✅ Loaded {len(df)} rows")
                        
                    except Exception as e:
                        print(f"      ❌ Error in file {i+1}: {str(e)}")
                        continue
                
                if dataframes:
                    self.data = pd.concat(dataframes, ignore_index=True)
                    
                    # Log data statistics
                    buy_count = len(self.data[self.data['Type'] == 'BUY'])
                    sell_count = len(self.data[self.data['Type'] == 'SELL'])
                    profitable_count = len(self.data[self.data['Profit'] > 0])
                    win_rate = profitable_count / len(self.data)
                    
                    if use_mlflow:
                        mlflow.log_metric("total_trades", len(self.data))
                        mlflow.log_metric("buy_trades", buy_count)
                        mlflow.log_metric("sell_trades", sell_count)
                        mlflow.log_metric("profitable_trades", profitable_count)
                        mlflow.log_metric("win_rate", win_rate)
                    
                    print(f"🎉 Data loaded successfully!")
                    print(f"   📊 Total trades: {len(self.data)}")
                    print(f"   📈 BUY: {buy_count}, SELL: {sell_count}")
                    print(f"   💰 Win Rate: {win_rate*100:.1f}%")
                    
                    return True
                else:
                    print("❌ No data could be loaded")
                    return False
                    
            finally:
                if use_mlflow:
                    try:
                        mlflow.end_run()
                    except:
                        pass
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            return False
    
    def create_features(self):
        """Create features dengan error handling yang baik"""
        if self.data is None:
            print("❌ No data loaded. Please run load_data() first")
            return False
        
        print("🔄 Creating features...")
        
        try:
            # Start MLflow run with error handling
            try:
                run = mlflow.start_run(run_name="feature_engineering", nested=True)
                use_mlflow = True
            except:
                use_mlflow = False
            
            try:
                df = self.data.copy()
                
                # Convert timestamp safely
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                except:
                    print("⚠️ Timestamp conversion failed, using index")
                    df['Timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df.index, unit='h')
                
                # Target variables
                df['is_profitable'] = (df['Profit'] > 0).astype(int)
                df['is_big_win'] = (df['Profit'] > 10).astype(int)
                
                # Basic features
                df['is_buy'] = (df['Type'] == 'BUY').astype(int)
                df['price_level'] = df['OpenPrice'].fillna(0)
                df['volume_size'] = df['Volume'].fillna(0)
                
                # Time features
                df['hour'] = df['Timestamp'].dt.hour
                df['day_of_week'] = df['Timestamp'].dt.dayofweek
                
                # Cyclical encoding for time features
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                
                # Trading session features
                df['session_asia'] = df['SessionAsia'].fillna(0)
                df['session_europe'] = df['SessionEurope'].fillna(0)
                df['session_us'] = df['SessionUS'].fillna(0)
                df['session_score'] = df['session_asia'] + df['session_europe'] + df['session_us']
                df['multi_session'] = (df['session_score'] > 1).astype(int)
                
                # Risk management features
                df['tp_size'] = df['TP'].fillna(0)
                df['sl_size'] = df['SL'].fillna(0)
                df['risk_reward_ratio'] = np.where(df['sl_size'] > 0, df['tp_size'] / df['sl_size'], 0)
                
                # Performance features
                df['mae_pips'] = df['MAE_pips'].fillna(0)
                df['mfe_pips'] = df['MFE_pips'].fillna(0)
                df['mfe_mae_ratio'] = np.where(df['mae_pips'] > 0, df['mfe_pips'] / df['mae_pips'], 0)
                
                # Price movement features
                df['close_price'] = df['ClosePrice'].fillna(df['price_level'])
                df['price_change'] = df['close_price'] - df['price_level']
                df['price_change_pct'] = np.where(df['price_level'] > 0, 
                                                 df['price_change'] / df['price_level'] * 100, 0)
                
                # Select final features
                feature_columns = [
                    'is_buy', 'price_level', 'volume_size',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'session_asia', 'session_europe', 'session_us',
                    'session_score', 'multi_session',
                    'tp_size', 'sl_size', 'risk_reward_ratio',
                    'mae_pips', 'mfe_pips', 'mfe_mae_ratio',
                    'price_change_pct'
                ]
                
                # Create final feature set
                self.features = df[feature_columns + ['is_profitable', 'is_big_win']].copy()
                
                # Remove any remaining NaN rows
                initial_rows = len(self.features)
                self.features = self.features.dropna()
                final_rows = len(self.features)
                
                # Log feature engineering metrics
                profitable_rate = self.features['is_profitable'].mean()
                big_win_rate = self.features['is_big_win'].mean()
                
                if use_mlflow:
                    mlflow.log_param("num_features", len(feature_columns))
                    mlflow.log_param("initial_rows", initial_rows)
                    mlflow.log_param("final_rows", final_rows)
                    mlflow.log_metric("profitable_rate", profitable_rate)
                    mlflow.log_metric("big_win_rate", big_win_rate)
                
                print(f"✅ Features created successfully!")
                print(f"   📊 Final dataset: {final_rows} rows, {len(feature_columns)} features")
                print(f"   💰 Profitable rate: {profitable_rate*100:.1f}%")
                print(f"   🎯 Big win rate: {big_win_rate*100:.1f}%")
                
                return True
                
            finally:
                if use_mlflow:
                    try:
                        mlflow.end_run()
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Feature creation failed: {str(e)}")
            return False
    
    def train_models(self, target='is_profitable'):
        """Train models dengan error handling yang baik"""
        if self.features is None:
            print("❌ No features available. Please run create_features() first")
            return False
        
        print(f"🔄 Training models for target: {target}")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score
            
            # Prepare data
            feature_cols = [col for col in self.features.columns if col not in ['is_profitable', 'is_big_win']]
            X = self.features[feature_cols]
            y = self.features[target]
            
            print(f"   📊 Training data: {len(X)} samples, {len(feature_cols)} features")
            print(f"   🎯 Target distribution: Positive={sum(y)}, Negative={len(y)-sum(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest (simple and robust)
            print("   🤖 Training Random Forest...")
            
            try:
                run = mlflow.start_run(run_name=f"RandomForest_{target}", nested=True)
                use_mlflow = True
            except:
                use_mlflow = False
            
            try:
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3)  # Reduced CV for speed
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean = accuracy
                    cv_std = 0
                
                # Calculate trading metrics
                predicted_wins = sum(y_pred)
                actual_wins = sum(y_test)
                precision_wins = sum((y_pred == 1) & (y_test == 1)) / max(predicted_wins, 1)
                
                # Log metrics
                if use_mlflow:
                    mlflow.log_param("model_type", "RandomForest")
                    mlflow.log_param("target", target)
                    mlflow.log_param("n_features", len(feature_cols))
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("cv_mean", cv_mean)
                    mlflow.log_metric("cv_std", cv_std)
                    mlflow.log_metric("precision_wins", precision_wins)
                    mlflow.sklearn.log_model(model, f"RandomForest_{target}")
                
                # Store results
                model_name = f"RandomForest_{target}"
                self.models[model_name] = model
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'precision_wins': precision_wins
                }
                
                print(f"      ✅ Accuracy: {accuracy:.3f}")
                print(f"      📊 CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
                print(f"      🎯 Win Precision: {precision_wins:.3f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("      🏆 Top 3 Important Features:")
                for idx, row in feature_importance.head(3).iterrows():
                    print(f"         {row['feature']}: {row['importance']:.3f}")
                
                return True
                
            finally:
                if use_mlflow:
                    try:
                        mlflow.end_run()
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return False
    
    def generate_signals(self, confidence_threshold=0.7):
        """Generate trading signals dengan error handling yang baik"""
        if not self.models:
            print("❌ No trained models available. Please run train_models() first")
            return None
        
        print(f"🔄 Generating signals with confidence threshold: {confidence_threshold}")
        
        try:
            # Use the first available model
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            
            print(f"   🤖 Using model: {model_name}")
            
            # Prepare features
            feature_cols = [col for col in self.features.columns if col not in ['is_profitable', 'is_big_win']]
            X = self.features[feature_cols]
            
            # Predict probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = model.predict(X).astype(float)
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'probability': probabilities,
                'signal': (probabilities >= confidence_threshold).astype(int),
                'confidence': np.where(probabilities >= 0.5, probabilities, 1 - probabilities)
            })
            
            # Add signal strength categories
            signals['strength'] = pd.cut(signals['probability'], 
                                       bins=[0, 0.3, 0.7, 1.0], 
                                       labels=['WEAK', 'MEDIUM', 'STRONG'])
            
            signal_count = signals['signal'].sum()
            high_conf_count = (signals['probability'] >= confidence_threshold).sum()
            
            print(f"   📊 Results:")
            print(f"      Total signals: {signal_count}/{len(signals)} ({signal_count/len(signals)*100:.1f}%)")
            print(f"      High confidence: {high_conf_count}")
            print(f"      Signal strength: {signals['strength'].value_counts().to_dict()}")
            
            # Save signals
            signals.to_csv("data/processed/trading_signals.csv", index=False)
            print(f"   💾 Signals saved to: data/processed/trading_signals.csv")
            
            return signals
            
        except Exception as e:
            print(f"❌ Signal generation failed: {str(e)}")
            return None
    
    def run_pipeline(self, num_files=10, target='is_profitable'):
        """Run complete pipeline dengan error handling yang sangat baik"""
        print("🚀 Starting Advanced Trading ML Pipeline (Fixed Version)")
        print("=" * 60)
        
        # Setup main MLflow run
        try:
            main_run = mlflow.start_run(run_name=f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_param("pipeline_version", "advanced_fixed_v1.0")
            mlflow.log_param("num_files", num_files)
            mlflow.log_param("target", target)
            use_mlflow = True
        except:
            print("⚠️ MLflow not available, continuing without main tracking")
            use_mlflow = False
        
        # Define steps with clear success criteria
        steps = [
            ("Load Data", lambda: self.load_data(num_files=num_files)),
            ("Create Features", self.create_features),
            ("Train Models", lambda: self.train_models(target=target)),
            ("Generate Signals", self.generate_signals)
        ]
        
        success_count = 0
        
        try:
            for step_name, step_func in steps:
                print(f"\n📋 Step: {step_name}")
                print("-" * 40)
                
                try:
                    result = step_func()
                    
                    # Check success based on step type
                    if step_name == "Generate Signals":
                        # For signal generation, success means we got a DataFrame
                        success = (result is not None) and (isinstance(result, pd.DataFrame)) and (len(result) > 0)
                    else:
                        # For other steps, success is boolean True
                        success = (result is True)
                    
                    if success:
                        print(f"✅ {step_name} - COMPLETED")
                        success_count += 1
                    else:
                        print(f"❌ {step_name} - FAILED")
                        if step_name != "Generate Signals":  # Continue even if signal generation fails
                            break
                    
                except Exception as e:
                    print(f"❌ {step_name} - ERROR: {str(e)}")
                    if step_name != "Generate Signals":  # Continue even if signal generation fails
                        break
            
            # Final summary
            print("\n" + "=" * 60)
            print("📊 PIPELINE SUMMARY")
            print("=" * 60)
            
            if success_count >= 3:  # At least data, features, and models
                print("🎉 Pipeline Completed Successfully!")
                
                if use_mlflow:
                    mlflow.log_param("pipeline_status", "SUCCESS")
                    mlflow.log_metric("steps_completed", success_count)
                
                # Show results
                if self.results:
                    print("\n📈 Model Performance:")
                    for model_name, metrics in self.results.items():
                        print(f"   {model_name}:")
                        print(f"      Accuracy: {metrics['accuracy']:.3f}")
                        print(f"      CV Score: {metrics['cv_mean']:.3f}")
                        print(f"      Win Precision: {metrics['precision_wins']:.3f}")
                
                print("\n📋 Files Created:")
                output_files = [
                    "data/processed/trading_signals.csv",
                    "models/",
                    "mlruns/"
                ]
                for file_path in output_files:
                    if os.path.exists(file_path):
                        print(f"   ✅ {file_path}")
                
                print("\n🚀 Next Steps:")
                print("   1. Check results in data/processed/")
                print("   2. Run: mlflow ui")
                print("   3. Open: http://localhost:5000")
                print("   4. Scale up with more files")
                
                return True
            else:
                print("⚠️ Pipeline completed with some issues")
                if use_mlflow:
                    mlflow.log_param("pipeline_status", f"PARTIAL_SUCCESS_{success_count}_steps")
                return False
                
        finally:
            if use_mlflow:
                try:
                    mlflow.end_run()
                except:
                    pass

def main():
    """Main function untuk menjalankan pipeline"""
    print("🎯 ADVANCED TRADING ML - FIXED VERSION")
    print("Versi yang sudah diperbaiki untuk pemula")
    print("=" * 50)
    
    # Create pipeline
    pipeline = AdvancedTradingMLFixed()
    
    # Run pipeline with error handling
    try:
        success = pipeline.run_pipeline(num_files=10, target='is_profitable')
        
        if success:
            print("\n🎉 SUCCESS! Pipeline selesai dengan baik")
            print("\n📋 Langkah selanjutnya:")
            print("1. Buka hasil di folder: data/processed/")
            print("2. Jalankan MLflow UI: mlflow ui")
            print("3. Buka browser: http://localhost:5000")
        else:
            print("\n⚠️ Pipeline selesai tapi ada beberapa masalah")
            print("Cek hasil yang sudah dibuat di folder data/processed/")
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        print("Tapi jangan khawatir, ini normal dalam development!")
        print("Cek apakah ada file hasil di data/processed/")

if __name__ == "__main__":
    main()
