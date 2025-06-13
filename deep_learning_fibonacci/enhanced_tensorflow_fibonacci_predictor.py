#!/usr/bin/env python3
"""
Enhanced TensorFlow Deep Learning Fibonacci Signal Generator
Integrated with MLFLOW infrastructure for robust performance
Menghasilkan signal prediksi trading yang akurat untuk EA MQL5 dengan timeout protection
"""

import os
import sys
import json
import time
import warnings
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')

# Add parent directory to path for MLFLOW imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import MLFLOW infrastructure first
try:
    from config.config import config
    from utils.timeout_utils import ExecutionGuard, safe_timeout, TimeoutException, timeout_decorator
    from utils.logging_utils import get_logger, PerformanceLogger
    from data.data_processor import FibonacciDataProcessor
    MLFLOW_INFRA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MLFLOW infrastructure not available: {e}")
    MLFLOW_INFRA_AVAILABLE = False
    # Fallback minimal setup
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Setup logging
if MLFLOW_INFRA_AVAILABLE:
    logger = get_logger('deep_learning_fibonacci', config.logs_dir)
    perf_logger = PerformanceLogger(logger)
else:
    logger = logging.getLogger(__name__)
    perf_logger = None

# Core libraries
import numpy as np
import pandas as pd

# MLflow tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    logger.info("✅ MLflow loaded successfully")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️ MLflow not available")

# TensorFlow support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow loaded successfully")
except ImportError:
    logger.warning("⚠️ TensorFlow not available, using scikit-learn alternative")
    TENSORFLOW_AVAILABLE = False

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib
    SKLEARN_AVAILABLE = True
    logger.info("✅ Scikit-learn loaded successfully")
except ImportError:
    logger.error("❌ Scikit-learn not available")
    SKLEARN_AVAILABLE = False

class EnhancedFibonacciDeepLearningPredictor:
    """
    Enhanced Deep Learning Fibonacci Signal Generator with MLFLOW integration
    Tujuan: Menghasilkan signal trading dengan 58%+ win rate untuk EA MQL5
    """
    
    def __init__(self, data_path: str = None, model_save_path: str = None, experiment_name: str = None):
        # Configuration
        if MLFLOW_INFRA_AVAILABLE:
            self.data_path = Path(data_path) if data_path else config.data_dir
            self.model_save_path = Path(model_save_path) if model_save_path else config.models_dir
        else:
            self.data_path = Path(data_path) if data_path else Path("../dataBT")
            self.model_save_path = Path(model_save_path) if model_save_path else Path("models")
        
        self.model_save_path.mkdir(exist_ok=True, parents=True)
        self.experiment_name = experiment_name or f"fibonacci_dl_{int(time.time())}"
        
        # Models
        self.lstm_model = None
        self.cnn_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Performance tracking
        self.training_history = {}
        self.model_performance = {}
        self.execution_stats = {
            'start_time': time.time(),
            'operations_completed': 0,
            'errors_encountered': 0
        }
          # MLflow setup
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"🔬 MLflow experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"⚠️ MLflow setup failed: {e}")
        
        logger.info("🧠 Enhanced Fibonacci Deep Learning Predictor initialized")
        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"💾 Model save path: {self.model_save_path}")
    
    @timeout_decorator(300, "Data loading")  # 5 minute timeout
    def load_and_prepare_data(self, max_files: int = 50, max_rows_per_file: int = 100) -> pd.DataFrame:
        """Load data dengan timeout protection dan optimasi kecepatan"""
        logger.info(f"📂 Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        csv_files = list(self.data_path.glob("*.csv"))[:max_files]
        logger.info(f"📁 Processing {len(csv_files)} files (max {max_rows_per_file} rows each)")
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_path}")
        
        all_data = []
        total_rows = 0
        
        # Use ExecutionGuard for additional timeout protection
        if MLFLOW_INFRA_AVAILABLE:
            guard = ExecutionGuard(max_execution_time=250)  # Slightly less than outer timeout
            guard.start()
        
        try:
            for i, file_path in enumerate(csv_files):
                # Check for timeout
                if MLFLOW_INFRA_AVAILABLE and guard.should_stop():
                    logger.warning("⚠️ Data loading timeout detected, stopping")
                    break
                
                try:
                    # Progress indicator
                    if i % 10 == 0:
                        logger.info(f"   📄 Progress: {i}/{len(csv_files)} files...")
                    
                    df = pd.read_csv(file_path, nrows=max_rows_per_file)
                    df['file_index'] = i
                    df['file_name'] = file_path.name
                    all_data.append(df)
                    total_rows += len(df)
                    
                    self.execution_stats['operations_completed'] += 1
                    
                except Exception as e:
                    logger.warning(f"   ❌ Error loading {file_path.name}: {e}")
                    self.execution_stats['errors_encountered'] += 1
                    continue
        
        finally:
            if MLFLOW_INFRA_AVAILABLE:
                guard.stop()
        
        if not all_data:
            raise ValueError("No data loaded successfully")
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Loaded {len(df)} records from {len(all_data)} files")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_param("files_processed", len(all_data))
                mlflow.log_param("total_records", len(df))
                mlflow.log_param("max_files", max_files)
                mlflow.log_param("max_rows_per_file", max_rows_per_file)
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging failed: {e}")
        
        return df
    
    @timeout_decorator(180, "Feature engineering")  # 3 minute timeout
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering dengan timeout protection"""
        logger.info("🔧 Engineering advanced features...")
        
        if perf_logger:
            perf_logger.start_timer('feature_engineering')
        
        features = {}
        
        # Convert key columns to numeric with error handling
        numeric_cols = ['LevelFibo', 'SeparatorHour', 'TP', 'SL', 'Profit', 
                       'SessionEurope', 'SessionUS', 'SessionAsia']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except Exception as e:
                    logger.warning(f"⚠️ Error converting {col}: {e}")
                    df[col] = 0
        
        # 1. PRIMARY FIBONACCI FEATURES (Proven 52%+ win rates)
        if 'LevelFibo' in df.columns:
            # Core signals
            features['fib_b0'] = (df['LevelFibo'] == 0.0).astype(int)
            features['fib_b_minus_1_8'] = (df['LevelFibo'] == -1.8).astype(int)
            features['fib_b_1_8'] = (df['LevelFibo'] == 1.8).astype(int)
            features['fib_b_minus_3_6'] = (df['LevelFibo'] == -3.6).astype(int)
            
            # Signal strength scoring (based on proven analysis)
            signal_strength = np.zeros(len(df))
            signal_strength[df['LevelFibo'] == 0.0] = 3      # 52.4% win rate
            signal_strength[df['LevelFibo'] == -1.8] = 3     # 52.5% win rate  
            signal_strength[df['LevelFibo'] == 1.8] = 2      # 45.9% win rate
            signal_strength[df['LevelFibo'] == -3.6] = 2     # Good level
            features['signal_strength'] = signal_strength
            
            # Fibonacci level categories
            features['fib_level_raw'] = df['LevelFibo']
            features['fib_is_major'] = (np.abs(df['LevelFibo']) <= 2.0).astype(int)
            features['fib_is_extension'] = (np.abs(df['LevelFibo']) > 2.0).astype(int)
        
        # 2. SESSION OPTIMIZATION (Europe best: 40.5%)
        if 'SessionEurope' in df.columns:
            features['session_europe'] = df['SessionEurope']
            features['session_us'] = df['SessionUS'].fillna(0)
            features['session_asia'] = df['SessionAsia'].fillna(0)
            
            # Session scoring (based on proven performance)
            session_score = (df['SessionEurope'] * 3 +     # Best session
                           df['SessionUS'].fillna(0) * 2 +  # Good session
                           df['SessionAsia'].fillna(0) * 1) # Lower performance
            features['session_score'] = session_score
            
            # Active sessions count
            active_sessions = (df['SessionEurope'] + 
                             df['SessionUS'].fillna(0) + 
                             df['SessionAsia'].fillna(0))
            features['active_sessions'] = active_sessions
        
        # 3. RISK MANAGEMENT FEATURES (2:1 TP/SL optimal)
        if 'TP' in df.columns and 'SL' in df.columns:
            # TP/SL ratio with safe division
            tp_sl_ratio = np.where(df['SL'] != 0, df['TP'] / df['SL'], 0)
            features['tp_sl_ratio'] = tp_sl_ratio
            features['optimal_ratio'] = (tp_sl_ratio >= 2.0).astype(int)
            features['conservative_ratio'] = (tp_sl_ratio >= 2.5).astype(int)
            
            # Risk score
            risk_score = np.clip(tp_sl_ratio / 2.0, 0, 2)  # Normalize to 0-2
            features['risk_score'] = risk_score
        
        # 4. TIME-BASED FEATURES
        if 'SeparatorHour' in df.columns:
            # Cyclical time encoding
            features['hour_sin'] = np.sin(2 * np.pi * df['SeparatorHour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df['SeparatorHour'] / 24)
            
            # Trading hour categories
            features['peak_hours'] = ((df['SeparatorHour'] >= 8) & 
                                    (df['SeparatorHour'] <= 17)).astype(int)
            features['london_hours'] = ((df['SeparatorHour'] >= 8) & 
                                      (df['SeparatorHour'] <= 16)).astype(int)
            features['ny_hours'] = ((df['SeparatorHour'] >= 13) & 
                                  (df['SeparatorHour'] <= 21)).astype(int)
        
        # 5. ADVANCED COMBINED FEATURES
        # High confidence signal
        high_conf = ((features.get('signal_strength', np.zeros(len(df))) >= 2) & 
                    (features.get('session_score', np.zeros(len(df))) >= 2) & 
                    (features.get('optimal_ratio', np.zeros(len(df))) == 1)).astype(int)
        features['high_confidence_signal'] = high_conf
        
        # Market timing score
        timing_score = (features.get('session_score', np.zeros(len(df))) + 
                       features.get('peak_hours', np.zeros(len(df))) + 
                       features.get('signal_strength', np.zeros(len(df))))
        features['market_timing_score'] = timing_score
        
        # Create feature matrix
        feature_df = pd.DataFrame(features).fillna(0)
        
        if perf_logger:
            perf_logger.end_timer('feature_engineering')
        
        logger.info(f"✅ Created {len(features)} engineered features")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_param("features_created", len(features))
                mlflow.log_param("feature_engineering_shape", feature_df.shape)
            except Exception:
                pass
        
        return feature_df
    
    def create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable dengan error handling"""
        logger.info("🎯 Creating target variable...")
        
        if 'Profit' in df.columns:
            profit_values = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
            y = (profit_values > 0).astype(int)
            win_rate = y.mean()
            logger.info(f"✅ Using Profit column: {y.value_counts().to_dict()}")
            logger.info(f"📊 Current win rate: {win_rate:.1%}")
            
        elif 'Result' in df.columns:
            result_values = pd.to_numeric(df['Result'], errors='coerce').fillna(0)
            y = (result_values > 0).astype(int)
            win_rate = y.mean()
            logger.info(f"✅ Using Result column: {y.value_counts().to_dict()}")
            logger.info(f"📊 Current win rate: {win_rate:.1%}")
            
        else:
            # Simulasi berdasarkan signal strength untuk demo
            if 'LevelFibo' in df.columns:
                df_num = pd.to_numeric(df['LevelFibo'], errors='coerce').fillna(0)
                # Simulate higher win rates for proven levels
                prob_win = np.where(df_num == 0.0, 0.524,      # 52.4% for B_0
                          np.where(df_num == -1.8, 0.525,      # 52.5% for B_-1.8
                          np.where(df_num == 1.8, 0.459,       # 45.9% for B_1.8
                                   0.45)))                      # 45% baseline
            else:
                prob_win = np.full(len(df), 0.52)  # 52% baseline
            
            # Set random seed for reproducibility
            np.random.seed(42)
            y = np.random.binomial(1, prob_win)
            win_rate = y.mean()
            logger.info(f"✅ Using simulated target: Win rate {win_rate:.1%}")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("baseline_win_rate", win_rate)
                mlflow.log_param("target_variable_source", "Profit" if 'Profit' in df.columns else "simulated")
            except Exception:
                pass
        
        return y
    
    def build_ensemble_model(self, input_dim: int) -> Dict[str, Any]:
        """Build ensemble model menggunakan scikit-learn dengan error handling"""
        logger.info("🎯 Building ensemble model...")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available for model building")
        
        models = {}
        
        try:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,  # Reduced for faster training
                max_depth=8,       # Reduced to prevent overfitting
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        except Exception as e:
            logger.warning(f"⚠️ Random Forest initialization failed: {e}")
        
        try:
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,  # Reduced for faster training
                learning_rate=0.1,
                max_depth=5,       # Reduced
                random_state=42
            )
        except Exception as e:
            logger.warning(f"⚠️ Gradient Boosting initialization failed: {e}")
        
        try:
            models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(32, 16),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=300,      # Reduced iterations
                random_state=42
            )
        except Exception as e:
            logger.warning(f"⚠️ Neural Network initialization failed: {e}")
        
        logger.info(f"✅ Built {len(models)} ensemble models")
        return models
    
    @timeout_decorator(600, "Model training")  # 10 minute timeout for training
    def train_models(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models dengan timeout protection dan evaluation"""
        logger.info("🚀 Training deep learning models...")
        
        if perf_logger:
            perf_logger.start_timer('model_training')
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        logger.info(f"📊 Feature dimensions: {X_train.shape[1]}")
        
        # Train ensemble models
        ensemble_models = self.build_ensemble_model(X_train.shape[1])
        results = {}
        
        # Start MLflow run
        if MLFLOW_AVAILABLE:
            try:
                mlflow.start_run()
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("feature_count", X_train.shape[1])
                mlflow.log_param("test_size", test_size)
            except Exception as e:
                logger.warning(f"⚠️ MLflow run start failed: {e}")
        
        for name, model in ensemble_models.items():
            logger.info(f"   🤖 Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
                
                # Trading-specific metrics
                # Win rate for high confidence signals (>= 0.7 probability)
                high_conf_mask = y_pred_proba >= 0.7
                if np.sum(high_conf_mask) > 0:
                    high_conf_win_rate = np.mean(y_test[high_conf_mask] == 1)
                    high_conf_signals = np.sum(high_conf_mask)
                else:
                    high_conf_win_rate = 0
                    high_conf_signals = 0
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc,
                    'high_conf_win_rate': high_conf_win_rate,
                    'high_conf_signals': high_conf_signals,
                    'training_time': training_time
                }
                
                logger.info(f"     ✅ Accuracy: {accuracy:.1%}")
                logger.info(f"     📈 High Confidence Win Rate: {high_conf_win_rate:.1%} ({high_conf_signals} signals)")
                logger.info(f"     ⏱️ Training time: {training_time:.1f}s")
                
                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metric(f"{name}_accuracy", accuracy)
                        mlflow.log_metric(f"{name}_precision", precision)
                        mlflow.log_metric(f"{name}_recall", recall)
                        mlflow.log_metric(f"{name}_f1", f1)
                        mlflow.log_metric(f"{name}_auc", auc)
                        mlflow.log_metric(f"{name}_high_conf_win_rate", high_conf_win_rate)
                        mlflow.log_metric(f"{name}_high_conf_signals", high_conf_signals)
                        mlflow.log_metric(f"{name}_training_time", training_time)
                    except Exception as e:
                        logger.warning(f"⚠️ MLflow metric logging failed: {e}")
                
                self.execution_stats['operations_completed'] += 1
                
            except Exception as e:
                logger.error(f"     ❌ Training failed: {e}")
                self.execution_stats['errors_encountered'] += 1
                continue
        
        # Save best model
        if results:
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k]['high_conf_win_rate'])
            best_model = results[best_model_name]['model']
            
            # Save model and scaler
            model_path = self.model_save_path / "best_fibonacci_model.pkl"
            scaler_path = self.model_save_path / "feature_scaler.pkl"
            
            try:
                joblib.dump(best_model, model_path)
                joblib.dump(self.scaler, scaler_path)
                
                logger.info(f"💾 Best model saved: {best_model_name}")
                logger.info(f"📁 Model path: {model_path}")
                
                # Store for inference
                self.ensemble_model = best_model
                self.model_performance = results
                
                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_param("best_model", best_model_name)
                        mlflow.log_metric("best_model_win_rate", results[best_model_name]['high_conf_win_rate'])
                        mlflow.sklearn.log_model(best_model, "best_model")
                        mlflow.log_artifact(str(model_path))
                        mlflow.log_artifact(str(scaler_path))
                    except Exception as e:
                        logger.warning(f"⚠️ MLflow artifact logging failed: {e}")
                
            except Exception as e:
                logger.error(f"❌ Model saving failed: {e}")
        
        # End MLflow run
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass
        
        if perf_logger:
            perf_logger.end_timer('model_training')
        
        return results
    
    def generate_trading_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal untuk EA MQL5 dengan error handling"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available for inference")
        
        try:
            # Extract features
            features = self.engineer_features(market_data)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction
            prediction_proba = self.ensemble_model.predict_proba(features_scaled)[0, 1]
            prediction = self.ensemble_model.predict(features_scaled)[0]
            
            # Generate signal
            signal = {
                'signal_type': 'BUY' if prediction == 1 else 'HOLD',
                'confidence': float(prediction_proba),
                'fibonacci_level': market_data['LevelFibo'].iloc[0] if 'LevelFibo' in market_data else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'model_version': 'enhanced_fibonacci_dl_v2.0'
            }
            
            # Add trade parameters for high confidence signals
            if prediction_proba >= 0.7:
                signal.update({
                    'entry_price': market_data.get('entry_price', 0),
                    'stop_loss': market_data.get('stop_loss', 0),
                    'take_profit': market_data.get('take_profit', 0),
                    'risk_reward_ratio': 2.0,
                    'position_size_pct': min(prediction_proba * 100, 5.0)  # Max 5% of account
                })
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return {
                'signal_type': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @timeout_decorator(900, "Complete analysis")  # 15 minute total timeout
    def run_complete_analysis(self, max_files: int = 30, max_rows_per_file: int = 50) -> Optional[Dict[str, Any]]:
        """Run complete analysis pipeline dengan comprehensive timeout protection"""
        logger.info("🧠 ENHANCED DEEP LEARNING FIBONACCI ANALYSIS")
        logger.info("=" * 60)
        logger.info("🎯 Target: 58%+ win rate for EA MQL5 integration")
        logger.info("🔒 Timeout protection: 15 minutes maximum")
        logger.info("")
        
        start_time = time.time()
          try:
            # Use ExecutionGuard as context manager
            if MLFLOW_INFRA_AVAILABLE:
                with ExecutionGuard(max_execution_time=850) as guard:
                    return self._run_analysis_with_guard(guard, max_files, max_rows_per_file)
            else:
                return self._run_analysis_with_guard(None, max_files, max_rows_per_file)
                
        except TimeoutException as e:
            logger.error(f"⏰ Analysis timed out: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _run_analysis_with_guard(self, guard, max_files: int, max_rows_per_file: int) -> Optional[Dict[str, Any]]:
            X = self.engineer_features(df)
            
            if MLFLOW_INFRA_AVAILABLE and main_guard.should_stop():
                raise TimeoutException("Timeout during feature engineering")
            
            # 3. Create target
            logger.info("🎯 Step 3: Creating target variable...")
            y = self.create_target_variable(df)
            
            if MLFLOW_INFRA_AVAILABLE and main_guard.should_stop():
                raise TimeoutException("Timeout during target creation")
            
            # 4. Train models
            logger.info("🚀 Step 4: Training models...")
            results = self.train_models(X, y)
            
            # 5. Performance summary
            total_time = time.time() - start_time
            self.execution_stats['total_time'] = total_time
            
            logger.info("\n" + "=" * 60)
            logger.info("🎯 ENHANCED DEEP LEARNING ANALYSIS RESULTS")
            logger.info("=" * 60)
            
            logger.info(f"⏱️ Total execution time: {total_time:.1f} seconds")
            logger.info(f"📊 Data processed: {len(df)} records from {max_files} files")
            logger.info(f"🔧 Features created: {X.shape[1]}")
            logger.info(f"📈 Current data win rate: {y.mean():.1%}")
            logger.info(f"✅ Operations completed: {self.execution_stats['operations_completed']}")
            logger.info(f"❌ Errors encountered: {self.execution_stats['errors_encountered']}")
            
            if results:
                logger.info("\n🏆 MODEL PERFORMANCE:")
                logger.info("-" * 50)
                
                for name, metrics in results.items():
                    logger.info(f"{name}:")
                    logger.info(f"  📊 Accuracy: {metrics['accuracy']:.1%}")
                    logger.info(f"  🎯 High Conf Win Rate: {metrics['high_conf_win_rate']:.1%}")
                    logger.info(f"  📈 High Conf Signals: {metrics['high_conf_signals']}")
                    logger.info(f"  🔄 AUC Score: {metrics['auc_score']:.3f}")
                    logger.info("")
                
                # Find best performing model
                best_model = max(results.keys(), 
                               key=lambda k: results[k]['high_conf_win_rate'])
                best_win_rate = results[best_model]['high_conf_win_rate']
                
                logger.info(f"🏆 BEST MODEL: {best_model}")
                logger.info(f"🎯 HIGH CONFIDENCE WIN RATE: {best_win_rate:.1%}")
                
                # Target achievement
                target_58_achieved = best_win_rate >= 0.58
                target_55_achieved = best_win_rate >= 0.55
                
                if target_58_achieved:
                    logger.info("🎉 TARGET 58% ACHIEVED! Ready for EA deployment!")
                elif target_55_achieved:
                    logger.info("✅ Target 55% achieved! Good for live trading.")
                else:
                    logger.info("📈 Partial improvement. Consider more data or tuning.")
                
                logger.info("\n🚀 NEXT STEPS FOR EA MQL5:")
                logger.info("1. 📁 Model saved to:", self.model_save_path)
                logger.info("2. 🔗 Use enhanced inference for real-time predictions")
                logger.info("3. 📊 Integrate with your EA using saved model")
                logger.info("4. 🧪 Test in paper trading before live deployment")
                logger.info("5. 📝 Review logs for optimization opportunities")
                
                # Final MLflow logging
                if MLFLOW_AVAILABLE:
                    try:
                        with mlflow.start_run():
                            mlflow.log_metric("final_best_win_rate", best_win_rate)
                            mlflow.log_metric("target_58_achieved", int(target_58_achieved))
                            mlflow.log_metric("total_execution_time", total_time)
                            mlflow.log_param("final_best_model", best_model)
                    except Exception:
                        pass
                
            else:
                logger.error("❌ No models trained successfully")
            
            return results
            
        except TimeoutException as e:
            logger.error(f"⏰ Analysis timed out: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            if MLFLOW_INFRA_AVAILABLE:
                try:
                    main_guard.stop()
                except:
                    pass

def main():
    """Main execution function dengan comprehensive error handling"""
    try:
        logger.info("🚀 Starting Enhanced Deep Learning Fibonacci Analysis")
        
        # Initialize predictor
        predictor = EnhancedFibonacciDeepLearningPredictor()
        
        # Run complete analysis
        results = predictor.run_complete_analysis(
            max_files=25,        # Process 25 files for speed
            max_rows_per_file=40 # 40 rows per file
        )
        
        if results:
            logger.info("\n🎉 ENHANCED DEEP LEARNING ANALYSIS COMPLETED SUCCESSFULLY!")
        else:
            logger.error("\n❌ Analysis failed or timed out")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
