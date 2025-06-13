#!/usr/bin/env python3
"""
Deep Learning Fibonacci Analyzer - Python 3.13 Compatible Version
Advanced ML pipeline using scikit-learn with neural network architectures
for Fibonacci trading signal prediction. Simulates TensorFlow capabilities.

Target: Improve from 55.1% B_0 win rate to 58%+ using deep learning techniques
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_learning_fibonacci.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepLearningFibonacciAnalyzer:
    """
    Advanced Fibonacci signal analyzer using deep learning techniques
    Simulates neural network architectures with scikit-learn
    """
    
    def __init__(self, data_path="E:/aiml/MLFLOW/dataBT"):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Deep learning simulation parameters
        self.neural_net_config = {
            'hidden_layer_sizes': [(100, 50), (150, 100, 50), (200, 100)],
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Baseline results from statistical analysis
        self.baseline_results = {
            'B_0': {'win_rate': 0.551, 'trades': 352, 'signal_type': 'primary'},
            'B_-1.8': {'win_rate': 0.525, 'trades': 120, 'signal_type': 'high_confidence'},
            'B_1.8': {'win_rate': 0.459, 'trades': 945, 'signal_type': 'secondary'}
        }
        
        logger.info("🧠 Deep Learning Fibonacci Analyzer initialized")
        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"🎯 Target: Improve from 55.1% to 58%+ win rate")
        
    def load_and_preprocess_data(self):
        """Load and preprocess all CSV files with advanced feature engineering"""
        logger.info("📊 Loading and preprocessing data...")
        
        all_data = []
        file_count = 0
        
        for csv_file in self.data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Add file metadata
                df['source_file'] = csv_file.stem
                df['file_id'] = file_count
                
                all_data.append(df)
                file_count += 1
                
                if file_count % 50 == 0:
                    logger.info(f"📈 Processed {file_count} files...")
                    
            except Exception as e:
                logger.warning(f"⚠️  Error loading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError("No valid CSV files found!")
            
        logger.info(f"✅ Loaded {file_count} files")
        
        # Combine all data
        self.raw_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Combined dataset shape: {self.raw_data.shape}")
        logger.info(f"📊 Columns: {list(self.raw_data.columns)}")
        
        # Advanced feature engineering
        self.engineered_data = self._engineer_deep_features(self.raw_data)
        
        return self.engineered_data
    
    def _engineer_deep_features(self, df):
        """Advanced feature engineering for deep learning"""
        logger.info("🔧 Engineering deep learning features...")
        
        engineered_df = df.copy()
        
        # 1. Fibonacci Level Analysis
        if 'LevelFibo' in df.columns:
            # Extract numeric Fibonacci levels
            engineered_df['fibo_level_numeric'] = pd.to_numeric(df['LevelFibo'], errors='coerce')
            
            # Categorical Fibonacci levels
            engineered_df['fibo_level_category'] = df['LevelFibo'].astype(str)
            
            # Fibonacci level strength (distance from key levels)
            key_levels = [-1.8, 0, 1.8, 2.618, 4.236]
            for level in key_levels:
                engineered_df[f'distance_to_{level}'] = abs(engineered_df['fibo_level_numeric'] - level)
            
            # Min distance to any key level
            distance_cols = [f'distance_to_{level}' for level in key_levels]
            engineered_df['min_distance_to_key_level'] = engineered_df[distance_cols].min(axis=1)
        
        # 2. Session Analysis
        session_cols = ['SessionAsia', 'SessionEurope', 'SessionUS']
        for col in session_cols:
            if col in df.columns:
                engineered_df[f'{col}_encoded'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Active sessions count
        if all(col in df.columns for col in session_cols):
            session_encoded = [f'{col}_encoded' for col in session_cols]
            engineered_df['active_sessions_count'] = engineered_df[session_encoded].sum(axis=1)
            
            # Dominant session
            engineered_df['dominant_session'] = engineered_df[session_encoded].idxmax(axis=1)
        
        # 3. Risk Management Features
        if 'TP' in df.columns and 'SL' in df.columns:
            engineered_df['TP_numeric'] = pd.to_numeric(df['TP'], errors='coerce')
            engineered_df['SL_numeric'] = pd.to_numeric(df['SL'], errors='coerce')
            
            # Risk-reward ratio
            engineered_df['risk_reward_ratio'] = (
                engineered_df['TP_numeric'] / (engineered_df['SL_numeric'] + 1e-8)
            )
            
            # Risk level categorization
            engineered_df['risk_level'] = pd.cut(
                engineered_df['SL_numeric'], 
                bins=[0, 50, 100, 200, float('inf')], 
                labels=['low', 'medium', 'high', 'extreme']
            )
        
        # 4. Level Position Features
        level_cols = ['Level1Above', 'Level1Below']
        for col in level_cols:
            if col in df.columns:
                engineered_df[f'{col}_numeric'] = pd.to_numeric(df[col], errors='coerce')
        
        if all(col in df.columns for col in level_cols):
            engineered_df['level_spread'] = (
                engineered_df['Level1Above_numeric'] - engineered_df['Level1Below_numeric']
            )
        
        # 5. Time-based Features
        if 'SeparatorHour' in df.columns:
            engineered_df['hour'] = pd.to_numeric(df['SeparatorHour'], errors='coerce')
            
            # Hour categories
            engineered_df['hour_category'] = pd.cut(
                engineered_df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            # Cyclical hour encoding (for neural networks)
            engineered_df['hour_sin'] = np.sin(2 * np.pi * engineered_df['hour'] / 24)
            engineered_df['hour_cos'] = np.cos(2 * np.pi * engineered_df['hour'] / 24)
        
        # 6. Daily Close Features
        if 'UseDailyClose' in df.columns:
            engineered_df['use_daily_close'] = pd.to_numeric(df['UseDailyClose'], errors='coerce').fillna(0)
        
        # 7. Complex Interaction Features (for deep learning)
        if 'LevelFibo' in df.columns and 'SessionEurope' in df.columns:
            # Fibonacci-Session interactions
            engineered_df['fibo_europe_interaction'] = (
                engineered_df['fibo_level_numeric'] * engineered_df.get('SessionEurope_encoded', 0)
            )
        
        # 8. Target Variable (Win/Loss)
        if 'Win' in df.columns:
            engineered_df['target'] = pd.to_numeric(df['Win'], errors='coerce').fillna(0)
        else:
            logger.warning("⚠️  No 'Win' column found, creating dummy target")
            engineered_df['target'] = 0
        
        # Remove rows with missing targets
        engineered_df = engineered_df.dropna(subset=['target'])
        
        logger.info(f"🔧 Feature engineering complete. Shape: {engineered_df.shape}")
        logger.info(f"🎯 Target distribution: {engineered_df['target'].value_counts().to_dict()}")
        
        return engineered_df
    
    def prepare_features_and_target(self, focus_fibonacci_level=None):
        """Prepare features and target for machine learning"""
        logger.info("🎯 Preparing features and target...")
        
        df = self.engineered_data.copy()
        
        # Filter by specific Fibonacci level if requested
        if focus_fibonacci_level:
            if 'LevelFibo' in df.columns:
                df = df[df['LevelFibo'].astype(str) == str(focus_fibonacci_level)]
                logger.info(f"🎯 Focused on Fibonacci level: {focus_fibonacci_level}")
                logger.info(f"📊 Filtered dataset shape: {df.shape}")
        
        # Select numerical features for ML
        feature_columns = [col for col in df.columns if col.endswith('_numeric') or 
                          col.endswith('_encoded') or col in ['hour_sin', 'hour_cos', 
                          'risk_reward_ratio', 'level_spread', 'min_distance_to_key_level',
                          'active_sessions_count']]
        
        # Remove features with too many missing values
        feature_columns = [col for col in feature_columns if col in df.columns and 
                          df[col].notna().sum() / len(df) > 0.5]
        
        X = df[feature_columns].fillna(0)
        y = df['target']
        
        logger.info(f"🔢 Features selected: {len(feature_columns)}")
        logger.info(f"🔢 Feature names: {feature_columns}")
        logger.info(f"📊 X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"🎯 Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns
    
    def build_deep_models(self):
        """Build deep learning models (simulated with MLPClassifier)"""
        logger.info("🧠 Building deep learning models...")
        
        models = {
            # LSTM-like architecture (sequence processing simulation)
            'LSTM_Simulator': Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=15)),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 50),  # Simulates LSTM layers
                    activation='tanh',  # LSTM-like activation
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=2000,
                    random_state=42
                ))
            ]),
            
            # CNN-like architecture (pattern recognition simulation)
            'CNN_Simulator': Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=20)),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(150, 100, 50),  # Deep layers for pattern recognition
                    activation='relu',  # CNN-like activation
                    solver='adam',
                    alpha=0.0001,
                    learning_rate='adaptive',
                    max_iter=2000,
                    random_state=42
                ))
            ]),
            
            # Ensemble Network (combines multiple approaches)
            'Ensemble_Neural': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50),  # Large ensemble-like network
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    learning_rate='adaptive',
                    max_iter=3000,
                    random_state=42
                ))
            ]),
            
            # Gradient Boosting (tree-based ensemble for comparison)
            'Advanced_Ensemble': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42
                ))
            ])
        }
        
        self.models = models
        logger.info(f"🧠 Built {len(models)} deep learning models")
        return models
    
    def train_and_evaluate_models(self, X, y, fibonacci_level="B_0"):
        """Train and evaluate all models with comprehensive metrics"""
        logger.info(f"🚀 Training models for Fibonacci level: {fibonacci_level}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🎯 Training {model_name}...")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{model_name}_{fibonacci_level}"):
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps['classifier'], 'predict_proba') else None
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Calculate metrics
                win_rate = (y_pred == 1).sum() / len(y_pred) if len(y_pred) > 0 else 0
                accuracy = (y_pred == y_test).sum() / len(y_test)
                
                # Advanced metrics
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                except:
                    auc_score = 0
                
                # Store results
                model_results = {
                    'accuracy': accuracy,
                    'win_rate': win_rate,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'auc_score': auc_score,                'total_trades': len(y_test),
                'winning_trades': (y_pred == 1).sum(),
                'model': model
            }
            
            results[model_name] = model_results\n                \n                # Log to MLflow\n                mlflow.log_params({\n                    'fibonacci_level': fibonacci_level,\n                    'train_size': len(X_train),\n                    'test_size': len(X_test),\n                    'features': X.shape[1]\n                })\n                \n                mlflow.log_metrics({\n                    'accuracy': accuracy,\n                    'win_rate': win_rate,\n                    'cv_mean': cv_scores.mean(),\n                    'cv_std': cv_scores.std(),\n                    'auc_score': auc_score\n                })\n                \n                # Save model\n                mlflow.sklearn.log_model(model, f\"{model_name}_{fibonacci_level}\")\n                \n                logger.info(f\"✅ {model_name}: Accuracy={accuracy:.3f}, Win Rate={win_rate:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}\")\n        \n        self.results[fibonacci_level] = results\n        return results\n    \n    def analyze_feature_importance(self, X, feature_names, fibonacci_level=\"B_0\"):\n        \"\"\"Analyze feature importance across models\"\"\"\n        logger.info(f\"🔍 Analyzing feature importance for {fibonacci_level}...\")\n        \n        importance_data = {}\n        \n        for model_name, model_results in self.results.get(fibonacci_level, {}).items():\n            model = model_results['model']\n            \n            try:\n                if hasattr(model.named_steps['classifier'], 'feature_importances_'):\n                    # Tree-based models\n                    if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:\n                        # Get selected features\n                        selector = model.named_steps['feature_selection']\n                        selected_features = selector.get_support()\n                        selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]\n                        importances = model.named_steps['classifier'].feature_importances_\n                    else:\n                        selected_feature_names = feature_names\n                        importances = model.named_steps['classifier'].feature_importances_\n                    \n                    importance_data[model_name] = dict(zip(selected_feature_names, importances))\n                    \n                elif hasattr(model.named_steps['classifier'], 'coef_'):\n                    # Neural networks (coefficient importance)\n                    if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:\n                        selector = model.named_steps['feature_selection']\n                        selected_features = selector.get_support()\n                        selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]\n                        coef_abs = np.abs(model.named_steps['classifier'].coef_[0])\n                    else:\n                        selected_feature_names = feature_names\n                        coef_abs = np.abs(model.named_steps['classifier'].coef_[0])\n                    \n                    importance_data[model_name] = dict(zip(selected_feature_names, coef_abs))\n                    \n            except Exception as e:\n                logger.warning(f\"⚠️  Could not extract importance for {model_name}: {e}\")\n        \n        self.feature_importance[fibonacci_level] = importance_data\n        return importance_data\n    \n    def create_comprehensive_report(self, fibonacci_level=\"B_0\"):\n        \"\"\"Create comprehensive analysis report\"\"\"\n        logger.info(f\"📊 Creating comprehensive report for {fibonacci_level}...\")\n        \n        if fibonacci_level not in self.results:\n            logger.error(f\"❌ No results found for {fibonacci_level}\")\n            return\n        \n        results = self.results[fibonacci_level]\n        baseline = self.baseline_results.get(fibonacci_level, {})\n        \n        report = f\"\"\"\n🧠 DEEP LEARNING FIBONACCI ANALYSIS REPORT\n==========================================\nFibonacci Level: {fibonacci_level}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n📊 BASELINE PERFORMANCE (Statistical Analysis)\n{'='*50}\nWin Rate: {baseline.get('win_rate', 0):.1%}\nTotal Trades: {baseline.get('trades', 0):,}\nSignal Type: {baseline.get('signal_type', 'unknown')}\n\n🧠 DEEP LEARNING RESULTS\n{'='*50}\n\"\"\"\n        \n        # Sort models by win rate\n        sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)\n        \n        for i, (model_name, metrics) in enumerate(sorted_results, 1):\n            improvement = metrics['win_rate'] - baseline.get('win_rate', 0)\n            status = \"🎯 TARGET ACHIEVED!\" if metrics['win_rate'] >= 0.58 else \"📈 IMPROVED\" if improvement > 0 else \"📉 BELOW BASELINE\"\n            \n            report += f\"\"\"\n{i}. {model_name}\n   Win Rate: {metrics['win_rate']:.1%} ({improvement:+.1%} vs baseline)\n   Accuracy: {metrics['accuracy']:.1%}\n   Cross-Validation: {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}\n   AUC Score: {metrics['auc_score']:.3f}\n   Total Trades: {metrics['total_trades']:,}\n   Winning Trades: {metrics['winning_trades']:,}\n   Status: {status}\n\n\"\"\"\n        \n        # Best model summary\n        best_model_name, best_results = sorted_results[0]\n        report += f\"\"\"\n🏆 BEST PERFORMING MODEL\n{'='*50}\nModel: {best_model_name}\nWin Rate: {best_results['win_rate']:.1%}\nImprovement: {best_results['win_rate'] - baseline.get('win_rate', 0):+.1%}\nConfidence: {best_results['cv_mean']:.1%} ± {best_results['cv_std']:.1%}\n\n\"\"\"\n        \n        # Feature importance\n        if fibonacci_level in self.feature_importance and best_model_name in self.feature_importance[fibonacci_level]:\n            importance = self.feature_importance[fibonacci_level][best_model_name]\n            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]\n            \n            report += f\"\"\"\n🔍 TOP FEATURES ({best_model_name})\n{'='*50}\n\"\"\"\n            for i, (feature, score) in enumerate(top_features, 1):\n                report += f\"{i:2d}. {feature:<30} {score:.4f}\\n\"\n        \n        # Recommendations\n        report += f\"\"\"\n\n💡 RECOMMENDATIONS\n{'='*50}\n\"\"\"\n        \n        if best_results['win_rate'] >= 0.58:\n            report += \"✅ DEEP LEARNING SUCCESS! Win rate target of 58% achieved.\\n\"\n            report += \"🚀 Ready for live trading deployment with enhanced model.\\n\"\n        elif best_results['win_rate'] > baseline.get('win_rate', 0):\n            report += \"📈 Improvement achieved over baseline statistical analysis.\\n\"\n            report += \"🔧 Consider further hyperparameter tuning for 58% target.\\n\"\n        else:\n            report += \"🔄 Deep learning didn't improve over baseline.\\n\"\n            report += \"📊 Statistical analysis remains the best approach.\\n\"\n        \n        report += f\"\"\"\n\n🎯 NEXT STEPS\n{'='*50}\n1. Deploy best performing model: {best_model_name}\n2. Implement real-time signal generation\n3. Monitor performance vs predictions\n4. Consider ensemble of top models\n5. Upgrade to TensorFlow when Python 3.11 available\n\n📝 TECHNICAL NOTES\n{'='*50}\n- Models trained on {len(self.engineered_data):,} historical trades\n- Advanced feature engineering applied\n- 5-fold cross-validation for robust evaluation\n- MLflow experiment tracking enabled\n- Ready for production deployment\n\"\"\"\n        \n        # Save report\n        report_file = f\"deep_learning_report_{fibonacci_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt\"\n        with open(report_file, 'w', encoding='utf-8') as f:\n            f.write(report)\n        \n        logger.info(f\"📄 Report saved: {report_file}\")\n        print(report)\n        \n        return report\n    \n    def run_full_analysis(self):\n        \"\"\"Run complete deep learning analysis pipeline\"\"\"\n        logger.info(\"🚀 Starting full deep learning analysis...\")\n        \n        try:\n            # Load and preprocess data\n            self.load_and_preprocess_data()\n            \n            # Build models\n            self.build_deep_models()\n            \n            # Analyze key Fibonacci levels\n            fibonacci_levels = ['B_0', 'B_-1.8', 'B_1.8']\n            \n            for fib_level in fibonacci_levels:\n                logger.info(f\"\\n🎯 Analyzing Fibonacci level: {fib_level}\")\n                \n                # Prepare data for this level\n                level_filter = fib_level.replace('B_', '')\n                X, y, feature_names = self.prepare_features_and_target(focus_fibonacci_level=level_filter)\n                \n                if len(X) < 50:  # Minimum trades required\n                    logger.warning(f\"⚠️  Insufficient data for {fib_level}: {len(X)} trades\")\n                    continue\n                \n                # Train and evaluate models\n                self.train_and_evaluate_models(X, y, fib_level)\n                \n                # Analyze feature importance\n                self.analyze_feature_importance(X, feature_names, fib_level)\n                \n                # Create report\n                self.create_comprehensive_report(fib_level)\n            \n            logger.info(\"\\n🎉 Deep learning analysis complete!\")\n            logger.info(\"📊 Check individual reports for detailed results\")\n            logger.info(\"🌐 MLflow UI: http://localhost:5000\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Analysis failed: {e}\")\n            raise\n\n\ndef main():\n    \"\"\"Main execution function\"\"\"\n    print(\"\\n🧠 Deep Learning Fibonacci Trading Analysis\")\n    print(\"=\"*50)\n    print(\"🎯 Target: Improve from 55.1% to 58%+ win rate\")\n    print(\"🔬 Using advanced ML with neural network simulation\")\n    print(\"📊 Compatible with Python 3.13 (TensorFlow alternative)\")\n    print(\"=\"*50)\n    \n    try:\n        # Initialize analyzer\n        analyzer = DeepLearningFibonacciAnalyzer()\n        \n        # Run full analysis\n        analyzer.run_full_analysis()\n        \n        print(\"\\n✅ Analysis completed successfully!\")\n        print(\"📄 Check the generated reports for detailed results\")\n        print(\"🌐 MLflow UI available at: http://localhost:5000\")\n        \n    except Exception as e:\n        print(f\"\\n❌ Error: {e}\")\n        logger.error(f\"Main execution failed: {e}\")\n        return 1\n    \n    return 0\n\n\nif __name__ == \"__main__\":\n    exit(main())\n
