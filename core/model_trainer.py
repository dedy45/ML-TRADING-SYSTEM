# Core machine learning model trainer dengan MLflow integration
# Robust training pipeline dengan timeout protection dan advanced ML

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.timeout_utils import safe_timeout, TimeoutException, ExecutionGuard, retry_with_timeout
from utils.logging_utils import get_logger, MLflowLogger, PerformanceLogger
from config.config import config

# MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = get_logger('model_trainer')

class FibonacciModelTrainer:
    """Advanced model trainer dengan MLflow tracking dan timeout protection"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or config.mlflow.experiment_name
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Setup logging
        self.mlflow_logger = MLflowLogger(logger) if MLFLOW_AVAILABLE else None
        self.perf_logger = PerformanceLogger(logger)
        
        logger.info(f"Model trainer initialized for experiment: {self.experiment_name}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping MLflow setup")
            return
            
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=config.mlflow.artifact_location,
                        tags=config.mlflow.default_tags
                    )
                    logger.info(f"Created MLflow experiment: {self.experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                    
                mlflow.set_experiment(self.experiment_name)
                
            except Exception as e:
                logger.error(f"Failed to setup MLflow experiment: {e}")
                
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
    
    def build_model_portfolio(self) -> Dict[str, Pipeline]:
        """Build portfolio of ML models optimized untuk Fibonacci trading"""
        logger.info("Building model portfolio...")
        
        models = {
            # Fast Neural Network (optimized untuk speed)
            'fast_neural_net': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=config.model.random_state
                ))
            ]),
            
            # Deep Neural Network
            'deep_neural_net': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    learning_rate='adaptive',
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=config.model.random_state
                ))
            ]),
            
            # Random Forest (robust baseline)
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    bootstrap=True,
                    n_jobs=config.model.n_jobs,
                    random_state=config.model.random_state
                ))
            ]),
            
            # Gradient Boosting (advanced ensemble)
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=config.model.random_state
                ))
            ]),
            
            # Logistic Regression (interpretable baseline)
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    solver='liblinear',
                    max_iter=1000,
                    random_state=config.model.random_state
                ))
            ]),
            
            # Support Vector Machine
            'svm_rbf': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,  # Untuk probability predictions
                    random_state=config.model.random_state
                ))
            ])
        }
        
        self.models = models
        logger.info(f"Built {len(models)} models in portfolio")
        return models
    
    def create_ensemble_model(self, base_models: Dict[str, Pipeline] = None) -> Pipeline:
        """Create ensemble model dari best performing models"""
        if base_models is None:
            base_models = self.trained_models
            
        if len(base_models) < 2:
            logger.warning("Insufficient models for ensemble, need at least 2")
            return None
            
        # Select top 3 models for ensemble
        model_scores = {}
        for name, model_data in self.results.items():
            if 'high_conf_win_rate' in model_data:
                model_scores[name] = model_data['high_conf_win_rate']
        
        if not model_scores:
            logger.warning("No model scores available for ensemble")
            return None
            
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        ensemble_estimators = []
        for model_name, score in top_models:
            if model_name in base_models:
                ensemble_estimators.append((model_name, base_models[model_name]))
        
        if len(ensemble_estimators) < 2:
            logger.warning("Insufficient trained models for ensemble")
            return None
            
        ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',  # Use probabilities
            n_jobs=config.model.n_jobs
        )
        
        ensemble_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ensemble', ensemble)
        ])
        
        logger.info(f"Created ensemble with {len(ensemble_estimators)} models")
        return ensemble_pipeline
    
    @retry_with_timeout(max_retries=2, timeout_per_attempt=300)
    def train_single_model(self, model_name: str, model: Pipeline, 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train single model dengan comprehensive evaluation"""
        
        with safe_timeout(config.model.model_timeout, f"Training {model_name}"):
            logger.info(f"Training {model_name}...")
            self.perf_logger.start_timer(f"train_{model_name}")
            
            # Start MLflow run
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"{model_name}_{int(time.time())}"):
                    results = self._train_and_evaluate_model(
                        model_name, model, X_train, y_train, X_test, y_test
                    )
            else:
                results = self._train_and_evaluate_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
            
            training_time = self.perf_logger.end_timer(f"train_{model_name}")
            results['training_time'] = training_time
            
            return results
    
    def _train_and_evaluate_model(self, model_name: str, model: Pipeline,
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Internal method untuk training dan evaluation"""
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=config.model.cv_folds, 
            scoring='accuracy',
            n_jobs=config.model.n_jobs
        )
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC score
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.5  # Random baseline
        
        # Trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(y_test, y_pred, y_pred_proba)
        
        # Compile results
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            **trading_metrics
        }
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and self.mlflow_logger:
            self._log_to_mlflow(model_name, model, results, X_train.shape)
        
        # Log performance
        logger.info(f"✅ {model_name} - Accuracy: {accuracy:.1%}, "
                   f"High Conf Win Rate: {trading_metrics['high_conf_win_rate']:.1%}")
        
        return results
    
    def _calculate_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        # High confidence predictions (>= 70% probability)
        high_conf_mask = y_pred_proba >= config.trading.high_confidence_threshold
        high_conf_count = np.sum(high_conf_mask)
        
        if high_conf_count > 0:
            high_conf_win_rate = np.mean(y_true.iloc[high_conf_mask] == 1)
            high_conf_accuracy = accuracy_score(y_true.iloc[high_conf_mask], y_pred[high_conf_mask])
        else:
            high_conf_win_rate = 0
            high_conf_accuracy = 0
        
        # Medium confidence predictions
        med_conf_mask = (y_pred_proba >= config.trading.medium_confidence_threshold) & \
                       (y_pred_proba < config.trading.high_confidence_threshold)
        med_conf_count = np.sum(med_conf_mask)
        
        if med_conf_count > 0:
            med_conf_win_rate = np.mean(y_true.iloc[med_conf_mask] == 1)
        else:
            med_conf_win_rate = 0
        
        # Overall win rate for positive predictions
        positive_pred_mask = y_pred == 1
        if np.sum(positive_pred_mask) > 0:
            positive_win_rate = np.mean(y_true.iloc[positive_pred_mask] == 1)
        else:
            positive_win_rate = 0
        
        # Signal distribution
        total_signals = len(y_pred)
        positive_signals = np.sum(y_pred == 1)
        signal_rate = positive_signals / total_signals if total_signals > 0 else 0
        
        return {
            'high_conf_win_rate': high_conf_win_rate,
            'high_conf_signals': high_conf_count,
            'high_conf_accuracy': high_conf_accuracy,
            'med_conf_win_rate': med_conf_win_rate,
            'med_conf_signals': med_conf_count,
            'positive_win_rate': positive_win_rate,
            'positive_signals': positive_signals,
            'signal_rate': signal_rate,
            'total_trades': total_signals
        }
    
    def _log_to_mlflow(self, model_name: str, model: Pipeline, 
                      results: Dict[str, Any], data_shape: Tuple[int, int]):
        """Log results to MLflow"""
        try:
            # Log parameters
            params = {
                'model_name': model_name,
                'data_shape': f"{data_shape[0]}x{data_shape[1]}",
                'cv_folds': config.model.cv_folds,
                'random_state': config.model.random_state
            }
            
            # Add model-specific parameters
            if hasattr(model.named_steps['classifier'], 'get_params'):
                model_params = model.named_steps['classifier'].get_params()
                # Limit parameter logging to avoid clutter
                key_params = ['hidden_layer_sizes', 'n_estimators', 'max_depth', 
                             'learning_rate', 'C', 'kernel']
                for param in key_params:
                    if param in model_params:
                        params[f'model_{param}'] = str(model_params[param])
            
            self.mlflow_logger.log_params(params)
            
            # Log metrics
            metrics = {k: v for k, v in results.items() 
                      if isinstance(v, (int, float)) and k != 'training_time'}
            self.mlflow_logger.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                f"model_{model_name}",
                registered_model_name=f"fibonacci_{model_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = None) -> Dict[str, Dict[str, Any]]:
        """Train all models in portfolio dengan parallel execution"""
        
        test_size = test_size or config.data.test_size
        
        logger.info("Starting comprehensive model training...")
        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        with ExecutionGuard(config.model.max_execution_time) as guard:
            
            # Split data
            guard.activity("Splitting data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=config.model.random_state,
                stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Build models if not already built
            if not self.models:
                guard.activity("Building model portfolio")
                self.build_model_portfolio()
            
            # Train models sequentially (more stable than parallel for this use case)
            results = {}
            
            for model_name, model in self.models.items():
                guard.activity(f"Training {model_name}")
                
                try:
                    result = self.train_single_model(
                        model_name, model, X_train, y_train, X_test, y_test
                    )
                    results[model_name] = result
                    self.trained_models[model_name] = result['model']
                    
                except TimeoutException:
                    logger.warning(f"⏱️ Training timeout for {model_name}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Training failed for {model_name}: {e}")
                    continue
            
            # Train ensemble if we have enough models
            if len(results) >= 2:
                guard.activity("Training ensemble model")
                try:
                    ensemble_model = self.create_ensemble_model()
                    if ensemble_model:
                        ensemble_result = self.train_single_model(
                            'ensemble', ensemble_model, X_train, y_train, X_test, y_test
                        )
                        results['ensemble'] = ensemble_result
                        self.trained_models['ensemble'] = ensemble_result['model']
                except Exception as e:
                    logger.error(f"Ensemble training failed: {e}")
            
            self.results = results
            
            # Identify best model
            if results:
                self._identify_best_model()
            
            return results
    
    def _identify_best_model(self):
        """Identify best performing model"""
        if not self.results:
            return
            
        # Sort by high confidence win rate (primary metric)
        model_scores = []
        for name, result in self.results.items():
            score = result.get('high_conf_win_rate', 0)
            signals = result.get('high_conf_signals', 0)
            # Penalize models with too few high confidence signals
            if signals < 10:  # Minimum threshold
                score *= 0.5
            model_scores.append((name, score, result))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        if model_scores:
            best_name, best_score, best_result = model_scores[0]
            self.best_model = {
                'name': best_name,
                'model': best_result['model'],
                'score': best_score,
                'result': best_result
            }
            
            logger.info(f"🏆 Best model: {best_name} "
                       f"(High Conf Win Rate: {best_score:.1%})")
    
    def save_models(self, save_dir: str = None) -> str:
        """Save trained models"""
        if not self.trained_models:
            raise ValueError("No trained models to save")
            
        save_dir = Path(save_dir) if save_dir else config.models_dir
        save_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        saved_files = []
        for name, model in self.trained_models.items():
            filename = f"fibonacci_{name}_{timestamp}.pkl"
            filepath = save_dir / filename
            joblib.dump(model, filepath)
            saved_files.append(str(filepath))
            logger.debug(f"Saved {name} model to: {filepath}")
        
        # Save results
        results_file = save_dir / f"training_results_{timestamp}.pkl"
        joblib.dump(self.results, results_file)
        saved_files.append(str(results_file))
        
        # Save best model separately
        if self.best_model:
            best_model_file = save_dir / f"fibonacci_best_model_{timestamp}.pkl"
            joblib.dump(self.best_model['model'], best_model_file)
            saved_files.append(str(best_model_file))
        
        logger.info(f"Saved {len(self.trained_models)} models to: {save_dir}")
        return str(save_dir)
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        if not self.results:
            return "No training results available"
            
        report = f"""
🧠 FIBONACCI DEEP LEARNING TRAINING REPORT
==========================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Experiment: {self.experiment_name}

📊 TRAINING SUMMARY
{'-'*50}
Models Trained: {len(self.results)}
Best Model: {self.best_model['name'] if self.best_model else 'None'}

🏆 MODEL PERFORMANCE RANKING
{'-'*50}
"""
        
        # Sort results by high confidence win rate
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('high_conf_win_rate', 0),
            reverse=True
        )
        
        for i, (name, result) in enumerate(sorted_results, 1):
            high_conf_wr = result.get('high_conf_win_rate', 0)
            high_conf_signals = result.get('high_conf_signals', 0)
            accuracy = result.get('accuracy', 0)
            auc = result.get('auc_score', 0)
            training_time = result.get('training_time', 0)
            
            status = "🎯 TARGET ACHIEVED!" if high_conf_wr >= config.trading.target_win_rate else \
                    "✅ GOOD" if high_conf_wr >= config.trading.min_acceptable_win_rate else \
                    "📈 ACCEPTABLE" if high_conf_wr >= 0.50 else \
                    "❌ POOR"
            
            report += f"""
{i}. {name.upper()}
   High Confidence Win Rate: {high_conf_wr:.1%}
   High Confidence Signals: {high_conf_signals}
   Overall Accuracy: {accuracy:.1%}
   AUC Score: {auc:.3f}
   Training Time: {training_time:.1f}s
   Status: {status}
"""
        
        # Add recommendations
        if self.best_model:
            best_score = self.best_model['score']
            target_achieved = best_score >= config.trading.target_win_rate
            
            report += f"""

💡 RECOMMENDATIONS
{'-'*50}
"""
            if target_achieved:
                report += f"""✅ TARGET ACHIEVED! {self.best_model['name']} reached {best_score:.1%} win rate
🚀 Ready for live trading deployment
📊 Continue monitoring performance
🔄 Consider ensemble methods for stability"""
            else:
                report += f"""📈 Best model: {self.best_model['name']} with {best_score:.1%} win rate
🔧 Recommended actions:
   - Collect more training data
   - Feature engineering optimization
   - Hyperparameter tuning
   - Try ensemble methods"""
        
        report += f"""

🎯 NEXT STEPS
{'-'*50}
1. Deploy best model: {self.best_model['name'] if self.best_model else 'None'}
2. Set up real-time inference server
3. Implement paper trading
4. Monitor live performance
5. Schedule model retraining

📝 TECHNICAL DETAILS
{'-'*50}
Target Win Rate: {config.trading.target_win_rate:.1%}
Min Acceptable: {config.trading.min_acceptable_win_rate:.1%}
High Confidence Threshold: {config.trading.high_confidence_threshold:.1%}
Models Directory: {config.models_dir}
MLflow Experiment: {self.experiment_name}
"""
        
        return report

# Export
__all__ = ['FibonacciModelTrainer']
