"""
Model training pipeline with MLflow integration
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path

from ..utils.helpers import setup_logging
from ..evaluation.metrics import TradingMetrics

logger = setup_logging()

class ModelTrainer:
    """Model training pipeline with MLflow tracking"""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str = "trading_signal_prediction"):
        self.config = config
        self.experiment_name = experiment_name
        self.trading_metrics = TradingMetrics()
        
        # Set up MLflow
        mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', './mlruns'))
        mlflow.set_experiment(experiment_name)
        
        # Model registry
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
    
    def prepare_data(self, X: pd.DataFrame, y: pd.DataFrame, target_column: str = 'is_profitable') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info(f"Preparing data for target: {target_column}")
        
        # Get target variable
        if target_column not in y.columns:
            raise ValueError(f"Target column '{target_column}' not found in targets")
        
        y_target = y[target_column].values
        
        # Handle missing values
        X_clean = X.fillna(0)
        
        # Remove infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        # Split data using time-aware split (last 20% for testing)
        test_size = self.config.get('model', {}).get('test_size', 0.2)
        
        # Use simple split for now (can be enhanced with time-based split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_target, 
            test_size=test_size, 
            random_state=self.config.get('model', {}).get('random_state', 42),
            stratify=y_target if len(np.unique(y_target)) > 1 else None
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                     params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train XGBoost model with MLflow tracking"""
        
        with mlflow.start_run(run_name="XGBoost_Trading_Model"):
            logger.info("Training XGBoost model...")
            
            # Default parameters
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            if params:
                default_params.update(params)
            
            # Log parameters
            mlflow.log_params(default_params)
            
            # Train model
            model = xgb.XGBClassifier(**default_params)
            model.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=False)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_train = model.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_train))
            y_proba_test = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros(len(y_test))
            
            # Calculate metrics
            train_metrics = self.trading_metrics.calculate_classification_metrics(y_train, y_pred_train, y_proba_train)
            test_metrics = self.trading_metrics.calculate_classification_metrics(y_test, y_pred_test, y_proba_test)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log model
            mlflow.xgboost.log_model(model, "xgboost_model")
            
            # Save feature importance
            feature_importance.to_csv("feature_importance_xgb.csv", index=False)
            mlflow.log_artifact("feature_importance_xgb.csv")
            
            result = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'y_pred_test': y_pred_test,
                    'y_proba_test': y_proba_test
                }
            }
            
            self.models['xgboost'] = result
            logger.info(f"XGBoost training completed. Test accuracy: {test_metrics['accuracy']:.4f}")
            
            return result
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train LightGBM model with MLflow tracking"""
        
        with mlflow.start_run(run_name="LightGBM_Trading_Model"):
            logger.info("Training LightGBM model...")
            
            # Default parameters
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
            
            if params:
                default_params.update(params)
            
            # Log parameters
            mlflow.log_params(default_params)
            
            # Train model
            model = lgb.LGBMClassifier(**default_params)
            model.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_train = model.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_train))
            y_proba_test = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros(len(y_test))
            
            # Calculate metrics
            train_metrics = self.trading_metrics.calculate_classification_metrics(y_train, y_pred_train, y_proba_train)
            test_metrics = self.trading_metrics.calculate_classification_metrics(y_test, y_pred_test, y_proba_test)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log model
            mlflow.lightgbm.log_model(model, "lightgbm_model")
            
            # Save feature importance
            feature_importance.to_csv("feature_importance_lgb.csv", index=False)
            mlflow.log_artifact("feature_importance_lgb.csv")
            
            result = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'y_pred_test': y_pred_test,
                    'y_proba_test': y_proba_test
                }
            }
            
            self.models['lightgbm'] = result
            logger.info(f"LightGBM training completed. Test accuracy: {test_metrics['accuracy']:.4f}")
            
            return result
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train Random Forest model with MLflow tracking"""
        
        with mlflow.start_run(run_name="RandomForest_Trading_Model"):
            logger.info("Training Random Forest model...")
            
            # Default parameters
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            
            if params:
                default_params.update(params)
            
            # Log parameters
            mlflow.log_params(default_params)
            
            # Train model
            model = RandomForestClassifier(**default_params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_train = model.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_train))
            y_proba_test = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros(len(y_test))
            
            # Calculate metrics
            train_metrics = self.trading_metrics.calculate_classification_metrics(y_train, y_pred_train, y_proba_train)
            test_metrics = self.trading_metrics.calculate_classification_metrics(y_test, y_pred_test, y_proba_test)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            # Save feature importance
            feature_importance.to_csv("feature_importance_rf.csv", index=False)
            mlflow.log_artifact("feature_importance_rf.csv")
            
            result = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'y_pred_test': y_pred_test,
                    'y_proba_test': y_proba_test
                }
            }
            
            self.models['random_forest'] = result
            logger.info(f"Random Forest training completed. Test accuracy: {test_metrics['accuracy']:.4f}")
            
            return result
    
    def hyperparameter_tuning(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Dict[str, List], cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning with cross-validation"""
        
        with mlflow.start_run(run_name=f"{model_type}_Hyperparameter_Tuning"):
            logger.info(f"Starting hyperparameter tuning for {model_type}...")
            
            # Select base model
            if model_type == 'xgboost':
                base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            elif model_type == 'lightgbm':
                base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            elif model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Time series split for cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=tscv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Log best parameters and score
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
    
    def train_all_models(self, X: pd.DataFrame, y: pd.DataFrame, target_column: str = 'is_profitable') -> Dict[str, Any]:
        """Train all configured models"""
        logger.info("Starting training for all models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, target_column)
        
        results = {}
        
        # Train each model type
        model_configs = self.config.get('models', [])
        
        for model_config in model_configs:
            model_name = model_config['name']
            model_params = model_config.get('params', {})
            
            try:
                if model_name == 'xgboost':
                    # Use first parameter set for now (can be enhanced with hyperparameter tuning)
                    params = {key: values[0] if isinstance(values, list) else values 
                             for key, values in model_params.items()}
                    result = self.train_xgboost(X_train, y_train, X_test, y_test, params)
                    
                elif model_name == 'lightgbm':
                    params = {key: values[0] if isinstance(values, list) else values 
                             for key, values in model_params.items()}
                    result = self.train_lightgbm(X_train, y_train, X_test, y_test, params)
                    
                elif model_name == 'random_forest':
                    params = {key: values[0] if isinstance(values, list) else values 
                             for key, values in model_params.items()}
                    result = self.train_random_forest(X_train, y_train, X_test, y_test, params)
                
                results[model_name] = result
                
                # Update best model
                current_score = result['test_metrics']['accuracy']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_model = result['model']
                    logger.info(f"New best model: {model_name} with accuracy {current_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Compare models
        if results:
            comparison = self._compare_models(results)
            logger.info("Model comparison completed")
            logger.info(f"Best performing model: {comparison.iloc[0]['model']}")
        
        return results
    
    def _compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Compare model results"""
        comparison_data = []
        
        for model_name, result in results.items():
            row = {'model': model_name}
            
            # Add test metrics
            for metric, value in result['test_metrics'].items():
                row[f'test_{metric}'] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
        
        return comparison_df
    
    def save_best_model(self, filepath: str = None) -> str:
        """Save the best model to file"""
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        if filepath is None:
            filepath = f"models/best_trading_model.joblib"
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, filepath)
        logger.info(f"Best model saved to {filepath}")
        
        return filepath
