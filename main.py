"""
Main pipeline for trading ML project
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import config
from src.utils.helpers import setup_logging, save_dataframe, load_dataframe
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.evaluation.metrics import TradingMetrics

class TradingMLPipeline:
    """Main trading ML pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Setup logging
        self.logger = setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize components
        self.data_loader = DataLoader(
            backtest_path=self.config['data']['backtest_path'],
            tick_path=self.config['data']['tick_path']
        )
        
        self.preprocessor = DataPreprocessor()
        
        self.feature_engineer = FeatureEngineer(
            config=self.config['features']
        )
        
        self.model_trainer = ModelTrainer(
            config=self.config,
            experiment_name=self.config['mlflow']['experiment_name']
        )
        
        self.trading_metrics = TradingMetrics()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.X = None
        self.y = None
        
        self.logger.info("Trading ML Pipeline initialized")
    
    def load_data(self, use_cached: bool = True, tick_limit: int = None) -> pd.DataFrame:
        """Load and cache data"""
        self.logger.info("Loading data...")
        
        processed_data_path = Path(self.config['data']['processed_path']) / "combined_backtest_data.csv"
        
        # Try to load cached data first
        if use_cached and processed_data_path.exists():
            self.logger.info("Loading cached processed data...")
            self.raw_data = load_dataframe(str(processed_data_path))
        else:
            # Load fresh data
            self.logger.info("Loading fresh data from source...")
            self.raw_data = self.data_loader.load_backtest_data()
            
            # Load tick data if available (optional)
            try:
                tick_data = self.data_loader.load_tick_data(limit_rows=tick_limit)
                if not tick_data.empty:
                    self.logger.info(f"Loaded {len(tick_data)} tick data records")
                    # Note: tick data integration can be added here if needed
            except Exception as e:
                self.logger.warning(f"Could not load tick data: {e}")
            
            # Save processed data
            save_dataframe(self.raw_data, str(processed_data_path))
            self.logger.info(f"Data cached to {processed_data_path}")
        
        # Data quality validation
        validation_report = self.data_loader.validate_backtest_data(self.raw_data)
        self.logger.info(f"Data validation report: {validation_report}")
        
        # Data summary
        summary = self.data_loader.get_data_summary(self.raw_data)
        self.logger.info(f"Data summary: {summary}")
        
        return self.raw_data
    
    def preprocess_data(self, use_cached: bool = True) -> pd.DataFrame:
        """Preprocess data"""
        self.logger.info("Preprocessing data...")
        
        if self.raw_data is None:
            self.load_data(use_cached)
        
        processed_data_path = Path(self.config['data']['processed_path']) / "preprocessed_data.csv"
        
        if use_cached and processed_data_path.exists():
            self.logger.info("Loading cached preprocessed data...")
            self.processed_data = load_dataframe(str(processed_data_path))
        else:
            # Preprocess data
            self.processed_data = self.preprocessor.preprocess_backtest_data(self.raw_data)
            
            # Handle missing values
            self.processed_data = self.preprocessor.handle_missing_values(
                self.processed_data, strategy='median'
            )
            
            # Remove outliers for critical columns
            outlier_columns = ['Profit', 'MAE_pips', 'MFE_pips']
            outlier_columns = [col for col in outlier_columns if col in self.processed_data.columns]
            if outlier_columns:
                self.processed_data = self.preprocessor.remove_outliers(
                    self.processed_data, outlier_columns, method='iqr'
                )
            
            # Save preprocessed data
            save_dataframe(self.processed_data, str(processed_data_path))
            self.logger.info(f"Preprocessed data cached to {processed_data_path}")
        
        self.logger.info(f"Preprocessed data shape: {self.processed_data.shape}")
        return self.processed_data
    
    def engineer_features(self, use_cached: bool = True) -> pd.DataFrame:
        """Engineer features"""
        self.logger.info("Engineering features...")
        
        if self.processed_data is None:
            self.preprocess_data(use_cached)
        
        features_data_path = Path(self.config['data']['features_path']) / "features_data.csv"
        
        if use_cached and features_data_path.exists():
            self.logger.info("Loading cached features data...")
            self.features_data = load_dataframe(str(features_data_path))
        else:
            # Engineer all features
            self.features_data = self.feature_engineer.engineer_all_features(self.processed_data)
            
            # Save features data
            save_dataframe(self.features_data, str(features_data_path))
            self.logger.info(f"Features data cached to {features_data_path}")
        
        self.logger.info(f"Features data shape: {self.features_data.shape}")
        self.logger.info(f"Total features created: {len(self.feature_engineer.feature_names)}")
        
        return self.features_data
    
    def prepare_model_data(self, target_columns: list = None) -> tuple:
        """Prepare data for modeling"""
        self.logger.info("Preparing data for modeling...")
        
        if self.features_data is None:
            self.engineer_features()
        
        if target_columns is None:
            target_columns = ['is_profitable', 'is_winning_trade']
        
        # Prepare features and targets
        self.X, self.y = self.preprocessor.prepare_features_targets(
            self.features_data, target_columns
        )
        
        self.logger.info(f"Model data prepared - Features: {self.X.shape}, Targets: {self.y.shape}")
        
        return self.X, self.y
    
    def train_models(self, target_column: str = 'is_profitable') -> dict:
        """Train all models"""
        self.logger.info(f"Training models for target: {target_column}")
        
        if self.X is None or self.y is None:
            self.prepare_model_data()
        
        # Train all models
        results = self.model_trainer.train_all_models(self.X, self.y, target_column)
        
        # Save best model
        if results:
            model_path = self.model_trainer.save_best_model()
            self.logger.info(f"Best model saved to {model_path}")
        
        return results
    
    def run_full_pipeline(self, use_cached: bool = True, target_column: str = 'is_profitable') -> dict:
        """Run the complete pipeline"""
        self.logger.info("Starting full trading ML pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data(use_cached=use_cached)
            
            # Step 2: Preprocess data
            self.preprocess_data(use_cached=use_cached)
            
            # Step 3: Engineer features
            self.engineer_features(use_cached=use_cached)
            
            # Step 4: Prepare model data
            self.prepare_model_data()
            
            # Step 5: Train models
            results = self.train_models(target_column=target_column)
            
            self.logger.info("Full pipeline completed successfully!")
            
            return {
                'success': True,
                'data_shape': self.features_data.shape,
                'feature_count': self.X.shape[1],
                'model_results': results,
                'best_model_score': self.model_trainer.best_score
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self, results: dict) -> dict:
        """Generate comprehensive report"""
        self.logger.info("Generating comprehensive report...")
        
        report = {
            'pipeline_summary': {
                'total_records': len(self.features_data) if self.features_data is not None else 0,
                'total_features': self.X.shape[1] if self.X is not None else 0,
                'data_date_range': {
                    'start': self.features_data['Timestamp'].min() if self.features_data is not None and 'Timestamp' in self.features_data.columns else None,
                    'end': self.features_data['Timestamp'].max() if self.features_data is not None and 'Timestamp' in self.features_data.columns else None
                }
            },
            'model_performance': results.get('model_results', {}),
            'best_model_score': results.get('best_model_score', 0),
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: dict) -> list:
        """Generate recommendations based on results"""
        recommendations = []
        
        best_score = results.get('best_model_score', 0)
        
        if best_score < 0.6:
            recommendations.append("Model accuracy is below 60%. Consider adding more features or collecting more data.")
        
        if best_score >= 0.7:
            recommendations.append("Good model performance achieved. Consider deploying for live testing.")
        
        if self.X is not None and self.X.shape[1] > 100:
            recommendations.append("High number of features detected. Consider feature selection to improve model interpretability.")
        
        recommendations.append("Monitor model performance over time and retrain periodically.")
        recommendations.append("Implement proper risk management when using model signals.")
        
        return recommendations

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = TradingMLPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(use_cached=True)
    
    # Generate report
    if results['success']:
        report = pipeline.generate_report(results)
        print("\n" + "="*50)
        print("TRADING ML PIPELINE REPORT")
        print("="*50)
        print(f"Total Records: {report['pipeline_summary']['total_records']}")
        print(f"Total Features: {report['pipeline_summary']['total_features']}")
        print(f"Best Model Score: {report['best_model_score']:.4f}")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        print("="*50)
    else:
        print(f"Pipeline failed: {results['error']}")

if __name__ == "__main__":
    main()
