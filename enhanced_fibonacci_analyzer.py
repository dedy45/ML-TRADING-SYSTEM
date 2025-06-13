# Main Deep Learning Fibonacci Analyzer - Versi Terbaru dengan Anaconda & MLflow
# Orchestrator utama yang menggabungkan semua komponen

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import configurations dan utilities
from config.config import config
from utils.timeout_utils import ExecutionGuard, safe_timeout, TimeoutException
from utils.logging_utils import get_logger, PerformanceLogger
from data.data_processor import FibonacciDataProcessor
from core.model_trainer import FibonacciModelTrainer

# Setup logging
logger = get_logger('fibonacci_analyzer', config.logs_dir)
perf_logger = PerformanceLogger(logger)

class DeepLearningFibonacciAnalyzer:
    """
    Main Fibonacci Deep Learning Analyzer
    Versi terbaru dengan Anaconda environment support dan MLflow tracking
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"fibonacci_analysis_{int(time.time())}"
        self.data_processor = None
        self.model_trainer = None
        self.results = {}
        self.execution_start = time.time()
        
        # Validate environment
        self._validate_environment()
        
        logger.info("🧠 Deep Learning Fibonacci Analyzer initialized")
        logger.info(f"🔬 Experiment: {self.experiment_name}")
        logger.info(f"📊 Environment: {self._get_environment_summary()}")
    
    def _validate_environment(self):
        """Validate environment dan configuration"""
        logger.info("🔍 Validating environment...")
        
        # Validate config
        if not config.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Check Python environment
        env_info = config.get_environment_info()
        logger.info(f"Python version: {env_info['python_version']}")
        logger.info(f"Anaconda detected: {env_info['is_anaconda']}")
        
        if env_info['is_anaconda']:
            logger.info(f"Conda environment: {env_info['conda_env']}")
        
        # Test essential imports
        try:
            import numpy as np
            import pandas as pd
            import sklearn
            logger.info("✅ Core ML libraries available")
        except ImportError as e:
            logger.error(f"❌ Missing essential libraries: {e}")
            raise
        
        # Test MLflow
        try:
            import mlflow
            logger.info(f"✅ MLflow available: {mlflow.__version__}")
        except ImportError:
            logger.warning("⚠️ MLflow not available, tracking disabled")
    
    def _get_environment_summary(self) -> str:
        """Get environment summary"""
        env_info = config.get_environment_info()
        return f"Python {env_info['python_version']} {'(Anaconda)' if env_info['is_anaconda'] else ''}"
    
    def initialize_components(self):
        """Initialize data processor dan model trainer"""
        logger.info("🚀 Initializing components...")
        
        with safe_timeout(60, "Component initialization"):
            # Initialize data processor
            self.data_processor = FibonacciDataProcessor(
                data_path=config.data.data_path,
                max_workers=4
            )
            
            # Initialize model trainer
            self.model_trainer = FibonacciModelTrainer(
                experiment_name=self.experiment_name
            )
            
        logger.info("✅ Components initialized successfully")
    
    def load_and_process_data(self) -> tuple:
        """Load dan process data dengan timeout protection"""
        logger.info("📊 Loading and processing data...")
        perf_logger.start_timer("data_processing")
        
        with safe_timeout(config.model.data_timeout, "Data processing"):
            # Load data
            logger.info("Loading raw data...")
            raw_data = self.data_processor.load_data_parallel(
                max_files=config.data.max_files,
                max_rows_per_file=config.data.max_rows_per_file
            )
            
            logger.info(f"Raw data shape: {raw_data.shape}")
            
            # Feature engineering
            logger.info("Engineering features...")
            processed_data = self.data_processor.engineer_features(raw_data)
            
            # Get feature matrix
            X, y, feature_names = self.data_processor.get_feature_matrix(processed_data)
            
            # Validate data quality
            self._validate_data_quality(X, y)
            
        data_processing_time = perf_logger.end_timer("data_processing")
        
        logger.info(f"✅ Data processing completed in {data_processing_time:.1f}s")
        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_names
    
    def _validate_data_quality(self, X, y):
        """Validate data quality"""
        logger.info("🔍 Validating data quality...")
        
        # Check minimum samples
        if len(X) < config.data.min_samples_required:
            raise ValueError(f"Insufficient data: {len(X)} < {config.data.min_samples_required}")
        
        # Check feature quality
        if X.shape[1] < 5:
            logger.warning(f"Few features available: {X.shape[1]}")
        
        # Check target balance
        target_balance = y.mean()
        if target_balance < 0.3 or target_balance > 0.7:
            logger.warning(f"Imbalanced target: {target_balance:.1%} positive class")
        
        # Check for missing values
        missing_features = X.isnull().sum().sum()
        if missing_features > 0:
            logger.warning(f"Missing values in features: {missing_features}")
        
        logger.info("✅ Data quality validation completed")
    
    def train_models(self, X, y, feature_names) -> Dict[str, Any]:
        """Train models dengan comprehensive evaluation"""
        logger.info("🤖 Training machine learning models...")
        perf_logger.start_timer("model_training")
        
        with safe_timeout(config.model.max_execution_time, "Model training"):
            # Log experiment start
            logger.log_experiment_start(self.experiment_name, {
                'data_shape': X.shape,
                'feature_count': len(feature_names),
                'target_distribution': y.value_counts().to_dict(),
                'config': {
                    'max_execution_time': config.model.max_execution_time,
                    'cv_folds': config.model.cv_folds,
                    'target_win_rate': config.trading.target_win_rate
                }
            })
            
            # Train all models
            training_results = self.model_trainer.train_all_models(X, y)
            
            # Save models
            if training_results:
                models_dir = self.model_trainer.save_models()
                logger.info(f"Models saved to: {models_dir}")
            
        training_time = perf_logger.end_timer("model_training")
        
        logger.info(f"✅ Model training completed in {training_time:.1f}s")
        logger.info(f"Trained {len(training_results)} models successfully")
        
        return training_results
    
    def evaluate_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate dan analyze results"""
        logger.info("📊 Evaluating results...")
        
        if not training_results:
            logger.error("❌ No training results to evaluate")
            return {}
        
        evaluation = {
            'total_models': len(training_results),
            'best_model': self.model_trainer.best_model,
            'target_achieved': False,
            'recommendations': []
        }
        
        # Check if target achieved
        if self.model_trainer.best_model:
            best_score = self.model_trainer.best_model['score']
            evaluation['best_score'] = best_score
            evaluation['target_achieved'] = best_score >= config.trading.target_win_rate
            
            if evaluation['target_achieved']:
                evaluation['recommendations'].append("🎯 Target achieved! Ready for live deployment")
                logger.info(f"🎉 TARGET ACHIEVED! Best win rate: {best_score:.1%}")
            elif best_score >= config.trading.min_acceptable_win_rate:
                evaluation['recommendations'].append("✅ Acceptable performance, proceed with caution")
                logger.info(f"✅ Acceptable performance: {best_score:.1%}")
            else:
                evaluation['recommendations'].append("📈 Performance below target, needs improvement")
                logger.warning(f"📈 Below target performance: {best_score:.1%}")
        
        # Model distribution analysis
        model_performance = {}
        for name, result in training_results.items():
            model_performance[name] = {
                'high_conf_win_rate': result.get('high_conf_win_rate', 0),
                'accuracy': result.get('accuracy', 0),
                'auc_score': result.get('auc_score', 0)
            }
        
        evaluation['model_performance'] = model_performance
        
        # Additional recommendations
        if len(training_results) >= 3:
            evaluation['recommendations'].append("🔄 Consider ensemble methods")
        
        if any(r.get('high_conf_signals', 0) < 50 for r in training_results.values()):
            evaluation['recommendations'].append("📊 Consider collecting more data")
        
        return evaluation
    
    def generate_comprehensive_report(self, training_results: Dict[str, Any], 
                                    evaluation: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        logger.info("📄 Generating comprehensive report...")
        
        total_time = time.time() - self.execution_start
        
        report = f"""
🧠 DEEP LEARNING FIBONACCI ANALYSIS REPORT
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment: {self.experiment_name}
Environment: {self._get_environment_summary()}
Total Execution Time: {total_time:.1f} seconds

📊 EXECUTION SUMMARY
{'-'*60}
✅ Data Processing: Completed
✅ Feature Engineering: Completed  
✅ Model Training: Completed
✅ Model Evaluation: Completed
✅ Report Generation: Completed

📈 DATASET INFORMATION
{'-'*60}
Data Source: {config.data.data_path}
Files Processed: {config.data.max_files}
Total Samples: {len(self.data_processor.processed_data) if self.data_processor.processed_data is not None else 'N/A'}
Features Created: {len(self.data_processor.feature_names)}
Target Win Rate: {self.data_processor.processed_data['target'].mean():.1% if self.data_processor.processed_data is not None else 'N/A'}

🏆 MODEL PERFORMANCE
{'-'*60}
"""
        
        # Add model performance details
        if training_results:
            # Sort by high confidence win rate
            sorted_models = sorted(
                training_results.items(),
                key=lambda x: x[1].get('high_conf_win_rate', 0),
                reverse=True
            )
            
            for i, (name, result) in enumerate(sorted_models, 1):
                high_conf_wr = result.get('high_conf_win_rate', 0)
                high_conf_signals = result.get('high_conf_signals', 0)
                accuracy = result.get('accuracy', 0)
                
                status = "🎯 EXCELLENT" if high_conf_wr >= config.trading.target_win_rate else \
                        "✅ GOOD" if high_conf_wr >= config.trading.min_acceptable_win_rate else \
                        "📈 FAIR" if high_conf_wr >= 0.50 else "❌ POOR"
                
                report += f"""
{i}. {name.upper().replace('_', ' ')}
   High Confidence Win Rate: {high_conf_wr:.1%}
   High Confidence Signals: {high_conf_signals}
   Overall Accuracy: {accuracy:.1%}
   Status: {status}
"""
        
        # Add best model details
        if evaluation.get('best_model'):
            best_model = evaluation['best_model']
            report += f"""

🏆 BEST PERFORMING MODEL
{'-'*60}
Model: {best_model['name'].upper().replace('_', ' ')}
High Confidence Win Rate: {best_model['score']:.1%}
Target Achievement: {'🎯 YES' if evaluation['target_achieved'] else '❌ NO'}
"""
        
        # Add recommendations
        if evaluation.get('recommendations'):
            report += f"""

💡 RECOMMENDATIONS
{'-'*60}
"""
            for rec in evaluation['recommendations']:
                report += f"• {rec}\n"
        
        # Add technical details
        report += f"""

🔧 TECHNICAL CONFIGURATION
{'-'*60}
Target Win Rate: {config.trading.target_win_rate:.1%}
Min Acceptable Win Rate: {config.trading.min_acceptable_win_rate:.1%}
High Confidence Threshold: {config.trading.high_confidence_threshold:.1%}
Cross-Validation Folds: {config.model.cv_folds}
Max Execution Time: {config.model.max_execution_time} seconds
Random State: {config.model.random_state}

📁 OUTPUT FILES
{'-'*60}
Models Directory: {config.models_dir}
Logs Directory: {config.logs_dir}
MLflow Tracking URI: {config.mlflow.tracking_uri}
Experiment Name: {self.experiment_name}

🚀 NEXT STEPS
{'-'*60}
1. Review model performance above
2. Deploy best model if target achieved
3. Set up real-time inference server
4. Implement paper trading
5. Monitor live performance
6. Schedule periodic retraining

⚠️  IMPORTANT NOTES
{'-'*60}
• Always test in paper trading before live deployment
• Monitor model performance continuously
• Retrain periodically with new data
• Maintain proper risk management
• Keep detailed trading logs
"""
        
        return report
    
    def save_report(self, report: str) -> str:
        """Save report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = config.logs_dir / f"fibonacci_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_file}")
        return str(report_file)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete deep learning analysis pipeline"""
        logger.info("🚀 Starting complete Fibonacci deep learning analysis...")
        logger.info("="*80)
        
        analysis_results = {
            'success': False,
            'error': None,
            'execution_time': 0,
            'results': {},
            'report_file': None
        }
        
        try:
            with ExecutionGuard(config.model.max_execution_time) as guard:
                
                # 1. Initialize components
                guard.activity("Initializing components")
                self.initialize_components()
                
                # 2. Load and process data
                guard.activity("Loading and processing data")
                X, y, feature_names = self.load_and_process_data()
                
                # 3. Train models
                guard.activity("Training models")
                training_results = self.train_models(X, y, feature_names)
                
                # 4. Evaluate results
                guard.activity("Evaluating results")
                evaluation = self.evaluate_results(training_results)
                
                # 5. Generate report
                guard.activity("Generating report")
                report = self.generate_comprehensive_report(training_results, evaluation)
                
                # 6. Save report
                report_file = self.save_report(report)
                
                # Update results
                analysis_results.update({
                    'success': True,
                    'execution_time': time.time() - self.execution_start,
                    'results': {
                        'training_results': training_results,
                        'evaluation': evaluation,
                        'best_model': self.model_trainer.best_model
                    },
                    'report_file': report_file
                })
                
                # Log success
                logger.log_experiment_end(
                    self.experiment_name,
                    analysis_results['results'],
                    success=True
                )
                
                print("\n" + report)
                print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
                print(f"⏱️ Total time: {analysis_results['execution_time']:.1f} seconds")
                print(f"📄 Full report: {report_file}")
                
        except TimeoutException as e:
            error_msg = f"Analysis timed out: {e}"
            logger.error(error_msg)
            analysis_results.update({
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - self.execution_start
            })
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            analysis_results.update({
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - self.execution_start
            })
            
            # Log failure
            logger.log_experiment_end(
                self.experiment_name,
                {'error': error_msg},
                success=False
            )
        
        return analysis_results

def main():
    """Main execution function"""
    print("\n🧠 DEEP LEARNING FIBONACCI TRADING ANALYZER")
    print("=" * 80)
    print("🎯 Target: Achieve 58%+ win rate with deep learning")
    print("🔬 Advanced ML pipeline with MLflow tracking")
    print("📊 Optimized for Anaconda environment")
    print("=" * 80)
    
    try:
        # Create analyzer
        analyzer = DeepLearningFibonacciAnalyzer()
        
        # Run analysis
        results = analyzer.run_complete_analysis()
        
        if results['success']:
            print(f"\n✅ MISSION ACCOMPLISHED!")
            print(f"🏆 Best model achieved target performance")
            return 0
        else:
            print(f"\n❌ ANALYSIS FAILED")
            print(f"💡 Error: {results['error']}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        return 2
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit(main())
