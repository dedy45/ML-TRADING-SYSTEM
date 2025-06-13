"""
Deep Learning Training Script for Fibonacci Trading Models
Main training pipeline with MLflow experiment tracking
"""

import os
import sys
import yaml
import logging
import numpy as np
import mlflow
import mlflow.tensorflow
from pathlib import Path
import h5py
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import FibonacciDataProcessor
from src.models.architectures import create_models, ModelTrainer
from src.utils.metrics import FibonacciMetrics
import tensorflow as tf

# Set GPU memory growth to avoid allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

class FibonacciTrainingPipeline:
    """
    Complete training pipeline for deep learning Fibonacci models.
    Integrates data processing, model training, and MLflow tracking.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize training pipeline."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_processor = FibonacciDataProcessor(config_path)
        self.metrics = FibonacciMetrics(self.config)
        self.trainer = ModelTrainer(self.config)
        
        # Setup MLflow
        self._setup_mlflow()
        
    def _load_config(self) -> dict:
        """Load configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.logger.info(f"MLflow tracking URI: {self.config['mlflow']['tracking_uri']}")
        self.logger.info(f"Experiment: {self.config['mlflow']['experiment_name']}")
    
    def prepare_data(self, max_files: int = None) -> dict:
        """
        Prepare data for training.
        
        Args:
            max_files: Limit number of files to process (for testing)
            
        Returns:
            Dictionary with train/val/test splits for both sequence and image data
        """
        self.logger.info("=== Data Preparation Phase ===")
        
        # Check if processed features exist
        seq_features_path = Path("data/features/sequence_features.h5")
        img_features_path = Path("data/features/image_features.h5")
        
        if seq_features_path.exists() and img_features_path.exists():
            self.logger.info("Loading existing processed features...")
            X_seq, y_seq = self.data_processor.load_features("sequence")
            X_img, y_img = self.data_processor.load_features("image")
        else:
            self.logger.info("Processing raw data...")
            
            # Process raw CSV files
            df = self.data_processor.process_fibonacci_data(max_files=max_files)
            
            # Engineer features
            df = self.data_processor.engineer_fibonacci_features(df)
            
            # Create sequence and image features
            X_seq, y_seq = self.data_processor.create_sequence_features(df)
            X_img, y_img = self.data_processor.create_image_features(df)
            
            # Save features
            self.data_processor.save_features(X_seq, y_seq, X_img, y_img)
        
        # Create train/val/test splits
        seq_splits = self.data_processor.prepare_train_test_split(X_seq, y_seq)
        img_splits = self.data_processor.prepare_train_test_split(X_img, y_img)
        
        # Log data statistics
        self.logger.info(f"Sequence data shape: {X_seq.shape}")
        self.logger.info(f"Image data shape: {X_img.shape}")
        self.logger.info(f"Positive samples: {np.sum(y_seq)}/{len(y_seq)} ({np.mean(y_seq):.1%})")
        
        return {
            'sequence': seq_splits,
            'image': img_splits,
            'sequence_shape': X_seq.shape[1:],
            'image_shape': X_img.shape[1:]
        }
    
    def train_lstm_model(self, data: dict) -> tuple:
        """Train LSTM model."""
        self.logger.info("=== LSTM Model Training ===")
        
        with mlflow.start_run(run_name="LSTM_Training", nested=True):
            # Log parameters
            mlflow.log_params(self.config['models']['lstm'])
            mlflow.log_params(self.config['training'])
            
            # Create and build LSTM model
            lstm_model, _, _ = create_models(self.config_path)
            model = lstm_model.build_model(data['sequence_shape'])
            
            # Log model summary
            model.summary()
            
            # Train model
            history = self.trainer.train_lstm(
                lstm_model,
                data['sequence']['X_train'], data['sequence']['y_train'],
                data['sequence']['X_val'], data['sequence']['y_val']
            )
            
            # Evaluate on test set
            test_results = model.evaluate(
                data['sequence']['X_test'], 
                {
                    'main_output': data['sequence']['y_test'],
                    'signal_strength': data['sequence']['y_test'],
                    'direction': (data['sequence']['y_test'] - 0.5) * 2,
                    'confidence': data['sequence']['y_test']
                },
                verbose=0
            )
            
            # Log metrics
            mlflow.log_metric("test_accuracy", test_results[5])  # main_output_accuracy
            mlflow.log_metric("test_loss", test_results[0])
            
            # Log training history
            for epoch, acc in enumerate(history.history['main_output_accuracy']):
                mlflow.log_metric("train_accuracy", acc, step=epoch)
            for epoch, val_acc in enumerate(history.history['val_main_output_accuracy']):
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Save model
            model_path = "models/saved_models/fibonacci_lstm_final.h5"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            self.logger.info(f"LSTM training complete. Test accuracy: {test_results[5]:.4f}")
            
            return lstm_model, history
    
    def train_cnn_model(self, data: dict) -> tuple:
        """Train CNN model."""
        self.logger.info("=== CNN Model Training ===")
        
        with mlflow.start_run(run_name="CNN_Training", nested=True):
            # Log parameters
            mlflow.log_params(self.config['models']['cnn'])
            mlflow.log_params(self.config['training'])
            
            # Create and build CNN model
            _, cnn_model, _ = create_models(self.config_path)
            model = cnn_model.build_model(data['image_shape'])
            
            # Log model summary
            model.summary()
            
            # Train model
            history = self.trainer.train_cnn(
                cnn_model,
                data['image']['X_train'], data['image']['y_train'],
                data['image']['X_val'], data['image']['y_val']
            )
            
            # Evaluate on test set
            test_results = model.evaluate(
                data['image']['X_test'], data['image']['y_test'],
                verbose=0
            )
            
            # Log metrics
            mlflow.log_metric("test_accuracy", test_results[1])
            mlflow.log_metric("test_loss", test_results[0])
            
            # Log training history
            for epoch, acc in enumerate(history.history['accuracy']):
                mlflow.log_metric("train_accuracy", acc, step=epoch)
            for epoch, val_acc in enumerate(history.history['val_accuracy']):
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Save model
            model_path = "models/saved_models/fibonacci_cnn_final.h5"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            self.logger.info(f"CNN training complete. Test accuracy: {test_results[1]:.4f}")
            
            return cnn_model, history
    
    def train_ensemble_model(self, lstm_model, cnn_model, data: dict) -> tuple:
        """Train ensemble model."""
        self.logger.info("=== Ensemble Model Training ===")
        
        with mlflow.start_run(run_name="Ensemble_Training", nested=True):
            # Log parameters
            mlflow.log_params(self.config['models']['ensemble'])
            
            # Create and build ensemble model
            _, _, ensemble_model = create_models(self.config_path)
            model = ensemble_model.build_ensemble(
                lstm_model.model, cnn_model.model,
                data['sequence_shape'], data['image_shape']
            )
            
            # Train ensemble (fine-tuning)
            history = model.fit(
                [data['sequence']['X_train'], data['image']['X_train']], 
                data['sequence']['y_train'],
                batch_size=self.config['training']['batch_size'],
                epochs=20,  # Shorter training for ensemble
                validation_data=(
                    [data['sequence']['X_val'], data['image']['X_val']], 
                    data['sequence']['y_val']
                ),
                verbose=1
            )
            
            # Evaluate on test set
            test_results = model.evaluate(
                [data['sequence']['X_test'], data['image']['X_test']], 
                data['sequence']['y_test'],
                verbose=0
            )
            
            # Log metrics
            mlflow.log_metric("test_accuracy", test_results[1])
            mlflow.log_metric("test_loss", test_results[0])
            
            # Calculate win rate improvement
            baseline_win_rate = self.config['baseline']['b_0_level']['win_rate']
            improvement = test_results[1] - baseline_win_rate
            mlflow.log_metric("win_rate_improvement", improvement)
            
            # Save model
            model_path = "models/saved_models/fibonacci_ensemble_final.h5"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            self.logger.info(f"Ensemble training complete. Test accuracy: {test_results[1]:.4f}")
            self.logger.info(f"Win rate improvement: {improvement:+.3f} ({improvement/baseline_win_rate:+.1%})")
            
            return ensemble_model, history
    
    def run_complete_training(self, max_files: int = None):
        """Run complete training pipeline."""
        self.logger.info("🚀 Starting Deep Learning Fibonacci Training Pipeline")
        self.logger.info(f"Timestamp: {datetime.now()}")
        
        with mlflow.start_run(run_name="Complete_Training_Pipeline"):
            # Log configuration
            mlflow.log_params({
                "max_files": max_files,
                "target_win_rate": self.config['targets']['win_rate_improvement'],
                "baseline_win_rate": self.config['baseline']['b_0_level']['win_rate']
            })
            
            try:
                # Step 1: Prepare data
                data = self.prepare_data(max_files=max_files)
                
                # Step 2: Train LSTM model
                lstm_model, lstm_history = self.train_lstm_model(data)
                
                # Step 3: Train CNN model  
                cnn_model, cnn_history = self.train_cnn_model(data)
                
                # Step 4: Train ensemble model
                ensemble_model, ensemble_history = self.train_ensemble_model(
                    lstm_model, cnn_model, data
                )
                
                # Step 5: Generate final report
                self._generate_training_report(data, lstm_history, cnn_history, ensemble_history)
                
                self.logger.info("✅ Training pipeline completed successfully!")
                
            except Exception as e:
                self.logger.error(f"❌ Training pipeline failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise
    
    def _generate_training_report(self, data, lstm_history, cnn_history, ensemble_history):
        """Generate comprehensive training report."""
        report_path = Path("reports")
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Deep Learning Fibonacci Training Report\n\n")
            f.write(f"**Training Date:** {datetime.now()}\n\n")
            
            f.write("## Data Summary\n")
            f.write(f"- Sequence data shape: {data['sequence']['X_train'].shape}\n")
            f.write(f"- Image data shape: {data['image']['X_train'].shape}\n")
            f.write(f"- Training samples: {len(data['sequence']['X_train'])}\n")
            f.write(f"- Validation samples: {len(data['sequence']['X_val'])}\n")
            f.write(f"- Test samples: {len(data['sequence']['X_test'])}\n\n")
            
            f.write("## Model Performance\n")
            f.write(f"- LSTM final accuracy: {max(lstm_history.history['val_main_output_accuracy']):.4f}\n")
            f.write(f"- CNN final accuracy: {max(cnn_history.history['val_accuracy']):.4f}\n")
            f.write(f"- Ensemble final accuracy: {max(ensemble_history.history['val_accuracy']):.4f}\n\n")
            
            f.write("## Baseline Comparison\n")
            baseline = self.config['baseline']['b_0_level']['win_rate']
            f.write(f"- Baseline win rate (B_0 level): {baseline:.1%}\n")
            f.write(f"- Target win rate: {self.config['targets']['win_rate_improvement']:.1%}\n\n")
            
            f.write("## Next Steps\n")
            f.write("1. Deploy best performing model\n")
            f.write("2. Set up real-time inference pipeline\n")
            f.write("3. Implement monitoring and alerts\n")
            f.write("4. Start live trading validation\n")
        
        mlflow.log_artifact(str(report_file))
        self.logger.info(f"Training report saved: {report_file}")

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Deep Learning Fibonacci Models')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize and run training pipeline
    pipeline = FibonacciTrainingPipeline(args.config)
    pipeline.run_complete_training(max_files=args.max_files)

if __name__ == "__main__":
    main()
