"""
Simple configuration for testing - without heavy dependencies
"""
import os
from pathlib import Path

class SimpleConfig:
    """Lightweight config for testing"""
    
    def __init__(self):
        # Data paths
        self.data_backtest_path = "./dataBT"
        self.data_tick_path = "./datatickxau"
        self.data_processed_path = "./data/processed"
        
        # Model settings
        self.test_size = 0.2
        self.random_state = 42
        
        # MLflow settings
        self.mlflow_experiment_name = "trading_signal_prediction"
        self.mlflow_tracking_uri = "./mlruns"
        
        # Data sampling (untuk testing)
        self.max_files_to_process = 5  # Hanya ambil 5 file pertama untuk testing
        self.sample_size = 1000  # Ambil 1000 baris per file
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_processed_path,
            "models",
            "experiments", 
            "logs",
            "mlruns"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory created: {directory}")

# Simple global config
simple_config = SimpleConfig()
