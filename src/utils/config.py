"""
Configuration module for trading ML pipeline
"""
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any

load_dotenv()

@dataclass
class DataConfig:
    """Data configuration"""
    backtest_path: str
    tick_path: str
    processed_path: str
    features_path: str

@dataclass
class ModelConfig:
    """Model configuration"""
    test_size: float
    validation_size: float
    random_state: int
    cv_folds: int

@dataclass
class TradingConfig:
    """Trading configuration"""
    min_profit_threshold: float
    max_loss_threshold: float
    risk_reward_ratio: float
    max_drawdown: float

@dataclass
class MLflowConfig:
    """MLflow configuration"""
    experiment_name: str
    tracking_uri: str

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        self.data = DataConfig(**config['data'])
        self.model = ModelConfig(**config['model'])
        self.trading = TradingConfig(**config['trading'])
        self.mlflow = MLflowConfig(**config['mlflow'])
        self.features = config['features']
        self.models = config['models']
        
    def get_env_var(self, key: str, default: str = None) -> str:
        """Get environment variable"""
        return os.getenv(key, default)
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.processed_path,
            self.data.features_path,
            "models",
            "experiments",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
# Global config instance
config = Config()
