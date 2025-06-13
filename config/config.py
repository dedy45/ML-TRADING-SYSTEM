# Configuration file untuk Deep Learning Fibonacci System
# Semua parameter sistem dikonfigurasi di sini

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    """Konfigurasi data processing"""
    # Path data
    data_path: str = "E:/aiml/MLFLOW/dataBT"
    backup_path: str = "./sample_data"
    
    # Limits untuk mencegah hang
    max_files: int = 30
    max_rows_per_file: int = 100
    max_total_rows: int = 3000
    
    # Preprocessing
    target_column: str = "Win"
    fibonacci_column: str = "LevelFibo"
    
    # Validation
    min_samples_required: int = 100
    test_size: float = 0.2
    validation_size: float = 0.1

@dataclass  
class ModelConfig:
    """Konfigurasi model training"""
    # Timeout settings (dalam detik)
    max_execution_time: int = 1800  # 30 menit
    model_timeout: int = 300        # 5 menit per model
    data_timeout: int = 180         # 3 menit untuk data loading
    
    # Model parameters
    cv_folds: int = 3               # Reduced untuk speed
    n_jobs: int = -1                # Gunakan semua cores
    random_state: int = 42
    
    # Neural network settings
    hidden_layers: List[tuple] = None
    max_iter: int = 500
    early_stopping: bool = True
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [(64, 32), (100, 50), (128, 64, 32)]

@dataclass
class MLflowConfig:
    """Konfigurasi MLflow tracking"""
    # MLflow settings
    tracking_uri: str = "./mlruns"
    experiment_name: str = "fibonacci_deep_learning"
    
    # Artifact storage
    artifact_location: str = "./mlflow_artifacts"
    
    # Logging settings
    log_models: bool = True
    log_artifacts: bool = True
    log_metrics: bool = True
    log_params: bool = True
    
    # Tags
    default_tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_tags is None:
            self.default_tags = {
                "project": "fibonacci_trading",
                "version": "2.0",
                "environment": "anaconda"
            }

@dataclass
class TradingConfig:
    """Konfigurasi trading strategy"""
    # Target performance
    target_win_rate: float = 0.58
    min_acceptable_win_rate: float = 0.55
    
    # Risk management
    default_risk_reward: float = 2.0
    max_position_size: float = 0.05  # 5% of account
    
    # Signal confidence thresholds
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.5
    
    # Fibonacci levels focus
    primary_levels: List[float] = None
    secondary_levels: List[float] = None
    
    def __post_init__(self):
        if self.primary_levels is None:
            self.primary_levels = [0.0, -1.8, 1.8]
        if self.secondary_levels is None:
            self.secondary_levels = [-3.6, 2.618, 4.236]

class SystemConfig:
    """Main configuration class"""
    
    def __init__(self):
        # Sub-configurations
        self.data = DataConfig()
        self.model = ModelConfig() 
        self.mlflow = MLflowConfig()
        self.trading = TradingConfig()
        
        # System paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.config_dir = self.project_root / "config"
        
        # Ensure directories exist
        self._create_directories()
        
        # Environment detection
        self.is_anaconda = self._detect_anaconda()
        self.python_version = self._get_python_version()
        
    def _create_directories(self):
        """Create required directories"""
        for directory in [self.models_dir, self.logs_dir, self.data_dir]:
            directory.mkdir(exist_ok=True)
            
    def _detect_anaconda(self) -> bool:
        """Detect if running in Anaconda environment"""
        return 'conda' in os.environ.get('CONDA_DEFAULT_ENV', '') or \
               'CONDA_PREFIX' in os.environ
               
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
    def get_environment_info(self) -> Dict:
        """Get environment information"""
        return {
            "python_version": self.python_version,
            "is_anaconda": self.is_anaconda,
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'Not detected'),
            "project_root": str(self.project_root),
            "mlflow_tracking_uri": self.mlflow.tracking_uri
        }
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Check data path
            data_path = Path(self.data.data_path)
            if not data_path.exists():
                print(f"⚠️  Primary data path tidak ditemukan: {data_path}")
                backup_path = Path(self.data.backup_path)
                if backup_path.exists():
                    print(f"✅ Menggunakan backup path: {backup_path}")
                    self.data.data_path = str(backup_path)
                else:
                    print("❌ Backup path juga tidak ditemukan")
                    return False
            
            # Check timeout values
            if self.model.max_execution_time <= 0:
                print("❌ Invalid max_execution_time")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False

# Global config instance
config = SystemConfig()

# Export untuk easy import
__all__ = ['config', 'SystemConfig', 'DataConfig', 'ModelConfig', 'MLflowConfig', 'TradingConfig']
