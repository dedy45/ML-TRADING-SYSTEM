# Logging utilities untuk Deep Learning Fibonacci System
# Structured logging dengan MLflow integration

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback

class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class StructuredLogger:
    """Structured logger dengan MLflow integration"""
    
    def __init__(self, name: str, log_dir: str = "./logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler dengan colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler untuk persistent logging
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler untuk structured logs
        json_file = self.log_dir / f"{name}_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.json_handler = JsonFileHandler(json_file)
        self.logger.addHandler(self.json_handler)
        
        self.logger.info(f"Logger initialized: {name}")
        
    def info(self, message: str, extra: Dict[str, Any] = None):
        """Log info message"""
        self._log_with_extra(logging.INFO, message, extra)
        
    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Log debug message"""
        self._log_with_extra(logging.DEBUG, message, extra)
        
    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Log warning message"""
        self._log_with_extra(logging.WARNING, message, extra)
        
    def error(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log error message"""
        if exc_info:
            extra = extra or {}
            extra['traceback'] = traceback.format_exc()
        self._log_with_extra(logging.ERROR, message, extra)
        
    def critical(self, message: str, extra: Dict[str, Any] = None):
        """Log critical message"""
        self._log_with_extra(logging.CRITICAL, message, extra)
        
    def _log_with_extra(self, level: int, message: str, extra: Dict[str, Any] = None):
        """Log dengan extra metadata"""
        if extra:
            # Log extra data as JSON in structured log
            self.json_handler.log_structured(level, message, extra)
        self.logger.log(level, message)
        
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], 
                            hyperparameters: Dict[str, Any] = None):
        """Log model performance metrics"""
        performance_data = {
            'model_name': model_name,
            'metrics': metrics,
            'hyperparameters': hyperparameters or {},
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_performance'
        }
        
        self.info(f"Model {model_name} performance logged", performance_data)
        
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """Log experiment start"""
        experiment_data = {
            'experiment_name': experiment_name,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'event_type': 'experiment_start'
        }
        
        self.info(f"Experiment started: {experiment_name}", experiment_data)
        
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any], 
                          success: bool = True):
        """Log experiment end"""
        experiment_data = {
            'experiment_name': experiment_name,
            'results': results,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'event_type': 'experiment_end'
        }
        
        level = logging.INFO if success else logging.ERROR
        message = f"Experiment {'completed' if success else 'failed'}: {experiment_name}"
        self._log_with_extra(level, message, experiment_data)

class JsonFileHandler(logging.Handler):
    """Custom handler untuk JSON structured logging"""
    
    def __init__(self, filename: Union[str, Path]):
        super().__init__()
        self.filename = Path(filename)
        
    def emit(self, record):
        """Emit log record as JSON"""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.format_exception(record.exc_info)
                
            with open(self.filename, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
                
        except Exception:
            self.handleError(record)
            
    def log_structured(self, level: int, message: str, extra: Dict[str, Any]):
        """Log structured data"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': logging.getLevelName(level),
                'message': message,
                'data': extra
            }
            
            with open(self.filename, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
                f.write('\n')
                
        except Exception as e:
            print(f"Error writing structured log: {e}")

class MLflowLogger:
    """Logger integration dengan MLflow"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self._mlflow_available = self._check_mlflow()
        
    def _check_mlflow(self) -> bool:
        """Check if MLflow is available"""
        try:
            import mlflow
            return True
        except ImportError:
            self.logger.warning("MLflow not available, skipping MLflow logging")
            return False
            
    def log_params(self, params: Dict[str, Any], run_id: str = None):
        """Log parameters to MLflow"""
        if not self._mlflow_available:
            return
            
        try:
            import mlflow
            
            for key, value in params.items():
                mlflow.log_param(key, value)
                
            self.logger.info(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters to MLflow: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        if not self._mlflow_available:
            return
            
        try:
            import mlflow
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
                
            self.logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics to MLflow: {e}")
            
    def log_artifact(self, artifact_path: str, local_path: str = None):
        """Log artifact to MLflow"""
        if not self._mlflow_available:
            return
            
        try:
            import mlflow
            
            if local_path:
                mlflow.log_artifact(local_path, artifact_path)
            else:
                mlflow.log_artifact(artifact_path)
                
            self.logger.info(f"Logged artifact to MLflow: {artifact_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to log artifact to MLflow: {e}")

# Performance logging utilities
class PerformanceLogger:
    """Logger untuk performance monitoring"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
        
    def end_timer(self, operation: str, log_level: int = logging.INFO):
        """End timing and log duration"""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return
            
        duration = datetime.now() - self.start_times[operation]
        duration_seconds = duration.total_seconds()
        
        performance_data = {
            'operation': operation,
            'duration_seconds': duration_seconds,
            'event_type': 'performance_timing'
        }
        
        self.logger._log_with_extra(
            log_level, 
            f"Operation {operation} completed in {duration_seconds:.2f}s",
            performance_data
        )
        
        del self.start_times[operation]
        return duration_seconds

# Factory function untuk easy logger creation
def get_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> StructuredLogger:
    """Factory function untuk membuat structured logger"""
    return StructuredLogger(name, log_dir, level)

# Export
__all__ = [
    'StructuredLogger',
    'MLflowLogger', 
    'PerformanceLogger',
    'get_logger'
]
