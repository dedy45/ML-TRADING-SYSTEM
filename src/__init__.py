# Trading ML Package
"""
Trading Machine Learning Package for Signal Prediction

This package provides:
- Data loading and preprocessing for trading data
- Feature engineering for technical indicators
- ML model training with MLflow tracking
- Trading signal generation and evaluation
"""

__version__ = "1.0.0"
__author__ = "Trading ML Team"

# Make core components available at package level
try:
    from .utils.simple_config import simple_config
    from .data.simple_data_loader import SimpleDataLoader
    
    __all__ = [
        'simple_config',
        'SimpleDataLoader',
        '__version__',
        '__author__'
    ]
    
    print("✅ Trading ML package loaded successfully")
    
except ImportError as e:
    print(f"⚠️ Some components not available: {e}")
    __all__ = ['__version__', '__author__']
