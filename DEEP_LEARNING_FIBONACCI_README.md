# Enhanced Deep Learning Fibonacci Module

## Overview

The Enhanced Deep Learning Fibonacci Module is a robust, production-ready machine learning system designed for forex trading signal generation. It integrates with the MLFLOW infrastructure to provide timeout protection, comprehensive logging, MLflow tracking, and scalable model training.

## 🎯 Goals

- **58%+ Win Rate**: Target high-confidence trading signals with 58% or higher win rate
- **Timeout Protection**: Comprehensive protection against hanging issues
- **MLflow Tracking**: Full experiment tracking and model management
- **EA MQL5 Integration**: Ready for MetaTrader Expert Advisor integration
- **Robust Architecture**: Modular, scalable, and maintainable codebase

## 📁 Directory Structure

```
MLFLOW/
├── deep_learning_fibonacci/
│   ├── enhanced_tensorflow_fibonacci_predictor.py  # Enhanced version with MLFLOW
│   └── tensorflow_fibonacci_predictor.py           # Compatibility wrapper
├── config/
│   └── config.py                                   # Configuration management
├── utils/
│   ├── timeout_utils.py                           # Timeout protection
│   └── logging_utils.py                           # Structured logging
├── data/
│   └── data_processor.py                          # Data processing pipeline
├── core/
│   └── model_trainer.py                           # Model training utilities
├── models/                                         # Saved models
├── logs/                                          # System logs
├── run_integrated_deep_learning_fibonacci.py      # Main integration script
├── test_deep_learning_fibonacci.py                # Test script
└── setup_deep_learning_fibonacci.ps1              # Setup script
```

## 🚀 Quick Start

### 1. Environment Setup

Run the setup script (PowerShell):
```powershell
.\setup_deep_learning_fibonacci.ps1
```

Or manually install dependencies:
```bash
pip install numpy pandas scikit-learn mlflow joblib tensorflow
```

### 2. Test Installation

```python
python test_deep_learning_fibonacci.py
```

### 3. Run Analysis

```python
python run_integrated_deep_learning_fibonacci.py
```

## 📊 Features

### Enhanced Predictor Features

- **Timeout Protection**: 15-minute maximum execution time
- **MLflow Integration**: Automatic experiment tracking
- **Advanced Feature Engineering**: 15+ proven fibonacci features
- **Multiple ML Models**: Random Forest, Gradient Boosting, Neural Networks
- **High-Confidence Filtering**: Focus on 70%+ confidence signals
- **Trading-Specific Metrics**: Win rate, signal count, risk metrics

### Compatibility Features

- **Fallback Mode**: Works without MLFLOW infrastructure
- **Wrapper Compatibility**: Backwards compatible with existing code
- **Graceful Degradation**: Continues working with missing dependencies
- **Error Recovery**: Robust error handling and recovery

## 🔧 Configuration

### Basic Configuration

```python
from deep_learning_fibonacci.enhanced_tensorflow_fibonacci_predictor import EnhancedFibonacciDeepLearningPredictor

predictor = EnhancedFibonacciDeepLearningPredictor(
    data_path="../dataBT",           # Path to CSV data files
    model_save_path="models/",       # Where to save trained models
    experiment_name="my_experiment"  # MLflow experiment name
)
```

### Analysis Parameters

```python
results = predictor.run_complete_analysis(
    max_files=25,          # Number of CSV files to process
    max_rows_per_file=40,  # Rows per file (for speed control)
)
```

## 📈 Usage Examples

### Basic Analysis

```python
from deep_learning_fibonacci.tensorflow_fibonacci_predictor import FibonacciDeepLearningPredictor

# Initialize predictor
predictor = FibonacciDeepLearningPredictor()

# Run analysis
results = predictor.run_complete_analysis(
    max_files=20,
    max_rows_per_file=30
)

# Check results
if results:
    best_model = max(results.keys(), key=lambda k: results[k]['high_conf_win_rate'])
    win_rate = results[best_model]['high_conf_win_rate']
    print(f"Best model: {best_model} with {win_rate:.1%} win rate")
```

### Trading Signal Generation

```python
import pandas as pd

# Sample market data
market_data = pd.DataFrame({
    'LevelFibo': [0.0],
    'SessionEurope': [1],
    'TP': [20],
    'SL': [10]
})

# Generate signal
signal = predictor.generate_trading_signal(market_data)
print(f"Signal: {signal['signal_type']} with {signal['confidence']:.1%} confidence")
```

### Integration Script

```python
from run_integrated_deep_learning_fibonacci import DeepLearningFibonacciIntegration

# Run integrated analysis
integration = DeepLearningFibonacciIntegration()
results = integration.run_integrated_analysis(
    max_files=25,
    max_rows_per_file=40,
    target_win_rate=0.58
)
```

## 🛡️ Timeout Protection

The system includes multiple layers of timeout protection:

1. **Function-level timeouts**: Individual functions have specific timeouts
2. **Execution guards**: Additional protection during long operations
3. **Global timeouts**: Overall analysis timeout (15 minutes)
4. **Progressive timeouts**: Shorter timeouts for sub-operations

### Timeout Configuration

```python
from utils.timeout_utils import safe_timeout, ExecutionGuard

# Function decorator
@safe_timeout(timeout_seconds=300)
def my_function():
    # Your code here
    pass

# Execution guard
guard = ExecutionGuard(timeout_seconds=600)
guard.start()
try:
    # Long running operation
    if guard.should_stop():
        break  # Exit gracefully
finally:
    guard.stop()
```

## 📊 MLflow Integration

### Experiment Tracking

- **Parameters**: Model hyperparameters, data configuration
- **Metrics**: Win rates, accuracy, precision, recall, AUC
- **Artifacts**: Trained models, feature scalers, logs
- **Models**: Model registry with versioning

### Viewing Results

```bash
# Start MLflow UI
mlflow ui

# View at http://localhost:5000
```

### Programmatic Access

```python
import mlflow

# Get experiment
experiment = mlflow.get_experiment_by_name("fibonacci_analysis")

# Get runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Load model
model = mlflow.sklearn.load_model("models:/fibonacci_model/latest")
```

## 🧪 Testing

### Test Suite

```python
# Run all tests
python test_deep_learning_fibonacci.py

# Expected output:
# MLFLOW: ✅ PASS
# ENHANCED: ✅ PASS  
# WRAPPER: ✅ PASS
```

### Manual Testing

```python
# Test MLFLOW infrastructure
from config.config import config
from utils.timeout_utils import ExecutionGuard
print("✅ MLFLOW infrastructure working")

# Test enhanced predictor
from deep_learning_fibonacci.enhanced_tensorflow_fibonacci_predictor import EnhancedFibonacciDeepLearningPredictor
predictor = EnhancedFibonacciDeepLearningPredictor()
print("✅ Enhanced predictor working")

# Test wrapper
from deep_learning_fibonacci.tensorflow_fibonacci_predictor import FibonacciDeepLearningPredictor
wrapper = FibonacciDeepLearningPredictor()
print("✅ Wrapper working")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'mlflow'
```
**Solution**: Install missing dependencies
```bash
pip install mlflow pandas scikit-learn numpy
```

#### 2. Timeout Issues
```
TimeoutException: Analysis timed out
```
**Solution**: Reduce data size or increase timeout
```python
results = predictor.run_complete_analysis(
    max_files=10,          # Reduce from 25
    max_rows_per_file=20   # Reduce from 40
)
```

#### 3. Data Path Issues
```
FileNotFoundError: Data path not found
```
**Solution**: Check data path
```python
predictor = EnhancedFibonacciDeepLearningPredictor(
    data_path="path/to/your/dataBT"  # Correct path
)
```

#### 4. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce data size or use chunking
```python
# Use smaller datasets
max_files=5
max_rows_per_file=10
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
predictor = EnhancedFibonacciDeepLearningPredictor()
results = predictor.run_complete_analysis(max_files=3, max_rows_per_file=10)
```

### Log Analysis

Check the logs directory for detailed information:
```
logs/
├── fibonacci_analyzer.log       # Main analysis logs
├── deep_learning_fibonacci.log  # DL-specific logs
└── performance.log              # Performance metrics
```

## 📊 Performance Optimization

### Speed Optimization

1. **Reduce data size**: Use fewer files and rows for testing
2. **Use parallel processing**: Enable n_jobs=-1 in models
3. **Feature selection**: Focus on most important features
4. **Model selection**: Use faster models for development

### Memory Optimization

1. **Chunked processing**: Process data in smaller chunks
2. **Feature caching**: Cache engineered features
3. **Model pruning**: Remove unnecessary model complexity
4. **Garbage collection**: Explicit memory cleanup

### Example Configuration

```python
# Fast configuration for development
results = predictor.run_complete_analysis(
    max_files=5,           # Small dataset
    max_rows_per_file=20   # Limited rows
)

# Production configuration
results = predictor.run_complete_analysis(
    max_files=50,          # Full dataset
    max_rows_per_file=100  # Full rows
)
```

## 🔗 EA MQL5 Integration

### Model Export

Trained models are saved in standard formats:
- **Scikit-learn models**: `.pkl` files using joblib
- **Feature scalers**: `.pkl` files for preprocessing
- **Model metadata**: JSON files with model information

### Integration Steps

1. **Train and save model**:
```python
results = predictor.run_complete_analysis()
# Models saved to models/ directory
```

2. **Load model in Python service**:
```python
import joblib
model = joblib.load("models/best_fibonacci_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
```

3. **Create REST API** (example):
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process data and return prediction
    return jsonify(signal)
```

4. **MQL5 Integration**:
```cpp
// Use WebRequest to call Python API
string url = "http://localhost:5000/predict";
string data = "{'LevelFibo': 0.0, 'SessionEurope': 1}";
string response = WebRequest(url, data);
// Parse response and use signal
```

## 📊 Model Performance Metrics

### Trading-Specific Metrics

- **Win Rate**: Percentage of profitable trades
- **High Confidence Win Rate**: Win rate for signals with 70%+ confidence
- **Signal Count**: Number of high-confidence signals generated
- **Risk-Reward Ratio**: Average TP/SL ratio
- **Drawdown**: Maximum consecutive losses

### ML Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Performance Targets

- **Minimum Win Rate**: 55%
- **Target Win Rate**: 58%
- **Excellent Win Rate**: 60%+
- **Minimum Confidence**: 70%
- **Target Signal Count**: 100+ per month

## 🔄 Version History

### v2.0 (Current)
- Enhanced with MLFLOW infrastructure
- Comprehensive timeout protection
- MLflow experiment tracking
- Advanced feature engineering
- Production-ready architecture

### v1.0 (Legacy)
- Basic TensorFlow implementation
- Simple scikit-learn fallback
- Limited error handling
- Basic feature engineering

## 📞 Support

### Getting Help

1. **Check logs**: Review logs/ directory for error details
2. **Run tests**: Use test_deep_learning_fibonacci.py
3. **Reduce complexity**: Start with smaller datasets
4. **Check dependencies**: Ensure all packages are installed

### Best Practices

1. **Start small**: Begin with 3-5 files for testing
2. **Monitor timeouts**: Watch for timeout warnings
3. **Check data quality**: Ensure CSV files are valid
4. **Use version control**: Track model versions with MLflow
5. **Test thoroughly**: Validate models before live trading

### Performance Guidelines

- **Development**: max_files=5, max_rows_per_file=20
- **Testing**: max_files=15, max_rows_per_file=30
- **Production**: max_files=25-50, max_rows_per_file=40-100

---

## ✅ Summary

The Enhanced Deep Learning Fibonacci Module provides a robust, scalable, and production-ready solution for forex trading signal generation. With comprehensive timeout protection, MLflow tracking, and backwards compatibility, it's designed to achieve 58%+ win rates while preventing hanging issues.

**Key Benefits:**
- 🛡️ Timeout protection prevents hanging
- 📊 MLflow tracking for experiment management  
- 🎯 Targets 58%+ win rate for trading signals
- 🔄 Backwards compatible with existing code
- 🧪 Comprehensive testing and validation
- 📈 Ready for EA MQL5 integration

**Next Steps:**
1. Run setup script: `.\setup_deep_learning_fibonacci.ps1`
2. Test installation: `python test_deep_learning_fibonacci.py`
3. Run analysis: `python run_integrated_deep_learning_fibonacci.py`
4. Integrate with your EA MQL5 system
