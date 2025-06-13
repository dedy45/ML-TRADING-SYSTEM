# Getting Started with Deep Learning Fibonacci Trading

This guide will help you set up and run the deep learning enhanced Fibonacci trading system that builds upon your successful 52%+ win rate analysis.

## 🎯 Project Overview

**Goal**: Improve Fibonacci trading signals from 52% to 55-58% win rate using deep learning

**Current Achievement**: Statistical analysis identified B_0 (52.4%) and B_-1.8 (52.5%) as top performing levels

**Deep Learning Enhancement**: Use TensorFlow/Keras for advanced pattern recognition and temporal analysis

## 📁 Project Structure

```
deep_learning_fibonacci/
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
├── quick_start.py           # Quick setup script
├── configs/
│   └── config.yaml          # Main configuration
├── data/                    # Data management
│   ├── raw/                # Original CSV files
│   ├── processed/          # Cleaned data
│   └── features/           # ML-ready features
├── src/                     # Source code
│   ├── data/
│   │   └── preprocessing.py # Data processing
│   ├── models/
│   │   └── architectures.py # Neural networks
│   └── utils/
│       └── metrics.py       # Trading metrics
├── scripts/
│   └── train_models.py      # Training pipeline
├── models/                  # Saved models
├── experiments/            # MLflow experiments
├── docs/                   # Documentation
├── logs/                   # Training logs
└── reports/                # Performance reports
```

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Setup
```bash
cd E:\aiml\MLFLOW\deep_learning_fibonacci
python quick_start.py
```

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow Server**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Run Quick Training (10 files)**
   ```bash
   python scripts/train_models.py --max-files 10
   ```

4. **View Results**
   - Open browser: http://localhost:5000
   - Check experiments and metrics

## 📊 Training Pipeline

### Phase 1: Data Preprocessing
```python
# Process 544 CSV files + 2.7GB tick data
# Engineer Fibonacci-specific features
# Create sequences for LSTM and images for CNN
python -c "from src.data.preprocessing import FibonacciDataProcessor; FibonacciDataProcessor().process_fibonacci_data()"
```

### Phase 2: Model Training
```python
# Train LSTM for temporal patterns
# Train CNN for chart pattern recognition  
# Create ensemble for optimal predictions
python scripts/train_models.py
```

### Phase 3: Evaluation & Deployment
- Compare with 52% baseline win rate
- Generate performance reports
- Deploy best performing model

## 🏗️ Model Architecture

### 1. LSTM Sequence Model
**Purpose**: Capture temporal dependencies in Fibonacci levels
- Input: 60-step sequences of trading features
- Architecture: 2 LSTM layers + dense layers
- Output: Signal strength, direction, confidence

### 2. CNN Pattern Recognition
**Purpose**: Visual pattern recognition in candlestick charts
- Input: 32x32x4 OHLC image representations
- Architecture: 3 conv layers + global pooling
- Output: Pattern confidence score

### 3. Ensemble Model
**Purpose**: Combine LSTM + CNN for robust predictions
- Method: Learned weighted averaging
- Target: 55-58% win rate improvement

## 📈 Key Features

### Fibonacci Level Analysis
- **B_0 Level**: Primary signal (52.4% baseline)
- **B_-1.8 Level**: High confidence (52.5% baseline)
- **B_1.8 Level**: Secondary signal (45.9% baseline)

### Trading Session Optimization
- **Europe Session**: Best performance (40.5%)
- **US Session**: Good performance (40.1%)
- **Asia Session**: Acceptable (39.7%)

### Risk Management
- **TP/SL Ratio**: 2:1 (proven from analysis)
- **Position Size**: 1-2% of account
- **Max Trades/Day**: 5
- **Confidence Threshold**: 70%

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Target Performance
targets:
  win_rate_improvement: 0.055  # 55% target
  win_rate_stretch: 0.058      # 58% stretch goal

# Model Settings
models:
  lstm:
    sequence_length: 60
    layers: [128, 64, 32, 16]
  
  cnn:
    input_shape: [32, 32, 4]
    filters: [32, 64, 128]

# Training
training:
  batch_size: 128
  epochs: 100
  learning_rate: 0.001
```

## 📊 Monitoring & Evaluation

### MLflow Dashboard
- Real-time training metrics
- Model comparison
- Experiment tracking
- Artifact management

### Trading Metrics
- **Win Rate**: Primary performance indicator
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Risk assessment
- **Expected Return**: Per-trade profitability

### Performance Reports
Automatic generation includes:
- Model comparison vs baseline
- Fibonacci level breakdown
- Session performance analysis
- Risk metrics assessment

## 🎯 Expected Results

### Baseline (Statistical Analysis)
- **B_0 Win Rate**: 52.4% (3,106 trades)
- **B_-1.8 Win Rate**: 52.5% (120 trades)
- **Overall Performance**: Profitable with 2:1 TP/SL

### Deep Learning Target
- **Improved Win Rate**: 55-58%
- **Better Signal Quality**: Higher confidence predictions
- **Reduced False Positives**: More selective signals
- **Enhanced Risk Management**: Dynamic position sizing

## 🔄 Next Steps After Training

1. **Model Validation**
   - Backtest on unseen data
   - Compare with baseline results
   - Validate on different market conditions

2. **Production Deployment**
   - Real-time inference pipeline
   - Automated signal generation
   - Performance monitoring

3. **Live Trading Integration**
   - Connect to trading platform
   - Implement risk management
   - Set up alerts and notifications

## 🆘 Troubleshooting

### Common Issues

**Memory Error with Tick Data**
```python
# Reduce chunk size in config.yaml
processing:
  chunk_size: 5000  # Reduce from 10000
  memory_limit_gb: 4  # Reduce limit
```

**MLflow Connection Error**
```bash
# Restart MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

**TensorFlow GPU Issues**
```python
# Disable GPU if issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Performance Optimization

**Speed Up Training**
- Use fewer files for testing (`--max-files 10`)
- Reduce sequence length in config
- Use smaller batch sizes

**Improve Accuracy**
- Increase training epochs
- Add more feature engineering
- Tune hyperparameters

## 📚 Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **TensorFlow Guide**: https://www.tensorflow.org/guide
- **Trading Strategy Guide**: `../COMPLETE_TRADING_STRATEGY.md`
- **Baseline Analysis**: `../FIBONACCI_ANALYSIS_FINAL_RESULTS.md`

## 🎉 Success Criteria

✅ **Training Complete** when:
- Models trained without errors
- Win rate > 55% achieved
- MLflow experiments logged
- Performance reports generated

✅ **Ready for Production** when:
- Ensemble model shows consistent 55%+ win rate
- Low false positive rate maintained
- Risk metrics within acceptable ranges
- Real-time inference < 100ms

---

**Remember**: Start with `python quick_start.py` for the fastest setup and testing!
