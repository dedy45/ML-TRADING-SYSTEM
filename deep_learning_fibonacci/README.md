# Deep Learning Fibonacci Trading Analysis

Advanced neural network implementation for Fibonacci trading signal detection with TensorFlow/Keras. Target: Improve win rate from 52% to 55-58%.

## Project Overview
This project upgrades the successful statistical Fibonacci analysis (52%+ win rate) using deep learning techniques for enhanced pattern recognition and signal accuracy.

## Key Achievements from Statistical Analysis
- **B_0 Level**: 52.4% win rate (3,106 trades) - Primary Signal
- **B_-1.8 Level**: 52.5% win rate (120 trades) - High Confidence
- **Europe Session**: Best performance (40.5% vs US 40.1% vs Asia 39.7%)
- **Risk Management**: Proven 2:1 TP/SL ratios

## Deep Learning Objectives
1. **Feature Enhancement**: Extract complex patterns from tick data using neural networks
2. **Sequence Learning**: LSTM networks for temporal dependencies in Fibonacci levels
3. **Pattern Recognition**: CNN for chart pattern identification
4. **Ensemble Methods**: Combine multiple models for robust predictions
5. **Real-time Processing**: Optimized inference for live trading signals

## Project Structure
```
deep_learning_fibonacci/
├── data/                    # Data management
│   ├── raw/                # Original 544 CSV files + tick data
│   ├── processed/          # Cleaned and engineered features
│   └── features/           # Deep learning feature sets
├── models/                 # Neural network architectures
│   ├── tensorflow/         # TensorFlow/Keras models
│   └── saved_models/       # Trained model artifacts
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data/              # Data processing utilities
│   ├── features/          # Feature engineering
│   ├── models/            # Model architectures
│   └── utils/             # Helper functions
├── experiments/           # MLflow experiment tracking
├── docs/                  # Documentation
├── configs/               # Configuration files
└── scripts/               # Training and inference scripts
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run data preprocessing: `python scripts/preprocess_data.py`
3. Train models: `python scripts/train_models.py`
4. Start MLflow UI: `mlflow ui`
5. Generate predictions: `python scripts/predict_signals.py`

## Deep Learning Models

### 1. LSTM Sequence Model
- **Purpose**: Capture temporal dependencies in Fibonacci level movements
- **Input**: Time series of price action around Fibonacci levels
- **Output**: Probability of successful trade signal

### 2. CNN Pattern Recognition
- **Purpose**: Identify visual chart patterns at Fibonacci levels
- **Input**: Candlestick data as image-like arrays
- **Output**: Pattern confidence scores

### 3. Ensemble Predictor
- **Purpose**: Combine LSTM and CNN predictions for robust signals
- **Method**: Weighted averaging with learned confidence weights
- **Target**: 55-58% win rate improvement

## Performance Targets
- **Current Baseline**: 52.4% win rate (B_0 level)
- **Target Improvement**: 55-58% win rate
- **Risk Management**: Maintain 2:1 TP/SL ratios
- **Processing Speed**: <100ms per signal for live trading

## Getting Started
See `docs/GETTING_STARTED.md` for detailed setup instructions.
