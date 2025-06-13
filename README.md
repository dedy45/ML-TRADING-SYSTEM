# ML-TRADING-SIGNALS
PROYEK Sistem Trading ML dengan Anaconda &amp; MLflow
# Trading Machine Learning Pipeline with MLflow

This project is a complete machine learning system for trading data analysis and signal prediction using MLflow for experiment tracking. Specifically designed for XAUUSD backtest data with focus on high win rate probabilities, clear risk-reward ratios, and long-term stability.

## 🎯 **Project Goals**

1. **High Win Rate Probability**: Identify trading patterns with win rate > 60%
2. **Clear Risk Reward**: Optimize risk-reward ratios for maximum profitability  
3. **Long-term Stability**: Consistent and reliable models
4. **Quality Signal Output**: Generate actionable trading signals

## ✅ **Current Status**

- ✅ **Advanced Pipeline Fixed**: All indentation errors resolved
- ✅ **MLflow UI Running**: Experiment tracking at http://127.0.0.1:5000
- ✅ **Multiple Models**: Random Forest, Gradient Boosting, Logistic Regression
- ✅ **Feature Engineering**: 12+ trading-specific features implemented
- ✅ **Experiment Tracking**: Full MLflow integration working
- ✅ **Easy-to-Use Scripts**: Beginner-friendly automation tools

## 🚀 **Quick Start - Choose Your Method**

### Method 1: Windows Batch File (Easiest for Beginners)
```batch
# Double-click this file in Windows Explorer
start_experiments.bat
```

### Method 2: Interactive Python Script
```bash
python run_experiments.py
```

### Method 3: Direct Pipeline Execution
```bash
# Simple pipeline (basic ML)
python simple_ml_pipeline.py

# Advanced pipeline with MLflow
python -c "from advanced_ml_pipeline import AdvancedTradingPipeline; p=AdvancedTradingPipeline(); p.run_complete_pipeline(10)"
```

### Method 4: Check Project Status
```bash
python check_status.py
```

## 📊 **View Your Results**

1. **Start MLflow UI** (if not already running):
   ```bash
   python -m mlflow ui --port 5000
   ```

2. **Open in browser**: http://127.0.0.1:5000

3. **Explore your experiments**:
   - Compare model performance
   - View feature importance
   - Analyze trading signals
   - Track experiment history

### **Menjalankan Pipeline Lengkap**
```bash
python main.py
```

### **Menjalankan MLflow UI**
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Buka browser ke `http://localhost:5000` untuk melihat experiment tracking.

## 📊 **Features yang Dibuat**

### **1. Technical Indicators**
- Simple Moving Average (SMA): 5, 10, 20, 50 periode
- Exponential Moving Average (EMA): 12, 26 periode
- RSI (Relative Strength Index): 14 periode
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands dengan deviasi standar 2

### **2. Time-based Features**
- Hour, day of week, month encoding
- Cyclical encoding (sin/cos transformations)
- Trading session indicators (Asia, Europe, US)
- Market overlap periods

### **3. Risk Features**
- Historical volatility (5, 10, 20 periode)
- MAE/MFE statistics
- Risk-reward ratios
- Skewness dan kurtosis

### **4. Statistical Features**
- Rolling statistics (mean, median, std, min, max)
- Lag features (1, 2, 3, 5, 10 periode)
- Quantile features (25%, 75%)

### **5. Interaction Features**
- Price-volume interactions
- Session-volatility interactions
- Time-price interactions

## 🤖 **Model yang Digunakan**

1. **XGBoost**: Gradient boosting dengan optimasi untuk trading
2. **LightGBM**: Fast gradient boosting dengan memory efficiency
3. **Random Forest**: Ensemble model untuk baseline comparison

Semua model dilacak menggunakan MLflow dengan:
- Parameter logging
- Metrics tracking
- Model versioning
- Feature importance analysis

## 📈 **Metrics Evaluasi**

### **Classification Metrics**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC untuk probabilitas prediction

### **Trading-Specific Metrics**
- **Win Rate**: Persentase trade yang profitable
- **Profit Factor**: Rasio total profit vs total loss
- **Risk-Reward Ratio**: Rata-rata profit vs rata-rata loss
- **Maximum Drawdown**: Kerugian maksimal dari peak
- **Sharpe Ratio**: Risk-adjusted return
- **Expectancy**: Expected value per trade

### **Signal Quality Metrics**
- Precision pada berbagai probability threshold
- Signal frequency dan coverage
- Stability metrics across time periods

## 🎯 **Target Variables**

1. **is_profitable**: Trade menghasilkan profit (binary)
2. **is_winning_trade**: Trade profit > 10 pips (binary)

## 📁 **Struktur Data**

### **Input Data (dataBT)**
- Symbol, Timestamp, Type (BUY/SELL)
- OpenPrice, ClosePrice, Volume
- Profit, MAE_pips, MFE_pips
- ExitReason, Trading sessions

### **Generated Features**
- 100+ technical indicators
- Time-based features
- Risk metrics
- Statistical features

## 🔧 **Konfigurasi Model**

```yaml
models:
  - name: "xgboost"
    params:
      n_estimators: [100, 200, 300]
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1, 0.2]
  
  - name: "lightgbm"
    params:
      n_estimators: [100, 200, 300]
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1, 0.2]
```

## 📊 **MLflow Tracking**

MLflow melacak semua eksperimen dengan informasi:

### **Parameters**
- Model hyperparameters
- Feature engineering settings
- Data preprocessing parameters

### **Metrics**
- Training dan validation metrics
- Trading performance metrics
- Cross-validation scores

### **Artifacts**
- Trained models
- Feature importance plots
- Performance reports

### **Model Registry**
- Model versioning
- Staging dan production deployment
- Model comparison

## 🔍 **Analisis dan Interpretasi**

### **Feature Importance**
- Identifikasi fitur paling berpengaruh
- Analisis kontribusi setiap kategori feature
- Elimination fitur yang tidak relevan

### **Model Performance**
- Perbandingan performa antar model
- Analisis stability across time
- Identification of overfitting

### **Trading Signal Analysis**
- Kualitas signal pada berbagai threshold
- Frekuensi signal vs accuracy trade-off
- Risk-adjusted performance metrics

## 📋 **Best Practices**

### **Data Management**
1. **Data Caching**: Automatic caching untuk faster iteration
2. **Data Validation**: Quality checks untuk data integrity
3. **Feature Versioning**: Track feature engineering changes

### **Model Development**
1. **Time-aware Validation**: TimeSeriesSplit untuk realistic evaluation
2. **Feature Selection**: Systematic feature importance analysis
3. **Hyperparameter Tuning**: Grid search dengan cross-validation

### **Risk Management**
1. **Drawdown Monitoring**: Continuous tracking of maximum drawdown
2. **Signal Filtering**: Quality thresholds untuk signal generation
3. **Performance Monitoring**: Regular model performance evaluation

## 🚨 **Peringatan dan Risiko**

1. **Past Performance**: Historical results tidak guarantee future performance
2. **Market Conditions**: Model performance dapat berubah dengan kondisi market
3. **Risk Management**: Selalu gunakan stop loss dan position sizing yang proper
4. **Overfitting**: Monitor untuk signs of overfitting pada unseen data

## 📈 **Rekomendasi Penggunaan**

### **Untuk Win Rate Tinggi**
- Gunakan threshold probability > 0.7 untuk high-confidence signals
- Focus pada session overlap periods (European-US)
- Consider volume confirmation untuk signal validation

### **Untuk Risk Management**
- Set maximum drawdown limit 10%
- Use risk-reward ratio minimal 2:1
- Implement position sizing berdasarkan signal confidence

### **Untuk Stabilitas Jangka Panjang**
- Retrain model setiap 3-6 bulan
- Monitor feature drift dan market regime changes
- Implement ensemble predictions untuk stability

## 🔄 **Monitoring dan Maintenance**

1. **Regular Retraining**: Schedule model retraining
2. **Performance Monitoring**: Track live performance vs backtest
3. **Feature Monitoring**: Monitor untuk feature drift
4. **Data Quality**: Continuous data quality checks

## 📞 **Support dan Development**

Untuk pengembangan lebih lanjut:
1. Implementasi real-time prediction pipeline
2. Integration dengan trading platform
3. Advanced ensemble methods
4. Multi-timeframe analysis
5. Portfolio optimization

---

**Note**: Project ini dirancang untuk educational dan research purposes. Selalu gunakan proper risk management dalam trading nyata.

# ML-TRADING-SIGNALS
PROYEK Sistem Trading ML dengan Anaconda &amp; MLflow
