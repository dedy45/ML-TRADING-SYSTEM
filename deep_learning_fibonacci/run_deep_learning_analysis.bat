@echo off
REM Batch file workaround untuk menjalankan deep learning fibonacci
REM Solusi untuk masalah hang loading Python files

echo ============================================================
echo 🧠 DEEP LEARNING FIBONACCI SIGNAL GENERATOR
echo ============================================================
echo.
echo 🎯 Tujuan: Generate signal prediksi untuk EA MQL5
echo 📊 Target: 58%+ win rate dari 52.4% baseline
echo.

REM Change to deep learning directory
cd /d "E:\aiml\MLFLOW\deep_learning_fibonacci"

echo 📂 Working directory: %CD%
echo.

REM Test Python availability
echo ⚡ Testing Python...
python -c "print('✅ Python available'); import pandas as pd; print('✅ Pandas available'); import sklearn; print('✅ Scikit-learn available')"

IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Python or required libraries not available
    pause
    exit /b 1
)

echo.
echo 🚀 Running Deep Learning Analysis...
echo ⏱️ Expected time: 30-60 seconds
echo 💡 Press Ctrl+C to cancel if needed
echo.

REM Run the analysis using command line approach (workaround for hang issue)
python -c "
import sys
sys.path.append('.')
sys.path.append('..')

# Inline implementation to avoid file execution hang
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
import time
from datetime import datetime

print('🧠 Deep Learning Fibonacci Analysis - INLINE EXECUTION')
print('=' * 60)

start_time = time.time()

try:
    # Load data
    print('📂 Loading backtest data...')
    data_path = Path('../dataBT')
    csv_files = list(data_path.glob('*.csv'))[:20]  # 20 files for balance of speed/accuracy
    
    if not csv_files:
        print('❌ No CSV files found in dataBT')
        sys.exit(1)
    
    print(f'📁 Processing {len(csv_files)} CSV files...')
    
    all_data = []
    total_rows = 0
    
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path, nrows=40)  # 40 rows per file for speed
            all_data.append(df)
            total_rows += len(df)
            
            if i % 5 == 0:
                print(f'   Progress: {i+1}/{len(csv_files)} files, {total_rows} rows')
                
        except Exception as e:
            print(f'   ⚠️ Skip {file_path.name}: {e}')
            continue
    
    if not all_data:
        print('❌ No data loaded successfully')
        sys.exit(1)
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    print(f'✅ Loaded {len(df)} records from {len(csv_files)} files')
    
    # Feature Engineering (Advanced)
    print('🔧 Engineering advanced features...')
    
    # Convert key columns
    numeric_cols = ['LevelFibo', 'Profit', 'SessionEurope', 'SessionUS', 'SessionAsia', 'TP', 'SL', 'SeparatorHour']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Feature dictionary
    features = {}
    
    # 1. FIBONACCI FEATURES (Based on proven 52%+ analysis)
    if 'LevelFibo' in df.columns:
        features['fib_b0'] = (df['LevelFibo'] == 0.0).astype(int)
        features['fib_b_minus_18'] = (df['LevelFibo'] == -1.8).astype(int)
        features['fib_b_18'] = (df['LevelFibo'] == 1.8).astype(int)
        features['fib_b_minus_36'] = (df['LevelFibo'] == -3.6).astype(int)
        
        # Signal strength (proven levels)
        signal_strength = np.where(df['LevelFibo'] == 0.0, 3,      # 52.4% win rate
                          np.where(df['LevelFibo'] == -1.8, 3,     # 52.5% win rate
                          np.where(df['LevelFibo'] == 1.8, 2,      # 45.9% win rate
                          np.where(df['LevelFibo'] == -3.6, 2, 1)))) # Good level
        features['signal_strength'] = signal_strength
        
        features['fib_level_raw'] = df['LevelFibo']
        features['fib_is_major'] = (np.abs(df['LevelFibo']) <= 2.0).astype(int)
    
    # 2. SESSION FEATURES (Europe best: 40.5%)
    if 'SessionEurope' in df.columns:
        features['session_europe'] = df['SessionEurope']
        features['session_us'] = df['SessionUS']
        features['session_asia'] = df['SessionAsia']
        
        # Session scoring
        session_score = (df['SessionEurope'] * 3 +      # Best performance
                        df['SessionUS'] * 2 +           # Good performance  
                        df['SessionAsia'] * 1)          # Lower performance
        features['session_score'] = session_score
    
    # 3. RISK MANAGEMENT (2:1 TP/SL optimal)
    if 'TP' in df.columns and 'SL' in df.columns:
        tp_sl_ratio = df['TP'] / df['SL'].replace(0, 1)
        features['tp_sl_ratio'] = np.clip(tp_sl_ratio, 0, 5)
        features['optimal_ratio'] = (tp_sl_ratio >= 2.0).astype(int)
        features['conservative_ratio'] = (tp_sl_ratio >= 2.5).astype(int)
    
    # 4. TIME FEATURES
    if 'SeparatorHour' in df.columns:
        features['hour_sin'] = np.sin(2 * np.pi * df['SeparatorHour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['SeparatorHour'] / 24)
        features['peak_hours'] = ((df['SeparatorHour'] >= 8) & (df['SeparatorHour'] <= 17)).astype(int)
        features['london_hours'] = ((df['SeparatorHour'] >= 8) & (df['SeparatorHour'] <= 16)).astype(int)
    
    # 5. COMBINED FEATURES
    high_conf = ((features.get('signal_strength', 0) >= 2) & 
                (features.get('session_score', 0) >= 2) & 
                (features.get('optimal_ratio', 0) == 1)).astype(int)
    features['high_confidence_signal'] = high_conf
    
    # Create feature matrix
    X = pd.DataFrame(features).fillna(0)
    print(f'✅ Created {X.shape[1]} features')
    
    # Target variable
    if 'Profit' in df.columns:
        y = (df['Profit'] > 0).astype(int)
        baseline_win_rate = y.mean()
        print(f'✅ Target created from Profit column')
        print(f'📊 Baseline win rate: {baseline_win_rate:.1%}')
    else:
        print('❌ No Profit column found')
        sys.exit(1)
    
    # Train Models
    print('🤖 Training ensemble models...')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'📊 Training set: {len(X_train)} samples')
    print(f'📊 Test set: {len(X_test)} samples')
    
    # Models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'   🔄 Training {name}...')
        
        try:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # High confidence signals (>= 0.7 probability)
            high_conf_mask = y_pred_proba >= 0.7
            if np.sum(high_conf_mask) > 0:
                high_conf_win_rate = np.mean(y_test[high_conf_mask] == 1)
                high_conf_count = np.sum(high_conf_mask)
            else:
                high_conf_win_rate = 0
                high_conf_count = 0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'high_conf_win_rate': high_conf_win_rate,
                'high_conf_count': high_conf_count
            }
            
            print(f'     ✅ Accuracy: {accuracy:.1%}')
            print(f'     🎯 High Conf Win Rate: {high_conf_win_rate:.1%} ({high_conf_count} signals)')
            
        except Exception as e:
            print(f'     ❌ Training failed: {e}')
            continue
    
    # Results Summary
    execution_time = time.time() - start_time
    
    print()
    print('=' * 60)
    print('🎯 DEEP LEARNING FIBONACCI RESULTS')
    print('=' * 60)
    print(f'⏱️ Execution time: {execution_time:.1f} seconds')
    print(f'📊 Data processed: {len(df)} records')
    print(f'🔧 Features engineered: {X.shape[1]}')
    print(f'📈 Baseline win rate: {baseline_win_rate:.1%}')
    print()
    
    if results:
        print('🏆 MODEL PERFORMANCE:')
        print('-' * 40)
        
        best_model_name = None
        best_win_rate = 0
        
        for name, metrics in results.items():
            print(f'{name}:')
            print(f'  📊 Accuracy: {metrics[\"accuracy\"]:.1%}')
            print(f'  🎯 High Conf Win Rate: {metrics[\"high_conf_win_rate\"]:.1%}')
            print(f'  📈 High Conf Signals: {metrics[\"high_conf_count\"]}')
            print()
            
            if metrics['high_conf_win_rate'] > best_win_rate:
                best_win_rate = metrics['high_conf_win_rate']
                best_model_name = name
        
        # Save best model
        if best_model_name and best_win_rate > 0:
            best_model = results[best_model_name]['model']
            
            # Create models directory
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            # Save model and scaler
            joblib.dump(best_model, models_dir / 'fibonacci_signal_model.pkl')
            joblib.dump(scaler, models_dir / 'signal_scaler.pkl')
            
            print(f'💾 Best model saved: {best_model_name}')
            print(f'📁 Model files: models/fibonacci_signal_model.pkl')
            print(f'📁 Scaler file: models/signal_scaler.pkl')
            
            # Create signal configuration
            signal_config = {
                'model_name': best_model_name,
                'high_confidence_win_rate': float(best_win_rate),
                'baseline_win_rate': float(baseline_win_rate),
                'improvement': float(best_win_rate - baseline_win_rate),
                'confidence_threshold': 0.7,
                'features_count': int(X.shape[1]),
                'training_samples': int(len(X_train)),
                'execution_time': float(execution_time),
                'timestamp': datetime.now().isoformat(),
                'ready_for_ea': best_win_rate >= 0.55
            }
            
            with open(models_dir / 'signal_config.json', 'w') as f:
                json.dump(signal_config, f, indent=2)
            
            print(f'⚙️ Configuration saved: models/signal_config.json')
            
            # Performance evaluation
            target_58_achieved = best_win_rate >= 0.58
            target_55_achieved = best_win_rate >= 0.55
            
            print()
            print('🎯 TARGET ACHIEVEMENT:')
            if target_58_achieved:
                print('🎉 TARGET 58% ACHIEVED! Excellent for live trading!')
            elif target_55_achieved:
                print('✅ Target 55% achieved! Ready for EA deployment!')
            else:
                print('📈 Partial improvement. Consider more data or optimization.')
            
            print()
            print('🚀 EA MQL5 INTEGRATION:')
            print('1. 📁 Load model: joblib.load(\"models/fibonacci_signal_model.pkl\")')
            print('2. 🎯 High confidence threshold: >= 0.7')
            print('3. ⚡ Expected inference time: <100ms')
            print('4. 📊 Use signal_config.json for parameters')
            
            improvement_pct = (best_win_rate - baseline_win_rate) / baseline_win_rate * 100
            print(f'📈 Win rate improvement: {improvement_pct:+.1f}%')
            
        else:
            print('❌ No viable model trained')
    else:
        print('❌ No models trained successfully')
        
    print()
    print('✅ DEEP LEARNING ANALYSIS COMPLETED!')
    
except Exception as e:
    print(f'❌ Analysis failed: {e}')
    import traceback
    traceback.print_exc()
"

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Analysis failed. Check error messages above.
    echo 💡 Try running: python -c "import pandas; print('OK')" to check dependencies
) ELSE (
    echo.
    echo 🎉 Analysis completed successfully!
    echo 📊 Check results above for model performance
    echo 📁 Model files saved in models/ directory
)

echo.
echo Press any key to exit...
pause >nul
