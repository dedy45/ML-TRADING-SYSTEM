#!/usr/bin/env python3
"""
Simple Fibonacci Signal Generator
Fast and reliable signal generation for EA MQL5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import time

def quick_fibonacci_signal_generator():
    """Generate trading signals quickly and reliably"""
    
    print("🚀 Fast Fibonacci Signal Generator")
    print("Target: Generate signals for EA MQL5")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load data (optimized for speed)
    data_path = Path("../dataBT")
    csv_files = list(data_path.glob("*.csv"))[:15]  # Only 15 files for speed
    
    print(f"📁 Loading {len(csv_files)} CSV files...")
    
    all_data = []
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path, nrows=30)  # Only 30 rows per file
            all_data.append(df)
            if i % 5 == 0:
                print(f"   Progress: {i+1}/{len(csv_files)} files")
        except Exception as e:
            print(f"   Error loading {file_path.name}: {e}")
            continue
    
    if not all_data:
        print("❌ No data loaded")
        return None
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(df)} records")
    
    # Feature engineering (simplified)
    features = {}
    
    # Convert to numeric
    numeric_cols = ['LevelFibo', 'Profit', 'SessionEurope', 'TP', 'SL']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Key features for signal generation
    if 'LevelFibo' in df.columns:
        features['fib_b0'] = (df['LevelFibo'] == 0.0).astype(int)
        features['fib_b_minus_18'] = (df['LevelFibo'] == -1.8).astype(int)
        features['fib_level'] = df['LevelFibo']
        
        # Signal strength
        signal_strength = np.where(df['LevelFibo'] == 0.0, 3,
                         np.where(df['LevelFibo'] == -1.8, 3, 1))
        features['signal_strength'] = signal_strength
    
    # Session features
    if 'SessionEurope' in df.columns:
        features['europe_session'] = df['SessionEurope']
    
    # Risk management
    if 'TP' in df.columns and 'SL' in df.columns:
        tp_sl_ratio = df['TP'] / df['SL'].replace(0, 1)
        features['tp_sl_ratio'] = np.clip(tp_sl_ratio, 0, 5)
    
    # Create feature matrix
    X = pd.DataFrame(features).fillna(0)
    
    # Create target (profitable trades)
    if 'Profit' in df.columns:
        y = (df['Profit'] > 0).astype(int)
        win_rate = y.mean()
        print(f"✅ Target win rate: {win_rate:.1%}")
    else:
        print("❌ No Profit column found")
        return None
    
    print(f"✅ Features: {X.shape[1]}")
    print(f"✅ Samples: {len(X)}")
    
    # Train model (fast)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest (fast and reliable)
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # High confidence signals
    high_conf_mask = y_pred_proba >= 0.7
    if np.sum(high_conf_mask) > 0:
        high_conf_win_rate = np.mean(y_test[high_conf_mask] == 1)
        high_conf_count = np.sum(high_conf_mask)
    else:
        high_conf_win_rate = 0
        high_conf_count = 0
    
    total_time = time.time() - start_time
    
    # Results
    print("\n" + "=" * 60)
    print("🎯 SIGNAL GENERATOR RESULTS")
    print("=" * 60)
    print(f"⏱️ Execution time: {total_time:.1f} seconds")
    print(f"📊 Model accuracy: {accuracy:.1%}")
    print(f"🎯 High confidence win rate: {high_conf_win_rate:.1%}")
    print(f"📈 High confidence signals: {high_conf_count}")
    print(f"📊 Data win rate: {win_rate:.1%}")
    
    # Save model for EA integration
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    
    joblib.dump(model, model_path / "fibonacci_signal_model.pkl")
    joblib.dump(scaler, model_path / "signal_scaler.pkl")
    
    print(f"💾 Model saved to: {model_path}")
    
    # Generate sample signal for EA
    sample_signal = {
        "signal_type": "BUY",
        "confidence": float(high_conf_win_rate),
        "model_accuracy": float(accuracy),
        "execution_time_seconds": float(total_time),
        "data_samples": int(len(X)),
        "high_confidence_threshold": 0.7,
        "model_version": "fibonacci_v1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save sample signal
    with open(model_path / "sample_signal.json", 'w') as f:
        json.dump(sample_signal, f, indent=2)
    
    print("📊 Sample signal saved for EA integration")
    
    # EA Integration instructions
    print("\n🔗 EA MQL5 INTEGRATION:")
    print("1. 📁 Use saved model: fibonacci_signal_model.pkl")
    print("2. 📊 Load with: joblib.load('models/fibonacci_signal_model.pkl')")
    print("3. 🎯 High confidence threshold: >= 0.7")
    print("4. ⚡ Expected response time: <1 second")
    
    target_achieved = high_conf_win_rate >= 0.55
    print(f"\n{'✅ READY FOR EA DEPLOYMENT' if target_achieved else '📈 NEEDS OPTIMIZATION'}")
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'high_conf_win_rate': high_conf_win_rate,
        'execution_time': total_time
    }

if __name__ == "__main__":
    try:
        result = quick_fibonacci_signal_generator()
        if result:
            print("\n🎉 Signal generator ready for EA integration!")
        else:
            print("\n❌ Signal generation failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
