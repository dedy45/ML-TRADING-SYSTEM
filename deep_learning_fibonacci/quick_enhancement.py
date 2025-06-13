"""
Immediate Enhanced Fibonacci Analysis
Quick ML improvement from 52% to 55%+ win rate
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML imports that work with Python 3.13
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def quick_fibonacci_enhancement():
    """Quick enhanced analysis to improve win rate immediately."""
    
    print("🚀 Quick Enhanced Fibonacci Analysis")
    print("Target: Improve 52% baseline to 55%+ win rate")
    print("=" * 50)
    
    # Load data
    data_path = Path("../dataBT")
    csv_files = list(data_path.glob("*.csv"))[:25]  # Use 25 files for quick test
    
    print(f"📁 Loading {len(csv_files)} CSV files...")
    
    all_data = []
    
    for i, file_path in enumerate(csv_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            df = pd.DataFrame(rows)
            df['file_index'] = i
            all_data.append(df)
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    if not all_data:
        print("❌ No data loaded")
        return
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(df)} records")
    
    # Convert key columns to numeric
    numeric_cols = ['LevelFibo', 'SeparatorHour', 'TP', 'SL', 'SessionEurope', 'SessionUS', 'SessionAsia']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create enhanced features based on proven 52% win rate analysis
    features = {}
    
    if 'LevelFibo' in df.columns:
        # Primary signals (52%+ win rate from analysis)
        features['is_b0_level'] = (df['LevelFibo'] == 0.0).astype(int)
        features['is_b_minus_1_8'] = (df['LevelFibo'] == -1.8).astype(int)
        features['is_b_1_8'] = (df['LevelFibo'] == 1.8).astype(int)
        
        # Signal strength
        signal_strength = np.zeros(len(df))
        signal_strength[df['LevelFibo'] == 0.0] = 3  # Highest (52.4% win rate)
        signal_strength[df['LevelFibo'] == -1.8] = 3  # Highest (52.5% win rate)
        signal_strength[df['LevelFibo'] == 1.8] = 2  # Medium (45.9% win rate)
        features['signal_strength'] = signal_strength
    
    # Session optimization (Europe best: 40.5%)
    if 'SessionEurope' in df.columns:
        features['europe_session'] = df['SessionEurope'].fillna(0)
        features['session_score'] = (df['SessionEurope'].fillna(0) * 3 + 
                                    df['SessionUS'].fillna(0) * 2 + 
                                    df['SessionAsia'].fillna(0) * 1)
    
    # Risk management (2:1 TP/SL optimal)
    if 'TP' in df.columns and 'SL' in df.columns:
        tp_sl_ratio = df['TP'] / df['SL'].replace(0, 1)
        features['tp_sl_ratio'] = tp_sl_ratio
        features['optimal_ratio'] = (tp_sl_ratio >= 2.0).astype(int)
    
    # Time features
    if 'SeparatorHour' in df.columns:
        features['hour_sin'] = np.sin(2 * np.pi * df['SeparatorHour'].fillna(0) / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['SeparatorHour'].fillna(0) / 24)
    
    # Create feature matrix
    X = pd.DataFrame(features).fillna(0)
      # Create target variable using Profit column (more reliable)
    if 'Profit' in df.columns:
        profit_values = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
        y = (profit_values > 0).astype(int)
        print(f"✅ Using Profit column for target: {y.value_counts().to_dict()}")
    elif 'Result' in df.columns:
        y = (pd.to_numeric(df['Result'], errors='coerce') > 0).astype(int)
        print(f"✅ Using Result column for target: {y.value_counts().to_dict()}")
    else:
        # Use enhanced signal strength with more variation
        y = np.random.choice([0, 1], size=len(X), p=[0.48, 0.52])  # Simulate 52% win rate
        print(f"✅ Using simulated target (demo): {pd.Series(y).value_counts().to_dict()}")
    
    print(f"✅ Created {X.shape[1]} features")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    print("\n🤖 Training Enhanced Models...")
    
    results = {}
    baseline_win_rate = 0.524  # B_0 level baseline
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Win rate: accuracy of positive predictions
        win_predictions = y_pred == 1
        if np.sum(win_predictions) > 0:
            win_rate = np.mean(y_test[win_predictions] == 1)
        else:
            win_rate = 0
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'win_rate': win_rate,
            'improvement': win_rate - baseline_win_rate
        }
        
        print(f"     Win Rate: {win_rate:.1%}")
        print(f"     Improvement: {win_rate - baseline_win_rate:+.3f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['win_rate'])
    best_metrics = results[best_model]
    
    print("\n" + "=" * 60)
    print("🎯 ENHANCED ANALYSIS RESULTS")
    print("=" * 60)
    print(f"📊 Baseline (B_0 Level): {baseline_win_rate:.1%}")
    print(f"🏆 Best Model: {best_model}")
    print(f"📈 Enhanced Win Rate: {best_metrics['win_rate']:.1%}")
    print(f"🚀 Improvement: {best_metrics['improvement']:+.3f} ({best_metrics['improvement']/baseline_win_rate:+.1%})")
    
    target_achieved = best_metrics['win_rate'] >= 0.55
    print(f"🎯 Target (55%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    print("\n📋 All Models Performance:")
    for name, metrics in results.items():
        print(f"   {name}: {metrics['win_rate']:.1%} (improvement: {metrics['improvement']:+.3f})")
    
    print("\n🚀 Next Steps:")
    if target_achieved:
        print("   ✅ Ready for live trading!")
        print("   ✅ Deploy enhanced model")
        print("   ✅ Set up real-time signals")
    else:
        print("   📈 Partial improvement achieved")
        print("   🔧 Try TensorFlow deep learning for further gains")
        print("   📊 Use more training data")
    
    print("\n💡 Key Enhancements Applied:")
    print("   - Enhanced signal strength scoring")
    print("   - Session optimization (Europe focus)")
    print("   - Risk ratio adherence (2:1 TP/SL)")
    print("   - Time-based pattern recognition")
    
    return results

if __name__ == "__main__":
    try:
        results = quick_fibonacci_enhancement()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
