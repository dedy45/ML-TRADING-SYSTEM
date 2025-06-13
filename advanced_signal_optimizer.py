#!/usr/bin/env python3
"""
ADVANCED SIGNAL OPTIMIZER
Mengoptimalkan sinyal trading menggunakan ML dan analisis mendalam
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedSignalOptimizer:
    """Optimizer untuk meningkatkan akurasi sinyal trading"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def load_and_prepare_data(self, max_files=20, sample_per_file=1000):
        """Load dan persiapkan data untuk training ML"""
        print("=" * 60)
        print("LOADING AND PREPARING DATA FOR ML")
        print("=" * 60)
        
        csv_files = glob.glob(f"{self.data_path}/*.csv")
        if not csv_files:
            print(f"[ERROR] No CSV files found in {self.data_path}")
            return None, None
            
        print(f"Found {len(csv_files)} files, processing {min(max_files, len(csv_files))}")
        
        all_data = []
        
        for i, file_path in enumerate(csv_files[:max_files]):
            try:
                filename = os.path.basename(file_path)
                print(f"[{i+1:2d}/{min(max_files, len(csv_files))}] Processing: {filename}")
                
                df = pd.read_csv(file_path, nrows=sample_per_file)
                
                # Filter valid trades
                df = df[df['Type'].isin(['BUY', 'SELL'])].copy()
                
                if len(df) > 0:
                    all_data.append(df)
                    print(f"        Added {len(df)} trades")
                
            except Exception as e:
                print(f"        [ERROR] {str(e)}")
                continue
        
        if not all_data:
            print("[ERROR] No data loaded")
            return None, None
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal trades loaded: {len(combined_df):,}")
        
        return self.engineer_features(combined_df)
    
    def engineer_features(self, df):
        """Feature engineering untuk meningkatkan prediksi"""
        print("\nENGINEERING FEATURES...")
        
        # Create target variable (profitable trade = 1)
        df['is_profitable'] = (df['Profit'] > 0).astype(int)
        
        # Basic features
        features = []
        
        # 1. Fibonacci Level features
        if 'LevelFibo' in df.columns:
            # Encode Fibonacci levels
            le_fibo = LabelEncoder()
            df['LevelFibo_encoded'] = le_fibo.fit_transform(df['LevelFibo'].fillna('UNKNOWN'))
            self.label_encoders['LevelFibo'] = le_fibo
            features.append('LevelFibo_encoded')
            
            # Create level type features
            df['is_buy_level'] = df['LevelFibo'].str.startswith('B_').astype(int)
            df['is_sell_level'] = df['LevelFibo'].str.startswith('S_').astype(int)
            features.extend(['is_buy_level', 'is_sell_level'])
        
        # 2. Trade Type
        if 'Type' in df.columns:
            le_type = LabelEncoder()
            df['Type_encoded'] = le_type.fit_transform(df['Type'])
            self.label_encoders['Type'] = le_type
            features.append('Type_encoded')
        
        # 3. Session features
        session_cols = ['SessionEurope', 'SessionUS', 'SessionAsia']
        for col in session_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                features.append(col)
        
        # 4. Price action features
        if all(col in df.columns for col in ['OpenPrice', 'TP', 'SL']):
            df['risk_reward_ratio'] = abs(df['TP'] - df['OpenPrice']) / abs(df['OpenPrice'] - df['SL'])
            df['risk_reward_ratio'] = df['risk_reward_ratio'].fillna(1.0)
            features.append('risk_reward_ratio')
            
            df['price_sl_distance'] = abs(df['OpenPrice'] - df['SL']) / df['OpenPrice'] * 10000  # in pips
            df['price_tp_distance'] = abs(df['TP'] - df['OpenPrice']) / df['OpenPrice'] * 10000  # in pips
            features.extend(['price_sl_distance', 'price_tp_distance'])
        
        # 5. Time-based features
        if 'Ticket' in df.columns:
            # Use ticket as proxy for time sequence
            df['trade_sequence'] = df.index
            df['trade_sequence_norm'] = df['trade_sequence'] / len(df)
            features.append('trade_sequence_norm')
        
        # 6. Volume/Lot features
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(df['Volume'].median())
            df['is_high_volume'] = (df['Volume'] > df['Volume'].quantile(0.75)).astype(int)
            features.extend(['Volume', 'is_high_volume'])
        
        # 7. Profit-based features for historical context
        if 'Profit' in df.columns:
            df['profit_magnitude'] = abs(df['Profit'])
            # Rolling statistics (for sequential data)
            df['profit_rolling_mean'] = df['Profit'].rolling(window=10, min_periods=1).mean()
            df['profit_rolling_std'] = df['Profit'].rolling(window=10, min_periods=1).std().fillna(0)
            features.extend(['profit_magnitude', 'profit_rolling_mean', 'profit_rolling_std'])
        
        # Clean features
        feature_df = df[features].fillna(0)
        target = df['is_profitable']
        
        print(f"Created {len(features)} features:")
        for i, feat in enumerate(features, 1):
            print(f"  {i:2d}. {feat}")
        
        print(f"Target distribution:")
        print(f"  Profitable trades: {target.sum():,} ({target.mean()*100:.1f}%)")
        print(f"  Losing trades: {(target==0).sum():,} ({(1-target.mean())*100:.1f}%)")
        
        return feature_df, target
    
    def train_model(self, X, y):
        """Train ML model untuk prediksi sinyal"""
        print("\n" + "=" * 60)
        print("TRAINING ML MODEL")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'RandomForest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_for_cv = X_train
                y_for_cv = y_train
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                X_for_cv = X_train_scaled
                y_for_cv = y_train
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            # Cross validation
            cv_scores = cross_val_score(model, X_for_cv, y_for_cv, cv=5, scoring='accuracy')
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                self.model = model
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
        
        print(f"\nBest model accuracy: {best_score:.3f}")
        
        return best_model
      def get_signal_strength(self, features_dict):
        """Get signal strength untuk trade baru"""
        if self.model is None:
            return {"error": "Model not trained"}
        
        try:
            # Convert to DataFrame
            df_input = pd.DataFrame([features_dict])
            
            # Apply same feature engineering that was used in training
            df_processed = self.engineer_features(df_input)
            
            # Apply label encoding for categorical columns
            for col, le in self.label_encoders.items():
                if col in df_input.columns:
                    try:
                        df_processed[f"{col}_encoded"] = le.transform(df_input[col])
                    except:
                        df_processed[f"{col}_encoded"] = 0  # Unknown category
            
            # Remove original categorical columns and keep only numeric features
            categorical_cols = ['LevelFibo', 'Type']  # Original string columns
            for col in categorical_cols:
                if col in df_processed.columns:
                    df_processed = df_processed.drop(columns=[col])
            
            # Ensure all columns are numeric
            df_processed = df_processed.select_dtypes(include=[np.number])
            
            # Fill any remaining NaN values
            df_processed = df_processed.fillna(0)
            
            # Predict using the appropriate model
            if hasattr(self.model, 'feature_importances_'):  # Tree-based model
                prediction = self.model.predict_proba(df_processed)
            else:  # Scaled model (LogisticRegression, SVM)
                prediction = self.model.predict_proba(self.scaler.transform(df_processed))            
            probability = prediction[0][1]  # Probability of profitable trade
            
            if probability >= 0.6:
                strength = "VERY_STRONG"
            elif probability >= 0.55:
                strength = "STRONG"
            elif probability >= 0.5:
                strength = "MEDIUM"
            else:
                strength = "WEAK"
            
            return {
                'signal_strength': strength,
                'win_probability': probability,
                'recommendation': 'TAKE_TRADE' if probability >= 0.55 else 'AVOID_TRADE'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'signal_strength': 'ERROR',
                'win_probability': 0.5,
                'recommendation': 'AVOID_TRADE'
            }
    
    def save_model(self, filepath="models/signal_optimizer.pkl"):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/signal_optimizer.pkl"):
        """Load trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data.get('feature_importance')
            print(f"Model loaded from {filepath}")
            return True
        return False

def main():
    """Main function untuk training dan testing"""
    print("ADVANCED SIGNAL OPTIMIZER")
    print("=" * 60)
    
    # Initialize
    optimizer = AdvancedSignalOptimizer()
    
    # Load data
    X, y = optimizer.load_and_prepare_data(max_files=15, sample_per_file=800)
    
    if X is not None and y is not None:
        # Train model
        model = optimizer.train_model(X, y)
        
        # Save model
        optimizer.save_model()
        
        # Show feature importance
        if optimizer.feature_importance is not None:
            print("\nTOP 10 MOST IMPORTANT FEATURES:")
            print(optimizer.feature_importance.head(10))
        
        # Test signal
        print("\n" + "=" * 60)
        print("TESTING SIGNAL PREDICTION")
        print("=" * 60)
        
        test_signals = [
            {'LevelFibo': 'B_0', 'Type': 'BUY', 'SessionEurope': 1, 'Volume': 0.1},
            {'LevelFibo': 'B_-1.8', 'Type': 'BUY', 'SessionUS': 1, 'Volume': 0.2},
            {'LevelFibo': 'S_1', 'Type': 'SELL', 'SessionAsia': 1, 'Volume': 0.1}
        ]
        
        for i, signal in enumerate(test_signals, 1):
            result = optimizer.get_signal_strength(signal)
            print(f"\nTest Signal {i}: {signal}")
            print(f"Result: {result}")

if __name__ == "__main__":
    main()
