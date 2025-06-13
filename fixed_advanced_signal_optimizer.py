#!/usr/bin/env python3
"""
FIXED ADVANCED SIGNAL OPTIMIZER
Perbaikan untuk mengatasi error string to float conversion
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

class FixedAdvancedSignalOptimizer:
    """Fixed optimizer untuk mengatasi error conversion"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.feature_names = []  # Track feature names used in training
        
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
    
    def engineer_features(self, df, is_training=True):
        """Feature engineering dengan handling yang benar untuk training dan prediction"""
        print("\nENGINEERING FEATURES...")
        
        # Create target variable (only for training)
        if 'Profit' in df.columns and is_training:
            df['is_profitable'] = (df['Profit'] > 0).astype(int)
        
        # Features that will be created
        features = []
        df_features = pd.DataFrame(index=df.index)
        
        # 1. Fibonacci Level features
        if 'LevelFibo' in df.columns:
            if is_training:
                # During training, fit the label encoder
                le_fibo = LabelEncoder()
                df_features['LevelFibo_encoded'] = le_fibo.fit_transform(df['LevelFibo'].fillna('UNKNOWN'))
                self.label_encoders['LevelFibo'] = le_fibo
            else:
                # During prediction, use existing encoder
                if 'LevelFibo' in self.label_encoders:
                    try:
                        df_features['LevelFibo_encoded'] = self.label_encoders['LevelFibo'].transform(df['LevelFibo'].fillna('UNKNOWN'))
                    except:
                        df_features['LevelFibo_encoded'] = 0  # Unknown level
                else:
                    df_features['LevelFibo_encoded'] = 0
            
            features.append('LevelFibo_encoded')
            
            # Level type indicators
            df_features['is_buy_level'] = df['LevelFibo'].str.startswith('B_').fillna(False).astype(int)
            df_features['is_sell_level'] = df['LevelFibo'].str.startswith('S_').fillna(False).astype(int)
            features.extend(['is_buy_level', 'is_sell_level'])
            
            # Strong level indicator
            strong_levels = ['B_0', 'B_-1.8']
            df_features['is_strong_level'] = df['LevelFibo'].isin(strong_levels).astype(int)
            features.append('is_strong_level')
        
        # 2. Trade Type
        if 'Type' in df.columns:
            if is_training:
                le_type = LabelEncoder()
                df_features['Type_encoded'] = le_type.fit_transform(df['Type'])
                self.label_encoders['Type'] = le_type
            else:
                if 'Type' in self.label_encoders:
                    try:
                        df_features['Type_encoded'] = self.label_encoders['Type'].transform(df['Type'])
                    except:
                        df_features['Type_encoded'] = 0  # Unknown type
                else:
                    df_features['Type_encoded'] = 0
            
            features.append('Type_encoded')
        
        # 3. Session features
        session_cols = ['SessionEurope', 'SessionUS', 'SessionAsia']
        for col in session_cols:
            df_features[col] = df[col].fillna(0) if col in df.columns else 0
            features.append(col)
        
        # Session strength
        df_features['session_strength'] = (
            df_features['SessionEurope'] * 0.405 +
            df_features['SessionUS'] * 0.401 +
            df_features['SessionAsia'] * 0.397
        )
        features.append('session_strength')
        
        # 4. Price action features
        price_cols = ['OpenPrice', 'TP', 'SL']
        if all(col in df.columns for col in price_cols):
            df_features['risk_reward_ratio'] = abs(df['TP'] - df['OpenPrice']) / abs(df['OpenPrice'] - df['SL'])
            df_features['risk_reward_ratio'] = df_features['risk_reward_ratio'].fillna(1.0)
            
            df_features['price_sl_distance'] = abs(df['OpenPrice'] - df['SL']) / df['OpenPrice'] * 10000
            df_features['price_tp_distance'] = abs(df['TP'] - df['OpenPrice']) / df['OpenPrice'] * 10000
            
            # Risk categories
            df_features['is_low_risk'] = (df_features['price_sl_distance'] <= 20).astype(int)
            df_features['is_medium_risk'] = ((df_features['price_sl_distance'] > 20) & (df_features['price_sl_distance'] <= 50)).astype(int)
            df_features['is_high_risk'] = (df_features['price_sl_distance'] > 50).astype(int)
            
            features.extend(['risk_reward_ratio', 'price_sl_distance', 'price_tp_distance', 
                           'is_low_risk', 'is_medium_risk', 'is_high_risk'])
        else:
            # Default values if price columns missing
            for col in ['risk_reward_ratio', 'price_sl_distance', 'price_tp_distance', 
                       'is_low_risk', 'is_medium_risk', 'is_high_risk']:
                df_features[col] = 0
                features.append(col)
        
        # 5. Volume features
        if 'Volume' in df.columns:
            df_features['Volume'] = df['Volume'].fillna(0.1)
            df_features['is_high_volume'] = (df_features['Volume'] > 0.2).astype(int)
            features.extend(['Volume', 'is_high_volume'])
        else:
            df_features['Volume'] = 0.1
            df_features['is_high_volume'] = 0
            features.extend(['Volume', 'is_high_volume'])
        
        # 6. Time-based features
        df_features['trade_sequence'] = df.index
        df_features['trade_sequence_norm'] = df_features['trade_sequence'] / max(len(df), 1)
        features.append('trade_sequence_norm')
        
        # 7. Interaction features
        df_features['buy_europe_interaction'] = df_features.get('is_buy_level', 0) * df_features.get('SessionEurope', 0)
        df_features['sell_us_interaction'] = df_features.get('is_sell_level', 0) * df_features.get('SessionUS', 0)
        features.extend(['buy_europe_interaction', 'sell_us_interaction'])
        
        # Clean and prepare final features
        feature_df = df_features[features].fillna(0)
        
        # Ensure all features are numeric
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        if is_training:
            self.feature_names = features
            target = df['is_profitable']
            
            print(f"Created {len(features)} features:")
            for i, feat in enumerate(features, 1):
                print(f"  {i:2d}. {feat}")
            
            print(f"Target distribution:")
            print(f"  Profitable trades: {target.sum():,} ({target.mean()*100:.1f}%)")
            print(f"  Losing trades: {(target==0).sum():,} ({(1-target.mean())*100:.1f}%)")
            
            return feature_df, target
        else:
            # For prediction, ensure same features as training
            missing_features = set(self.feature_names) - set(feature_df.columns)
            for feat in missing_features:
                feature_df[feat] = 0
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]
            return feature_df
    
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
            
            # Tree-based models don't need scaling
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                self.model = model
                
                # Feature importance
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
            
            # Apply feature engineering (not training mode)
            df_processed = self.engineer_features(df_input, is_training=False)
            
            # Predict
            prediction = self.model.predict_proba(df_processed)
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
    
    def save_model(self, filepath="models/fixed_signal_optimizer.pkl"):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/fixed_signal_optimizer.pkl"):
        """Load trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data.get('feature_importance')
            self.feature_names = model_data.get('feature_names', [])
            print(f"Model loaded from {filepath}")
            return True
        return False

def main():
    """Main function untuk training dan testing"""
    print("FIXED ADVANCED SIGNAL OPTIMIZER")
    print("=" * 60)
    
    # Initialize
    optimizer = FixedAdvancedSignalOptimizer()
    
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
            {
                'LevelFibo': 'B_0', 
                'Type': 'BUY', 
                'SessionEurope': 1,
                'SessionUS': 0,
                'SessionAsia': 0, 
                'OpenPrice': 2650.0,
                'TP': 2655.0,
                'SL': 2648.0,
                'Volume': 0.1
            },
            {
                'LevelFibo': 'B_-1.8', 
                'Type': 'BUY', 
                'SessionEurope': 0,
                'SessionUS': 1,
                'SessionAsia': 0,
                'OpenPrice': 2651.0,
                'TP': 2656.0,
                'SL': 2649.0,
                'Volume': 0.2
            },
            {
                'LevelFibo': 'S_1', 
                'Type': 'SELL', 
                'SessionEurope': 0,
                'SessionUS': 0,
                'SessionAsia': 1,
                'OpenPrice': 2652.0,
                'TP': 2647.0,
                'SL': 2654.0,
                'Volume': 0.1
            }
        ]
        
        for i, signal in enumerate(test_signals, 1):
            result = optimizer.get_signal_strength(signal)
            print(f"\nTest Signal {i}: {signal}")
            print(f"Result: {result}")
    else:
        print("Failed to load data. Check if dataBT folder contains CSV files.")

if __name__ == "__main__":
    main()
