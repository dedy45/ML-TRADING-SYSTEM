#!/usr/bin/env python3
"""
ENSEMBLE SIGNAL DETECTOR
Menggunakan multiple models untuk akurasi maksimal
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
warnings.filterwarnings('ignore')

class EnsembleSignalDetector:
    """Ensemble model untuk deteksi sinyal dengan akurasi tinggi"""
    
    def __init__(self, data_path="dataBT"):
        self.data_path = data_path
        self.ensemble_model = None
        self.individual_models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Performance tracking
        self.training_time = 0
        self.prediction_time = 0
        self.model_performance = {}
        
        # Cache untuk optimasi
        self.feature_cache = {}
        
    @lru_cache(maxsize=128)
    def _cached_feature_calculation(self, data_hash):
        """Cache expensive feature calculations"""
        return self.feature_cache.get(data_hash, None)
        
    def prepare_advanced_features(self, df):
        """Buat feature engineering yang lebih canggih dengan optimasi"""
        start_time = time.time()
        print("Creating advanced features...")
        
        # Target variable
        df['is_profitable'] = (df['Profit'] > 0).astype(int)
        
        features = []
        
        # 1. Fibonacci Level Analysis
        if 'LevelFibo' in df.columns:
            # Encode levels
            le_fibo = LabelEncoder()
            df['LevelFibo_encoded'] = le_fibo.fit_transform(df['LevelFibo'].fillna('UNKNOWN'))
            self.label_encoders['LevelFibo'] = le_fibo
            features.append('LevelFibo_encoded')
            
            # Level type indicators
            df['is_buy_level'] = df['LevelFibo'].str.startswith('B_').fillna(False).astype(int)
            df['is_sell_level'] = df['LevelFibo'].str.startswith('S_').fillna(False).astype(int)
            features.extend(['is_buy_level', 'is_sell_level'])
            
            # Level strength (based on historical performance)
            strong_levels = ['B_0', 'B_-1.8', 'S_0']  # From previous analysis
            df['is_strong_level'] = df['LevelFibo'].isin(strong_levels).astype(int)
            features.append('is_strong_level')
        
        # 2. Trade Type
        if 'Type' in df.columns:
            le_type = LabelEncoder()
            df['Type_encoded'] = le_type.fit_transform(df['Type'])
            self.label_encoders['Type'] = le_type
            features.append('Type_encoded')
        
        # 3. Session Analysis
        session_cols = ['SessionEurope', 'SessionUS', 'SessionAsia']
        for col in session_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                features.append(col)
        
        # Session strength (based on historical performance)
        if all(col in df.columns for col in session_cols):
            df['session_strength'] = (
                df['SessionEurope'] * 0.405 +  # Europe: 40.5% win rate
                df['SessionUS'] * 0.401 +      # US: 40.1% win rate
                df['SessionAsia'] * 0.397      # Asia: 39.7% win rate
            )
            features.append('session_strength')
        
        # 4. Risk-Reward Analysis
        if all(col in df.columns for col in ['OpenPrice', 'TP', 'SL']):
            # Risk-reward ratio
            df['risk_reward_ratio'] = abs(df['TP'] - df['OpenPrice']) / abs(df['OpenPrice'] - df['SL'])
            df['risk_reward_ratio'] = df['risk_reward_ratio'].fillna(1.0)
            
            # Distance to SL and TP (in pips)
            df['sl_distance_pips'] = abs(df['OpenPrice'] - df['SL']) / df['OpenPrice'] * 10000
            df['tp_distance_pips'] = abs(df['TP'] - df['OpenPrice']) / df['OpenPrice'] * 10000
            
            # Risk categories
            df['low_risk'] = (df['sl_distance_pips'] <= 20).astype(int)
            df['medium_risk'] = ((df['sl_distance_pips'] > 20) & (df['sl_distance_pips'] <= 50)).astype(int)
            df['high_risk'] = (df['sl_distance_pips'] > 50).astype(int)
            
            features.extend(['risk_reward_ratio', 'sl_distance_pips', 'tp_distance_pips', 
                           'low_risk', 'medium_risk', 'high_risk'])
        
        # 5. Volume Analysis
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(df['Volume'].median())
            
            # Volume categories
            volume_q25 = df['Volume'].quantile(0.25)
            volume_q75 = df['Volume'].quantile(0.75)
            
            df['low_volume'] = (df['Volume'] <= volume_q25).astype(int)
            df['high_volume'] = (df['Volume'] >= volume_q75).astype(int)
            df['volume_normalized'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
            
            features.extend(['Volume', 'low_volume', 'high_volume', 'volume_normalized'])
        
        # 6. Profit Pattern Analysis
        if 'Profit' in df.columns:
            # Profit magnitude and patterns
            df['profit_magnitude'] = abs(df['Profit'])
            
            # Rolling statistics (for sequential patterns)
            window_size = min(20, len(df) // 10)
            df['profit_rolling_mean'] = df['Profit'].rolling(window=window_size, min_periods=1).mean()
            df['profit_rolling_std'] = df['Profit'].rolling(window=window_size, min_periods=1).std().fillna(0)
            df['profit_trend'] = df['Profit'].rolling(window=window_size, min_periods=1).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
            ).fillna(0)
            
            features.extend(['profit_magnitude', 'profit_rolling_mean', 'profit_rolling_std', 'profit_trend'])
        
        # 7. Time-based Features
        df['trade_sequence'] = df.index
        df['trade_sequence_norm'] = df['trade_sequence'] / len(df)
        features.append('trade_sequence_norm')
        
        # 8. Interaction Features
        if 'is_buy_level' in df.columns and 'SessionEurope' in df.columns:
            df['buy_europe_interaction'] = df['is_buy_level'] * df['SessionEurope']
            df['sell_us_interaction'] = df['is_sell_level'] * df['SessionUS']
            features.extend(['buy_europe_interaction', 'sell_us_interaction'])
        
        # Clean and prepare final dataset
        feature_df = df[features].fillna(0)
        target = df['is_profitable']
        
        self.feature_names = features
        
        # Performance tracking
        processing_time = time.time() - start_time
        print(f"Feature engineering completed in {processing_time:.2f} seconds")
        
        print(f"Created {len(features)} features:")
        for i, feat in enumerate(features, 1):
            print(f"  {i:2d}. {feat}")
        
        return feature_df, target
    
    def create_ensemble_model_optimized(self):
        """Create optimized ensemble with better hyperparameters"""
        
        # Optimized individual models based on your data characteristics
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,        # Increased for better performance
                max_depth=12,            # Deeper trees for complex patterns
                min_samples_split=3,     # More sensitive to patterns
                min_samples_leaf=1,      # Allow more granular decisions
                max_features='sqrt',     # Optimal feature selection
                bootstrap=True,
                oob_score=True,          # Out-of-bag evaluation
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,        # More boosting rounds
                max_depth=8,             # Deeper trees
                learning_rate=0.08,      # Slower learning for better generalization
                subsample=0.8,           # Prevent overfitting
                max_features='sqrt',     # Feature randomness
                random_state=42,
                validation_fraction=0.1, # Early stopping validation
                n_iter_no_change=10     # Early stopping patience
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=2000,           # More iterations
                solver='liblinear',      # Better for small datasets
                C=0.5,                   # L2 regularization
                class_weight='balanced', # Handle class imbalance
                penalty='l2'
            ),
            'extra_trees': ExtraTreesClassifier(  # Added ExtraTrees for diversity
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        }
        
        # Create weighted voting classifier with optimized weights
        estimators = [(name, model) for name, model in models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=-1,
            weights=[0.3, 0.3, 0.2, 0.2]  # RF and GB get higher weights
        )
        
        self.individual_models = models
        self.ensemble_model = ensemble
        
        return ensemble
    
    def train_ensemble(self, max_files=20, sample_per_file=1000):
        """Train ensemble model dengan data yang ada"""
        print("=" * 70)
        print("ENSEMBLE SIGNAL DETECTOR TRAINING")
        print("=" * 70)
        
        # Load data
        csv_files = glob.glob(f"{self.data_path}/*.csv")
        if not csv_files:
            print(f"[ERROR] No CSV files found in {self.data_path}")
            return None
        
        print(f"Found {len(csv_files)} files, processing {min(max_files, len(csv_files))}")
        
        all_data = []
        for i, file_path in enumerate(csv_files[:max_files]):
            try:
                filename = os.path.basename(file_path)
                print(f"[{i+1:2d}] Loading: {filename}")
                
                df = pd.read_csv(file_path, nrows=sample_per_file)
                df = df[df['Type'].isin(['BUY', 'SELL'])].copy()
                
                if len(df) > 0:
                    all_data.append(df)
                    print(f"     Added {len(df)} trades")
                
            except Exception as e:
                print(f"     [ERROR] {str(e)}")
                continue
        
        if not all_data:
            print("[ERROR] No data loaded")
            return None
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal trades: {len(combined_df):,}")
        
        # Feature engineering
        X, y = self.prepare_advanced_features(combined_df)
        
        print(f"\nTarget distribution:")
        print(f"  Profitable: {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"  Losing: {(y==0).sum():,} ({(1-y.mean())*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create ensemble
        ensemble = self.create_ensemble_model_optimized()
        
        print(f"\nTraining ensemble with {len(self.individual_models)} models...")
        
        # Train individual models and evaluate
        individual_scores = {}
        
        for name, model in self.individual_models.items():
            print(f"\nTraining {name}...")
            
            if name in ['logistic_regression', 'svm']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                X_cv = X_train_scaled
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                X_cv = X_train
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='accuracy')
            
            individual_scores[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  AUC: {auc:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train ensemble
        print(f"\nTraining ensemble model...")
        
        # Prepare data for ensemble (mix of scaled and unscaled)
        X_train_ensemble = X_train.copy()
        X_test_ensemble = X_test.copy()
        
        ensemble.fit(X_train_ensemble, y_train)
        y_pred_ensemble = ensemble.predict(X_test_ensemble)
        y_prob_ensemble = ensemble.predict_proba(X_test_ensemble)[:, 1]
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_auc = roc_auc_score(y_test, y_prob_ensemble)
        
        print(f"\n" + "=" * 50)
        print("ENSEMBLE RESULTS")
        print("=" * 50)
        print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")
        print(f"Ensemble AUC: {ensemble_auc:.3f}")
        
        print(f"\nIndividual Model Performance:")
        for name, scores in individual_scores.items():
            print(f"  {name:<20}: Acc={scores['accuracy']:.3f}, AUC={scores['auc']:.3f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_ensemble))
        
        # Cache model performance
        self.model_performance = {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'individual_scores': individual_scores
        }
        
        return ensemble
    
    def predict_signal_strength(self, market_data):
        """Predict signal strength menggunakan ensemble"""
        if self.ensemble_model is None:
            return {"error": "Model not trained"}
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df_input = pd.DataFrame([market_data])
            
            # Apply same feature engineering
            for col, le in self.label_encoders.items():
                if col in df_input.columns:
                    try:
                        df_input[f"{col}_encoded"] = le.transform(df_input[col])
                    except:
                        df_input[f"{col}_encoded"] = 0
            
            # Create all features (simplified for prediction)
            feature_values = []
            for feature in self.feature_names:
                if feature in df_input.columns:
                    feature_values.append(df_input[feature].iloc[0])
                else:
                    feature_values.append(0)  # Default value
            
            X_input = np.array(feature_values).reshape(1, -1)
            
            # Predict
            probability = self.ensemble_model.predict_proba(X_input)[0][1]
            prediction = self.ensemble_model.predict(X_input)[0]
            
            # Individual model predictions for analysis
            individual_predictions = {}
            for name, model in self.individual_models.items():
                try:
                    if name in ['logistic_regression', 'svm']:
                        X_scaled = self.scaler.transform(X_input)
                        prob = model.predict_proba(X_scaled)[0][1]
                    else:
                        prob = model.predict_proba(X_input)[0][1]
                    individual_predictions[name] = prob
                except:
                    individual_predictions[name] = 0.5
            
            # Signal strength classification
            if probability >= 0.7:
                strength = "VERY_STRONG"
                recommendation = "STRONG_BUY" if prediction == 1 else "STRONG_SELL"
            elif probability >= 0.6:
                strength = "STRONG"
                recommendation = "TAKE_TRADE"
            elif probability >= 0.55:
                strength = "MEDIUM"
                recommendation = "CONSIDER_TRADE"
            else:
                strength = "WEAK"
                recommendation = "AVOID_TRADE"
            
            end_time = time.time()
            self.prediction_time += end_time - start_time
            
            return {
                'ensemble_probability': probability,
                'prediction': int(prediction),
                'signal_strength': strength,
                'recommendation': recommendation,
                'individual_models': individual_predictions,
                'confidence': abs(probability - 0.5) * 2  # 0 to 1 scale
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def save_ensemble_model(self, filepath="models/ensemble_signal_detector.pkl"):
        """Save ensemble model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'individual_models': self.individual_models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Ensemble model saved to {filepath}")
    
    def load_ensemble_model(self, filepath="models/ensemble_signal_detector.pkl"):
        """Load ensemble model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.ensemble_model = model_data['ensemble_model']
            self.individual_models = model_data['individual_models']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            print(f"Ensemble model loaded from {filepath}")
            return True
        return False

def main():
    """Main function untuk training ensemble"""
    detector = EnsembleSignalDetector()
    
    # Train ensemble
    ensemble = detector.train_ensemble(max_files=15, sample_per_file=800)
    
    if ensemble:
        # Save model
        detector.save_ensemble_model()
        
        # Test predictions
        print(f"\n" + "=" * 70)
        print("TESTING ENSEMBLE PREDICTIONS")
        print("=" * 70)
        
        test_cases = [
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
                'Volume': 0.15
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {test_case}")
            
            result = detector.predict_signal_strength(test_case)
            print(f"Ensemble Result:")
            for key, value in result.items():
                if key != 'individual_models':
                    print(f"  {key}: {value}")
            
            print(f"Individual Models:")
            for model, prob in result.get('individual_models', {}).items():
                print(f"  {model}: {prob:.3f}")

if __name__ == "__main__":
    main()
