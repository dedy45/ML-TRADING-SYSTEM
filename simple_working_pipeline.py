#!/usr/bin/env python3
"""
🎯 SIMPLE WORKING ML PIPELINE
=============================

A simplified, working version of the ML trading pipeline that works with our sample data.
This is designed to be a quick test of the system without complex dependencies.

Author: ML Trading System
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.END}")

def print_header(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

class SimpleWorkingPipeline:
    """Simple ML pipeline that actually works with our data"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {}
    
    def load_data(self):
        """Load sample data"""
        print_header("LOADING DATA")
        
        try:
            data_file = Path('dataBT/sample_trading_data.csv')
            if not data_file.exists():
                print_error(f"Data file not found: {data_file}")
                return False
            
            self.data = pd.read_csv(data_file)
            print_success(f"Loaded {len(self.data)} trading records")
            print_info(f"Columns: {list(self.data.columns)}")
            
            # Quick data info
            profitable = (self.data['Profit'] > 0).sum()
            print_info(f"Profitable trades: {profitable}/{len(self.data)} ({profitable/len(self.data)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print_error(f"Error loading data: {e}")
            return False
    
    def create_features(self):
        """Create simple features from the data"""
        print_header("FEATURE ENGINEERING")
        
        try:
            df = self.data.copy()
            
            # Basic price features
            df['price_change'] = df['ClosePrice'] - df['OpenPrice']
            df['price_change_pct'] = (df['price_change'] / df['OpenPrice']) * 100
            df['price_range'] = abs(df['price_change'])
            
            print_success("Created price features")
            
            # Risk features
            df['risk_reward'] = df['MFE_pips'] / (df['MAE_pips'] + 0.01)  # Add small value to avoid division by zero
            df['mae_normalized'] = df['MAE_pips'] / df['OpenPrice'] * 10000  # Convert to basis points
            df['mfe_normalized'] = df['MFE_pips'] / df['OpenPrice'] * 10000
            
            print_success("Created risk features")
            
            # Session encoding
            session_dummies = pd.get_dummies(df['Session'], prefix='session')
            df = pd.concat([df, session_dummies], axis=1)
            
            # Type encoding
            df['is_buy'] = (df['Type'] == 'BUY').astype(int)
            
            print_success("Created categorical features")
            
            # Target variable
            df['is_profitable'] = (df['Profit'] > 0).astype(int)
            df['is_big_win'] = (df['Profit'] > 50).astype(int)  # Profits > 50
            
            print_success("Created target variables")
            
            self.data = df
            
            print_info(f"Final dataset shape: {df.shape}")
            print_info(f"Features created: {df.shape[1] - 11} new features")  # Original had 11 columns
            
            return True
            
        except Exception as e:
            print_error(f"Error creating features: {e}")
            return False
    
    def train_model(self):
        """Train a simple model"""
        print_header("TRAINING MODEL")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            
            # Select features
            feature_cols = [
                'OpenPrice', 'Volume', 'price_change', 'price_change_pct', 
                'price_range', 'risk_reward', 'mae_normalized', 'mfe_normalized',
                'is_buy'
            ]
            
            # Add session features if they exist
            session_cols = [col for col in self.data.columns if col.startswith('session_')]
            feature_cols.extend(session_cols)
            
            # Make sure all features exist
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            X = self.data[available_features]
            y = self.data['is_profitable']
            
            print_info(f"Using {len(available_features)} features: {available_features}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            print_info(f"Training set: {len(X_train)} samples")
            print_info(f"Test set: {len(X_test)} samples")
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            print_success("Model trained successfully")
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print_success(f"Test Accuracy: {accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print_info("Top 5 Most Important Features:")
            for _, row in feature_importance.head().iterrows():
                print(f"  {row['feature']:20} {row['importance']:.3f}")
            
            # Store results
            self.results = {
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': len(available_features)
            }
            
            return True
            
        except Exception as e:
            print_error(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_signals(self):
        """Generate trading signals"""
        print_header("GENERATING SIGNALS")
        
        try:
            if self.model is None:
                print_error("No trained model available")
                return False
            
            # Get feature columns from training
            feature_cols = [
                'OpenPrice', 'Volume', 'price_change', 'price_change_pct', 
                'price_range', 'risk_reward', 'mae_normalized', 'mfe_normalized',
                'is_buy'
            ]
            session_cols = [col for col in self.data.columns if col.startswith('session_')]
            feature_cols.extend(session_cols)
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            X = self.data[available_features]
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of profitable trade
            predictions = self.model.predict(X)
            
            # Create signals
            signals_df = self.data[['Symbol', 'Timestamp', 'Type', 'OpenPrice', 'Profit']].copy()
            signals_df['predicted_profitable'] = predictions
            signals_df['confidence'] = probabilities
            
            # Signal classification
            signals_df['signal_strength'] = pd.cut(
                signals_df['confidence'], 
                bins=[0, 0.4, 0.6, 0.8, 1.0],
                labels=['WEAK', 'MEDIUM', 'STRONG', 'VERY_STRONG']
            )
            
            # High confidence signals
            high_confidence = signals_df[signals_df['confidence'] >= 0.7]
            
            print_success(f"Generated signals for {len(signals_df)} trades")
            print_info(f"High confidence signals (>=70%): {len(high_confidence)}")
            print_info(f"Average confidence: {signals_df['confidence'].mean():.2%}")
            
            # Signal quality analysis
            correct_predictions = (signals_df['predicted_profitable'] == (signals_df['Profit'] > 0)).sum()
            signal_accuracy = correct_predictions / len(signals_df)
            
            print_success(f"Signal accuracy: {signal_accuracy:.2%}")
            
            # Save signals
            signals_file = 'results/trading_signals.csv'
            signals_df.to_csv(signals_file, index=False)
            print_success(f"Signals saved to: {signals_file}")
            
            return True
            
        except Exception as e:
            print_error(f"Error generating signals: {e}")
            return False
    
    def save_model(self):
        """Save the trained model"""
        print_header("SAVING MODEL")
        
        try:
            if self.model is None:
                print_error("No model to save")
                return False
            
            import joblib
            
            model_file = 'models/simple_trading_model.joblib'
            joblib.dump(self.model, model_file)
            print_success(f"Model saved to: {model_file}")
            
            # Save results summary
            results_file = 'results/model_summary.txt'
            with open(results_file, 'w') as f:
                f.write("SIMPLE ML TRADING MODEL SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Accuracy: {self.results.get('accuracy', 0):.2%}\n")
                f.write(f"Training samples: {self.results.get('train_size', 0)}\n")
                f.write(f"Test samples: {self.results.get('test_size', 0)}\n")
                f.write(f"Features used: {self.results.get('n_features', 0)}\n\n")
                
                if 'feature_importance' in self.results:
                    f.write("TOP FEATURES:\n")
                    for _, row in self.results['feature_importance'].head().iterrows():
                        f.write(f"  {row['feature']:20} {row['importance']:.3f}\n")
            
            print_success(f"Results saved to: {results_file}")
            
            return True
            
        except Exception as e:
            print_error(f"Error saving model: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print(f"{Colors.BOLD}{Colors.BLUE}")
        print("🎯 SIMPLE WORKING ML TRADING PIPELINE")
        print("=====================================")
        print(f"{Colors.END}")
        
        steps = [
            ("Load Data", self.load_data),
            ("Create Features", self.create_features),
            ("Train Model", self.train_model),
            ("Generate Signals", self.generate_signals),
            ("Save Model", self.save_model)
        ]
        
        success_count = 0
        
        for step_name, step_func in steps:
            if step_func():
                success_count += 1
            else:
                print_error(f"Pipeline stopped at: {step_name}")
                break
        
        # Summary
        print_header("PIPELINE SUMMARY")
        
        if success_count == len(steps):
            print_success("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print_info("Next steps:")
            print_info("1. Check results/ folder for outputs")
            print_info("2. Review trading_signals.csv for signal quality")
            print_info("3. Try: mlflow ui --port 5000 for experiment tracking")
            print_info("4. Run: python advanced_ml_pipeline_working.py for more features")
        else:
            print_error(f"Pipeline completed {success_count}/{len(steps)} steps")

def main():
    """Main function"""
    pipeline = SimpleWorkingPipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()