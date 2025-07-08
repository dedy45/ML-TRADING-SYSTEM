"""
Simple ML Pipeline untuk Trading Data - Version 1.0
Focus: Proof of concept dengan sample data yang kecil
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SimpleTradingML:
    """Simple trading ML pipeline untuk testing"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.features = None
        
        # Create directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
    def load_sample_data(self, num_files=3):
        """Load sample data dari beberapa file CSV pertama"""
        print(f"🔄 Loading sample data from {num_files} files...")
        
        # Get CSV files
        csv_files = glob.glob("dataBT/*.csv")
        if not csv_files:
            print("❌ No CSV files found in dataBT directory")
            return False
            
        print(f"📁 Found {len(csv_files)} total files")
        
        # Take only first few files
        sample_files = csv_files[:num_files]
        
        dataframes = []
        for i, file in enumerate(sample_files):
            try:
                print(f"📖 Reading file {i+1}: {os.path.basename(file)}")
                df = pd.read_csv(file)
                
                # Skip INIT_SUCCESS rows (they're system messages)
                df = df[df['Type'] != 'INIT_SUCCESS']
                
                if len(df) > 0:
                    dataframes.append(df)
                    print(f"   ✅ Loaded {len(df)} trading records")
                else:
                    print(f"   ⚠️ No trading data found")
                    
            except Exception as e:
                print(f"   ❌ Error reading {file}: {str(e)}")
                continue
        
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            print(f"🎉 Total data loaded: {len(self.data)} trading records")
            
            # Show basic info
            print(f"📊 Data info:")
            print(f"   - Columns: {len(self.data.columns)}")
            print(f"   - BUY trades: {len(self.data[self.data['Type'] == 'BUY'])}")
            print(f"   - SELL trades: {len(self.data[self.data['Type'] == 'SELL'])}")
            print(f"   - Profitable trades: {len(self.data[self.data['Profit'] > 0])}")
            print(f"   - Loss trades: {len(self.data[self.data['Profit'] < 0])}")
            
            return True
        else:
            print("❌ No data could be loaded")
            return False
    
    def basic_feature_engineering(self):
        """Create basic features untuk ML"""
        if self.data is None:
            print("❌ No data loaded. Run load_sample_data() first")
            return False
            
        print("🔄 Creating basic features...")
        
        df = self.data.copy()
        
        # 1. Target variable: is trade profitable?
        df['is_profitable'] = (df['Profit'] > 0).astype(int)
        
        # 2. Binary features
        df['is_buy'] = (df['Type'] == 'BUY').astype(int)
        df['is_sell'] = (df['Type'] == 'SELL').astype(int)
        
        # 3. Trading sessions
        df['session_asia'] = df['SessionAsia']
        df['session_europe'] = df['SessionEurope'] 
        df['session_us'] = df['SessionUS']
        
        # 4. Price features
        df['price_level'] = df['OpenPrice']
        df['volume_size'] = df['Volume']
        
        # 5. Risk features
        df['tp_size'] = df['TP']
        df['sl_size'] = df['SL']
        df['risk_reward'] = np.where(df['SL'] > 0, df['TP'] / df['SL'], 0)
        
        # 6. Performance features
        df['mae_pips'] = df['MAE_pips']
        df['mfe_pips'] = df['MFE_pips']
        
        # Select features for ML
        feature_columns = [
            'is_buy', 'is_sell', 
            'session_asia', 'session_europe', 'session_us',
            'price_level', 'volume_size',
            'tp_size', 'sl_size', 'risk_reward',
            'mae_pips', 'mfe_pips'
        ]
        
        # Remove any rows with missing target
        df = df.dropna(subset=['is_profitable'])
        
        # Handle missing values in features
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        self.features = df[feature_columns + ['is_profitable']].copy()
        
        print(f"✅ Features created: {len(self.features)} rows, {len(feature_columns)} features")
        print(f"📊 Target distribution:")
        print(f"   - Profitable: {sum(self.features['is_profitable'])} ({sum(self.features['is_profitable'])/len(self.features)*100:.1f}%)")
        print(f"   - Loss: {len(self.features) - sum(self.features['is_profitable'])} ({(len(self.features) - sum(self.features['is_profitable']))/len(self.features)*100:.1f}%)")
        
        return True
    
    def train_simple_model(self):
        """Train simple model dengan scikit-learn"""
        if self.features is None:
            print("❌ No features created. Run basic_feature_engineering() first")
            return False
            
        print("🔄 Training simple model...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            
            # Prepare data
            feature_cols = [col for col in self.features.columns if col != 'is_profitable']
            X = self.features[feature_cols]
            y = self.features['is_profitable']
            
            print(f"📊 Training data: {len(X)} samples, {len(feature_cols)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Test Accuracy: {accuracy:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🎯 Top 5 Important Features:")
            for idx, row in feature_importance.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            # Trading metrics
            profitable_pred = sum(y_pred)
            actual_profitable = sum(y_test)
            print(f"\n📈 Trading Analysis:")
            print(f"   - Predicted profitable trades: {profitable_pred}/{len(y_test)} ({profitable_pred/len(y_test)*100:.1f}%)")
            print(f"   - Actual profitable trades: {actual_profitable}/{len(y_test)} ({actual_profitable/len(y_test)*100:.1f}%)")
            
            return True
            
        except ImportError:
            print("❌ Scikit-learn not available. Please install: pip install scikit-learn")
            return False
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save results untuk inspection"""
        if self.data is not None:
            output_file = "data/processed/sample_trading_data.csv"
            self.data.to_csv(output_file, index=False)
            print(f"💾 Raw data saved to: {output_file}")
        
        if self.features is not None:
            output_file = "data/processed/sample_features.csv"
            self.features.to_csv(output_file, index=False)
            print(f"💾 Features saved to: {output_file}")
        
        if self.model is not None:
            try:
                import joblib
                model_file = "models/simple_trading_model.joblib"
                joblib.dump(self.model, model_file)
                print(f"💾 Model saved to: {model_file}")
            except ImportError:
                print("⚠️ Joblib not available for model saving")
    
    def run_complete_pipeline(self):
        """Run complete simple pipeline"""
        print("Starting Simple Trading ML Pipeline")
        print("=" * 50)
        
        steps = [
            ("Load Sample Data", lambda: self.load_sample_data(num_files=3)),
            ("Feature Engineering", self.basic_feature_engineering),
            ("Train Model", self.train_simple_model),
            ("Save Results", self.save_results)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            print("-" * 30)
            
            success = step_func()
            if not success:
                print(f"❌ Pipeline stopped at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 Simple Pipeline Completed Successfully!")
        print("\n📋 What was created:")
        print("   - Sample trading data processed")
        print("   - Basic features engineered") 
        print("   - Simple ML model trained")
        print("   - Results saved to data/processed/")
        print("\n📋 Next Steps:")
        print("   1. Review results in data/processed/")
        print("   2. If satisfied, expand to more data")
        print("   3. Add MLflow tracking")
        print("   4. Implement advanced features")
        
        return True

def main():
    """Main function untuk testing"""
    pipeline = SimpleTradingML()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()
