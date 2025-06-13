"""
Clean Advanced Trading ML Pipeline - No Unicode Issues
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

class CleanAdvancedTradingML:
    """Clean advanced trading ML without unicode issues"""
    
    def __init__(self, experiment_name="clean_trading_prediction"):
        print("Starting Clean Advanced Trading ML Pipeline")
        print("=" * 60)
        
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        try:
            mlflow.set_tracking_uri("./mlruns")
            mlflow.set_experiment(experiment_name)
            print("MLflow setup successful")
        except Exception as e:
            print(f"MLflow setup warning: {e}")
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data/processed", "data/features", "models", 
            "experiments", "logs", "mlruns"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print("Directories created")
    
    def load_data(self, num_files=15, max_rows_per_file=1000):
        """Load data with proper filtering"""
        print(f"Loading data from {num_files} files...")
        
        with mlflow.start_run(run_name="data_loading", nested=True):
            csv_files = glob.glob("dataBT/*.csv")
            
            if not csv_files:
                print("No CSV files found")
                return False
            
            # Sample files randomly
            import random
            random.seed(42)
            sample_files = random.sample(csv_files, min(num_files, len(csv_files)))
            
            dataframes = []
            total_rows = 0
            
            for i, file in enumerate(sample_files):
                try:
                    df = pd.read_csv(file)
                    
                    # Remove system messages
                    system_messages = ['INIT_SUCCESS', 'SYSTEM_READY', 'SESSION_COMPLETE', 'DEINIT_COMPLETE']
                    for msg in system_messages:
                        df = df[df['Type'] != msg]
                        if 'LevelFibo' in df.columns:
                            df = df[df['LevelFibo'] != msg]
                    
                    if len(df) > 0:
                        if len(df) > max_rows_per_file:
                            df = df.sample(n=max_rows_per_file, random_state=42)
                        
                        df['source_file'] = os.path.basename(file)
                        dataframes.append(df)
                        total_rows += len(df)
                        
                        print(f"   File {i+1}: {len(df)} rows")
                    
                except Exception as e:
                    print(f"   Error in file {i+1}: {str(e)}")
                    continue
            
            if dataframes:
                self.data = pd.concat(dataframes, ignore_index=True)
                
                # Log basic metrics
                buy_count = len(self.data[self.data['Type'] == 'BUY'])
                sell_count = len(self.data[self.data['Type'] == 'SELL'])
                profitable_count = len(self.data[self.data['Profit'] > 0])
                
                mlflow.log_metric("total_trades", len(self.data))
                mlflow.log_metric("buy_trades", buy_count)
                mlflow.log_metric("sell_trades", sell_count)
                mlflow.log_metric("profitable_trades", profitable_count)
                mlflow.log_metric("win_rate", profitable_count / len(self.data))
                
                print(f"Data loaded: {len(self.data)} trades from {len(dataframes)} files")
                print(f"   BUY: {buy_count}, SELL: {sell_count}")
                print(f"   Win Rate: {profitable_count/len(self.data)*100:.1f}%")
                
                return True
            else:
                print("No data loaded")
                return False
    
    def create_advanced_features(self):
        """Create advanced features with Fibonacci analysis"""
        if self.data is None:
            print("No data loaded")
            return False
        
        print("Creating advanced features with Fibonacci analysis...")
        
        with mlflow.start_run(run_name="feature_engineering", nested=True):
            df = self.data.copy()
            
            # Convert timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Target variables
            df['is_profitable'] = (df['Profit'] > 0).astype(int)
            df['is_big_win'] = (df['Profit'] > 10).astype(int)
            
            # Basic features
            df['is_buy'] = (df['Type'] == 'BUY').astype(int)
            df['price_level'] = df['OpenPrice']
            df['volume_size'] = df['Volume']
            
            # Time features
            df['hour'] = df['Timestamp'].dt.hour
            df['day_of_week'] = df['Timestamp'].dt.dayofweek
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Fibonacci features with proper encoding
            if 'LevelFibo' in df.columns:
                # Handle categorical LevelFibo properly
                df['level_fibo_raw'] = df['LevelFibo'].astype(str)
                
                # Extract numeric values (e.g., 'B_0' -> 0, 'S_-0.9' -> -0.9)
                df['level_fibo_numeric'] = 0
                for idx, value in enumerate(df['level_fibo_raw']):
                    try:
                        if '_' in value:
                            numeric_part = value.split('_')[1]
                            df.loc[idx, 'level_fibo_numeric'] = float(numeric_part)
                    except:
                        pass
                
                # Direction features
                df['is_buy_fibo'] = df['level_fibo_raw'].str.contains('B_', na=False).astype(int)
                df['is_sell_fibo'] = df['level_fibo_raw'].str.contains('S_', na=False).astype(int)
                
                # Popular fibonacci levels
                fib_counts = df['level_fibo_raw'].value_counts()
                top_levels = fib_counts.head(5).index.tolist()
                
                for level in top_levels:
                    clean_name = level.replace('-', 'neg').replace('.', 'dot').replace('_', '')
                    df[f'fibo_{clean_name}'] = (df['level_fibo_raw'] == level).astype(int)
            
            # Support/Resistance features
            if 'Level1Above' in df.columns and 'Level1Below' in df.columns:
                df['level_above'] = pd.to_numeric(df['Level1Above'], errors='coerce').fillna(0)
                df['level_below'] = pd.to_numeric(df['Level1Below'], errors='coerce').fillna(0)
                df['distance_to_above'] = abs(df['OpenPrice'] - df['level_above'])
                df['distance_to_below'] = abs(df['OpenPrice'] - df['level_below'])
                df['nearest_level_distance'] = np.minimum(df['distance_to_above'], df['distance_to_below'])
            
            # Session features
            session_cols = ['SessionAsia', 'SessionEurope', 'SessionUS']
            for col in session_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['session_score'] = sum(df[col] for col in session_cols if col in df.columns)
            df['multi_session'] = (df['session_score'] > 1).astype(int)
            
            # Risk management features
            df['tp_size'] = pd.to_numeric(df['TP'], errors='coerce').fillna(0)
            df['sl_size'] = pd.to_numeric(df['SL'], errors='coerce').fillna(0)
            df['risk_reward_ratio'] = np.where(df['sl_size'] > 0, df['tp_size'] / df['sl_size'], 0)
            
            # Performance features
            df['mae_pips'] = pd.to_numeric(df['MAE_pips'], errors='coerce').fillna(0)
            df['mfe_pips'] = pd.to_numeric(df['MFE_pips'], errors='coerce').fillna(0)
            df['mfe_mae_ratio'] = np.where(df['mae_pips'] > 0, df['mfe_pips'] / df['mae_pips'], 0)
            
            # Price movement features
            df['price_change'] = df['ClosePrice'] - df['OpenPrice']
            df['price_change_pct'] = df['price_change'] / df['OpenPrice'] * 100
            
            # Select final features
            base_features = [
                'is_buy', 'price_level', 'volume_size',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'session_score', 'multi_session',
                'tp_size', 'sl_size', 'risk_reward_ratio',
                'mae_pips', 'mfe_pips', 'mfe_mae_ratio',
                'price_change_pct'
            ]
            
            # Add fibonacci features if available
            fib_features = [col for col in df.columns if 'fibo' in col or 'level_' in col]
            feature_columns = base_features + fib_features
            
            # Ensure all features are numeric
            for col in feature_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create final feature set
            available_features = [col for col in feature_columns if col in df.columns]
            self.features = df[available_features + ['is_profitable', 'is_big_win']].copy()
            
            # Remove any remaining NaN rows
            self.features = self.features.dropna()
            
            # Log metrics
            mlflow.log_param("num_features", len(available_features))
            mlflow.log_param("final_rows", len(self.features))
            mlflow.log_metric("profitable_rate", self.features['is_profitable'].mean())
            mlflow.log_metric("big_win_rate", self.features['is_big_win'].mean())
            
            print(f"Features created: {len(self.features)} rows, {len(available_features)} features")
            print(f"   Fibonacci features: {len(fib_features)}")
            print(f"   Profitable rate: {self.features['is_profitable'].mean()*100:.1f}%")
            
            return True
    
    def train_models(self, target='is_profitable'):
        """Train multiple models"""
        if self.features is None:
            print("No features available")
            return False
        
        print(f"Training models for target: {target}")
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, classification_report
        
        # Prepare data
        feature_cols = [col for col in self.features.columns if col not in ['is_profitable', 'is_big_win']]
        X = self.features[feature_cols]
        y = self.features[target]
        
        print(f"   Training data: {len(X)} samples, {len(feature_cols)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models_to_train = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        
        for model_name, model in models_to_train.items():
            with mlflow.start_run(run_name=f"{model_name}_{target}", nested=True):
                print(f"   Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                cv_mean = cv_scores.mean()
                
                # Log metrics
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("target", target)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("cv_mean", cv_mean)
                
                # Save model
                mlflow.sklearn.log_model(model, f"{model_name}_{target}")
                
                # Store results
                self.models[f"{model_name}_{target}"] = model
                self.results[f"{model_name}_{target}"] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_mean
                }
                
                print(f"      Accuracy: {accuracy:.3f}")
                print(f"      CV Score: {cv_mean:.3f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = f"{model_name}_{target}"
        
        print(f"Best model: {best_model} (Accuracy: {best_score:.3f})")
        return True
    
    def run_pipeline(self, num_files=15, target='is_profitable'):
        """Run complete pipeline"""
        print("Starting Clean Advanced Trading ML Pipeline")
        print("=" * 60)
        
        with mlflow.start_run(run_name=f"clean_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            steps = [
                ("Load Data", lambda: self.load_data(num_files=num_files)),
                ("Create Features", self.create_advanced_features),
                ("Train Models", lambda: self.train_models(target=target))
            ]
            
            for step_name, step_func in steps:
                print(f"\nStep: {step_name}")
                print("-" * 40)
                
                try:
                    result = step_func()
                    if not result:
                        print(f"Pipeline failed at: {step_name}")
                        return False
                    print(f"{step_name} completed successfully")
                except Exception as e:
                    print(f"Error in {step_name}: {str(e)}")
                    return False
            
            print("\n" + "=" * 60)
            print("Pipeline Completed Successfully!")
            
            if self.results:
                print("\nModel Performance Summary:")
                for model_name, metrics in self.results.items():
                    print(f"   {model_name}: Accuracy={metrics['accuracy']:.3f}")
            
            print("\nNext Steps:")
            print("   1. Run: mlflow ui")
            print("   2. Open: http://localhost:5000")
            
            return True

def main():
    """Main function"""
    pipeline = CleanAdvancedTradingML()
    success = pipeline.run_pipeline(num_files=20, target='is_profitable')
    
    if success:
        print("\nTo view results, run: mlflow ui")

if __name__ == "__main__":
    main()
