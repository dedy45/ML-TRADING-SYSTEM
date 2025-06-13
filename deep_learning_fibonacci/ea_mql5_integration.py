#!/usr/bin/env python3
"""
EA MQL5 Integration Helper for Fibonacci Deep Learning
Provides seamless integration between Python models and MQL5 EA
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class FibonacciEAIntegration:
    """
    Integration helper untuk EA MQL5
    Handles signal generation, file communication, and model inference
    """
    
    def __init__(self, model_path="models/fibonacci_signal_model.pkl", 
                 scaler_path="models/signal_scaler.pkl"):
        """Initialize EA integration"""
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.signal_file = "fibonacci_signal.json"
        self.request_file = "fibonacci_request.json"
        
        # Load model jika ada
        self.load_model()
        
        print("🔗 Fibonacci EA Integration initialized")
    
    def load_model(self):
        """Load trained model dan scaler"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print(f"✅ Model loaded: {self.model_path}")
            else:
                print(f"⚠️  Model not found: {self.model_path}")
                
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Scaler loaded: {self.scaler_path}")
            else:
                print(f"⚠️  Scaler not found: {self.scaler_path}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def extract_features_from_ea_data(self, market_data):
        """
        Extract features dari data yang dikirim EA MQL5
        
        Expected market_data format:
        {
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "current_price": 1.2345,
            "fibonacci_level": 0.0,  # B_0, B_-1.8, etc
            "session_asia": 0,
            "session_europe": 1,
            "session_us": 0,
            "tp": 1.2400,
            "sl": 1.2300,
            "hour": 14
        }
        """
        
        features = {}
        
        # Fibonacci features
        fib_level = market_data.get('fibonacci_level', 0.0)
        features['fib_b0'] = 1 if fib_level == 0.0 else 0
        features['fib_b_minus_18'] = 1 if fib_level == -1.8 else 0
        features['fib_level'] = fib_level
        
        # Signal strength (berdasarkan proven analysis)
        if fib_level == 0.0:
            features['signal_strength'] = 3      # 52.4% win rate
        elif fib_level == -1.8:
            features['signal_strength'] = 3     # 52.5% win rate
        elif fib_level == 1.8:
            features['signal_strength'] = 2     # 45.9% win rate
        else:
            features['signal_strength'] = 1
        
        # Session features
        features['europe_session'] = market_data.get('session_europe', 0)
        
        # Risk management
        tp = market_data.get('tp', 0)
        sl = market_data.get('sl', 1)  # Avoid division by zero
        if sl > 0:
            tp_sl_ratio = tp / sl
            features['tp_sl_ratio'] = min(tp_sl_ratio, 5.0)  # Cap at 5
        else:
            features['tp_sl_ratio'] = 2.0  # Default 2:1
        
        # Time features (simplified)
        hour = market_data.get('hour', 12)
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Additional derived features
        features['high_confidence_signal'] = (
            features['signal_strength'] >= 2 and 
            features['europe_session'] == 1 and
            features['tp_sl_ratio'] >= 2.0
        )
        
        # Convert to DataFrame untuk compatibility
        feature_df = pd.DataFrame([features])
        
        return feature_df
    
    def generate_signal(self, market_data):
        """
        Generate trading signal untuk EA MQL5
        
        Returns:
        {
            "signal_type": "BUY|SELL|HOLD",
            "confidence": 0.85,
            "entry_price": 1.2345,
            "stop_loss": 1.2300, 
            "take_profit": 1.2400,
            "position_size_pct": 2.0,
            "fibonacci_level": "B_0",
            "session": "Europe",
            "timestamp": "2025-06-12T15:30:00Z",
            "validity_seconds": 300
        }
        """
        
        if self.model is None or self.scaler is None:
            return {
                "signal_type": "HOLD",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Extract features
            features = self.extract_features_from_ea_data(market_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0, 1]
            prediction = self.model.predict(features_scaled)[0]
            
            # Determine signal type
            signal_type = "BUY" if prediction == 1 and prediction_proba >= 0.7 else "HOLD"
            
            # Calculate position size based on confidence
            if prediction_proba >= 0.9:
                position_size = 3.0  # High confidence
            elif prediction_proba >= 0.8:
                position_size = 2.0  # Medium confidence  
            elif prediction_proba >= 0.7:
                position_size = 1.0  # Low confidence
            else:
                position_size = 0.0  # No position
            
            # Create signal
            signal = {
                "signal_type": signal_type,
                "confidence": float(prediction_proba),
                "entry_price": market_data.get('current_price', 0),
                "stop_loss": market_data.get('sl', 0),
                "take_profit": market_data.get('tp', 0),
                "position_size_pct": position_size,
                "fibonacci_level": f"B_{market_data.get('fibonacci_level', 0)}",
                "session": self.get_active_session(market_data),
                "timestamp": datetime.now().isoformat(),
                "validity_seconds": 300,  # Valid for 5 minutes
                "model_version": "fibonacci_dl_v1.0"
            }
            
            return signal
            
        except Exception as e:
            return {
                "signal_type": "HOLD",
                "confidence": 0.0,
                "error": f"Prediction failed: {str(e)}"
            }
    
    def get_active_session(self, market_data):
        """Determine active trading session"""
        if market_data.get('session_europe', 0) == 1:
            return "Europe"
        elif market_data.get('session_us', 0) == 1:
            return "US"
        elif market_data.get('session_asia', 0) == 1:
            return "Asia"
        else:
            return "Unknown"
    
    def save_signal_for_ea(self, signal):
        """Save signal ke file untuk dibaca EA"""
        try:
            with open(self.signal_file, 'w') as f:
                json.dump(signal, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving signal: {e}")
            return False
    
    def read_ea_request(self):
        """Read request dari EA MQL5"""
        try:
            if os.path.exists(self.request_file):
                with open(self.request_file, 'r') as f:
                    request = json.load(f)
                # Delete request file setelah dibaca
                os.remove(self.request_file)
                return request
            return None
        except Exception as e:
            print(f"❌ Error reading EA request: {e}")
            return None
    
    def start_signal_server(self, polling_interval=1.0):
        """
        Start signal server untuk komunikasi dengan EA
        EA writes request → Python processes → Python writes signal
        """
        print("🚀 Starting Fibonacci Signal Server...")
        print(f"📁 Watching for requests: {self.request_file}")
        print(f"📁 Writing signals to: {self.signal_file}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Check for EA request
                request = self.read_ea_request()
                
                if request:
                    print(f"📨 Request received: {request.get('symbol', 'Unknown')}")
                    
                    # Generate signal
                    signal = self.generate_signal(request)
                    
                    # Save signal for EA
                    if self.save_signal_for_ea(signal):
                        print(f"📤 Signal sent: {signal['signal_type']} (confidence: {signal['confidence']:.1%})")
                    else:
                        print("❌ Failed to save signal")
                
                # Wait before next check
                time.sleep(polling_interval)
                
        except KeyboardInterrupt:
            print("\n⚠️  Signal server stopped by user")
        except Exception as e:
            print(f"❌ Signal server error: {e}")

def create_sample_ea_request():
    """Create sample request untuk testing"""
    sample_request = {
        "symbol": "XAUUSD",
        "timeframe": "M15", 
        "current_price": 2035.50,
        "fibonacci_level": 0.0,  # B_0 level
        "session_asia": 0,
        "session_europe": 1,  # Europe session active
        "session_us": 0,
        "tp": 2040.00,  # 2:1 ratio
        "sl": 2033.25,
        "hour": 14,
        "request_time": datetime.now().isoformat()
    }
    
    with open("fibonacci_request.json", 'w') as f:
        json.dump(sample_request, f, indent=2)
    
    print("📝 Sample EA request created: fibonacci_request.json")
    return sample_request

def test_integration():
    """Test EA integration"""
    print("🧪 Testing EA Integration...")
    
    # Initialize integration
    ea_integration = FibonacciEAIntegration()
    
    # Create sample request
    sample_request = create_sample_ea_request()
    
    # Generate signal
    signal = ea_integration.generate_signal(sample_request)
    
    # Save signal
    ea_integration.save_signal_for_ea(signal)
    
    print("\n📊 Test Results:")
    print(f"Signal Type: {signal['signal_type']}")
    print(f"Confidence: {signal['confidence']:.1%}")
    print(f"Fibonacci Level: {signal['fibonacci_level']}")
    print(f"Session: {signal['session']}")
    print(f"Position Size: {signal['position_size_pct']}%")
    
    print("\n✅ Integration test completed!")
    print("📁 Check fibonacci_signal.json for EA to read")

def main():
    """Main function"""
    print("🔗 Fibonacci EA MQL5 Integration Helper")
    print("=" * 50)
    
    choice = input("""
Choose option:
1. Test integration 
2. Start signal server
3. Create sample request
Enter choice (1-3): """).strip()
    
    if choice == "1":
        test_integration()
    elif choice == "2":
        ea_integration = FibonacciEAIntegration()
        ea_integration.start_signal_server()
    elif choice == "3":
        create_sample_ea_request()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
