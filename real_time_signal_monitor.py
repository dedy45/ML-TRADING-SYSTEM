#!/usr/bin/env python3
"""
REAL-TIME SIGNAL MONITOR
Monitor sinyal trading secara real-time dengan ML
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json

class RealTimeSignalMonitor:
    """Monitor sinyal trading real-time"""
    
    def __init__(self):
        self.signal_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'avg_profit_per_signal': 0.0
        }
        
        # Load trained model if available
        self.optimizer = None
        try:
            from advanced_signal_optimizer import AdvancedSignalOptimizer
            self.optimizer = AdvancedSignalOptimizer()
            if self.optimizer.load_model():
                print("[INFO] ML Model loaded successfully")
            else:
                print("[WARN] No trained ML model found")
        except Exception as e:
            print(f"[WARN] Could not load ML optimizer: {e}")
    
    def analyze_current_market(self, market_data):
        """Analisis kondisi pasar saat ini"""
        
        # Basic Fibonacci signal detection
        from fibonacci_signal_detector import FibonacciSignalDetector
        fib_detector = FibonacciSignalDetector()
        
        signal_result = fib_detector.detect_signal(market_data)
        
        # Enhanced with ML if available
        if self.optimizer and self.optimizer.model:
            try:
                ml_result = self.optimizer.get_signal_strength(market_data)
                signal_result['ml_probability'] = ml_result['win_probability']
                signal_result['ml_strength'] = ml_result['signal_strength']
                signal_result['ml_recommendation'] = ml_result['recommendation']
            except Exception as e:
                signal_result['ml_error'] = str(e)
        
        # Timestamp
        signal_result['timestamp'] = datetime.now().isoformat()
        
        return signal_result
    
    def log_signal(self, signal_data, actual_result=None):
        """Log sinyal dan hasil aktual"""
        signal_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal_data': signal_data,
            'actual_result': actual_result
        }
        
        self.signal_history.append(signal_entry)
        
        # Update metrics if actual result is provided
        if actual_result:
            self.performance_metrics['total_signals'] += 1
            if actual_result.get('profit', 0) > 0:
                self.performance_metrics['profitable_signals'] += 1
            
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['profitable_signals'] / 
                self.performance_metrics['total_signals']
            )
            
            profit = actual_result.get('profit', 0)
            self.performance_metrics['total_profit'] += profit
            self.performance_metrics['avg_profit_per_signal'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['total_signals']
            )
    
    def get_performance_report(self):
        """Generate performance report"""
        report = {
            'current_time': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'recent_signals': self.signal_history[-10:] if self.signal_history else [],
            'total_signals_logged': len(self.signal_history)
        }
        
        return report
    
    def save_performance_log(self, filename=None):
        """Save performance log to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/signal_performance_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.get_performance_report(), f, indent=2)
        
        print(f"Performance log saved to {filename}")

def demo_real_time_monitoring():
    """Demo monitoring real-time"""
    monitor = RealTimeSignalMonitor()
    
    # Simulasi data pasar
    sample_market_conditions = [
        {
            'LevelFibo': 'B_0',
            'Type': 'BUY',
            'SessionEurope': 1,
            'SessionUS': 0,
            'SessionAsia': 0,
            'OpenPrice': 2650.50,
            'TP': 2655.50,
            'SL': 2648.50,
            'Volume': 0.1
        },
        {
            'LevelFibo': 'B_-1.8',
            'Type': 'BUY',
            'SessionEurope': 0,
            'SessionUS': 1,
            'SessionAsia': 0,
            'OpenPrice': 2651.20,
            'TP': 2656.20,
            'SL': 2649.20,
            'Volume': 0.2
        },
        {
            'LevelFibo': 'S_1',
            'Type': 'SELL',
            'SessionEurope': 0,
            'SessionUS': 0,
            'SessionAsia': 1,
            'OpenPrice': 2652.80,
            'TP': 2647.80,
            'SL': 2654.80,
            'Volume': 0.1
        }
    ]
    
    print("=" * 70)
    print("REAL-TIME SIGNAL MONITORING DEMO")
    print("=" * 70)
    
    for i, market_data in enumerate(sample_market_conditions, 1):
        print(f"\n--- SIGNAL {i} ---")
        print(f"Market Data: {market_data}")
        
        # Analyze signal
        signal_result = monitor.analyze_current_market(market_data)
        
        print("Signal Analysis:")
        for key, value in signal_result.items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
        
        # Simulate actual result (for demo)
        simulated_result = {
            'profit': np.random.choice([5.0, -2.0], p=[0.6, 0.4]),  # 60% win rate simulation
            'duration_minutes': np.random.randint(15, 120)
        }
        
        # Log signal
        monitor.log_signal(signal_result, simulated_result)
        
        print(f"Simulated Result: {simulated_result}")
        
        time.sleep(1)  # Simulate time between signals
    
    # Performance report
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    report = monitor.get_performance_report()
    metrics = report['performance_metrics']
    
    print(f"Total Signals: {metrics['total_signals']}")
    print(f"Profitable Signals: {metrics['profitable_signals']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Total Profit: {metrics['total_profit']:.2f}")
    print(f"Avg Profit/Signal: {metrics['avg_profit_per_signal']:.2f}")
    
    # Save report
    monitor.save_performance_log()

if __name__ == "__main__":
    demo_real_time_monitoring()
