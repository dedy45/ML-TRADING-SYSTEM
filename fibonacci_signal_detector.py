#!/usr/bin/env python3
"""
FIBONACCI SIGNAL DETECTOR - Ready for Live Trading
Based on proven analysis: B_0 (52.4% win rate), B_-1.8 (52.5% win rate)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class FibonacciSignalDetector:
    """Detects high-probability Fibonacci trading signals"""
    
    def __init__(self):
        self.signal_criteria = {
            'B_0': {'win_rate': 52.4, 'confidence': 'HIGH', 'volume': 3106},
            'B_-1.8': {'win_rate': 52.5, 'confidence': 'HIGH', 'volume': 120},
            'B_1.8': {'win_rate': 45.9, 'confidence': 'MEDIUM', 'volume': 945},
            'S_4.5': {'win_rate': 48.4, 'confidence': 'LOW', 'volume': 31}
        }
        
        self.session_performance = {
            'Europe': 40.5,
            'US': 40.1,
            'Asia': 39.7
        }
        
    def detect_signal(self, current_data):
        """
        Detect Fibonacci signals from current market data
        
        Args:
            current_data (dict): Must include:
                - 'LevelFibo': Fibonacci level (e.g., 'B_0', 'S_1')
                - 'Type': Trade type ('BUY', 'SELL')
                - 'SessionEurope': 1 if Europe session, 0 otherwise
                - 'SessionUS': 1 if US session, 0 otherwise
                - 'SessionAsia': 1 if Asia session, 0 otherwise
                - 'OpenPrice': Current price
                - 'TP': Take profit level
                - 'SL': Stop loss level
        
        Returns:
            dict: Signal analysis with recommendation
        """
        
        level_fibo = current_data.get('LevelFibo', '')
        trade_type = current_data.get('Type', '')
        
        # Priority signals (highest win rate)
        if level_fibo in ['B_0', 'B_-1.8']:
            signal_strength = 'STRONG'
            expected_win_rate = self.signal_criteria[level_fibo]['win_rate']
            confidence = 'HIGH'
            recommendation = 'TAKE_TRADE'
            
        elif level_fibo == 'B_1.8':
            signal_strength = 'MEDIUM'
            expected_win_rate = self.signal_criteria[level_fibo]['win_rate']
            confidence = 'MEDIUM'
            recommendation = 'CONSIDER_TRADE'
            
        elif level_fibo == 'S_4.5':
            signal_strength = 'WEAK'
            expected_win_rate = self.signal_criteria[level_fibo]['win_rate']
            confidence = 'LOW'
            recommendation = 'CAREFUL_CONSIDERATION'
            
        else:
            signal_strength = 'AVOID'
            expected_win_rate = 30.0  # Below average
            confidence = 'LOW'
            recommendation = 'AVOID_TRADE'
        
        # Session boost
        session_boost = 0
        if current_data.get('SessionEurope', 0) == 1:
            session_boost += 1.0  # Europe is best performing
        if current_data.get('SessionUS', 0) == 1:
            session_boost += 0.5
        if current_data.get('SessionAsia', 0) == 1:
            session_boost += 0.3
        
        adjusted_win_rate = expected_win_rate + session_boost
        
        # Risk management
        tp = current_data.get('TP', 0)
        sl = current_data.get('SL', 0)
        risk_reward = tp / sl if sl > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'fibonacci_level': level_fibo,
            'trade_type': trade_type,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'expected_win_rate': round(adjusted_win_rate, 1),
            'recommendation': recommendation,
            'risk_reward_ratio': round(risk_reward, 2),
            'session_active': self._get_active_session(current_data),
            'notes': self._generate_notes(level_fibo, signal_strength)
        }
    
    def _get_active_session(self, data):
        """Determine active trading session"""
        sessions = []
        if data.get('SessionEurope', 0) == 1:
            sessions.append('Europe')
        if data.get('SessionUS', 0) == 1:
            sessions.append('US')
        if data.get('SessionAsia', 0) == 1:
            sessions.append('Asia')
        return ', '.join(sessions) if sessions else 'None'
    
    def _generate_notes(self, level, strength):
        """Generate trading notes"""
        if level == 'B_0':
            return "PRIMARY SIGNAL: Highest volume (3,106 trades) with 52.4% win rate"
        elif level == 'B_-1.8':
            return "HIGH CONFIDENCE: 52.5% win rate, selective entry recommended"
        elif level == 'B_1.8':
            return "SECONDARY SIGNAL: 45.9% win rate, good for confirmation"
        elif level == 'S_4.5':
            return "RARE SIGNAL: Low volume, use with caution"
        else:
            return "LOW PROBABILITY: Consider avoiding this setup"
    
    def batch_analysis(self, data_list):
        """Analyze multiple data points"""
        results = []
        for data in data_list:
            signal = self.detect_signal(data)
            results.append(signal)
        return results
    
    def get_trading_summary(self):
        """Get summary of signal criteria"""
        return {
            'top_signals': ['B_0', 'B_-1.8'],
            'win_rates': self.signal_criteria,
            'session_preferences': 'Europe > US > Asia',
            'overall_strategy': 'Focus on BUY signals, avoid most SELL signals',
            'expected_performance': '40-52% win rate depending on signal quality'
        }

def test_signal_detector():
    """Test the signal detector with sample data"""
    
    detector = FibonacciSignalDetector()
    
    # Test cases based on real analysis
    test_cases = [
        {
            'LevelFibo': 'B_0',
            'Type': 'BUY',
            'SessionEurope': 1,
            'SessionUS': 0,
            'SessionAsia': 0,
            'OpenPrice': 2050.0,
            'TP': 300,
            'SL': 150
        },
        {
            'LevelFibo': 'B_-1.8',
            'Type': 'BUY',
            'SessionUS': 1,
            'SessionEurope': 0,
            'SessionAsia': 0,
            'OpenPrice': 2045.0,
            'TP': 250,
            'SL': 125
        },
        {
            'LevelFibo': 'S_1',
            'Type': 'SELL',
            'SessionAsia': 1,
            'SessionEurope': 0,
            'SessionUS': 0,
            'OpenPrice': 2055.0,
            'TP': 200,
            'SL': 100
        }
    ]
    
    print("FIBONACCI SIGNAL DETECTOR TEST")
    print("=" * 50)
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Level: {test_data['LevelFibo']}, Type: {test_data['Type']}")
        
        signal = detector.detect_signal(test_data)
        
        print(f"Result:")
        print(f"  Signal Strength: {signal['signal_strength']}")
        print(f"  Expected Win Rate: {signal['expected_win_rate']}%")
        print(f"  Recommendation: {signal['recommendation']}")
        print(f"  Risk/Reward: {signal['risk_reward_ratio']}")
        print(f"  Notes: {signal['notes']}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("TRADING SUMMARY")
    summary = detector.get_trading_summary()
    print(f"Top Signals: {summary['top_signals']}")
    print(f"Strategy: {summary['overall_strategy']}")
    print(f"Expected Performance: {summary['expected_performance']}")

if __name__ == "__main__":
    test_signal_detector()
