#!/usr/bin/env python3
"""
TRADING SIGNAL DASHBOARD
Dashboard komprehensif untuk monitoring semua sinyal trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time

class TradingSignalDashboard:
    """Dashboard utama untuk monitoring sinyal trading"""
    
    def __init__(self):
        self.signals_today = []
        self.performance_history = []
        self.active_trades = {}
        
        # Load all available models
        self.models = {}
        self.load_all_models()
        
    def load_all_models(self):
        """Load semua model yang tersedia"""
        print("Loading available models...")
        
        # 1. Fibonacci Signal Detector
        try:
            from fibonacci_signal_detector import FibonacciSignalDetector
            self.models['fibonacci'] = FibonacciSignalDetector()
            print("✅ Fibonacci Signal Detector loaded")
        except Exception as e:
            print(f"❌ Fibonacci Signal Detector: {e}")
        
        # 2. Advanced Signal Optimizer
        try:
            from advanced_signal_optimizer import AdvancedSignalOptimizer
            optimizer = AdvancedSignalOptimizer()
            if optimizer.load_model():
                self.models['advanced_ml'] = optimizer
                print("✅ Advanced ML Optimizer loaded")
            else:
                print("⚠️  Advanced ML Optimizer: No saved model found")
        except Exception as e:
            print(f"❌ Advanced ML Optimizer: {e}")
        
        # 3. Ensemble Signal Detector
        try:
            from ensemble_signal_detector import EnsembleSignalDetector
            ensemble = EnsembleSignalDetector()
            if ensemble.load_ensemble_model():
                self.models['ensemble'] = ensemble
                print("✅ Ensemble Signal Detector loaded")
            else:
                print("⚠️  Ensemble Signal Detector: No saved model found")
        except Exception as e:
            print(f"❌ Ensemble Signal Detector: {e}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
    
    def analyze_market_signal(self, market_data):
        """Analisis sinyal menggunakan semua model yang tersedia"""
        timestamp = datetime.now()
        
        signal_analysis = {
            'timestamp': timestamp.isoformat(),
            'market_data': market_data,
            'models_results': {},
            'consensus': {},
            'final_recommendation': 'HOLD'
        }
        
        model_predictions = []
        model_probabilities = []
        model_recommendations = []
        
        # Test dengan semua model
        for model_name, model in self.models.items():
            try:
                if model_name == 'fibonacci':
                    result = model.detect_signal(market_data)
                    probability = self.convert_fibonacci_to_probability(result)
                    recommendation = result.get('recommendation', 'HOLD')
                    
                elif model_name == 'advanced_ml':
                    result = model.get_signal_strength(market_data)
                    probability = result.get('win_probability', 0.5)
                    recommendation = result.get('recommendation', 'HOLD')
                    
                elif model_name == 'ensemble':
                    result = model.predict_signal_strength(market_data)
                    probability = result.get('ensemble_probability', 0.5)
                    recommendation = result.get('recommendation', 'HOLD')
                
                signal_analysis['models_results'][model_name] = {
                    'result': result,
                    'probability': probability,
                    'recommendation': recommendation
                }
                
                model_predictions.append(1 if probability > 0.5 else 0)
                model_probabilities.append(probability)
                model_recommendations.append(recommendation)
                
            except Exception as e:
                signal_analysis['models_results'][model_name] = {'error': str(e)}
        
        # Consensus analysis
        if model_probabilities:
            avg_probability = np.mean(model_probabilities)
            consensus_strength = len([p for p in model_probabilities if p > 0.6]) / len(model_probabilities)
            
            # Agreement score
            agreement_score = 1 - np.std(model_probabilities)  # Higher when models agree
            
            signal_analysis['consensus'] = {
                'average_probability': avg_probability,
                'consensus_strength': consensus_strength,
                'agreement_score': agreement_score,
                'model_count': len(model_probabilities)
            }
            
            # Final recommendation based on consensus
            if avg_probability >= 0.6 and consensus_strength >= 0.5 and agreement_score >= 0.7:
                final_rec = 'STRONG_TAKE_TRADE'
            elif avg_probability >= 0.55 and consensus_strength >= 0.3:
                final_rec = 'TAKE_TRADE'
            elif avg_probability >= 0.5:
                final_rec = 'CONSIDER_TRADE'
            else:
                final_rec = 'AVOID_TRADE'
            
            signal_analysis['final_recommendation'] = final_rec
        
        return signal_analysis
    
    def convert_fibonacci_to_probability(self, fib_result):
        """Convert Fibonacci result ke probability"""
        strength = fib_result.get('signal_strength', 'WEAK')
        
        strength_map = {
            'STRONG': 0.65,
            'MEDIUM': 0.55,
            'WEAK': 0.45,
            'AVOID': 0.3
        }
        
        return strength_map.get(strength, 0.5)
    
    def generate_daily_report(self):
        """Generate laporan harian"""
        today = datetime.now().date()
        
        # Filter signals hari ini
        today_signals = [s for s in self.signals_today 
                        if datetime.fromisoformat(s['timestamp']).date() == today]
        
        report = {
            'date': today.isoformat(),
            'total_signals': len(today_signals),
            'strong_signals': len([s for s in today_signals 
                                 if s['final_recommendation'] in ['STRONG_TAKE_TRADE', 'TAKE_TRADE']]),
            'signals_breakdown': {},
            'model_performance': {},
            'recommendations': []
        }
        
        # Breakdown by recommendation
        for signal in today_signals:
            rec = signal['final_recommendation']
            if rec not in report['signals_breakdown']:
                report['signals_breakdown'][rec] = 0
            report['signals_breakdown'][rec] += 1
        
        # Model availability
        report['model_performance'] = {
            'models_loaded': len(self.models),
            'models_list': list(self.models.keys())
        }
        
        # Recommendations untuk tomorrow
        report['recommendations'] = [
            f"Generated {report['total_signals']} signals today",
            f"Strong signals: {report['strong_signals']}",
            f"Models active: {len(self.models)}",
            "Continue monitoring for high-probability setups"
        ]
        
        return report
    
    def save_signal_log(self, signal_analysis):
        """Save signal ke log file"""
        self.signals_today.append(signal_analysis)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/signals_{timestamp}.json"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump(self.signals_today, f, indent=2, default=str)
    
    def display_signal_analysis(self, signal_analysis):
        """Display signal analysis dalam format yang mudah dibaca"""
        print("=" * 80)
        print("🎯 TRADING SIGNAL ANALYSIS")
        print("=" * 80)
        
        # Market data
        market_data = signal_analysis['market_data']
        print(f"📊 Market Data:")
        print(f"   Level: {market_data.get('LevelFibo', 'N/A')}")
        print(f"   Type: {market_data.get('Type', 'N/A')}")
        print(f"   Session: EU:{market_data.get('SessionEurope', 0)} US:{market_data.get('SessionUS', 0)} AS:{market_data.get('SessionAsia', 0)}")
        print(f"   Price: {market_data.get('OpenPrice', 'N/A')}")
        print(f"   Volume: {market_data.get('Volume', 'N/A')}")
        
        # Models results
        print(f"\n🤖 Models Analysis:")
        for model_name, result in signal_analysis['models_results'].items():
            if 'error' in result:
                print(f"   ❌ {model_name}: {result['error']}")
            else:
                prob = result.get('probability', 0.5)
                rec = result.get('recommendation', 'HOLD')
                print(f"   ✅ {model_name}: {prob:.1%} confidence → {rec}")
        
        # Consensus
        consensus = signal_analysis.get('consensus', {})
        if consensus:
            print(f"\n🎯 Consensus Analysis:")
            print(f"   Average Probability: {consensus.get('average_probability', 0):.1%}")
            print(f"   Consensus Strength: {consensus.get('consensus_strength', 0):.1%}")
            print(f"   Agreement Score: {consensus.get('agreement_score', 0):.1%}")
            print(f"   Models Count: {consensus.get('model_count', 0)}")
        
        # Final recommendation
        final_rec = signal_analysis['final_recommendation']
        print(f"\n🏆 FINAL RECOMMENDATION: {final_rec}")
        
        # Risk assessment
        if final_rec in ['STRONG_TAKE_TRADE', 'TAKE_TRADE']:
            print(f"   🟢 Action: Consider taking this trade")
            print(f"   ⚠️  Risk: Use proper position sizing")
            print(f"   📈 Setup: Good probability setup detected")
        elif final_rec == 'CONSIDER_TRADE':
            print(f"   🟡 Action: Analyze additional factors")
            print(f"   ⚠️  Risk: Medium confidence signal")
            print(f"   📊 Setup: Wait for confirmation")
        else:
            print(f"   🔴 Action: Avoid this trade")
            print(f"   ⚠️  Risk: Low probability setup")
            print(f"   📉 Setup: Not recommended")
        
        print("=" * 80)
    
    def run_live_monitoring(self, demo_mode=True):
        """Run live monitoring (demo mode untuk testing)"""
        print("🚀 STARTING LIVE SIGNAL MONITORING")
        print("=" * 80)
        
        if demo_mode:
            print("📝 DEMO MODE: Using simulated market data")
            
            # Demo market conditions
            demo_signals = [
                {
                    'LevelFibo': 'B_0',
                    'Type': 'BUY',
                    'SessionEurope': 1,
                    'SessionUS': 0,
                    'SessionAsia': 0,
                    'OpenPrice': 2650.75,
                    'TP': 2655.75,
                    'SL': 2648.25,
                    'Volume': 0.1
                },
                {
                    'LevelFibo': 'B_-1.8',
                    'Type': 'BUY',
                    'SessionEurope': 0,
                    'SessionUS': 1,
                    'SessionAsia': 0,
                    'OpenPrice': 2651.20,
                    'TP': 2656.70,
                    'SL': 2648.70,
                    'Volume': 0.2
                },
                {
                    'LevelFibo': 'S_1',
                    'Type': 'SELL',
                    'SessionEurope': 0,
                    'SessionUS': 0,
                    'SessionAsia': 1,
                    'OpenPrice': 2652.80,
                    'TP': 2647.30,
                    'SL': 2655.30,
                    'Volume': 0.15
                }
            ]
            
            for i, market_data in enumerate(demo_signals, 1):
                print(f"\n🎲 Demo Signal {i}/3")
                
                # Analyze signal
                signal_analysis = self.analyze_market_signal(market_data)
                
                # Display analysis
                self.display_signal_analysis(signal_analysis)
                
                # Save signal
                self.save_signal_log(signal_analysis)
                
                if i < len(demo_signals):
                    print(f"\n⏳ Waiting 3 seconds for next signal...")
                    time.sleep(3)
            
            # Generate daily report
            print(f"\n📋 DAILY REPORT")
            print("=" * 80)
            daily_report = self.generate_daily_report()
            
            for key, value in daily_report.items():
                if key != 'recommendations':
                    print(f"{key}: {value}")
            
            print(f"\n💡 Recommendations:")
            for rec in daily_report['recommendations']:
                print(f"   • {rec}")
        
        else:
            print("🔴 LIVE MODE: Connect to real market data feed")
            print("   (Implementation needed for real broker API)")

def main():
    """Main dashboard function"""
    print("🎯 TRADING SIGNAL DASHBOARD v1.0")
    print("=" * 80)
    
    # Initialize dashboard
    dashboard = TradingSignalDashboard()
    
    # Check if we have any models
    if not dashboard.models:
        print("❌ No models loaded. Please train models first:")
        print("   1. Run: python advanced_signal_optimizer.py")
        print("   2. Run: python ensemble_signal_detector.py")
        return
    
    # Run monitoring
    try:
        dashboard.run_live_monitoring(demo_mode=True)
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in monitoring: {e}")
    
    print(f"\n✅ Dashboard session completed")
    print(f"📊 Signals generated: {len(dashboard.signals_today)}")
    print(f"📁 Logs saved to: logs/signals_{datetime.now().strftime('%Y%m%d')}.json")

if __name__ == "__main__":
    main()
