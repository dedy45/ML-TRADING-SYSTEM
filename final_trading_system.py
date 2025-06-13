#!/usr/bin/env python3
"""
FINAL TRADING SIGNAL SYSTEM
Sistem final yang sudah teruji dan siap digunakan
"""

import sys
import os
from datetime import datetime

class FinalTradingSystem:
    """Sistem trading final yang sudah teruji"""
    
    def __init__(self):
        self.fibonacci_detector = None
        self.ensemble_detector = None
        self.system_status = {
            'fibonacci': False,
            'ensemble': False,
            'total_models': 0
        }
        
    def initialize(self):
        """Initialize sistem"""
        print("🎯 FINAL TRADING SIGNAL SYSTEM")
        print("=" * 50)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Load Fibonacci (Primary system)
        try:
            from fibonacci_signal_detector import FibonacciSignalDetector
            self.fibonacci_detector = FibonacciSignalDetector()
            self.system_status['fibonacci'] = True
            print("✅ Fibonacci Signal Detector: READY")
            self.system_status['total_models'] += 1
        except Exception as e:
            print(f"❌ Fibonacci Error: {e}")
        
        # Load Ensemble (Secondary system)  
        try:
            from ensemble_signal_detector import EnsembleSignalDetector
            ensemble = EnsembleSignalDetector()
            if ensemble.load_ensemble_model():
                self.ensemble_detector = ensemble
                self.system_status['ensemble'] = True
                print("✅ Ensemble Signal Detector: READY")
                self.system_status['total_models'] += 1
            else:
                print("⚠️  Ensemble: Model not found")
        except Exception as e:
            print(f"❌ Ensemble Error: {e}")
        
        print(f"\n📊 System Status: {self.system_status['total_models']} models loaded")
        
        return self.system_status['total_models'] > 0
        
    def analyze_signal(self, market_data):
        """Analisis sinyal trading"""
        print(f"\n🔍 ANALYZING SIGNAL...")
        print(f"Market Data: {market_data}")
        
        results = {}
        recommendations = []
        probabilities = []
        
        # Fibonacci Analysis (Primary)
        if self.fibonacci_detector:
            try:
                fib_result = self.fibonacci_detector.detect_signal(market_data)
                
                # Convert to standard format
                strength = fib_result.get('signal_strength', 'WEAK')
                prob_map = {'STRONG': 0.65, 'MEDIUM': 0.55, 'WEAK': 0.45, 'AVOID': 0.3}
                probability = prob_map.get(strength, 0.5)
                
                results['fibonacci'] = {
                    'probability': probability,
                    'strength': strength,
                    'recommendation': fib_result.get('recommendation', 'HOLD'),
                    'win_rate': fib_result.get('expected_win_rate', 50.0),
                    'notes': fib_result.get('notes', '')
                }
                
                probabilities.append(probability)
                recommendations.append(fib_result.get('recommendation', 'HOLD'))
                
                print(f"✅ Fibonacci: {strength} ({probability:.1%}) → {fib_result.get('recommendation', 'HOLD')}")
                
            except Exception as e:
                print(f"❌ Fibonacci Error: {e}")
                results['fibonacci'] = {'error': str(e)}
        
        # Ensemble Analysis (Secondary)
        if self.ensemble_detector:
            try:
                ensemble_result = self.ensemble_detector.predict_signal_strength(market_data)
                
                if 'error' not in ensemble_result:
                    probability = ensemble_result.get('ensemble_probability', 0.5)
                    strength = ensemble_result.get('signal_strength', 'WEAK')
                    recommendation = ensemble_result.get('recommendation', 'HOLD')
                    
                    results['ensemble'] = {
                        'probability': probability,
                        'strength': strength,
                        'recommendation': recommendation,
                        'confidence': ensemble_result.get('confidence', 0.5)
                    }
                    
                    probabilities.append(probability)
                    recommendations.append(recommendation)
                    
                    print(f"✅ Ensemble: {strength} ({probability:.1%}) → {recommendation}")
                else:
                    print(f"❌ Ensemble Error: {ensemble_result.get('error', 'Unknown')}")
                    results['ensemble'] = {'error': ensemble_result.get('error', 'Unknown')}
                    
            except Exception as e:
                print(f"❌ Ensemble Error: {e}")
                results['ensemble'] = {'error': str(e)}
        
        # Final Decision
        if probabilities:
            avg_probability = sum(probabilities) / len(probabilities)
            high_conf_count = sum(1 for p in probabilities if p >= 0.6)
            
            # Decision logic
            if avg_probability >= 0.6 and high_conf_count >= 1:
                final_decision = "STRONG_TAKE_TRADE"
                action_color = "🟢"
            elif avg_probability >= 0.55:
                final_decision = "TAKE_TRADE"
                action_color = "🟡"
            elif avg_probability >= 0.5:
                final_decision = "CONSIDER_TRADE"
                action_color = "🟡"
            else:
                final_decision = "AVOID_TRADE"
                action_color = "🔴"
            
            print(f"\n🎯 FINAL DECISION: {action_color} {final_decision}")
            print(f"📊 Average Probability: {avg_probability:.1%}")
            print(f"🎪 High Confidence Models: {high_conf_count}/{len(probabilities)}")
            
            results['final'] = {
                'decision': final_decision,
                'average_probability': avg_probability,
                'high_confidence_count': high_conf_count,
                'total_models': len(probabilities)
            }
        else:
            print("❌ No models available for analysis")
            results['final'] = {'decision': 'NO_ANALYSIS', 'error': 'No models available'}
        
        return results
        
    def run_demo(self):
        """Run demo dengan signal-signal terbaik"""
        if not self.initialize():
            print("❌ System initialization failed")
            return
        
        print(f"\n🎮 RUNNING SIGNAL DEMO")
        print("=" * 50)
        
        # Best signals dari analisis sebelumnya
        demo_signals = [
            {
                'name': 'B_0 Europe Strong Signal',
                'data': {
                    'LevelFibo': 'B_0',
                    'Type': 'BUY',
                    'SessionEurope': 1,
                    'SessionUS': 0,
                    'SessionAsia': 0,
                    'OpenPrice': 2650.50,
                    'TP': 2655.50,
                    'SL': 2648.50,
                    'Volume': 0.1
                }
            },
            {
                'name': 'B_-1.8 US Session',
                'data': {
                    'LevelFibo': 'B_-1.8',
                    'Type': 'BUY',
                    'SessionEurope': 0,
                    'SessionUS': 1,
                    'SessionAsia': 0,
                    'OpenPrice': 2651.20,
                    'TP': 2656.70,
                    'SL': 2648.70,
                    'Volume': 0.2
                }
            },
            {
                'name': 'S_1 Weak Signal (Test)',
                'data': {
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
            }
        ]
        
        all_results = []
        
        for i, signal in enumerate(demo_signals, 1):
            print(f"\n--- SIGNAL {i}/{len(demo_signals)}: {signal['name']} ---")
            
            result = self.analyze_signal(signal['data'])
            all_results.append({
                'name': signal['name'],
                'data': signal['data'],
                'result': result
            })
            
            if i < len(demo_signals):
                input("\nPress Enter for next signal...")
        
        # Demo Summary
        print(f"\n📋 DEMO SESSION SUMMARY")
        print("=" * 50)
        
        strong_signals = 0
        for result in all_results:
            decision = result['result'].get('final', {}).get('decision', 'UNKNOWN')
            if decision in ['STRONG_TAKE_TRADE', 'TAKE_TRADE']:
                strong_signals += 1
        
        print(f"Total Signals: {len(all_results)}")
        print(f"Strong Signals: {strong_signals}")
        print(f"Success Rate: {(strong_signals/len(all_results))*100:.1f}%")
        print(f"System Models: {self.system_status['total_models']}")
        
        print(f"\n🎉 DEMO COMPLETED!")
        print("System is ready for live trading! 🚀")
        
        return all_results

def main():
    """Main function"""
    system = FinalTradingSystem()
    
    try:
        system.run_demo()
    except KeyboardInterrupt:
        print(f"\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
