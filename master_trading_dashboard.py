#!/usr/bin/env python3
"""
MASTER TRADING DASHBOARD
Dashboard utama untuk menjalankan semua sistem trading
"""

import os
import sys
import time
from datetime import datetime
import json

class MasterTradingDashboard:
    """Dashboard utama untuk sistem trading"""
    
    def __init__(self):
        self.status = {
            'fibonacci_detector': False,
            'ensemble_detector': False,
            'fixed_optimizer': False,
            'models_loaded': 0,
            'last_update': None
        }
        self.models = {}
        
    def initialize_systems(self):
        """Initialize semua sistem trading"""
        print("🚀 INITIALIZING MASTER TRADING DASHBOARD")
        print("=" * 70)
        
        # 1. Fibonacci Signal Detector (Selalu tersedia)
        try:
            from fibonacci_signal_detector import FibonacciSignalDetector
            self.models['fibonacci'] = FibonacciSignalDetector()
            self.status['fibonacci_detector'] = True
            print("✅ Fibonacci Signal Detector: LOADED")
        except Exception as e:
            print(f"❌ Fibonacci Signal Detector: {e}")
        
        # 2. Ensemble Signal Detector (Jika model tersedia)
        try:
            from ensemble_signal_detector import EnsembleSignalDetector
            ensemble = EnsembleSignalDetector()
            if ensemble.load_ensemble_model():
                self.models['ensemble'] = ensemble
                self.status['ensemble_detector'] = True
                print("✅ Ensemble Signal Detector: LOADED")
            else:
                print("⚠️  Ensemble Signal Detector: No model found")
        except Exception as e:
            print(f"❌ Ensemble Signal Detector: {e}")
        
        # 3. Fixed Advanced Optimizer (Jika model tersedia)
        try:
            from fixed_advanced_signal_optimizer import FixedAdvancedSignalOptimizer
            optimizer = FixedAdvancedSignalOptimizer()
            if optimizer.load_model():
                self.models['optimizer'] = optimizer
                self.status['fixed_optimizer'] = True
                print("✅ Fixed Advanced Optimizer: LOADED")
            else:
                print("⚠️  Fixed Advanced Optimizer: No model found")
        except Exception as e:
            print(f"❌ Fixed Advanced Optimizer: {e}")
        
        self.status['models_loaded'] = len(self.models)
        self.status['last_update'] = datetime.now().isoformat()
        
        print(f"\n📊 SYSTEMS STATUS:")
        print(f"   Models Loaded: {self.status['models_loaded']}")
        print(f"   Fibonacci: {'✅' if self.status['fibonacci_detector'] else '❌'}")
        print(f"   Ensemble: {'✅' if self.status['ensemble_detector'] else '❌'}")
        print(f"   Optimizer: {'✅' if self.status['fixed_optimizer'] else '❌'}")
        
        return self.status['models_loaded'] > 0
    
    def analyze_signal(self, market_data):
        """Analisis sinyal menggunakan semua model yang tersedia"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'models_results': {},
            'consensus': {},
            'final_recommendation': 'HOLD'
        }
        
        probabilities = []
        recommendations = []
        
        # Test dengan semua model
        for model_name, model in self.models.items():
            try:
                if model_name == 'fibonacci':
                    result = model.detect_signal(market_data)
                    # Convert to probability
                    strength = result.get('signal_strength', 'WEAK')
                    prob = {'STRONG': 0.65, 'MEDIUM': 0.55, 'WEAK': 0.45, 'AVOID': 0.3}.get(strength, 0.5)
                    recommendation = result.get('recommendation', 'HOLD')
                    
                elif model_name == 'ensemble':
                    result = model.predict_signal_strength(market_data)
                    prob = result.get('ensemble_probability', 0.5)
                    recommendation = result.get('recommendation', 'HOLD')
                    
                elif model_name == 'optimizer':
                    result = model.get_signal_strength(market_data)
                    prob = result.get('win_probability', 0.5)
                    recommendation = result.get('recommendation', 'HOLD')
                
                results['models_results'][model_name] = {
                    'result': result,
                    'probability': prob,
                    'recommendation': recommendation
                }
                
                probabilities.append(prob)
                recommendations.append(recommendation)
                
            except Exception as e:
                results['models_results'][model_name] = {'error': str(e)}
        
        # Consensus analysis
        if probabilities:
            avg_prob = sum(probabilities) / len(probabilities)
            high_conf_count = len([p for p in probabilities if p > 0.6])
            agreement_score = 1 - (max(probabilities) - min(probabilities))  # Higher when models agree
            
            results['consensus'] = {
                'average_probability': avg_prob,
                'high_confidence_models': high_conf_count,
                'total_models': len(probabilities),
                'agreement_score': agreement_score
            }
            
            # Final recommendation
            if avg_prob >= 0.6 and high_conf_count >= 1 and agreement_score >= 0.7:
                final_rec = 'STRONG_TAKE_TRADE'
            elif avg_prob >= 0.55 and high_conf_count >= 1:
                final_rec = 'TAKE_TRADE'
            elif avg_prob >= 0.5:
                final_rec = 'CONSIDER_TRADE'
            else:
                final_rec = 'AVOID_TRADE'
            
            results['final_recommendation'] = final_rec
        
        return results
    
    def display_signal_analysis(self, analysis):
        """Display analisis dalam format yang mudah dibaca"""
        print("=" * 80)
        print("🎯 MASTER TRADING SIGNAL ANALYSIS")
        print("=" * 80)
        
        # Market data
        market_data = analysis['market_data']
        print(f"📊 MARKET CONDITIONS:")
        print(f"   Level: {market_data.get('LevelFibo', 'N/A')}")
        print(f"   Type: {market_data.get('Type', 'N/A')}")
        print(f"   Price: {market_data.get('OpenPrice', 'N/A')}")
        print(f"   Session: EU:{market_data.get('SessionEurope', 0)} US:{market_data.get('SessionUS', 0)} AS:{market_data.get('SessionAsia', 0)}")
        
        # Models results
        print(f"\n🤖 MODELS ANALYSIS:")
        for model_name, result in analysis['models_results'].items():
            if 'error' in result:
                print(f"   ❌ {model_name.upper()}: {result['error']}")
            else:
                prob = result.get('probability', 0.5)
                rec = result.get('recommendation', 'HOLD')
                confidence = "🟢 HIGH" if prob >= 0.6 else "🟡 MED" if prob >= 0.55 else "🔴 LOW"
                print(f"   ✅ {model_name.upper()}: {prob:.1%} {confidence} → {rec}")
        
        # Consensus
        consensus = analysis.get('consensus', {})
        if consensus:
            avg_prob = consensus.get('average_probability', 0)
            high_conf = consensus.get('high_confidence_models', 0)
            total = consensus.get('total_models', 0)
            agreement = consensus.get('agreement_score', 0)
            
            print(f"\n🎯 CONSENSUS ANALYSIS:")
            print(f"   Average Probability: {avg_prob:.1%}")
            print(f"   High Confidence Models: {high_conf}/{total}")
            print(f"   Model Agreement: {agreement:.1%}")
        
        # Final recommendation
        final_rec = analysis['final_recommendation']
        print(f"\n🏆 FINAL RECOMMENDATION: {final_rec}")
        
        if final_rec in ['STRONG_TAKE_TRADE', 'TAKE_TRADE']:
            print(f"   🟢 ACTION: Execute trade")
            print(f"   ⚠️  RISK: Use proper position sizing")
            print(f"   📈 SETUP: High probability signal detected")
        elif final_rec == 'CONSIDER_TRADE':
            print(f"   🟡 ACTION: Wait for additional confirmation")
            print(f"   ⚠️  RISK: Medium confidence signal")
            print(f"   📊 SETUP: Monitor closely")
        else:
            print(f"   🔴 ACTION: Avoid this trade")
            print(f"   ⚠️  RISK: Low probability setup")
            print(f"   📉 SETUP: Wait for better opportunity")
        
        print("=" * 80)
        return analysis
    
    def run_live_demo(self):
        """Run demo monitoring dengan sample data"""
        if not self.initialize_systems():
            print("❌ No systems available. Cannot run demo.")
            return
        
        print(f"\n🎮 STARTING LIVE DEMO MODE")
        print("=" * 70)
        
        # Sample signals berdasarkan analisis terbaik
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
            },
            {
                'LevelFibo': 'B_1.8',
                'Type': 'BUY',
                'SessionEurope': 1,
                'SessionUS': 0,
                'SessionAsia': 0,
                'OpenPrice': 2649.50,
                'TP': 2654.50,
                'SL': 2647.00,
                'Volume': 0.12
            }
        ]
        
        all_analyses = []
        
        for i, signal in enumerate(demo_signals, 1):
            print(f"\n🎲 SIGNAL {i}/{len(demo_signals)}")
            print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Analyze signal
            analysis = self.analyze_signal(signal)
            
            # Display analysis
            self.display_signal_analysis(analysis)
            
            all_analyses.append(analysis)
            
            if i < len(demo_signals):
                print(f"\n⏳ Next signal in 3 seconds...")
                time.sleep(3)
        
        # Final summary
        self.generate_session_summary(all_analyses)
        
        return all_analyses
    
    def generate_session_summary(self, analyses):
        """Generate summary dari session"""
        print(f"\n📋 SESSION SUMMARY")
        print("=" * 80)
        
        total_signals = len(analyses)
        strong_signals = len([a for a in analyses if a['final_recommendation'] in ['STRONG_TAKE_TRADE', 'TAKE_TRADE']])
        
        # Breakdown by recommendation
        rec_breakdown = {}
        for analysis in analyses:
            rec = analysis['final_recommendation']
            rec_breakdown[rec] = rec_breakdown.get(rec, 0) + 1
        
        print(f"📊 SIGNAL STATISTICS:")
        print(f"   Total Signals Analyzed: {total_signals}")
        print(f"   Strong Signals: {strong_signals}")
        print(f"   Win Rate Estimate: {(strong_signals/total_signals)*100:.1f}%")
        
        print(f"\n📈 RECOMMENDATION BREAKDOWN:")
        for rec, count in rec_breakdown.items():
            percentage = (count/total_signals)*100
            print(f"   {rec}: {count} ({percentage:.1f}%)")
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   Models Active: {self.status['models_loaded']}")
        print(f"   Fibonacci Detector: {'✅ Active' if self.status['fibonacci_detector'] else '❌ Inactive'}")
        print(f"   Ensemble Detector: {'✅ Active' if self.status['ensemble_detector'] else '❌ Inactive'}")
        print(f"   Fixed Optimizer: {'✅ Active' if self.status['fixed_optimizer'] else '❌ Inactive'}")
        
        print(f"\n💡 TRADING RECOMMENDATIONS:")
        if strong_signals >= 2:
            print("   • Excellent session with multiple high-probability setups")
            print("   • Focus on risk management and position sizing")
            print("   • Consider increasing position size for strongest signals")
        elif strong_signals >= 1:
            print("   • Good session with some quality setups")
            print("   • Be selective and wait for high-confidence signals")
            print("   • Use standard position sizing")
        else:
            print("   • Challenging session with limited opportunities")
            print("   • Consider sitting out or reducing position sizes")
            print("   • Wait for better market conditions")
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"reports/session_summary_{timestamp}.json"
        
        os.makedirs("reports", exist_ok=True)
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_signals': total_signals,
            'strong_signals': strong_signals,
            'recommendation_breakdown': rec_breakdown,
            'system_status': self.status,
            'analyses': analyses
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"\n💾 Session saved: {summary_file}")

def main():
    """Main dashboard function"""
    print("🎯 MASTER TRADING DASHBOARD v2.0")
    print("=" * 80)
    print(f"🕒 Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    dashboard = MasterTradingDashboard()
    
    try:
        # Run live demo
        analyses = dashboard.run_live_demo()
        
        print(f"\n✅ Demo session completed successfully!")
        print(f"📊 {len(analyses)} signals analyzed")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Session stopped by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Session ended at: {datetime.now().strftime('%H:%M:%S')}")
    print("Thank you for using Master Trading Dashboard! 🚀")

if __name__ == "__main__":
    main()
