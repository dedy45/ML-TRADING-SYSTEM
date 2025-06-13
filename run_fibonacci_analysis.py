#!/usr/bin/env python3
"""
Runner untuk analisis komprehensif data trading Fibonacci
Menjalankan analisis semua file CSV dan training ML model
"""

import sys
import os
from datetime import datetime
from fibonacci_analyzer import FibonacciAnalyzer
from advanced_ml_pipeline import AdvancedTradingML

def print_banner():
    """Print banner"""
    print("=" * 70)
    print("🎯 COMPREHENSIVE FIBONACCI TRADING ANALYSIS")
    print("=" * 70)
    print("📊 Analyze all your trading data for best Fibonacci levels")
    print("🤖 Train ML models with enhanced Fibonacci features")
    print("📈 Generate detailed statistical reports")
    print("=" * 70)

def get_analysis_scope():
    """Get user choice for analysis scope"""
    print("\n🎯 Choose analysis scope:")
    print("1. Quick Test (50 files) - Fast preview")
    print("2. Medium Test (200 files) - Balanced analysis") 
    print("3. Large Test (400 files) - Comprehensive")
    print("4. Full Analysis (ALL 544 files) - Complete dataset")
    print("5. Custom - Choose your own number")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ Please enter a number between 1 and 5")
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled!")
            sys.exit(0)

def get_file_count(choice):
    """Get number of files based on choice"""
    if choice == '1':
        return 50
    elif choice == '2':
        return 200
    elif choice == '3':
        return 400
    elif choice == '4':
        return None  # All files
    elif choice == '5':
        while True:
            try:
                count = int(input("Enter number of files to process (1-544): "))
                if 1 <= count <= 544:
                    return count
                else:
                    print("❌ Please enter a number between 1 and 544")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Analysis cancelled!")
                sys.exit(0)

def run_fibonacci_analysis(max_files=None):
    """Run comprehensive Fibonacci analysis"""
    print("\n🔢 STEP 1: FIBONACCI STATISTICAL ANALYSIS")
    print("=" * 50)
    
    analyzer = FibonacciAnalyzer()
    report = analyzer.run_complete_analysis(max_files=max_files)
    
    return analyzer, report

def run_ml_training(num_files=None):
    """Run ML training with enhanced features"""
    print("\n🤖 STEP 2: MACHINE LEARNING TRAINING")
    print("=" * 45)
    
    # Use same number of files for ML training
    if num_files is None:
        ml_files = 50  # Default for full analysis to avoid memory issues
    else:
        ml_files = min(num_files, 50)  # Cap at 50 for ML training
        
    print(f"🔄 Training ML models with {ml_files} files...")
    
    pipeline = AdvancedTradingML(experiment_name="fibonacci_enhanced_trading")
    success = pipeline.run_complete_pipeline(num_files=ml_files, target='is_profitable')
    
    if success:
        print("✅ ML training completed successfully!")
        return pipeline
    else:
        print("❌ ML training failed!")
        return None

def generate_combined_report(analyzer, ml_pipeline, report):
    """Generate combined analysis report"""
    print("\n📋 STEP 3: GENERATING COMBINED REPORT")
    print("=" * 45)
    
    # Create comprehensive summary
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_trades_analyzed': report['total_trades'],
        'fibonacci_insights': {},
        'ml_performance': {},
        'recommendations': []
    }
    
    # Add Fibonacci insights
    if hasattr(analyzer, 'fibonacci_stats') and not analyzer.fibonacci_stats.empty:
        top_fib_levels = analyzer.fibonacci_stats.head(5)
        summary['fibonacci_insights'] = {
            'best_fibonacci_levels': [],
            'session_performance': analyzer.session_stats,
            'hourly_patterns': {}
        }
        
        # Top performing Fibonacci levels
        for level, row in top_fib_levels.iterrows():
            level_info = {
                'level': level,
                'total_profit': row['Profit_sum'],
                'win_rate': row['win_rate'],
                'avg_profit': row['Profit_mean'],
                'total_trades': row['Profit_count'],
                'risk_reward_ratio': row['risk_reward_ratio']
            }
            summary['fibonacci_insights']['best_fibonacci_levels'].append(level_info)
    
    # Add ML performance
    if ml_pipeline and ml_pipeline.results:
        summary['ml_performance'] = {
            'best_model': None,
            'best_accuracy': 0,
            'all_models': {}
        }
        
        for model_name, metrics in ml_pipeline.results.items():
            summary['ml_performance']['all_models'][model_name] = {
                'accuracy': metrics['accuracy'],
                'cv_score': metrics['cv_mean'],
                'win_precision': metrics['precision_wins']
            }
            
            # Track best model
            if metrics['accuracy'] > summary['ml_performance']['best_accuracy']:
                summary['ml_performance']['best_accuracy'] = metrics['accuracy']
                summary['ml_performance']['best_model'] = model_name
    
    # Generate recommendations
    recommendations = []
    
    if 'fibonacci_insights' in summary and summary['fibonacci_insights']['best_fibonacci_levels']:
        best_fib = summary['fibonacci_insights']['best_fibonacci_levels'][0]
        recommendations.append(f"🎯 Focus on Fibonacci level {best_fib['level']} - highest profitability ({best_fib['total_profit']:.2f} total profit)")
        recommendations.append(f"📊 Best win rate: {best_fib['win_rate']:.1f}% with {best_fib['total_trades']:.0f} trades")
    
    if summary['ml_performance']['best_model']:
        recommendations.append(f"🤖 Use {summary['ml_performance']['best_model']} model - {summary['ml_performance']['best_accuracy']:.1%} accuracy")
    
    # Add session recommendations
    if analyzer.session_stats:
        best_session = max(analyzer.session_stats.items(), key=lambda x: x[1]['win_rate'])
        recommendations.append(f"🌍 Trade during {best_session[0]} session - {best_session[1]['win_rate']:.1f}% win rate")
    
    summary['recommendations'] = recommendations
    
    # Save report
    report_file = f"data/processed/combined_fibonacci_ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    import json
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"💾 Combined report saved to: {report_file}")
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 30)
    print(f"📈 Total trades analyzed: {summary['total_trades_analyzed']:,}")
    
    if summary['fibonacci_insights']['best_fibonacci_levels']:
        best_fib = summary['fibonacci_insights']['best_fibonacci_levels'][0]
        print(f"🎯 Best Fibonacci level: {best_fib['level']}")
        print(f"💰 Total profit: {best_fib['total_profit']:,.2f}")
        print(f"📊 Win rate: {best_fib['win_rate']:.1f}%")
    
    if summary['ml_performance']['best_model']:
        print(f"🤖 Best ML model: {summary['ml_performance']['best_model']}")
        print(f"🎯 Accuracy: {summary['ml_performance']['best_accuracy']:.1%}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    return summary

def main():
    """Main function"""
    print_banner()
    
    # Get analysis scope
    choice = get_analysis_scope()
    max_files = get_file_count(choice)
    
    print(f"\n📋 ANALYSIS PLAN:")
    files_desc = f"{max_files} files" if max_files else "ALL 544 files"
    print(f"   📁 Processing: {files_desc}")
    print(f"   🔢 Fibonacci analysis: Complete statistical analysis")
    print(f"   🤖 ML training: Enhanced models with Fibonacci features")
    print(f"   📊 Reports: Comprehensive analysis reports")
    
    confirm = input(f"\n🚀 Ready to start comprehensive analysis? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Analysis cancelled!")
        return
    
    start_time = datetime.now()
    print(f"\n⏰ Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Fibonacci Analysis
        analyzer, report = run_fibonacci_analysis(max_files=max_files)
        
        # Step 2: ML Training
        ml_pipeline = run_ml_training(num_files=max_files)
        
        # Step 3: Combined Report
        summary = generate_combined_report(analyzer, ml_pipeline, report)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print("=" * 35)
        print(f"⏰ Duration: {duration}")
        print(f"📁 Files processed: {files_desc}")
        print(f"💾 Reports saved in: data/processed/")
        print(f"🌐 View ML results at: http://127.0.0.1:5000")
        
        print(f"\n📋 NEXT STEPS:")
        print("   1. Review the generated reports")
        print("   2. Implement the recommended Fibonacci levels")
        print("   3. Use the best ML model for signal generation")
        print("   4. Backtest with recommended settings")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("💡 Try running with fewer files or check your data")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Analysis stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("💡 Please check your setup and try again")
