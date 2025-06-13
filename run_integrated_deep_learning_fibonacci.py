#!/usr/bin/env python3
"""
Deep Learning Fibonacci Integration Script
Integrates enhanced deep learning module with MLFLOW infrastructure
Prevents hanging issues and provides robust performance
"""

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import MLFLOW infrastructure
try:
    from config.config import config
    from utils.timeout_utils import ExecutionGuard, safe_timeout, TimeoutException, timeout_decorator
    from utils.logging_utils import get_logger, PerformanceLogger
    MLFLOW_INFRA_AVAILABLE = True
except ImportError:
    MLFLOW_INFRA_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.INFO)

# Import deep learning module
try:
    from deep_learning_fibonacci.enhanced_tensorflow_fibonacci_predictor import EnhancedFibonacciDeepLearningPredictor
    ENHANCED_DL_AVAILABLE = True
except ImportError:
    try:
        from deep_learning_fibonacci.tensorflow_fibonacci_predictor import FibonacciDeepLearningPredictor
        ENHANCED_DL_AVAILABLE = False
    except ImportError:
        ENHANCED_DL_AVAILABLE = None

# Setup logging
if MLFLOW_INFRA_AVAILABLE:
    logger = get_logger('dl_fibonacci_integration', config.logs_dir)
    perf_logger = PerformanceLogger(logger)
else:
    logger = logging.getLogger(__name__)
    perf_logger = None

class DeepLearningFibonacciIntegration:
    """
    Integrated Deep Learning Fibonacci System
    Combines MLFLOW infrastructure with deep learning capabilities
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"dl_fibonacci_integration_{int(time.time())}"
        self.predictor = None
        self.integration_stats = {
            'start_time': time.time(),
            'operations_completed': 0,
            'errors_encountered': 0,
            'timeouts_handled': 0
        }
        
        # Initialize predictor based on availability
        self._initialize_predictor()
        
        logger.info("🧠 Deep Learning Fibonacci Integration initialized")
        logger.info(f"🔬 Experiment: {self.experiment_name}")
        logger.info(f"📊 Enhanced DL Available: {ENHANCED_DL_AVAILABLE}")
        logger.info(f"📊 MLFLOW Infrastructure: {MLFLOW_INFRA_AVAILABLE}")
    
    def _initialize_predictor(self):
        """Initialize the appropriate predictor based on availability"""
        try:
            if ENHANCED_DL_AVAILABLE:
                self.predictor = EnhancedFibonacciDeepLearningPredictor(
                    experiment_name=self.experiment_name
                )
                logger.info("✅ Using Enhanced Deep Learning Predictor")
            elif ENHANCED_DL_AVAILABLE is False:
                self.predictor = FibonacciDeepLearningPredictor()
                logger.info("⚠️ Using Basic Deep Learning Predictor (fallback)")
            else:
                raise ImportError("No deep learning predictors available")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize predictor: {e}")
            self.predictor = None
    
    @timeout_decorator(1200, "Integrated analysis")  # 20 minute timeout
    def run_integrated_analysis(self, 
                               max_files: int = 25, 
                               max_rows_per_file: int = 40,
                               target_win_rate: float = 0.58) -> Optional[Dict[str, Any]]:
        """
        Run integrated deep learning fibonacci analysis
        With comprehensive timeout protection and error handling
        """
        logger.info("🚀 INTEGRATED DEEP LEARNING FIBONACCI ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"🎯 Target Win Rate: {target_win_rate:.1%}")
        logger.info(f"📁 Max Files: {max_files}")
        logger.info(f"📄 Max Rows per File: {max_rows_per_file}")
        logger.info(f"⏰ Timeout Protection: 20 minutes")
        logger.info("")
        
        if self.predictor is None:
            logger.error("❌ No predictor available")
            return None
        
        start_time = time.time()
        
        # Initialize execution guard for additional protection
        if MLFLOW_INFRA_AVAILABLE:
            main_guard = ExecutionGuard(max_execution_time=1150)  # Slightly less than outer timeout
            main_guard.start()
        
        try:
            if perf_logger:
                perf_logger.start_timer('integrated_analysis')
            
            # Run the analysis
            logger.info("🔄 Starting deep learning analysis...")
            results = self.predictor.run_complete_analysis(
                max_files=max_files,
                max_rows_per_file=max_rows_per_file
            )
            
            # Check for timeout during execution
            if MLFLOW_INFRA_AVAILABLE and main_guard.should_stop():
                raise TimeoutException("Analysis timed out during execution")
            
            # Process results
            if results:
                self._process_analysis_results(results, target_win_rate)
                self.integration_stats['operations_completed'] += 1
            else:
                logger.error("❌ Analysis returned no results")
                self.integration_stats['errors_encountered'] += 1
            
            # Final performance summary
            total_time = time.time() - start_time
            self.integration_stats['total_time'] = total_time
            
            self._log_integration_summary(total_time, results)
            
            return results
            
        except TimeoutException as e:
            logger.error(f"⏰ Analysis timed out: {e}")
            self.integration_stats['timeouts_handled'] += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Integrated analysis failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.integration_stats['errors_encountered'] += 1
            return None
            
        finally:
            if MLFLOW_INFRA_AVAILABLE:
                try:
                    main_guard.stop()
                except:
                    pass
            
            if perf_logger:
                perf_logger.end_timer('integrated_analysis')
    
    def _process_analysis_results(self, results: Dict[str, Any], target_win_rate: float):
        """Process and enhance analysis results"""
        logger.info("\n🔍 PROCESSING ANALYSIS RESULTS")
        logger.info("-" * 50)
        
        if not results:
            logger.warning("⚠️ No results to process")
            return
        
        # Find best model
        best_model_name = max(results.keys(), 
                            key=lambda k: results[k].get('high_conf_win_rate', 0))
        best_win_rate = results[best_model_name].get('high_conf_win_rate', 0)
        
        # Performance classification
        if best_win_rate >= target_win_rate:
            performance_level = "🎉 EXCELLENT"
            ready_for_production = True
        elif best_win_rate >= 0.55:
            performance_level = "✅ GOOD"
            ready_for_production = True
        elif best_win_rate >= 0.50:
            performance_level = "⚠️ FAIR"
            ready_for_production = False
        else:
            performance_level = "❌ POOR"
            ready_for_production = False
        
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"📈 Win Rate: {best_win_rate:.1%}")
        logger.info(f"🎯 Performance: {performance_level}")
        logger.info(f"🚀 Production Ready: {ready_for_production}")
        
        # Trading recommendations
        self._generate_trading_recommendations(best_win_rate, ready_for_production)
    
    def _generate_trading_recommendations(self, win_rate: float, production_ready: bool):
        """Generate trading recommendations based on performance"""
        logger.info("\n📋 TRADING RECOMMENDATIONS")
        logger.info("-" * 40)
        
        if production_ready:
            logger.info("✅ APPROVED FOR LIVE TRADING")
            logger.info("📊 Recommended Settings:")
            logger.info("   • Position Size: 1-2% per trade")
            logger.info("   • Risk/Reward: 1:2 minimum")
            logger.info("   • High Confidence Threshold: 70%+")
            
            if win_rate >= 0.60:
                logger.info("   • Maximum Position Size: 3-5%")
                logger.info("   • Aggressive Strategy: Enabled")
            else:
                logger.info("   • Conservative Strategy: Recommended")
                
        else:
            logger.info("⚠️ NOT RECOMMENDED FOR LIVE TRADING")
            logger.info("📊 Improvement Suggestions:")
            logger.info("   • Collect more training data")
            logger.info("   • Tune hyperparameters")
            logger.info("   • Feature engineering optimization")
            logger.info("   • Consider ensemble methods")
        
        logger.info("\n🧪 TESTING RECOMMENDATIONS:")
        logger.info("   • Paper trading: 2-4 weeks minimum")
        logger.info("   • Demo account: 1-2 months")
        logger.info("   • Live micro-lots: Start with 0.01")
    
    def _log_integration_summary(self, total_time: float, results: Optional[Dict]):
        """Log comprehensive integration summary"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 INTEGRATED ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"⏱️ Total Execution Time: {total_time:.1f} seconds")
        logger.info(f"✅ Operations Completed: {self.integration_stats['operations_completed']}")
        logger.info(f"❌ Errors Encountered: {self.integration_stats['errors_encountered']}")
        logger.info(f"⏰ Timeouts Handled: {self.integration_stats['timeouts_handled']}")
        
        # System status
        if results and self.integration_stats['errors_encountered'] == 0:
            system_status = "🎉 EXCELLENT"
        elif results:
            system_status = "✅ GOOD"
        else:
            system_status = "❌ NEEDS ATTENTION"
        
        logger.info(f"🏥 System Health: {system_status}")
        
        # Next steps
        logger.info("\n🚀 NEXT STEPS:")
        if results:
            logger.info("1. 📁 Review saved models and logs")
            logger.info("2. 🧪 Run validation tests")
            logger.info("3. 📊 Integrate with EA MQL5")
            logger.info("4. 📈 Monitor performance in paper trading")
        else:
            logger.info("1. 🔧 Check system configuration")
            logger.info("2. 📁 Verify data availability")
            logger.info("3. 🐛 Debug any error messages")
            logger.info("4. 🔄 Retry with adjusted parameters")
    
    def quick_validation_test(self) -> bool:
        """Run quick validation test"""
        logger.info("🧪 Running Quick Validation Test...")
        
        try:
            if self.predictor is None:
                logger.error("❌ No predictor available for testing")
                return False
            
            # Test basic functionality
            test_results = self.predictor.run_complete_analysis(
                max_files=3,
                max_rows_per_file=20
            )
            
            if test_results:
                logger.info("✅ Quick validation test passed")
                return True
            else:
                logger.warning("⚠️ Quick validation test returned no results")
                return False
                
        except Exception as e:
            logger.error(f"❌ Quick validation test failed: {e}")
            return False

def main():
    """Main execution function"""
    try:
        logger.info("🚀 STARTING INTEGRATED DEEP LEARNING FIBONACCI SYSTEM")
        
        # Initialize integration
        integration = DeepLearningFibonacciIntegration()
        
        # Run quick validation first
        logger.info("\n🧪 STEP 1: QUICK VALIDATION")
        validation_passed = integration.quick_validation_test()
        
        if not validation_passed:
            logger.error("❌ Quick validation failed. Aborting full analysis.")
            return
        
        # Run full integrated analysis
        logger.info("\n🚀 STEP 2: FULL INTEGRATED ANALYSIS")
        results = integration.run_integrated_analysis(
            max_files=20,          # Balanced for performance
            max_rows_per_file=35,  # Balanced for performance
            target_win_rate=0.58   # Target win rate
        )
        
        # Final summary
        if results:
            logger.info("\n🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
            logger.info("📋 Check logs for detailed results and recommendations")
        else:
            logger.error("\n❌ INTEGRATION FAILED")
            logger.error("📋 Check error logs and system configuration")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Integration interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
