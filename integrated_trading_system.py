#!/usr/bin/env python3
"""
COMPLETE INTEGRATED TRADING SYSTEM
Production-ready ML trading system with MLflow tracking
"""

import asyncio
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class IntegratedTradingSystem:
    """Complete integrated trading system"""
    
    def __init__(self, mode: str = "paper"):
        self.mode = mode  # paper, demo, or live
        self.is_running = False
        self.performance_metrics = {}
        
        # System components
        self.signal_detector = None
        self.data_feed = None
        self.paper_trader = None
        self.broadcaster = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # MLflow experiment
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("integrated_trading_system")
            
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "integrated_system.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("🔧 Initializing integrated trading system components...")
        
        try:
            # Initialize signal detection
            from final_trading_system import FinalTradingSystem
            self.signal_detector = FinalTradingSystem()
            
            if self.signal_detector.initialize():
                self.logger.info("✅ Signal detector initialized")
            else:
                self.logger.error("❌ Signal detector initialization failed")
                
        except Exception as e:
            self.logger.error(f"❌ Signal detector error: {e}")
            
        try:
            # Initialize data feed
            from real_time_data_feed import RealTimeDataFeed
            self.data_feed = RealTimeDataFeed(data_source="demo")
            
            if self.data_feed.is_connected:
                self.logger.info("✅ Data feed initialized")
            else:
                self.logger.error("❌ Data feed initialization failed")
                
        except Exception as e:
            self.logger.error(f"❌ Data feed error: {e}")
            
        try:
            # Initialize paper trading (if in paper mode)
            if self.mode == "paper":
                from paper_trading_system import PaperTradingSystem
                self.paper_trader = PaperTradingSystem(initial_balance=10000.0)
                self.logger.info("✅ Paper trading initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Paper trading error: {e}")
            
        try:
            # Initialize signal broadcaster
            from real_time_data_feed import TradingSignalBroadcaster
            self.broadcaster = TradingSignalBroadcaster()
            self.logger.info("✅ Signal broadcaster initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Broadcaster error: {e}")
            
    def validate_system_health(self) -> Dict[str, bool]:
        """Validate system health before starting"""
        health_check = {
            'signal_detector': self.signal_detector is not None,
            'data_feed': self.data_feed is not None and self.data_feed.is_connected,
            'paper_trader': self.paper_trader is not None if self.mode == "paper" else True,
            'broadcaster': self.broadcaster is not None,
            'mlflow': MLFLOW_AVAILABLE
        }
        
        all_healthy = all(health_check.values())
        
        self.logger.info("🔍 System Health Check:")
        for component, status in health_check.items():
            status_icon = "✅" if status else "❌"
            self.logger.info(f"  {status_icon} {component}: {'OK' if status else 'FAILED'}")
            
        return health_check
        
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming market data and generate signals"""
        
        try:
            # Generate trading signal
            if self.signal_detector:
                signal = self.signal_detector.analyze_signal(market_data)
                
                # Add market data to signal
                signal['market_data'] = market_data
                
                return signal
            else:
                self.logger.error("No signal detector available")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
            
    async def execute_trading_decision(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute trading decision based on signal"""
        
        decision = signal.get('final_decision', 'HOLD')
        confidence = signal.get('confidence', 0.0)
        
        # Only execute high-confidence signals
        if confidence < 0.55 or decision == 'HOLD':
            return None
            
        trade_result = None
        
        try:
            if self.mode == "paper" and self.paper_trader:
                # Execute paper trade
                market_data = signal['market_data']
                trade_result = self.paper_trader.execute_paper_trade(signal, market_data)
                
                if trade_result:
                    self.logger.info(f"📈 Paper trade executed: {decision} at {market_data['ask']:.2f}")
                    
            elif self.mode == "demo":
                # Demo mode - just log the signal
                self.logger.info(f"🎯 Demo signal: {decision} (Confidence: {confidence:.1%})")
                trade_result = {
                    'mode': 'demo',
                    'signal': decision,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
            elif self.mode == "live":
                # Live trading mode (implement with actual broker API)
                self.logger.warning("🚨 LIVE TRADING MODE - Not implemented yet")
                # TODO: Implement live trading with MT5/broker API
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            
        return trade_result
        
    async def broadcast_signal(self, signal: Dict[str, Any]):
        """Broadcast signal to external systems"""
        
        try:
            if self.broadcaster:
                # Broadcast to subscribers
                await self.broadcaster.broadcast_signal(signal)
                
                # Save to file for EA integration
                self.broadcaster.save_signal_to_file(signal)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting signal: {e}")
            
    def update_performance_metrics(self):
        """Update system performance metrics"""
        
        try:
            if self.mode == "paper" and self.paper_trader:
                metrics = self.paper_trader.calculate_performance_metrics()
                self.performance_metrics.update(metrics)
                
                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    with mlflow.start_run():
                        for key, value in metrics.items():
                            mlflow.log_metric(f"system_{key}", value)
                            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            
    async def main_trading_loop(self, duration_minutes: int = 60):
        """Main trading loop"""
        
        self.logger.info(f"🚀 Starting integrated trading system ({self.mode} mode)")
        self.logger.info(f"⏱️ Duration: {duration_minutes} minutes")
        
        # Validate system health
        health_check = self.validate_system_health()
        if not all(health_check.values()):
            self.logger.error("❌ System health check failed. Cannot start trading.")
            return False
            
        # Start MLflow run
        mlflow_run_id = None
        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name=f"integrated_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow_run_id = run.info.run_id
            
            # Log system configuration
            mlflow.log_param("mode", self.mode)
            mlflow.log_param("duration_minutes", duration_minutes)
            mlflow.log_param("start_time", datetime.now().isoformat())
            
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        signal_count = 0
        trade_count = 0
        error_count = 0
        
        try:
            while datetime.now() < end_time and self.is_running:
                loop_start = time.time()
                
                try:
                    # Get current market data
                    if self.data_feed:
                        market_data = self.data_feed.get_current_price("XAUUSD")
                        
                        if market_data:
                            # Process market data and generate signal
                            signal = await self.process_market_data(market_data)
                            
                            if signal:
                                signal_count += 1
                                
                                # Broadcast signal
                                await self.broadcast_signal(signal)
                                
                                # Execute trading decision
                                trade_result = await self.execute_trading_decision(signal)
                                
                                if trade_result:
                                    trade_count += 1
                                    
                                # Check exit conditions for existing positions
                                if self.paper_trader:
                                    self.paper_trader.check_exit_conditions(market_data)
                                    
                    # Update performance metrics every 5 minutes
                    if datetime.now().minute % 5 == 0:
                        self.update_performance_metrics()
                        
                        # Log status
                        self.logger.info(f"📊 Status: Signals={signal_count}, Trades={trade_count}, Errors={error_count}")
                        
                except Exception as e:
                    error_count += 1
                    self.logger.error(f"Error in trading loop: {e}")
                    
                # Control loop timing
                loop_duration = time.time() - loop_start
                sleep_time = max(0, 5 - loop_duration)  # Target 5-second intervals
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading system stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Fatal error in trading loop: {e}")
        finally:
            self.is_running = False
            
        # Final performance update
        self.update_performance_metrics()
        
        # Close all positions (paper trading)
        if self.paper_trader:
            for position in self.paper_trader.positions[:]:
                market_data = self.data_feed.get_current_price("XAUUSD")
                if market_data:
                    self.paper_trader._close_position(
                        position, 
                        market_data['ask'], 
                        "SYSTEM_SHUTDOWN", 
                        datetime.now()
                    )
                    
        # Final metrics
        total_runtime = (datetime.now() - start_time).total_seconds() / 60
        
        self.logger.info("🎯 INTEGRATED TRADING SYSTEM SESSION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Runtime: {total_runtime:.1f} minutes")
        self.logger.info(f"📡 Signals Generated: {signal_count}")
        self.logger.info(f"📈 Trades Executed: {trade_count}")
        self.logger.info(f"❌ Errors: {error_count}")
        
        if self.performance_metrics:
            self.logger.info(f"💰 Final Balance: ${self.performance_metrics.get('current_balance', 0):,.2f}")
            self.logger.info(f"📊 Win Rate: {self.performance_metrics.get('win_rate', 0):.1%}")
            self.logger.info(f"📈 Total Return: {self.performance_metrics.get('total_return', 0):.2f}%")
            
        # End MLflow run
        if MLFLOW_AVAILABLE and mlflow_run_id:
            mlflow.log_metric("total_signals", signal_count)
            mlflow.log_metric("total_trades", trade_count)
            mlflow.log_metric("error_count", error_count)
            mlflow.log_metric("runtime_minutes", total_runtime)
            
            mlflow.end_run()
            
        return True
        
    def stop_system(self):
        """Stop the trading system"""
        self.logger.info("🛑 Stopping integrated trading system...")
        self.is_running = False
        
        # Disconnect data feed
        if self.data_feed:
            self.data_feed.disconnect()
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'performance_metrics': self.performance_metrics,
            'components': {
                'signal_detector': self.signal_detector is not None,
                'data_feed': self.data_feed is not None and self.data_feed.is_connected,
                'paper_trader': self.paper_trader is not None if self.mode == "paper" else True,
                'broadcaster': self.broadcaster is not None
            }
        }

def main():
    """Main execution function"""
    
    print("🎯 INTEGRATED ML TRADING SYSTEM")
    print("=" * 60)
    print("Choose operating mode:")
    print("1. Paper Trading (Recommended)")
    print("2. Demo Mode (Signal Generation Only)")
    print("3. Live Trading (Not Implemented)")
    print("=" * 60)
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        mode_map = {
            '1': 'paper',
            '2': 'demo', 
            '3': 'live'
        }
        
        mode = mode_map.get(choice, 'paper')
        
        if mode == 'live':
            print("⚠️ Live trading mode is not yet implemented")
            print("🔄 Switching to paper trading mode")
            mode = 'paper'
            
        # Get session duration
        try:
            duration = int(input("Session duration in minutes (default 30): ") or 30)
        except ValueError:
            duration = 30
            
        print(f"\n🚀 Starting system in {mode} mode for {duration} minutes...")
        
        # Create and run system
        trading_system = IntegratedTradingSystem(mode=mode)
        
        # Run async trading loop
        asyncio.run(trading_system.main_trading_loop(duration_minutes=duration))
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
