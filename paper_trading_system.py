#!/usr/bin/env python3
"""
PAPER TRADING SYSTEM with MLflow Integration
Real-time signal testing and performance tracking
"""

import asyncio
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class PaperTradingSystem:
    """Paper trading system with MLflow tracking"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Load trained models
        self.fibonacci_detector = None
        self.ensemble_detector = None
        self._load_models()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # MLflow experiment for paper trading
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("paper_trading_live")
            self.experiment_id = mlflow.create_run().info.run_id
            
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/paper_trading.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _load_models(self):
        """Load trained signal detection models"""
        try:
            # Load Fibonacci detector
            sys.path.append('.')
            from fibonacci_signal_detector import FibonacciSignalDetector
            self.fibonacci_detector = FibonacciSignalDetector()
            self.logger.info("✅ Fibonacci detector loaded")
            
            # Load ensemble detector
            from ensemble_signal_detector import EnsembleSignalDetector
            ensemble = EnsembleSignalDetector()
            if ensemble.load_ensemble_model():
                self.ensemble_detector = ensemble
                self.logger.info("✅ Ensemble detector loaded")
                
        except Exception as e:
            self.logger.error(f"❌ Model loading error: {e}")
            
    def generate_mock_market_data(self) -> Dict[str, Any]:
        """Generate realistic market data for testing"""
        # Simulate XAUUSD price movement
        base_price = 2000.0
        volatility = np.random.normal(0, 0.5)  # 0.5% average volatility
        
        current_price = base_price + volatility
        
        return {
            'symbol': 'XAUUSD',
            'bid': current_price - 0.1,
            'ask': current_price + 0.1,
            'spread': 0.2,
            'timestamp': datetime.now(),
            'volume': np.random.randint(100, 1000),
            # Technical indicators (mock)
            'rsi': np.random.uniform(30, 70),
            'ema_fast': current_price * 0.999,
            'ema_slow': current_price * 1.001,
            'bb_upper': current_price * 1.002,
            'bb_lower': current_price * 0.998,
        }
        
    def analyze_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading signal using loaded models"""
        results = {
            'timestamp': market_data['timestamp'],
            'price': market_data['ask'],
            'signals': {},
            'final_decision': 'HOLD',
            'confidence': 0.0
        }
        
        # Fibonacci analysis
        if self.fibonacci_detector:
            try:
                fib_result = self.fibonacci_detector.detect_signal(market_data)
                results['signals']['fibonacci'] = fib_result
            except Exception as e:
                self.logger.error(f"Fibonacci analysis error: {e}")
                
        # Ensemble analysis
        if self.ensemble_detector:
            try:
                ensemble_result = self.ensemble_detector.predict_signal_strength(market_data)
                results['signals']['ensemble'] = ensemble_result
            except Exception as e:
                self.logger.error(f"Ensemble analysis error: {e}")
                
        # Decision logic
        confidences = []
        recommendations = []
        
        for signal_name, signal_data in results['signals'].items():
            if 'error' not in signal_data:
                confidence = signal_data.get('probability', 0.5)
                recommendation = signal_data.get('recommendation', 'HOLD')
                
                confidences.append(confidence)
                recommendations.append(recommendation)
                
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            results['confidence'] = avg_confidence
            
            # Decision thresholds
            if avg_confidence >= 0.65:
                results['final_decision'] = 'STRONG_BUY' if 'BUY' in recommendations else 'STRONG_SELL'
            elif avg_confidence >= 0.55:
                results['final_decision'] = 'BUY' if 'BUY' in recommendations else 'SELL'
            elif avg_confidence >= 0.45:
                results['final_decision'] = 'WEAK_BUY' if 'BUY' in recommendations else 'WEAK_SELL'
                
        return results
        
    def execute_paper_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute paper trade based on signal"""
        
        decision = signal['final_decision']
        confidence = signal['confidence']
        
        # Only trade on high confidence signals
        if confidence < 0.55 or decision == 'HOLD':
            return None
            
        # Position sizing based on confidence
        risk_per_trade = 0.02  # 2% risk per trade
        position_size = self.current_balance * risk_per_trade * confidence
        
        # Create trade
        trade = {
            'id': len(self.trade_history) + 1,
            'timestamp': market_data['timestamp'],
            'symbol': market_data['symbol'],
            'direction': 'BUY' if 'BUY' in decision else 'SELL',
            'entry_price': market_data['ask'] if 'BUY' in decision else market_data['bid'],
            'position_size': position_size,
            'confidence': confidence,
            'signal_source': list(signal['signals'].keys()),
            'status': 'OPEN'
        }
        
        # Set stop loss and take profit
        if 'BUY' in decision:
            trade['stop_loss'] = trade['entry_price'] * 0.99  # 1% SL
            trade['take_profit'] = trade['entry_price'] * 1.02  # 2% TP
        else:
            trade['stop_loss'] = trade['entry_price'] * 1.01  # 1% SL
            trade['take_profit'] = trade['entry_price'] * 0.98  # 2% TP
            
        self.positions.append(trade)
        self.trade_history.append(trade)
        
        self.logger.info(f"📈 Paper trade executed: {decision} at {trade['entry_price']:.2f} (Confidence: {confidence:.1%})")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_id=self.experiment_id):
                mlflow.log_metric("trades_executed", len(self.trade_history))
                mlflow.log_metric("avg_confidence", confidence)
                
        return trade
        
    def check_exit_conditions(self, market_data: Dict[str, Any]):
        """Check exit conditions for open positions"""
        current_price = market_data['ask']
        
        for position in self.positions[:]:  # Copy list for safe iteration
            if position['status'] != 'OPEN':
                continue
                
            should_close = False
            close_reason = ""
            
            if position['direction'] == 'BUY':
                if current_price <= position['stop_loss']:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif current_price >= position['take_profit']:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
            else:  # SELL
                if current_price >= position['stop_loss']:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif current_price <= position['take_profit']:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
                    
            # Time-based exit (24 hours max)
            if (market_data['timestamp'] - position['timestamp']).total_seconds() > 86400:
                should_close = True
                close_reason = "TIME_EXIT"
                
            if should_close:
                self._close_position(position, current_price, close_reason, market_data['timestamp'])
                
    def _close_position(self, position: Dict[str, Any], exit_price: float, reason: str, exit_time: datetime):
        """Close position and calculate P&L"""
        
        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * (position['position_size'] / position['entry_price'])
        else:
            pnl = (position['entry_price'] - exit_price) * (position['position_size'] / position['entry_price'])
            
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = exit_time
        position['pnl'] = pnl
        position['status'] = 'CLOSED'
        position['close_reason'] = reason
        
        # Update balance
        self.current_balance += pnl
        
        # Remove from open positions
        if position in self.positions:
            self.positions.remove(position)
            
        self.logger.info(f"💰 Position closed: {reason} | P&L: {pnl:.2f} | Balance: {self.current_balance:.2f}")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_id=self.experiment_id):
                mlflow.log_metric("current_balance", self.current_balance)
                mlflow.log_metric("total_pnl", self.current_balance - self.initial_balance)
                
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {}
            
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum([t['pnl'] for t in closed_trades])
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in closed_trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_return': (self.current_balance / self.initial_balance - 1) * 100,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 and losing_trades > 0 else 0,
            'current_balance': self.current_balance,
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        return metrics
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.trade_history) < 2:
            return 0.0
            
        balance_history = [self.initial_balance]
        running_balance = self.initial_balance
        
        for trade in self.trade_history:
            if trade['status'] == 'CLOSED':
                running_balance += trade['pnl']
                balance_history.append(running_balance)
                
        if len(balance_history) < 2:
            return 0.0
            
        peak = balance_history[0]
        max_dd = 0.0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            else:
                drawdown = (peak - balance) / peak
                max_dd = max(max_dd, drawdown)
                
        return max_dd * 100  # Return as percentage
        
    async def run_paper_trading_session(self, duration_minutes: int = 60):
        """Run paper trading session"""
        self.logger.info(f"🎯 Starting {duration_minutes}-minute paper trading session")
        self.logger.info(f"💰 Initial balance: ${self.initial_balance:,.2f}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # MLflow run
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_id=self.experiment_id):
                mlflow.log_param("initial_balance", self.initial_balance)
                mlflow.log_param("session_duration_minutes", duration_minutes)
                
        trade_count = 0
        
        while datetime.now() < end_time:
            try:
                # Generate market data
                market_data = self.generate_mock_market_data()
                
                # Check exit conditions for open positions
                self.check_exit_conditions(market_data)
                
                # Analyze new signals
                signal = self.analyze_signal(market_data)
                
                # Execute trades
                new_trade = self.execute_paper_trade(signal, market_data)
                if new_trade:
                    trade_count += 1
                    
                # Log current status every 5 minutes
                if datetime.now().minute % 5 == 0:
                    metrics = self.calculate_performance_metrics()
                    self.logger.info(f"📊 Status: Balance=${self.current_balance:.2f} | Open Positions: {len(self.positions)} | Total Trades: {trade_count}")
                    
                # Wait before next iteration
                await asyncio.sleep(10)  # 10 seconds between checks
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Session stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retry
                
        # Close all open positions at session end
        for position in self.positions[:]:
            market_data = self.generate_mock_market_data()
            self._close_position(position, market_data['ask'], "SESSION_END", datetime.now())
            
        # Calculate final metrics
        final_metrics = self.calculate_performance_metrics()
        
        # Log final results
        self.logger.info("🎯 PAPER TRADING SESSION COMPLETE")
        self.logger.info("=" * 50)
        self.logger.info(f"💰 Final Balance: ${self.current_balance:,.2f}")
        self.logger.info(f"📈 Total Return: {final_metrics.get('total_return', 0):.2f}%")
        self.logger.info(f"🎯 Win Rate: {final_metrics.get('win_rate', 0):.1%}")
        self.logger.info(f"📊 Total Trades: {final_metrics.get('total_trades', 0)}")
        self.logger.info(f"📉 Max Drawdown: {final_metrics.get('max_drawdown', 0):.2f}%")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_id=self.experiment_id):
                for key, value in final_metrics.items():
                    mlflow.log_metric(f"final_{key}", value)
                    
                # Save trade history
                trades_df = pd.DataFrame(self.trade_history)
                trades_df.to_csv('paper_trades.csv', index=False)
                mlflow.log_artifact('paper_trades.csv')
                
        return final_metrics

def main():
    """Main paper trading execution"""
    print("🎯 PAPER TRADING SYSTEM")
    print("=" * 50)
    
    # Create paper trading system
    paper_trader = PaperTradingSystem(initial_balance=10000.0)
    
    # Run session
    try:
        asyncio.run(paper_trader.run_paper_trading_session(duration_minutes=30))
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
