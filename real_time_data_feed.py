#!/usr/bin/env python3
"""
REAL-TIME DATA FEED INTEGRATION
MT5 and broker API integration for live trading
"""

import asyncio
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# MetaTrader 5 integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available. Install with: pip install MetaTrader5")

# Alternative data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import websocket
    import requests
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

class RealTimeDataFeed:
    """Real-time data feed manager supporting multiple sources"""
    
    def __init__(self, data_source: str = "mt5"):
        self.data_source = data_source
        self.is_connected = False
        self.current_data = {}
        self.data_callbacks = []
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize connection
        self._initialize_connection()
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_feed.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _initialize_connection(self):
        """Initialize data source connection"""
        if self.data_source == "mt5":
            self._initialize_mt5()
        elif self.data_source == "demo":
            self._initialize_demo_feed()
        elif self.data_source == "yfinance":
            self._initialize_yfinance()
        else:
            self.logger.error(f"Unknown data source: {self.data_source}")
            
    def _initialize_mt5(self):
        """Initialize MetaTrader 5 connection"""
        if not MT5_AVAILABLE:
            self.logger.error("MT5 not available")
            return False
            
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to connect to MT5 account")
                return False
                
            self.is_connected = True
            self.logger.info(f"✅ MT5 connected. Account: {account_info.login}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
            
    def _initialize_demo_feed(self):
        """Initialize demo data feed (for testing)"""
        self.is_connected = True
        self.logger.info("✅ Demo data feed initialized")
        return True
        
    def _initialize_yfinance(self):
        """Initialize Yahoo Finance feed"""
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance not available")
            return False
            
        try:
            # Test connection
            test_data = yf.download("GC=F", period="1d", interval="1m")
            if test_data.empty:
                self.logger.error("Failed to fetch test data from Yahoo Finance")
                return False
                
            self.is_connected = True
            self.logger.info("✅ Yahoo Finance feed initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance connection error: {e}")
            return False
            
    def get_current_price(self, symbol: str = "XAUUSD") -> Optional[Dict[str, Any]]:
        """Get current price data"""
        if not self.is_connected:
            return None
            
        try:
            if self.data_source == "mt5":
                return self._get_mt5_price(symbol)
            elif self.data_source == "demo":
                return self._get_demo_price(symbol)
            elif self.data_source == "yfinance":
                return self._get_yfinance_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
            
    def _get_mt5_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price from MT5"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'timestamp': datetime.fromtimestamp(tick.time),
                'volume': tick.volume_real if hasattr(tick, 'volume_real') else 0
            }
        except Exception as e:
            self.logger.error(f"MT5 price error: {e}")
            return None
            
    def _get_demo_price(self, symbol: str) -> Dict[str, Any]:
        """Get demo price (simulated)"""
        # Simulate realistic XAUUSD price movement
        base_price = 2000.0
        volatility = np.random.normal(0, 0.5)
        spread = 0.2
        
        bid = base_price + volatility
        ask = bid + spread
        
        return {
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'timestamp': datetime.now(),
            'volume': np.random.randint(100, 1000)
        }
        
    def _get_yfinance_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price from Yahoo Finance"""
        try:
            # Map symbol to Yahoo Finance format
            yf_symbol = "GC=F" if symbol == "XAUUSD" else symbol
            
            # Get latest data
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
                
            latest = data.iloc[-1]
            spread = 0.2  # Estimated spread
            
            return {
                'symbol': symbol,
                'bid': latest['Close'] - spread/2,
                'ask': latest['Close'] + spread/2,
                'spread': spread,
                'timestamp': datetime.now(),
                'volume': latest['Volume']
            }
        except Exception as e:
            self.logger.error(f"Yahoo Finance error: {e}")
            return None
            
    def get_historical_data(self, symbol: str = "XAUUSD", timeframe: str = "M1", count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data"""
        if not self.is_connected:
            return None
            
        try:
            if self.data_source == "mt5":
                return self._get_mt5_historical(symbol, timeframe, count)
            elif self.data_source == "yfinance":
                return self._get_yfinance_historical(symbol, timeframe, count)
            elif self.data_source == "demo":
                return self._get_demo_historical(symbol, timeframe, count)
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
            
    def _get_mt5_historical(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get historical data from MT5"""
        try:
            # Map timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"MT5 historical data error: {e}")
            return None
            
    def _get_yfinance_historical(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            # Map symbol and timeframe
            yf_symbol = "GC=F" if symbol == "XAUUSD" else symbol
            
            tf_map = {
                "M1": "1m",
                "M5": "5m",
                "M15": "15m",
                "H1": "1h",
                "H4": "4h",
                "D1": "1d"
            }
            
            interval = tf_map.get(timeframe, "1m")
            
            # Determine period based on count and timeframe
            if timeframe in ["M1", "M5", "M15"]:
                period = "7d"  # Max for minute data
            elif timeframe in ["H1", "H4"]:
                period = "60d"
            else:
                period = "1y"
                
            # Get data
            data = yf.download(yf_symbol, period=period, interval=interval)
            
            if data.empty:
                return None
                
            # Rename columns to match MT5 format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Return last 'count' rows
            return data.tail(count)
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance historical error: {e}")
            return None
            
    def _get_demo_historical(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Generate demo historical data"""
        # Generate realistic price data
        dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
        
        # Generate OHLC data with realistic patterns
        np.random.seed(42)  # For reproducibility
        base_price = 2000.0
        
        prices = []
        current_price = base_price
        
        for i in range(count):
            # Random walk with trend
            change = np.random.normal(0, 0.3)
            current_price += change
            
            # Create OHLC
            open_price = current_price
            high_price = open_price + abs(np.random.normal(0, 0.2))
            low_price = open_price - abs(np.random.normal(0, 0.2))
            close_price = open_price + np.random.normal(0, 0.1)
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
            
            current_price = close_price
            
        df = pd.DataFrame(prices, index=dates)
        return df
        
    def add_data_callback(self, callback):
        """Add callback for real-time data updates"""
        self.data_callbacks.append(callback)
        
    async def start_real_time_feed(self, symbol: str = "XAUUSD", interval_seconds: int = 1):
        """Start real-time data feed"""
        self.logger.info(f"🚀 Starting real-time feed for {symbol}")
        
        while self.is_connected:
            try:
                # Get current price
                price_data = self.get_current_price(symbol)
                
                if price_data:
                    self.current_data = price_data
                    
                    # Call all callbacks
                    for callback in self.data_callbacks:
                        try:
                            await callback(price_data)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
                            
                # Wait for next update
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Real-time feed stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Feed error: {e}")
                await asyncio.sleep(5)  # Wait before retry
                
    def disconnect(self):
        """Disconnect from data source"""
        if self.data_source == "mt5" and MT5_AVAILABLE:
            mt5.shutdown()
            
        self.is_connected = False
        self.logger.info("📴 Data feed disconnected")

class TradingSignalBroadcaster:
    """Broadcast trading signals to external systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.subscribers = []
        
    def add_subscriber(self, callback):
        """Add signal subscriber"""
        self.subscribers.append(callback)
        
    async def broadcast_signal(self, signal: Dict[str, Any]):
        """Broadcast signal to all subscribers"""
        self.logger.info(f"📡 Broadcasting signal: {signal['final_decision']} (Confidence: {signal['confidence']:.1%})")
        
        for subscriber in self.subscribers:
            try:
                await subscriber(signal)
            except Exception as e:
                self.logger.error(f"Broadcast error: {e}")
                
    def save_signal_to_file(self, signal: Dict[str, Any]):
        """Save signal to file for EA integration"""
        signal_file = "signals/latest_signal.json"
        
        try:
            import os
            os.makedirs("signals", exist_ok=True)
            
            with open(signal_file, 'w') as f:
                json.dump({
                    'timestamp': signal['timestamp'].isoformat(),
                    'symbol': 'XAUUSD',
                    'decision': signal['final_decision'],
                    'confidence': signal['confidence'],
                    'price': signal['price'],
                    'signals': signal['signals']
                }, f, indent=2)
                
            self.logger.info(f"📄 Signal saved to {signal_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")

def main():
    """Test real-time data feed"""
    print("🔄 REAL-TIME DATA FEED TEST")
    print("=" * 50)
    
    # Test different data sources
    sources = ["demo", "yfinance"] if YFINANCE_AVAILABLE else ["demo"]
    if MT5_AVAILABLE:
        sources.insert(0, "mt5")
        
    for source in sources:
        print(f"\n🧪 Testing {source} data source...")
        
        feed = RealTimeDataFeed(data_source=source)
        
        if feed.is_connected:
            # Test current price
            price = feed.get_current_price("XAUUSD")
            if price:
                print(f"✅ Current Price: {price['bid']:.2f}/{price['ask']:.2f}")
            
            # Test historical data
            hist_data = feed.get_historical_data("XAUUSD", "M1", 10)
            if hist_data is not None:
                print(f"✅ Historical Data: {len(hist_data)} records")
                print(hist_data.tail(3))
            
        feed.disconnect()

if __name__ == "__main__":
    main()
