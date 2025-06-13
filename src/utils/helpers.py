"""
Helper utilities for trading ML pipeline
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_ml.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def calculate_pips(entry_price: float, exit_price: float, is_buy: bool = True) -> float:
    """Calculate pips for forex trading"""
    if is_buy:
        return (exit_price - entry_price) * 10000
    else:
        return (entry_price - exit_price) * 10000

def calculate_win_rate(predictions: pd.Series, actual: pd.Series) -> float:
    """Calculate win rate"""
    correct_predictions = (predictions == actual).sum()
    total_predictions = len(predictions)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

def calculate_risk_reward_ratio(profits: List[float], losses: List[float]) -> float:
    """Calculate risk-reward ratio"""
    if not profits or not losses:
        return 0.0
    
    avg_profit = np.mean([p for p in profits if p > 0])
    avg_loss = np.mean([abs(l) for l in losses if l < 0])
    
    return avg_profit / avg_loss if avg_loss > 0 else 0.0

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown"""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns / returns.std() if returns.std() > 0 else 0.0

def get_file_list(directory: str, extension: str = ".csv") -> List[str]:
    """Get list of files in directory with specific extension"""
    path = Path(directory)
    if not path.exists():
        return []
    
    return [str(f) for f in path.glob(f"*{extension}")]

def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if not"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save dataframe to file"""
    ensure_directory_exists(os.path.dirname(filepath))
    
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=index)
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath, index=index)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load dataframe from file"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    return f"{currency} {amount:,.2f}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage"""
    return f"{value:.{decimal_places}%}"

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate data quality"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'missing_values': {},
        'data_types': {},
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Check for missing required columns
    missing_cols = set(required_columns) - set(df.columns)
    report['missing_columns'] = list(missing_cols)
    
    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            report['missing_values'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }
    
    # Data types
    report['data_types'] = df.dtypes.to_dict()
    
    return report
