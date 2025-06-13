"""
Trading-specific evaluation metrics
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ..utils.helpers import setup_logging, calculate_win_rate, calculate_risk_reward_ratio, calculate_max_drawdown, calculate_sharpe_ratio

logger = setup_logging()

class TradingMetrics:
    """Trading-specific evaluation metrics"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate standard classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def calculate_trading_performance(self, df: pd.DataFrame, predictions: np.ndarray, actual_profits: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive trading performance metrics"""
        
        # Convert to pandas Series for easier manipulation
        pred_series = pd.Series(predictions)
        profit_series = pd.Series(actual_profits)
        
        # Basic trading metrics
        total_trades = len(predictions)
        winning_trades = len(profit_series[profit_series > 0])
        losing_trades = len(profit_series[profit_series < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = profit_series.sum()
        avg_profit = profit_series.mean()
        avg_winning_trade = profit_series[profit_series > 0].mean() if winning_trades > 0 else 0
        avg_losing_trade = profit_series[profit_series < 0].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = (
            abs(profit_series[profit_series > 0].sum()) / abs(profit_series[profit_series < 0].sum())
            if abs(profit_series[profit_series < 0].sum()) > 0 else 0
        )
        
        # Risk-reward ratio
        risk_reward = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else 0
        
        # Maximum drawdown
        equity_curve = profit_series.cumsum()
        max_drawdown = calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio (assuming daily returns)
        if len(profit_series) > 1 and profit_series.std() > 0:
            sharpe_ratio = calculate_sharpe_ratio(profit_series)
        else:
            sharpe_ratio = 0.0
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_runs(profit_series > 0)
        consecutive_losses = self._calculate_consecutive_runs(profit_series <= 0)
        
        # Recovery factor
        recovery_factor = abs(total_profit / max_drawdown) if max_drawdown != 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_winning_trade) + ((1 - win_rate) * avg_losing_trade)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy
        }
        
        return metrics
    
    def _calculate_consecutive_runs(self, boolean_series: pd.Series) -> int:
        """Calculate maximum consecutive runs in a boolean series"""
        if len(boolean_series) == 0:
            return 0
        
        max_run = 0
        current_run = 0
        
        for value in boolean_series:
            if value:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def calculate_signal_quality_metrics(self, y_true: np.ndarray, y_proba: np.ndarray, thresholds: List[float] = None) -> Dict[str, Dict[str, float]]:
        """Calculate signal quality metrics at different probability thresholds"""
        
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold_metrics = {}
        
        for threshold in thresholds:
            y_pred_threshold = (y_proba >= threshold).astype(int)
            
            # Only calculate for samples above threshold
            above_threshold_mask = y_proba >= threshold
            
            if above_threshold_mask.sum() > 0:
                y_true_filtered = y_true[above_threshold_mask]
                y_pred_filtered = y_pred_threshold[above_threshold_mask]
                
                if len(np.unique(y_true_filtered)) > 1:
                    precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
                    recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
                    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
                else:
                    precision = recall = accuracy = 0.0
                
                threshold_metrics[f'threshold_{threshold}'] = {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'signal_count': above_threshold_mask.sum(),
                    'signal_percentage': above_threshold_mask.mean() * 100
                }
            else:
                threshold_metrics[f'threshold_{threshold}'] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'accuracy': 0.0,
                    'signal_count': 0,
                    'signal_percentage': 0.0
                }
        
        return threshold_metrics
    
    def calculate_stability_metrics(self, results_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate stability metrics across multiple CV folds or time periods"""
        
        if not results_list:
            return {}
        
        # Convert list of dicts to DataFrame for easier calculation
        results_df = pd.DataFrame(results_list)
        
        stability_metrics = {}
        
        for metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 1:
                stability_metrics[f'{metric}_mean'] = values.mean()
                stability_metrics[f'{metric}_std'] = values.std()
                stability_metrics[f'{metric}_cv'] = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                stability_metrics[f'{metric}_min'] = values.min()
                stability_metrics[f'{metric}_max'] = values.max()
        
        return stability_metrics
    
    def create_performance_report(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                y_proba: np.ndarray,
                                actual_profits: np.ndarray,
                                df: pd.DataFrame = None) -> Dict[str, Any]:
        """Create comprehensive performance report"""
        
        logger.info("Creating comprehensive performance report...")
        
        report = {}
        
        # Classification metrics
        report['classification_metrics'] = self.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        # Trading performance metrics
        report['trading_metrics'] = self.calculate_trading_performance(df, y_pred, actual_profits)
        
        # Signal quality at different thresholds
        if y_proba is not None:
            report['signal_quality'] = self.calculate_signal_quality_metrics(y_true, y_proba)
        
        # Feature importance metrics (if available)
        if df is not None:
            report['data_summary'] = {
                'total_samples': len(df),
                'feature_count': len([col for col in df.columns if col not in ['Symbol', 'Timestamp', 'Type', 'ExitReason']]),
                'date_range': {
                    'start': df['Timestamp'].min() if 'Timestamp' in df.columns else None,
                    'end': df['Timestamp'].max() if 'Timestamp' in df.columns else None
                }
            }
        
        # Overall score (weighted combination of key metrics)
        report['overall_score'] = self._calculate_overall_score(report)
        
        return report
    
    def _calculate_overall_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall model score based on multiple criteria"""
        
        weights = {
            'win_rate': 0.25,
            'profit_factor': 0.20,
            'sharpe_ratio': 0.15,
            'accuracy': 0.15,
            'max_drawdown': 0.10,  # Lower is better, so we'll invert
            'f1_score': 0.10,
            'expectancy': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        trading_metrics = report.get('trading_metrics', {})
        classification_metrics = report.get('classification_metrics', {})
        
        for metric, weight in weights.items():
            if metric in trading_metrics:
                value = trading_metrics[metric]
                if metric == 'max_drawdown':
                    # Invert max drawdown (lower is better)
                    normalized_value = max(0, 1 + value)  # Since max_drawdown is negative
                else:
                    normalized_value = max(0, min(1, value))  # Clamp between 0 and 1
                
                score += weight * normalized_value
                total_weight += weight
                
            elif metric in classification_metrics:
                value = classification_metrics[metric]
                normalized_value = max(0, min(1, value))
                score += weight * normalized_value
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models and rank them"""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Add trading metrics
            if 'trading_metrics' in results:
                for metric, value in results['trading_metrics'].items():
                    row[f'trading_{metric}'] = value
            
            # Add classification metrics
            if 'classification_metrics' in results:
                for metric, value in results['classification_metrics'].items():
                    row[f'classification_{metric}'] = value
            
            # Add overall score
            row['overall_score'] = results.get('overall_score', 0.0)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by overall score (descending)
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        
        return comparison_df
