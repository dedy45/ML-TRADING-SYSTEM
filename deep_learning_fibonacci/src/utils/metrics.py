"""
Fibonacci Trading Metrics and Evaluation Utilities
Custom metrics for trading performance evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

class FibonacciMetrics:
    """
    Comprehensive metrics for Fibonacci trading model evaluation.
    Focus on trading-specific metrics beyond standard ML metrics.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trading-specific thresholds
        self.baseline_win_rate = config['baseline']['b_0_level']['win_rate']
        self.target_win_rate = config['targets']['win_rate_improvement']
        self.tp_sl_ratio = config['risk_management']['tp_sl_ratio']
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray = None) -> Dict:
        """Calculate basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: np.ndarray = None) -> Dict:
        """
        Calculate trading-specific performance metrics.
        
        Args:
            y_true: Actual outcomes (1=win, 0=loss)
            y_pred: Predicted outcomes (1=win, 0=loss)
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of trading metrics
        """
        # Basic win rate
        win_rate = np.mean(y_pred[y_pred == 1] == y_true[y_pred == 1]) if np.sum(y_pred) > 0 else 0
        
        # Signal accuracy (how often we're right when we predict a win)
        signal_accuracy = accuracy_score(y_true[y_pred == 1], y_pred[y_pred == 1]) if np.sum(y_pred) > 0 else 0
        
        # True positive rate (sensitivity)
        true_positive_rate = recall_score(y_true, y_pred)
        
        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Expected return calculation (assuming 2:1 TP/SL ratio)
        tp_ratio = self.tp_sl_ratio
        sl_ratio = 1.0
        
        expected_return_per_trade = (win_rate * tp_ratio) - ((1 - win_rate) * sl_ratio)
        
        # Sharpe-like ratio for trading
        returns = []
        for i in range(len(y_true)):
            if y_pred[i] == 1:  # We took the trade
                if y_true[i] == 1:  # Trade won
                    returns.append(tp_ratio)
                else:  # Trade lost
                    returns.append(-sl_ratio)
        
        avg_return = np.mean(returns) if returns else 0
        return_std = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Maximum drawdown simulation
        cumulative_returns = np.cumsum(returns) if returns else [0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if drawdown else 0
        
        # Win rate improvement over baseline
        win_rate_improvement = win_rate - self.baseline_win_rate
        improvement_percentage = (win_rate_improvement / self.baseline_win_rate) * 100
        
        # Confidence-adjusted metrics (if probabilities available)
        high_confidence_metrics = {}
        if y_prob is not None:
            confidence_threshold = self.config['targets']['model_confidence_threshold']
            high_conf_mask = y_prob >= confidence_threshold
            
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
                high_conf_win_rate = np.mean(y_true[high_conf_mask])
                
                high_confidence_metrics = {
                    'high_confidence_accuracy': high_conf_accuracy,
                    'high_confidence_win_rate': high_conf_win_rate,
                    'high_confidence_sample_count': np.sum(high_conf_mask),
                    'high_confidence_percentage': np.mean(high_conf_mask)
                }
        
        trading_metrics = {
            'win_rate': win_rate,
            'signal_accuracy': signal_accuracy,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'expected_return_per_trade': expected_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate_improvement': win_rate_improvement,
            'improvement_percentage': improvement_percentage,
            'total_signals': np.sum(y_pred),
            'signal_frequency': np.mean(y_pred),
            **high_confidence_metrics
        }
        
        return trading_metrics
    
    def calculate_fibonacci_level_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        fibonacci_levels: np.ndarray) -> Dict:
        """
        Calculate performance metrics for each Fibonacci level.
        Compare with baseline performance from statistical analysis.
        """
        level_metrics = {}
        
        for level in np.unique(fibonacci_levels):
            level_mask = fibonacci_levels == level
            
            if np.sum(level_mask) == 0:
                continue
                
            level_y_true = y_true[level_mask]
            level_y_pred = y_pred[level_mask]
            
            level_win_rate = np.mean(level_y_true[level_y_pred == 1]) if np.sum(level_y_pred) > 0 else 0
            level_accuracy = accuracy_score(level_y_true, level_y_pred)
            
            # Compare with baseline for known levels
            baseline_comparison = {}
            if level == 0.0:  # B_0 level
                baseline_comparison['baseline_win_rate'] = self.config['baseline']['b_0_level']['win_rate']
                baseline_comparison['baseline_trades'] = self.config['baseline']['b_0_level']['total_trades']
            elif level == -1.8:  # B_-1.8 level
                baseline_comparison['baseline_win_rate'] = self.config['baseline']['b_minus_1_8_level']['win_rate']
                baseline_comparison['baseline_trades'] = self.config['baseline']['b_minus_1_8_level']['total_trades']
            elif level == 1.8:  # B_1.8 level
                baseline_comparison['baseline_win_rate'] = self.config['baseline']['b_1_8_level']['win_rate']
                baseline_comparison['baseline_trades'] = self.config['baseline']['b_1_8_level']['total_trades']
            
            level_metrics[f'level_{level}'] = {
                'win_rate': level_win_rate,
                'accuracy': level_accuracy,
                'sample_count': np.sum(level_mask),
                'signal_count': np.sum(level_y_pred),
                **baseline_comparison
            }
        
        return level_metrics
    
    def generate_performance_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray = None, fibonacci_levels: np.ndarray = None,
                                  model_name: str = "Model") -> str:
        """Generate comprehensive performance report."""
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred, y_prob)
        
        # Fibonacci level metrics if available
        level_metrics = {}
        if fibonacci_levels is not None:
            level_metrics = self.calculate_fibonacci_level_metrics(y_true, y_pred, fibonacci_levels)
        
        # Generate report
        report = f"""
# {model_name} Performance Report

## Basic Classification Metrics
- **Accuracy**: {basic_metrics['accuracy']:.4f}
- **Precision**: {basic_metrics['precision']:.4f}
- **Recall**: {basic_metrics['recall']:.4f}
- **F1-Score**: {basic_metrics['f1_score']:.4f}
"""
        
        if 'auc_roc' in basic_metrics:
            report += f"- **AUC-ROC**: {basic_metrics['auc_roc']:.4f}\n"
        
        report += f"""
## Trading Performance Metrics
- **Win Rate**: {trading_metrics['win_rate']:.1%}
- **Signal Accuracy**: {trading_metrics['signal_accuracy']:.1%}
- **Expected Return per Trade**: {trading_metrics['expected_return_per_trade']:.3f}
- **Sharpe Ratio**: {trading_metrics['sharpe_ratio']:.3f}
- **Maximum Drawdown**: {trading_metrics['max_drawdown']:.3f}
- **Total Signals Generated**: {trading_metrics['total_signals']}
- **Signal Frequency**: {trading_metrics['signal_frequency']:.1%}

## Baseline Comparison
- **Baseline Win Rate (B_0)**: {self.baseline_win_rate:.1%}
- **Win Rate Improvement**: {trading_metrics['win_rate_improvement']:+.3f} ({trading_metrics['improvement_percentage']:+.1f}%)
- **Target Win Rate**: {self.target_win_rate:.1%}
- **Target Achievement**: {'✅ ACHIEVED' if trading_metrics['win_rate'] >= self.target_win_rate else '❌ NOT ACHIEVED'}
"""
        
        # High confidence metrics
        if 'high_confidence_accuracy' in trading_metrics:
            report += f"""
## High Confidence Predictions (≥{self.config['targets']['model_confidence_threshold']:.0%})
- **High Confidence Accuracy**: {trading_metrics['high_confidence_accuracy']:.1%}
- **High Confidence Win Rate**: {trading_metrics['high_confidence_win_rate']:.1%}
- **High Confidence Samples**: {trading_metrics['high_confidence_sample_count']} ({trading_metrics['high_confidence_percentage']:.1%})
"""
        
        # Fibonacci level breakdown
        if level_metrics:
            report += "\n## Fibonacci Level Performance\n"
            for level_name, metrics in level_metrics.items():
                report += f"\n### {level_name.replace('_', ' ').title()}\n"
                report += f"- **Win Rate**: {metrics['win_rate']:.1%}\n"
                report += f"- **Accuracy**: {metrics['accuracy']:.1%}\n"
                report += f"- **Sample Count**: {metrics['sample_count']}\n"
                report += f"- **Signal Count**: {metrics['signal_count']}\n"
                
                if 'baseline_win_rate' in metrics:
                    improvement = metrics['win_rate'] - metrics['baseline_win_rate']
                    report += f"- **Baseline Win Rate**: {metrics['baseline_win_rate']:.1%}\n"
                    report += f"- **Improvement**: {improvement:+.3f}\n"
        
        return report
    
    def plot_performance_charts(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: np.ndarray = None, save_path: str = None):
        """Generate performance visualization charts."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fibonacci Trading Model Performance', fontsize=16)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Win Rate by Confidence (if probabilities available)
        if y_prob is not None:
            confidence_bins = np.arange(0, 1.1, 0.1)
            win_rates = []
            sample_counts = []
            
            for i in range(len(confidence_bins)-1):
                mask = (y_prob >= confidence_bins[i]) & (y_prob < confidence_bins[i+1])
                if np.sum(mask) > 0:
                    win_rate = np.mean(y_true[mask])
                    win_rates.append(win_rate)
                    sample_counts.append(np.sum(mask))
                else:
                    win_rates.append(0)
                    sample_counts.append(0)
            
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            axes[0, 1].bar(bin_centers, win_rates, width=0.08, alpha=0.7)
            axes[0, 1].axhline(y=self.baseline_win_rate, color='r', linestyle='--', 
                              label=f'Baseline ({self.baseline_win_rate:.1%})')
            axes[0, 1].set_title('Win Rate by Confidence Level')
            axes[0, 1].set_xlabel('Prediction Confidence')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].legend()
        
        # Cumulative Returns Simulation
        returns = []
        for i in range(len(y_true)):
            if y_pred[i] == 1:  # We took the trade
                if y_true[i] == 1:  # Trade won
                    returns.append(self.tp_sl_ratio)
                else:  # Trade lost
                    returns.append(-1.0)
        
        if returns:
            cumulative_returns = np.cumsum(returns)
            axes[1, 0].plot(cumulative_returns)
            axes[1, 0].set_title('Cumulative Returns Simulation')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Cumulative Return')
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Performance Metrics Comparison
        metrics_names = ['Win Rate', 'Accuracy', 'Precision', 'Recall']
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
        
        current_values = [
            trading_metrics['win_rate'],
            basic_metrics['accuracy'],
            basic_metrics['precision'],
            basic_metrics['recall']
        ]
        
        baseline_values = [self.baseline_win_rate, 0.5, 0.5, 0.5]  # Rough baselines
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, current_values, width, label='Current Model', alpha=0.8)
        axes[1, 1].bar(x + width/2, baseline_values, width, label='Baseline', alpha=0.8)
        axes[1, 1].set_title('Metrics Comparison')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_names)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def evaluate_model_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray = None, fibonacci_levels: np.ndarray = None,
                                   model_name: str = "Model") -> Dict:
        """Complete model evaluation with all metrics."""
        
        self.logger.info(f"Evaluating {model_name} performance...")
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred, y_prob)
        
        level_metrics = {}
        if fibonacci_levels is not None:
            level_metrics = self.calculate_fibonacci_level_metrics(y_true, y_pred, fibonacci_levels)
        
        # Generate report
        report = self.generate_performance_report(y_true, y_pred, y_prob, fibonacci_levels, model_name)
        
        # Combine all results
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'trading_metrics': trading_metrics,
            'fibonacci_level_metrics': level_metrics,
            'performance_report': report,
            'model_name': model_name,
            'target_achieved': trading_metrics['win_rate'] >= self.target_win_rate
        }
        
        # Log key findings
        self.logger.info(f"{model_name} Win Rate: {trading_metrics['win_rate']:.1%}")
        self.logger.info(f"Win Rate Improvement: {trading_metrics['win_rate_improvement']:+.3f}")
        self.logger.info(f"Expected Return per Trade: {trading_metrics['expected_return_per_trade']:.3f}")
        
        return evaluation_results
