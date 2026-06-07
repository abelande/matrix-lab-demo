"""
Comparison analyzer for strategy performance vs buy-and-hold.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.metrics.quantitative import MetricsCalculator


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single strategy."""
    name: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            "win_rate": self.win_rate,
        }


@dataclass
class ComparisonReport:
    """Report comparing strategy to buy-and-hold."""
    strategy_metrics: PerformanceMetrics
    buyhold_metrics: PerformanceMetrics
    
    # Value time series
    strategy_values: pd.Series = field(default_factory=pd.Series)
    buyhold_values: pd.Series = field(default_factory=pd.Series)
    
    # Differential analysis
    excess_return: float = 0.0
    excess_sharpe: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    # Period analysis
    periods_outperformed: int = 0
    periods_underperformed: int = 0
    outperformance_rate: float = 0.0
    
    # Summary
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_text(self) -> str:
        """Generate a text report."""
        lines = [
            "=" * 70,
            "PERFORMANCE COMPARISON REPORT",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"{'Metric':<25} {'Strategy':>15} {'Buy & Hold':>15}",
            "-" * 40,
            f"{'Total Return':<25} {self.strategy_metrics.total_return:>14.2%} {self.buyhold_metrics.total_return:>14.2%}",
            f"{'Annualized Return':<25} {self.strategy_metrics.annualized_return:>14.2%} {self.buyhold_metrics.annualized_return:>14.2%}",
            f"{'Sharpe Ratio':<25} {self.strategy_metrics.sharpe_ratio:>15.2f} {self.buyhold_metrics.sharpe_ratio:>15.2f}",
            f"{'Sortino Ratio':<25} {self.strategy_metrics.sortino_ratio:>15.2f} {self.buyhold_metrics.sortino_ratio:>15.2f}",
            f"{'Max Drawdown':<25} {self.strategy_metrics.max_drawdown:>14.2%} {self.buyhold_metrics.max_drawdown:>14.2%}",
            f"{'Volatility':<25} {self.strategy_metrics.volatility:>14.2%} {self.buyhold_metrics.volatility:>14.2%}",
            f"{'VaR (95%)':<25} {self.strategy_metrics.var_95:>14.2%} {self.buyhold_metrics.var_95:>14.2%}",
            "-" * 40,
            "",
            "DIFFERENTIAL ANALYSIS",
            "-" * 40,
            f"Excess Return:          {self.excess_return:>+.2%}",
            f"Excess Sharpe:          {self.excess_sharpe:>+.2f}",
            f"Tracking Error:         {self.tracking_error:>.2%}",
            f"Information Ratio:      {self.information_ratio:>.2f}",
            "",
            "PERIOD ANALYSIS",
            "-" * 40,
            f"Periods Outperformed:   {self.periods_outperformed}",
            f"Periods Underperformed: {self.periods_underperformed}",
            f"Outperformance Rate:    {self.outperformance_rate:.1%}",
            "",
            "=" * 70,
        ]
        
        # Add conclusion
        if self.excess_return > 0 and self.excess_sharpe > 0:
            lines.append("✓ Strategy OUTPERFORMED buy-and-hold on both return and risk-adjusted basis")
        elif self.excess_return > 0:
            lines.append("◐ Strategy had higher returns but lower risk-adjusted performance")
        elif self.excess_sharpe > 0:
            lines.append("◐ Strategy had better risk-adjusted returns but lower total returns")
        else:
            lines.append("✗ Strategy UNDERPERFORMED buy-and-hold")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


class ComparisonAnalyzer:
    """
    Analyzes and compares strategy performance against buy-and-hold.
    
    Can work with either:
    1. Simulation results (from SimulationEngine)
    2. Live broker data (comparing actual vs hypothetical)
    """
    
    def __init__(
        self,
        broker=None,
        config=None,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ):
        """
        Initialize the analyzer.
        
        Args:
            broker: Broker instance for live data.
            config: Configuration object.
            risk_free_rate: Annual risk-free rate.
            trading_days: Trading days per year.
        """
        self.broker = broker
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def compare(
        self,
        strategy_returns: pd.Series,
        buyhold_returns: pd.Series,
        strategy_name: str = "Strategy",
    ) -> ComparisonReport:
        """
        Compare strategy returns to buy-and-hold returns.
        
        Args:
            strategy_returns: Strategy daily returns.
            buyhold_returns: Buy-and-hold daily returns.
            strategy_name: Name for the strategy.
            
        Returns:
            ComparisonReport with full analysis.
        """
        # Align the series
        combined = pd.concat([strategy_returns, buyhold_returns], axis=1).dropna()
        strategy_returns = combined.iloc[:, 0]
        buyhold_returns = combined.iloc[:, 1]
        
        # Calculate metrics for each
        strategy_calc = MetricsCalculator(
            strategy_returns,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
        )
        
        buyhold_calc = MetricsCalculator(
            buyhold_returns,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
        )
        
        strategy_metrics = PerformanceMetrics(
            name=strategy_name,
            total_return=strategy_calc.total_return(),
            annualized_return=strategy_calc.annualized_return(),
            sharpe_ratio=strategy_calc.sharpe_ratio(),
            sortino_ratio=strategy_calc.sortino_ratio(),
            max_drawdown=strategy_calc.max_drawdown(),
            volatility=strategy_calc.volatility(),
            calmar_ratio=strategy_calc.calmar_ratio(),
            var_95=strategy_calc.var(0.95),
            win_rate=strategy_calc.win_rate(),
        )
        
        buyhold_metrics = PerformanceMetrics(
            name="Buy & Hold",
            total_return=buyhold_calc.total_return(),
            annualized_return=buyhold_calc.annualized_return(),
            sharpe_ratio=buyhold_calc.sharpe_ratio(),
            sortino_ratio=buyhold_calc.sortino_ratio(),
            max_drawdown=buyhold_calc.max_drawdown(),
            volatility=buyhold_calc.volatility(),
            calmar_ratio=buyhold_calc.calmar_ratio(),
            var_95=buyhold_calc.var(0.95),
            win_rate=buyhold_calc.win_rate(),
        )
        
        # Calculate cumulative values
        strategy_values = (1 + strategy_returns).cumprod()
        buyhold_values = (1 + buyhold_returns).cumprod()
        
        # Differential analysis
        excess_returns = strategy_returns - buyhold_returns
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days)
        
        excess_return = strategy_metrics.total_return - buyhold_metrics.total_return
        excess_sharpe = strategy_metrics.sharpe_ratio - buyhold_metrics.sharpe_ratio
        
        information_ratio = 0.0
        if tracking_error > 0:
            ann_excess = strategy_metrics.annualized_return - buyhold_metrics.annualized_return
            information_ratio = ann_excess / tracking_error
        
        # Period analysis
        periods_outperformed = (excess_returns > 0).sum()
        periods_underperformed = (excess_returns < 0).sum()
        total_periods = len(excess_returns)
        outperformance_rate = periods_outperformed / total_periods if total_periods > 0 else 0
        
        return ComparisonReport(
            strategy_metrics=strategy_metrics,
            buyhold_metrics=buyhold_metrics,
            strategy_values=strategy_values,
            buyhold_values=buyhold_values,
            excess_return=excess_return,
            excess_sharpe=excess_sharpe,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            periods_outperformed=periods_outperformed,
            periods_underperformed=periods_underperformed,
            outperformance_rate=outperformance_rate,
        )
    
    def compare_simulation_to_broker(
        self,
        simulation_results,
        start_date: Optional[datetime] = None,
    ) -> ComparisonReport:
        """
        Compare simulation results to buy-and-hold of broker holdings.
        
        Args:
            simulation_results: SimulationResults from engine.
            start_date: Start date for comparison period.
            
        Returns:
            ComparisonReport.
        """
        if self.broker is None:
            raise ValueError("Broker required for this comparison")
        
        # Get broker holdings
        holdings = self.broker.get_holdings()
        if not holdings:
            raise ValueError("No broker holdings found")
        
        # Calculate buy-and-hold returns for broker portfolio
        symbols = [h.symbol for h in holdings]
        weights = self.broker.get_holdings_summary()
        
        # Get historical data for buy-and-hold simulation
        try:
            import yfinance as yf
            data = yf.download(symbols, period="1y", progress=False)
            
            if data.empty:
                raise ValueError("Could not fetch historical data")
            
            # Calculate weighted returns
            buyhold_returns = pd.Series(0.0, index=data.index)
            
            for symbol in symbols:
                if symbol in weights and symbol in data.columns.get_level_values(0):
                    symbol_returns = data[symbol]["Close"].pct_change()
                    buyhold_returns += symbol_returns * weights[symbol]
            
            buyhold_returns = buyhold_returns.dropna()
            
        except Exception as e:
            raise ValueError(f"Error calculating buy-and-hold returns: {e}")
        
        # Get strategy returns
        strategy_returns = simulation_results.returns
        
        return self.compare(strategy_returns, buyhold_returns, "Simulation")
    
    def generate_report(
        self,
        strategy_returns: Optional[pd.Series] = None,
        buyhold_returns: Optional[pd.Series] = None,
    ) -> str:
        """
        Generate a comparison report.
        
        If returns not provided, attempts to use broker data.
        
        Returns:
            Text report.
        """
        if strategy_returns is None or buyhold_returns is None:
            if self.broker is None:
                return "Error: No data available for comparison. Provide returns or connect a broker."
            
            # Get broker data and create synthetic comparison
            holdings = self.broker.get_holdings()
            if not holdings:
                return "Error: No holdings found in broker account."
            
            # For now, return a placeholder
            return "Comparison requires simulation results or return series."
        
        report = self.compare(strategy_returns, buyhold_returns)
        return report.to_text()


class BuyAndHoldSimulator:
    """
    Simulates a buy-and-hold strategy for comparison.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the simulator.
        
        Args:
            initial_capital: Starting capital.
            weights: Portfolio weights {symbol: weight}.
        """
        self.initial_capital = initial_capital
        self.weights = weights or {}
    
    def simulate(
        self,
        price_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Simulate buy-and-hold returns.
        
        Args:
            price_data: DataFrame with price data.
            symbols: List of symbols to hold.
            
        Returns:
            Series of daily returns.
        """
        if symbols is None:
            if isinstance(price_data.columns, pd.MultiIndex):
                symbols = list(price_data.columns.get_level_values(0).unique())
            else:
                symbols = ["DEFAULT"]
        
        # Equal weight if not specified
        if not self.weights:
            n = len(symbols)
            self.weights = {s: 1.0 / n for s in symbols}
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=price_data.index)
        
        for symbol in symbols:
            weight = self.weights.get(symbol, 0)
            
            try:
                if isinstance(price_data.columns, pd.MultiIndex):
                    prices = price_data[symbol]["Close"]
                else:
                    prices = price_data["Close"]
                
                returns = prices.pct_change()
                portfolio_returns += returns * weight
                
            except Exception:
                continue
        
        return portfolio_returns.dropna()
    
    def get_value_series(
        self,
        price_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Get portfolio value over time.
        
        Returns:
            Series of portfolio values.
        """
        returns = self.simulate(price_data, symbols)
        cumulative = (1 + returns).cumprod()
        return cumulative * self.initial_capital