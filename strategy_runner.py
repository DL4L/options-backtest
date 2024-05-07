from datetime import datetime
from typing import List, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from model import TradeParams, OptionType
from risk_manager import RiskManager
from data_loaders import DataLoader, get_underlying_data
from model import LiquidityCheck
from strategy import Strategy



@dataclass
class StrategyParams:
    strategy: Strategy
    param: TradeParams
    risk_manager: List[RiskManager]
    desc: str = ""

    def __hash__(self) -> int:
        return hash((self.strategy.NAME, str(self.param), self._risk_manager_str(), self.desc))
    def __repr__(self):
        return f"{self.strategy.NAME} : {self.param} : {self._risk_manager_str()} : {self.desc}"
    
    def _risk_manager_str(self):
        return ", ".join([str(rm) for rm in self.risk_manager])

@dataclass
class ResultData:
    params: StrategyParams
    trades: pd.DataFrame
    daily_sums: pd.DataFrame
    cumulative_sums: pd.DataFrame
    portfolio: pd.DataFrame
    events: dict


@dataclass
class SummaryStats:
    params: StrategyParams
    end_return: float
    number_of_trades: int
    avg_open_price: float
    avg_close_price: float
    avg_call_open_price: float
    avg_call_close_price: float
    avg_put_open_price: float
    avg_put_close_price: float
    max_drawdown: float
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    max_unrl_drawdown: float
    max_unrl_drawdown_start: datetime
    max_unrl_drawdown_end: datetime
    num_rolls: int = 0


class PlotManager:
    def __init__(self):
        self.figures = []  # List to store figure objects

    def add_plot(self, fig):
        self.figures.append(fig)

    def show(self):
        for fig in self.figures:
            plt.figure(fig.number)  # Activate the figure
            plt.show()

    def get_figures(self):
        # Generator that yields figures
        for fig in self.figures:
            yield fig

class StrategyRunner:

    def __init__(self, start_date: datetime, end_date: datetime, data_loader: DataLoader, ticker: str):
        self.start_date = start_date
        self.end_date = end_date
        self.starting_cash = 10000000
        self.strategies: List[StrategyParams] = []
        self.data_loader: DataLoader = data_loader
        self.ticker = ticker
        self.results: dict[tuple[int, StrategyParams], Strategy] = {}
        self.data_snapshot = pd.DataFrame()
        self.results_data: dict[int, ResultData] = {}
        self.summary_stats: dict[int, SummaryStats] = {}
    
    def get_dates(self):
        return (date for date in get_underlying_data(self.ticker).index if date >= self.start_date)
    
    def add_strategies(self, strategy_params: List[StrategyParams]):
        for strategy_param in strategy_params:
            self.strategies.append(strategy_param)
    
    def run(self):
        for strat_id, strategy_param in enumerate(self.strategies):
            self.run_strategy(strategy_param, strat_id + 1)
    
    def run_strategy(self, strategy_param: StrategyParams, strat_id: int):
        data_loader = self.data_loader(self.start_date, self.end_date, ticker=self.ticker)
        data = data_loader.load_data(self.start_date)
        self.data_snapshot = data
        date_range = self.get_dates()

        strategy, param, risk_manager = strategy_param.strategy, strategy_param.param, strategy_param.risk_manager
        strategy = strategy(param, risk_manager, ticker=self.ticker)
        strategy.initialize(self.starting_cash, self.start_date)

        for current_date in date_range:
            if data_loader.is_batch and data_loader.check_if_new_data_needed(current_date):
                data = data_loader.load_data(current_date)
                self.data_snapshot = data

            strategy.next(data, current_date)
        
        self.results[(strat_id, strategy_param)] = strategy
        self.populate_results_data(strategy_param, strategy, strat_id)
        self.populate_summary_stats(strategy_param, strategy, strat_id)
    
    def populate_summary_stats(self,strategy_param, strategy, strat_id):
        
        trades = [trade for trade_list in strategy.trades.values() for trade in trade_list]
        opens = [trade for trade in trades if not trade.to_close]
        closes = [trade for trade in trades if trade.to_close]

        
        number_of_trades = len(trades)
        avg_open_price = np.mean([trade.price for trade in opens])
        avg_close_price = np.mean([trade.price for trade in closes])

        avg_call_open_price = np.mean([trade.price for trade in opens if trade.option.option_type == OptionType.CALL])
        avg_call_close_price = np.mean([trade.price for trade in closes if trade.option.option_type == OptionType.CALL])
        avg_put_open_price = np.mean([trade.price for trade in opens if trade.option.option_type == OptionType.PUT])
        avg_put_close_price = np.mean([trade.price for trade in closes if trade.option.option_type == OptionType.PUT])

        # Calculate the drawdown
        portfolio = self.results_data[strat_id].portfolio.copy(deep=True)
        end_return = portfolio["cash"].iloc[-1]

        portfolio["running_max"] = portfolio["cash"].cummax()
        portfolio["cash_drawdown"] = portfolio["running_max"] - portfolio["cash"]

        max_cash_drawdown = portfolio["cash_drawdown"].max()
        max_cash_drawdown_end = portfolio["cash_drawdown"].idxmax()
        max_cash_drawdown_start = portfolio.loc[:max_cash_drawdown_end, "cash"].idxmax()

        portfolio["cash_plus_unrealized"] = portfolio["cash"] + portfolio["unrealized"]
        portfolio["unrl_drawdown"] = portfolio["running_max"] - portfolio["cash_plus_unrealized"]
        max_unrl_drawdown = portfolio["unrl_drawdown"].max()
        max_unrl_drawdown_end = portfolio["unrl_drawdown"].idxmax()
        max_unrl_drawdown_start = portfolio.loc[:max_unrl_drawdown_end, "cash"].idxmax()

        self.summary_stats[strat_id] = SummaryStats(strategy_param, end_return, number_of_trades, avg_open_price, avg_close_price,
                                                            avg_call_open_price, avg_call_close_price, avg_put_open_price, 
                                                            avg_put_close_price, max_cash_drawdown, max_cash_drawdown_start, max_cash_drawdown_end,
                                                            max_unrl_drawdown, max_unrl_drawdown_start, max_unrl_drawdown_end)

    def get_summary_stats_dataframe(self):
        summary_stats = []
        for strat_id, stats in self.summary_stats.items():
            summary_stats.append({"id":strat_id, "param": stats.params.param, "risk_manager": stats.params._risk_manager_str(), "end_return": stats.end_return, "number_of_trades": stats.number_of_trades,
                                  "avg_open_price": stats.avg_open_price, "avg_close_price": stats.avg_close_price, "avg_call_open_price": stats.avg_call_open_price,
                                  "avg_call_close_price": stats.avg_call_close_price, "avg_put_open_price": stats.avg_put_open_price, "avg _put_close_price": stats.avg_put_close_price,
                                  "max_drawdown": stats.max_drawdown, "max_drawdown_start": stats.max_drawdown_start, "max_drawdown_end": stats.max_drawdown_end,
                                  "max_unrl_drawdown": stats.max_unrl_drawdown, "max_unrl_drawdown_start": stats.max_unrl_drawdown_start, "max_unrl_drawdown_end": stats.max_unrl_drawdown_end,}
                                  )
        return pd.DataFrame(summary_stats)
    
    def populate_results_data(self,strategy_param: StrategyParams, strategy:Strategy, strat_id: int):
        
        flattened_data = []
        for date, trades in strategy.trades.items():
            for trade in trades:
                flattened_data.append({'trade_date': date, 'price': trade.price, 
                                        'quantity': trade.quantity, "type": trade.option.option_type,
                                        'strike': trade.option.strike, 'expiration': trade.option.expiration,
                                        'underlying': trade.underlying_price,})

        # Convert to DataFrame
        df_trades = pd.DataFrame(flattened_data)

        # Calculate the product of price and negative quantity
        df_trades['price_x_neg_quantity'] = df_trades['price'] * df_trades['quantity'] * -1 * 100
        # Aggregate by date and sum the products
        daily_sum = df_trades.groupby('trade_date')['price_x_neg_quantity'].sum()
        
        # Compute the cumulative sum
        cumulative_sum = daily_sum.cumsum()
        cumulative_sum = cumulative_sum.to_frame().rename(columns={'price_x_neg_quantity': 'cumulative_return'})
        cumulative_sum['portfolio_return'] = cumulative_sum['cumulative_return'] + self.starting_cash
        daily_sum = daily_sum.to_frame().rename(columns={'price_x_neg_quantity': 'daily_return'})

        dates = []
        cash_values = []
        unrealized_values = []

        # Populate the lists with data from the dictionary
        for date, portfolio in strategy.portfolio_values.items():
            dates.append(date)
            cash_values.append(portfolio.cash)
            unrealized_values.append(portfolio.unrealized)

        # Create DataFrame
        portfolio = pd.DataFrame({
            'date': dates,
            'cash': cash_values,
            'unrealized': unrealized_values
        })

        # Optionally set 'Date' as the index
        portfolio.set_index('date', inplace=True)

        self.results_data[strat_id] = ResultData(strategy_param, df_trades, daily_sum, cumulative_sum, portfolio, strategy.events)
    
    def portfolio_returns_plot(self, strategy_ids: List[int] = None, show_underlying: bool = True):
        plot_manager = PlotManager()
        param_to_portfolio = {(strat_id, result_data.params): result_data.portfolio 
                              for strat_id, result_data in self.results_data.items()
                              if strategy_ids is None or strat_id in strategy_ids}

        fig, ax = plt.subplots(figsize=(20, 12))
        for (strat_id, param), portfolio in param_to_portfolio.items():
            portfolio["Cash + Unrealized"] = portfolio["cash"] + portfolio["unrealized"]
            portfolio["Cash + Unrealized"].plot(ax=ax, kind='line', marker='o', label=f"{strat_id} {param}")

        underlying = get_underlying_data(self.ticker).loc[self.start_date: self.end_date]
        starting_shares = self.starting_cash / underlying["Close"].iloc[0]
        underlying["portfolio"] = underlying["Close"] * starting_shares
        
        if show_underlying:
            underlying["portfolio"].plot(ax=ax, kind='line', marker='o', label=f"Benchmark {self.ticker}")
        
        ax.set_title('Portfolio Cash + Unrealized')
        ax.set_xlabel('Trade Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plot_manager.add_plot(fig)
        return plot_manager

    def portfolio_cash_plot(self, strategy_ids: List[int] = None):
        plot_manager = PlotManager()
        param_to_portfolio = {(strat_id, result_data.params): result_data.portfolio 
                              for strat_id, result_data in self.results_data.items()
                              if strategy_ids is None or strat_id in strategy_ids}

        fig, ax = plt.subplots(figsize=(20, 12))
        for (strat_id, param), portfolio in param_to_portfolio.items():
            portfolio["cash"].plot(ax=ax, kind='line', marker='o', label=f"{strat_id} {param}")
        

        ax.set_title('Portfolio Cash')
        ax.set_xlabel('Trade Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plot_manager.add_plot(fig)
        return plot_manager
    
    
    def returns_plot(self) -> PlotManager:
        plot_manager = PlotManager()
        param_to_cumulative_sum = {(strat_id, result_data.params): result_data.cumulative_sums for strat_id, result_data in self.results_data.items()}

        fig, ax = plt.subplots(figsize=(20, 12))
        for (strat_id, param), cumulative_sum in param_to_cumulative_sum.items():
            cumulative_sum["portfolio_return"].plot(ax=ax, kind='line', marker='o', label=f"{strat_id} {param}")

        ax.set_title('Portfolio Value')
        ax.set_xlabel('Trade Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plot_manager.add_plot(fig)
        return plot_manager
    
    def trades_scatter_plot(self, strategy_ids: List[int] = None) -> PlotManager:
        plot_manager = PlotManager()
        for (strat_id, strategy_param), strategy in self.results.items():

            if strategy_ids and strat_id not in strategy_ids:
                continue

            trades = [trade for trade_list in strategy.trades.values() for trade in trade_list]
            call_opens = [trade for trade in trades if not trade.to_close and trade.option.option_type == OptionType.CALL]
            call_closes = [trade for trade in trades if trade.to_close and trade.option.option_type == OptionType.CALL]
            put_opens = [trade for trade in trades if not trade.to_close and trade.option.option_type == OptionType.PUT]
            put_closes = [trade for trade in trades if trade.to_close and trade.option.option_type == OptionType.PUT]
            closes = [trade for trade in trades if trade.to_close]
            opens = [trade for trade in trades if not trade.to_close]

            fig, ax = plt.subplots(figsize=(20, 12))
            ax.set_title(f"""id: {strat_id},
                        param: {strategy_param.param}, 
                        number of trades: {len(trades)},  
                        avg close price: {np.mean([trade.price for trade in closes])}, 
                        avg open price: {np.mean([trade.price for trade in opens])}
                        """)
            ax.scatter([trade.trade_date for trade in call_closes], [trade.price for trade in call_closes], marker="$C$", color='red', label="Closing Call")
            ax.scatter([trade.trade_date for trade in call_opens], [trade.price for trade in call_opens], marker="$C$", color='blue', label="Opening Call")
            ax.scatter([trade.trade_date for trade in put_closes], [trade.price for trade in put_closes], marker="$P$", color='red', label="Closing Put")
            ax.scatter([trade.trade_date for trade in put_opens], [trade.price for trade in put_opens], marker="$P$", color='blue', label="Opening Put")
            ax.legend()
            plt.tight_layout()
            plot_manager.add_plot(fig)
        
        return plot_manager

    def plot_liquidity_checks(self, strategy_ids: List[int] = None) -> PlotManager:
        plot_manager = PlotManager()
        for (strat_id, strategy_param), strategy in self.results.items():

            if strategy_ids and strat_id not in strategy_ids:
                continue

            # split liquidity checks between calls and puts into dicts where key is trade_date
            call_liquidity_checks_open = {}
            call_liquidity_checks_close = {}
            put_liquidity_checks_open = {}
            put_liquidity_checks_close = {}
            for date, events in strategy.events.items():
                for event in events:
                    if isinstance(event, LiquidityCheck):
                        if event.option_type == OptionType.CALL:
                            if event.to_close:
                                call_liquidity_checks_close[date] = event
                            else:
                                call_liquidity_checks_open[date] = event
                        else:
                            if event.to_close:
                                put_liquidity_checks_close[date] = event
                            else:
                                put_liquidity_checks_open[date] = event
            
            # plot separate scatter plots for calls and puts
            # each date should plot the option quantity as green circle, the single option volume as blue star, and the dispersion volume as red triangle
            fig, ax = plt.subplots(figsize=(20, 12))
            ax.set_title(f"""id: {strat_id},
                        Call Trade Volumes Open
                        """)
            _scatter_liquidity_checks(ax, call_liquidity_checks_open)
            
            plt.tight_layout()
            plot_manager.add_plot(fig)

            fig, ax = plt.subplots(figsize=(20, 12))
            ax.set_title(f"""id: {strat_id},
                        Call Trade Volumes Close
                        """)
            _scatter_liquidity_checks(ax, call_liquidity_checks_close)
            plt.tight_layout()
            plot_manager.add_plot(fig)
            
            fig, ax = plt.subplots(figsize=(20, 12))
            ax.set_title(f"""id: {strat_id},
                        Put Trade Volumes Open
                        """)
            _scatter_liquidity_checks(ax, put_liquidity_checks_open)
            plt.tight_layout()
            plot_manager.add_plot(fig)

            fig, ax = plt.subplots(figsize=(20, 12))
            ax.set_title(f"""id: {strat_id},
                        Put Trade Volumes Close
                        """)
            _scatter_liquidity_checks(ax, put_liquidity_checks_close)
            plt.tight_layout()
            plot_manager.add_plot(fig)

        
        return plot_manager
    
        

    def plot_for_strategies(self, strategy_ids:List[int], plot_functions: List[Callable]):
        for plot_func in plot_functions:
            plot_manager = plot_func(strategy_ids=strategy_ids)
            plot_manager.show()

    def save_plots_to_pdf(self, filename, plot_functions):
        with PdfPages(filename) as pdf:
            for plot_func in plot_functions:
                plot_manager = plot_func()
                for fig in plot_manager.get_figures():
                    pdf.savefig(fig)
                    plt.close(fig)
            pdf.close()
            
from matplotlib.lines import Line2D

def _scatter_liquidity_checks(ax, liquidity_checks):
    
    mean_single_option_volume = np.mean([event.single_option_volume for event in liquidity_checks.values()])
    mean_dispersed_option_volume = np.mean([event.dispersed_option_volume for event in liquidity_checks.values()])

    for date, event in liquidity_checks.items():
        ax.scatter(date, event.option_quantity, marker="o", color='green')
        ax.scatter(date, event.single_option_volume, marker="*", color='blue')
        ax.scatter(date, event.dispersed_option_volume, marker="^", color='red')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Option Quantity', markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='*', color='w', label='Single Option Volume', markerfacecolor='blue', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='Dispersion Volume', markerfacecolor='red', markersize=10)]
    
    ax.set_ylim(top= max(mean_single_option_volume, mean_dispersed_option_volume))
    ax.legend(handles=legend_elements)