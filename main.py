from model import TradeParams2
from data_loaders import CboeOptionChainDataLoader
from risk_manager import VolatilitySpreadRiskManager, NoRiskManager
from datetime import datetime
from strategy_runner import StrategyRunner, StrategyParams
from strategy import CboeStrategy

cboe_trade_params = TradeParams2(40, 50, 0.05, 0.15, 0.009, 0.02, call_notional_multiplier=1, put_notional_multiplier=1) # Not optimal Params, just example
cboe_assymetric_no_roll_risk_manager_cfg = [
    (cboe_trade_params, [NoRiskManager()]),
    (cboe_trade_params, [VolatilitySpreadRiskManager(50, 20, 1.0)]),
]

START_DATE = datetime(2019, 1, 2)
END_DATE = datetime(2023, 12, 29)

strat_params = [StrategyParams(CboeStrategy, param, risk_manager) for param, risk_manager in cboe_assymetric_no_roll_risk_manager_cfg]

strategy_cboe_assymetric = StrategyRunner(START_DATE, END_DATE, CboeOptionChainDataLoader, ticker="QQQ")
strategy_cboe_assymetric.add_strategies(strat_params)
strategy_cboe_assymetric.run()


summary_stats = strategy_cboe_assymetric.get_summary_stats_dataframe()
strategy_cboe_assymetric.portfolio_returns_plot(show_underlying=True)