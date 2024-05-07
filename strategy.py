from datetime import datetime
from dataclasses import dataclass,replace
from enum import Enum
from typing import List, Union
from collections import defaultdict
from model import Option, Trade, TradeParams, TradeParams2, OptionType, Portfolio, LiquidityCheck
from data_loaders import get_underlying_data
from risk_manager import RiskManager, RiskManagerResult, RiskManagerResultType
import pandas as pd


class Strategy:

    NAME = "Option Sell Strategy"
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2023, 12, 29)

    def __init__(self, params: Union[TradeParams, TradeParams2], risk_manager: List[RiskManager], ticker: str):
        self.can_open_positions = True
        self.close_all_positions = False
        self.original_params = replace(params)
        self.params = replace(params)
        self.positions: dict[Option, int] = {}
        self.trades: dict[datetime, List[Trade]] = defaultdict(list)
        self.ticker = ticker
        self.underlying_data = get_underlying_data(ticker)
        self.underlying_price: float = None
        self.events = defaultdict(list)
        self.call_count = 0
        self.put_count = 0
        self.portfolio_values: dict[datetime, Portfolio] = None
        self.portfolio_value: float = None
        self.risk_manager: List[RiskManager] = risk_manager
    
    def initialize(self, starting_cash: float, trade_date: datetime):
        self.portfolio_values = {}
        self.portfolio_values[trade_date] = Portfolio(starting_cash, 0)
        self.portfolio_value = starting_cash
        for rm in self.risk_manager:
            rm.initialize(self.underlying_data)
    
    def next(self, chain: pd.DataFrame, trade_date: datetime):

        try:
            self._update_underlying_price(chain, trade_date)
        except KeyError:
            print(f"NO UNDERLYING PRICE FOUND ON {trade_date}")
            return
        
        self._close_positions(chain, trade_date)
        self._update_portfolio(chain, trade_date)
        if self.can_open_positions:
            self._open_positions(chain, trade_date)
        
        self.run_risk_manager(trade_date)
    
    def _update_portfolio(self, chain: pd.DataFrame, trade_date: datetime):
        unrealized = 0
        for position in self.positions:
            try:
                live_option = self._find_live_option_exact(chain, trade_date, position)
            except:
                print(f"Could not find {position} on {trade_date}")
                continue
                
            mid_price = self._calc_mid_price(live_option, position.option_type)
            
            unrealized += (position.open_price - mid_price) * abs(position.quantity) * 100
        
        cash = self.portfolio_value
        for trade in self.trades[trade_date]:
            if trade.to_close:
                realized = (trade.option.open_price - trade.price) * abs(trade.quantity) * 100
                cash += realized
        
        self.portfolio_values[trade_date] = Portfolio(cash, unrealized)
        self.portfolio_value = cash

    def _find_live_option_exact(self, chain, trade_date, option: Option):
        
        live_option = chain.loc[(trade_date, option.expiration, option.strike)]
        return live_option
    
    def _update_underlying_price(self, chain: pd.DataFrame, trade_date: datetime):
        self.underlying_price = chain.loc[trade_date].iloc[0]["UNDERLYING_LAST"]


    
    def _close_positions(self, chain: pd.DataFrame, trade_date: datetime):
        
        if self.close_all_positions:
            options = self.positions.keys()
            try:
                self._close_options(set(options), chain, trade_date)
                self.close_all_positions = False
            except ValueError:
                print(f"Unable to close all positions on {trade_date}")
            
        items = list(self.positions.items())
        for option, quantity in items:
            if self._should_close_position(option, chain, trade_date):
                self._close_option(option, chain, trade_date)


    def _open_positions(self, chain: pd.DataFrame, trade_date: datetime):
        if self._should_open_position(chain, trade_date):
            print(f"opening positions on {trade_date}")
            live_call, live_put, max_call_size, max_put_size = self._get_option(chain, trade_date) # TODO Hacky
            
            if live_call is None and live_put is None:
                return
            
            call_size, put_size = self._get_position_size()
            call_size = min(call_size, max_call_size)
            put_size = min(put_size, max_put_size)

            if self.call_count == 0:
                call_mid_price = self._calc_mid_price(live_call, OptionType.CALL)
                call = Option(OptionType.CALL, live_call["STRIKE"], live_call["EXPIRE_DATE"], -1 * call_size, call_mid_price)
                self.positions[call] = call_size
                
                call_trade = Trade(trade_date, call, call_mid_price, -1 * call_size, to_close= False, underlying_price=self.underlying_price)
                self._add_trade(call_trade, chain)
            
            if self.put_count == 0:
                put_mid_price = self._calc_mid_price(live_put, OptionType.PUT)
                put = Option(OptionType.PUT, live_put["STRIKE"], live_put["EXPIRE_DATE"], -1 * put_size, put_mid_price)
                self.positions[put] = put_size
                put_trade = Trade(trade_date, put, put_mid_price, -1 * put_size, to_close=False, underlying_price=self.underlying_price)
                self._add_trade(put_trade, chain)
    
    def _get_position_size(self) -> tuple[int, int]:
        cash = self.portfolio_value
        underlying_notional = self.underlying_price * 100
        call_size = (cash * self.params.call_notional_multiplier) // underlying_notional
        put_size = (cash * self.params.put_notional_multiplier) // underlying_notional
        return call_size, put_size

    def _should_open_position(self, chain: pd.DataFrame, trade_date: datetime):
        if len(self.positions) != 2:
            return True

    def _get_option(self, chain: pd.DataFrame, trade_date: datetime):
        # get calls with QUOTE_DATE = trade_date and DTE between dte_min and dte_max and delta between delta_min and delta_max
        options = chain[(chain["QUOTE_DATE"] == trade_date) & (chain["DTE"] >= self.params.dte_min) 
                        & (chain["DTE"] <= self.params.dte_max) & (chain["EXPIRE_DATE"].isin(chain["QUOTE_DATE"]))]
        if options.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with expiry between {self.params.dte_min} and {self.params.dte_max}")
            return None, None, None, None
        
        calls = options[(options["C_DELTA"] >= self.params.call_delta_min) & (options["C_DELTA"] <= self.params.call_delta_max)]
        puts = options[(options["P_DELTA"] >= self.params.put_delta_min) & (options["P_DELTA"] <= self.params.put_delta_max)]

        if calls.empty or puts.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with delta between {self.params.call_delta_min} and {self.params.call_delta_max}")
            return None, None, None, None
        
        call_size = sum(calls["C_VOLUME"])//3
        put_size = sum(puts["P_VOLUME"])//3

        call = self._get_single_option(calls.sort_values(by="C_BID", ascending=False), chain)
        put = self._get_single_option(puts.sort_values(by="P_BID", ascending=False), chain)

        if call.empty or put.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} which are closeable between {self.params.dte_min} and {self.params.dte_max} days")
            return None, None, None, None
            
        
        return call, put, call_size, put_size
    
    def _get_single_option(self, calls_or_puts: pd.DataFrame, chain: pd.DataFrame):

        for index, row in calls_or_puts.iterrows():
            match = chain[
                (chain["EXPIRE_DATE"] == row["EXPIRE_DATE"]) &
                (chain["STRIKE"] == row["STRIKE"]) &
                (chain["QUOTE_DATE"] == row["EXPIRE_DATE"])
            ]
        
            if not match.empty:
                return row
        
        else:
            return pd.Series()
        
    def _should_roll_option(self, option: Option, chain: pd.DataFrame, trade_date: datetime):
        should_roll = False

        underlying_price = self.underlying_price
        
        if option.option_type == OptionType.CALL:
            if underlying_price >= option.strike:
                should_roll = True
        else:
            if underlying_price <= option.strike:
                should_roll = True
        
        return should_roll
    
    def _should_close_position(self, option: Option, chain: pd.DataFrame, trade_date: datetime):
        
        if self.close_all_positions:
            return True
        
        if option.expiration == trade_date:
            return True

    def _close_options(self, options: set[Option], chain: pd.DataFrame, trade_date: datetime):
        for option in options:
            self._close_option(option, chain, trade_date)

    def _close_option(self, option: Option, chain: pd.DataFrame, trade_date: datetime):
        print(f"closing option: {option}")
        try:
            live_option = chain.loc[(trade_date, option.expiration, option.strike)] # assumes it is a series
        except IndexError:
            live_option = chain[(chain["QUOTE_DATE"] == trade_date) & (chain["EXPIRE_DATE"] == option.expiration) 
                                & (chain["STRIKE"] >= option.strike -1) & (chain["STRIKE"] <= option.strike +1)]
            if not live_option.empty:
                live_option["strike_diff"] = abs(live_option["STRIKE"] - option.strike)
                live_option = live_option.sort_values(by='strike_diff').iloc[0]
            else:
                raise ValueError("No suitable option found within ±1 strike range")


        mid_price = self._calc_mid_price(live_option, option.option_type)

        trade = Trade(trade_date, option, mid_price, option.quantity * -1, to_close=True, underlying_price=self.underlying_price) # quantity is neg, so make pos for closing trade (Assumes buying)
        self._add_trade(trade, chain)
        del self.positions[option]

    def _liquidity_check(self, trade: Trade, chain: pd.DataFrame, trade_date: datetime):
        pass

    def _add_trade(self, trade: Trade, chain: pd.DataFrame):
        print(f"adding trade: {trade}")
        self.trades[trade.trade_date].append(trade)
        
        self._liquidity_check(trade, chain, trade.trade_date)

        if trade.option.option_type == OptionType.CALL:
            self.call_count += trade.quantity * -1
        else:
            self.put_count += trade.quantity * -1
        
    
    
    def run_risk_manager(self, trade_date: datetime):
        if not self.risk_manager:
            return

        results = []
        for risk_manager in self.risk_manager:
            result = risk_manager.evaluate(trade_date)
            results.append(result)

        self.events[trade_date].extend(results)

        for result in results:
            if result.result_type == RiskManagerResultType.STOP_OPEN_AND_CLOSE_POSITIONS:
                self.can_open_positions = False
                self.close_all_positions = True
                return
            
            if result.result_type == RiskManagerResultType.STOP_OPEN_POSITIONS:
                self.can_open_positions = False
                return
        
            if result.result_type == RiskManagerResultType.CAN_OPEN_POSITIONS:
                self.can_open_positions = True
            
            if result.result_type == RiskManagerResultType.UPDATE_PARAMS:
                self.params = result.metadata["new_params"]
        
    
    def _calc_mid_price(self, live_option: pd.Series, option_type: OptionType) -> float:
        assert type(live_option) == pd.Series
        if option_type == OptionType.CALL:
            mid_price = (live_option["C_BID"] + live_option["C_ASK"]) / 2
        else:
            mid_price = (live_option["P_BID"] + live_option["P_ASK"]) / 2
        
        return mid_price


class CboeStrategy(Strategy):
    
    def _update_underlying_price(self, chain: pd.DataFrame, trade_date: datetime):
        row = chain.loc[trade_date].iloc[0]
        bid = row["underlying_bid_1545"]
        ask = row["underlying_ask_1545"]
        self.underlying_price = (bid + ask) / 2

    def _find_live_option_exact(self, chain, trade_date, option: Option):
        
        live_option = chain.loc[(trade_date, option.expiration, option.strike, option.option_type)]
        return live_option
    

    def _get_option(self, chain: pd.DataFrame, trade_date: datetime):
        # get calls with QUOTE_DATE = trade_date and DTE between dte_min and dte_max and delta between delta_min and delta_max
        options = chain[(chain["QUOTE_DATE"] == trade_date) & (chain["DTE"] >= self.params.dte_min) 
                        & (chain["DTE"] <= self.params.dte_max)]
        if options.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with expiry between {self.params.dte_min} and {self.params.dte_max}")
            return None, None, None, None
        
        min_call_strike = self.underlying_price * (1 + self.params.call_pct_dist_min)
        max_call_strike = self.underlying_price * (1 + self.params.call_pct_dist_max)
        min_put_strike = self.underlying_price * (1 - self.params.put_pct_dist_max)
        max_put_strike = self.underlying_price * (1 - self.params.put_pct_dist_min)

        calls = options[(options["option_type"] == OptionType.CALL) & (options["STRIKE"] >= min_call_strike) & (options["STRIKE"] <= max_call_strike)]
        puts = options[(options["option_type"] == OptionType.PUT) & (options["STRIKE"] >= min_put_strike) & (options["STRIKE"] <= max_put_strike)]

        if calls.empty or puts.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with strikes between {min_call_strike} and {max_call_strike} and {min_put_strike} and {max_put_strike}")
            return None, None, None, None
        
        call_size = sum(calls["trade_volume"])//3
        put_size = sum(puts["trade_volume"])//3

        call = calls.sort_values(by="bid_1545", ascending=False).iloc[0]
        put = puts.sort_values(by="bid_1545", ascending=False).iloc[0]

        if call.empty or put.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} which are closeable between {self.params.dte_min} and {self.params.dte_max} days")
            return None, None, None, None
            
        
        return call, put, call_size, put_size
    
    def _close_option(self, option: Option, chain: pd.DataFrame, trade_date: datetime):
        print(f"closing option: {option}")

        live_option = self._find_live_option_exact(chain, trade_date, option) # assumes it is a series


        mid_price = self._calc_mid_price(live_option, option.option_type)

        trade = Trade(trade_date, option, mid_price, option.quantity * -1, to_close=True, underlying_price=self.underlying_price) # quantity is neg, so make pos for closing trade (Assumes buying)
        self._add_trade(trade, chain)
        del self.positions[option]

    def _calc_mid_price(self, live_option: pd.Series, option_type: OptionType) -> float:
        assert type(live_option) == pd.Series

        mid_price = (live_option["bid_1545"] + live_option["ask_1545"]) / 2

        return mid_price

    def _liquidity_check(self, trade: Trade, chain: pd.DataFrame, trade_date: datetime):
        option = trade.option
        single_option_volume = self._find_live_option_exact(chain, trade_date, option)["trade_volume"]
        min_call_strike = option.strike * (1 + self.params.call_pct_dist_min)
        max_call_strike = option.strike * (1 + self.params.call_pct_dist_max)
        min_put_strike = option.strike * (1 - self.params.put_pct_dist_max)
        max_put_strike = option.strike * (1 - self.params.put_pct_dist_min)
        filtered_chain = chain.loc[trade_date]
        if option.option_type == OptionType.CALL:
            volume = filtered_chain[(filtered_chain["STRIKE"] >= min_call_strike) & (filtered_chain["STRIKE"] <= max_call_strike)]["trade_volume"].sum()
        else:
            volume = filtered_chain[(filtered_chain["STRIKE"] >= min_put_strike) & (filtered_chain["STRIKE"] <= max_put_strike)]["trade_volume"].sum()
        
        check = LiquidityCheck(option.option_type, abs(option.quantity), single_option_volume, volume, trade.to_close)
        self.events[trade_date].append(check)



                
    
class CboeStrategyDisperse(CboeStrategy):

    def _get_option(self, chain: pd.DataFrame, trade_date: datetime):
        # get calls with QUOTE_DATE = trade_date and DTE between dte_min and dte_max and delta between delta_min and delta_max
        options = chain[(chain["QUOTE_DATE"] == trade_date) & (chain["DTE"] >= self.params.dte_min) 
                        & (chain["DTE"] <= self.params.dte_max)]
        if options.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with expiry between {self.params.dte_min} and {self.params.dte_max}")
            return None, None
        
        min_call_strike = self.underlying_price * (1 + self.params.call_pct_dist_min)
        max_call_strike = self.underlying_price * (1 + self.params.call_pct_dist_max)
        min_put_strike = self.underlying_price * (1 - self.params.put_pct_dist_max)
        max_put_strike = self.underlying_price * (1 - self.params.put_pct_dist_min)

        calls = options[(options["option_type"] == OptionType.CALL) & (options["STRIKE"] >= min_call_strike) & (options["STRIKE"] <= max_call_strike)]
        puts = options[(options["option_type"] == OptionType.PUT) & (options["STRIKE"] >= min_put_strike) & (options["STRIKE"] <= max_put_strike)]

        if calls.empty or puts.empty:
            print(f"NO OPTIONS FOUND ON {trade_date} with strikes between {min_call_strike} and {max_call_strike} and {min_put_strike} and {max_put_strike}")
            return None, None
        

        return calls, puts
    
    def _open_positions(self, chain: pd.DataFrame, trade_date: datetime):
        if self._should_open_position(chain, trade_date):
            print(f"opening positions on {trade_date}")
            call_size, put_size = self._get_position_size()
            live_calls, live_puts = self._get_option(chain, trade_date) # TODO Hacky
            

            if self.call_count == 0 and live_calls is not None:
                for live_call, size in self._get_positions_from_target(live_calls, call_size - self.call_count):
                    call_mid_price = self._calc_mid_price(live_call, OptionType.CALL)
                    call = Option(OptionType.CALL, live_call["STRIKE"], live_call["EXPIRE_DATE"], -1 * size, call_mid_price)
                    self.positions[call] = size
                    
                    call_trade = Trade(trade_date, call, call_mid_price, -1 * size, to_close= False, underlying_price=self.underlying_price)
                    self._add_trade(call_trade, chain)

            
            if self.put_count == 0 and live_puts is not None:
                for live_put, size in self._get_positions_from_target(live_puts, put_size - self.put_count):
                    put_mid_price = self._calc_mid_price(live_put, OptionType.PUT)
                    put = Option(OptionType.PUT, live_put["STRIKE"], live_put["EXPIRE_DATE"], -1 * size, put_mid_price)
                    self.positions[put] = size
                    put_trade = Trade(trade_date, put, put_mid_price, -1 * size, to_close=False, underlying_price=self.underlying_price)
                    self._add_trade(put_trade, chain)
                    
    def _get_positions_from_target(self, calls_or_puts: pd.DataFrame, target_size: int):

        for index, row in calls_or_puts.sort_values(by="bid_1545", ascending=False).iterrows():
            max_fillable = row["trade_volume"]//3
            if max_fillable == 0:
                continue

            fill_size = min(max_fillable, target_size)

            yield row, fill_size

            target_size -= fill_size

            if target_size <= 0:
                break
            
    def _get_positions_from_target_2(self, calls_or_puts: pd.DataFrame, target_size: int):
       
        target_size_strike = calls_or_puts["STRIKE"].median() * target_size

        for index, row in calls_or_puts.sort_values(by="trade_volume", ascending=False).iterrows():
            max_fillable = row["trade_volume"]//2

            max_fillable_size_strike = max_fillable * row["STRIKE"]

            if max_fillable_size_strike < target_size_strike:
                fill_size = max_fillable
            else:
                fill_size = target_size_strike // row["STRIKE"]

            if fill_size == 0:
                break

            yield row, fill_size

            target_size_strike -= row["STRIKE"] * fill_size

            if target_size_strike <= 0:
                break
    
    def _should_open_position(self, chain: pd.DataFrame, trade_date: datetime):
        return self.call_count == 0 or self.put_count == 0

    def _liquidity_check(self, trade: Trade, chain: pd.DataFrame, trade_date: datetime):
        option = trade.option
        single_option_volume = self._find_live_option_exact(chain, trade_date, option)["trade_volume"]
        in_the_money = self.underlying_price >= option.strike if option.option_type == OptionType.CALL else self.underlying_price <= option.strike
        check = LiquidityCheck(option.option_type, abs(option.quantity), single_option_volume, None, trade.to_close, in_the_money)
        self.events[trade_date].append(check)