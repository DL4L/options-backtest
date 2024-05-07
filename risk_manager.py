from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import pandas as pd
from model import TradeParams
from data_loaders import get_spread_data

class RiskManagerResultType(str, Enum):
    CAN_OPEN_POSITIONS = "CAN_OPEN_POSITIONS"
    STOP_OPEN_POSITIONS = "STOP_OPEN_POSITIONS"
    STOP_OPEN_AND_CLOSE_POSITIONS = "STOP_OPEN_AND_CLOSE_POSITIONS"
    UPDATE_PARAMS = "UPDATE_PARAMS"

@dataclass
class RiskManagerResult:
    result_type: RiskManagerResultType
    metadata: dict

    def __post_init__(self):
        if self.result_type == RiskManagerResultType.UPDATE_PARAMS:
            assert self.metadata

class RiskManager:
    NAME = "RiskManager"
    def __init__(self):
        pass

    def initialize(self, underlying_data: pd.DataFrame):
        self.underlying_data = underlying_data

    def evaluate(self, current_date: datetime) -> RiskManagerResult:
        pass

    def __repr__(self):
        attrs = vars(self).copy()
        del attrs["underlying_data"]
        return f"{self.NAME} {str(attrs)}"
    
class NoRiskManager(RiskManager):
    NAME = "No Risk Manager"

    def evaluate(self, current_date: datetime):
        return RiskManagerResult(RiskManagerResultType.CAN_OPEN_POSITIONS, {})

class VolatilityRiskManager(RiskManager):
    NAME = "VolatilityRiskManager"
    # TODO later can add params such as days lookback and multiplier threshold etc

    def evaluate(self, current_date: datetime):
        row = self.underlying_data.loc[current_date]

        if row["20-Day Volatility"] > row["+2 cum_std_vol"]:
            return RiskManagerResult(RiskManagerResultType.STOP_OPEN_POSITIONS, {})

        return RiskManagerResult(RiskManagerResultType.CAN_OPEN_POSITIONS, {})
    
class SoftVolatilityRiskManager(RiskManager):
    NAME = "SoftVolatilityRiskManager"
    # TODO later can add params such as days lookback and multiplier threshold etc
    def __init__(self, original_params: TradeParams):
        self.original_params = original_params

    def evaluate(self, current_date: datetime):

        row = self.underlying_data.loc[current_date]

        if row["20-Day Volatility"] > row["+2 cum_std_vol"]:
            new_params = replace(self.original_params)
            # calculate percentage diff between current vol and +2 std vol
            diff = (row["20-Day Volatility"] - row["+2 cum_std_vol"]) / row["+2 cum_std_vol"]
            # increase the delta range by the percentage difference
            new_params.put_delta_max = self.original_params.put_delta_max * (1 - (diff * 1.5))
            new_params.put_delta_min = self.original_params.put_delta_min * (1 - (diff * 1.5))

            return RiskManagerResult(RiskManagerResultType.UPDATE_PARAMS, {"new_params": new_params})


        return RiskManagerResult(RiskManagerResultType.UPDATE_PARAMS, {"new_params": self.original_params})

class VolatilitySpreadRiskManager(RiskManager):
    NAME = "VolatilitySpreadRiskManager"

    def __init__(self, ticker: str, window_days_1: int, window_days_2: int, threshold_mul: float = 1.0):
        self.ticker = ticker # TODO hacky
        self.window_days_1 = window_days_1
        self.window_days_2 = window_days_2
        self.threshold_mul = threshold_mul
    
    def initialize(self, underlying_data: pd.DataFrame):
        self.underlying_data = get_spread_data(self.window_days_1, self.window_days_2, self.ticker)

    def evaluate(self, current_date: datetime):
        row = self.underlying_data.loc[current_date]

        spread = row["volatility_spread"]
        spread_low = row["-1 cum_std_vol_spread"] * self.threshold_mul

        if spread < spread_low:
            return RiskManagerResult(RiskManagerResultType.STOP_OPEN_POSITIONS, {})

        return RiskManagerResult(RiskManagerResultType.CAN_OPEN_POSITIONS, {})