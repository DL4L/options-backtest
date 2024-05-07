from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"

class EventType(str, Enum):
    PARAM_CHANGE = "PARAM_CHANGE"
    ROLL = "ROLL"

@dataclass
class Event:
    event_type: EventType
    description: str

@dataclass
class Option:
    option_type: OptionType
    strike: float
    expiration: datetime
    quantity: int
    open_price: float

    def __hash__(self):
        return hash((self.option_type, self.strike, self.expiration))

@dataclass
class Trade:
    trade_date: datetime
    option: Option
    price: float
    quantity: int
    to_close: bool
    underlying_price: float

@dataclass
class TradeParams:
    dte_min: int
    dte_max: int
    call_delta_min: float
    call_delta_max: float
    put_delta_min: float
    put_delta_max: float
    roll_on_breach: bool = False
    keep_balanced: bool = True
    call_notional_multiplier: float = 1
    put_notional_multiplier: float = 1

@dataclass
class TradeParams2:
    dte_min: int
    dte_max: int
    call_pct_dist_min: float
    call_pct_dist_max: float
    put_pct_dist_min: float
    put_pct_dist_max: float
    roll_on_breach: bool = False
    keep_balanced: bool = True
    call_notional_multiplier: float = 1
    put_notional_multiplier: float = 1

class OPEN_POSITION(str, Enum):
    BOTH = "BOTH"
    CALL = "CALL"
    PUT = "PUT"
    DO_NOTHING = "DO_NOTHING"

@dataclass
class Portfolio: # TODO should maybe separate cash and realized
    cash: float
    unrealized: float

@dataclass
class LiquidityCheck:
    option_type: OptionType
    option_quantity: int
    single_option_volume: int
    dispersed_option_volume: int
    to_close: bool
    in_the_money: bool