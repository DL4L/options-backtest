import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np
from model import OptionType

class DataLoader:

    def __init__(self, start_date: datetime, end_date: datetime, is_batch: bool = False):
        self.start_date = start_date
        self.end_date = end_date
        self.data: pd.DataFrame = None
        self.is_batch = is_batch

    def initialize(self):
        pass

    def load_data(self):
        pass

    def check_if_new_data_needed(self, current_date: datetime):
        pass

class OptionChainDataLoader(DataLoader):

    def __init__(self, start_date: datetime, end_date: datetime,  ticker: str, is_batch: bool = True):
        super().__init__(start_date, end_date, is_batch)
        self.data = pd.DataFrame()
        self.earliest_loaded_date = None
        self.ticker = ticker
    
    def initialize(self):
        # load data for the current month, next month, and the month after that
        current_month = self.start_date.strftime("%Y%m")
        next_month = (self.start_date + pd.DateOffset(months=1)).strftime("%Y%m")
        month_after_next = (self.start_date + pd.DateOffset(months=2)).strftime("%Y%m")
        month_after_next_next = (self.start_date + pd.DateOffset(months=3)).strftime("%Y%m")
        

        current_data = self.load_data_for_month(current_month)
        next_data = self.load_data_for_month(next_month)
        month_after_next_data = self.load_data_for_month(month_after_next)
        month_after_next_next_data = self.load_data_for_month(month_after_next_next)

        self.data = pd.concat([current_data, next_data, month_after_next_data, month_after_next_next_data])

        self.earliest_loaded_date = self.start_date

    def load_data(self, current_date: datetime):

        if self.data.empty:
            self.initialize()
            return self.data
        

        self.data = self.data[self.data["QUOTE_DATE"] >= current_date]

        # load data for the latest month
        latest_month = (current_date + pd.DateOffset(months=3)).strftime("%Y%m")
        latest_month_data = self.load_data_for_month(latest_month)

        self.data = pd.concat([self.data, latest_month_data])

        self.earliest_loaded_date = current_date
        return self.data


    def load_data_for_month(self, year_month):
        file_path = f"chains/{self.ticker.lower()}/{self.ticker.lower()}_eod_{year_month}.txt"
        try:
            # Assuming the file exists and has a 'Date' column to use as index
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path,sep=",")
            # remove brackets from the column names
            df.columns = df.columns.str.replace('[', '')
            df.columns = df.columns.str.replace(']', '')
            df.columns = df.columns.str.replace(' ', '')
            # change QUOTE_DATE and EXPIRE_DATE to datetime
            df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
            df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
            # cast C_DELTA and P_DELTA to float got ValueError: could not convert string to float: ''

            df['C_DELTA'] = pd.to_numeric(df['C_DELTA'], errors='coerce')
            df['P_DELTA'] = pd.to_numeric(df['P_DELTA'], errors='coerce')
            df['C_BID'] = pd.to_numeric(df['C_BID'], errors='coerce')
            df['P_BID'] = pd.to_numeric(df['P_BID'], errors='coerce')
            df['C_ASK'] = pd.to_numeric(df['C_ASK'], errors='coerce')
            df['P_ASK'] = pd.to_numeric(df['P_ASK'], errors='coerce')
            df['C_VOLUME'] = pd.to_numeric(df['C_VOLUME'], errors='coerce')
            df['P_VOLUME'] = pd.to_numeric(df['P_VOLUME'], errors='coerce')
            df["UNDERLYING_LAST"] = pd.to_numeric(df["UNDERLYING_LAST"], errors='coerce').ffill()
            df["STRIKE"] = pd.to_numeric(df["STRIKE"], errors='coerce')

            df.set_index(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], drop=False, inplace=True)


            return df
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
            return None
    
    def check_if_new_data_needed(self, current_date: datetime):
        
        return current_date.month != self.earliest_loaded_date.month
    

class CboeOptionChainDataLoader(OptionChainDataLoader):

    def load_data_for_month(self, year_month):
        # reformat year_month to be year-month, it is currently yearmonth
        year_month = year_month[:4] + "-" + year_month[4:]
        file_pref = "hanweck.UnderlyingOptionsEODQuotesHanweck_"
        file_path = f"chains/cboe/{self.ticker.lower()}/{file_pref}{year_month}.csv"

        try:
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            #rename quote_date to QUOTE_DATE and expiration to EXPIRE_DATE
            df.rename(columns={"quote_date": "QUOTE_DATE", "expiration": "EXPIRE_DATE", "strike": "STRIKE"}, inplace=True)
            df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
            df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
            # add column DTE
            df["DTE"] = (df["EXPIRE_DATE"] - df["QUOTE_DATE"]).dt.days
            # replace values in option_type column ["P", "C"] with OptionType.PUT and OptionType.CALL
            df["option_type"] = df["option_type"].map({"P": OptionType.PUT, "C": OptionType.CALL})
            # set index to be QUOTE_DATE, EXPIRE_DATE, STRIKE, and OPTION_TYPE
            df.set_index(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "option_type"], drop=False, inplace=True)
            return df
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
            return None


def get_underlying_data(ticker:str):

    # Load the data
    tick = yf.Ticker(ticker)
    data = tick.history(start="2018-01-01", end="2023-12-29")
    data.index = data.index.tz_localize(None)

    return data


def get_spread_data(window_days_1: int, window_days_2: int, data: pd.DataFrame):

    # Calculate daily logarithmic returns
    data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    data[f"{window_days_1}-Day Volatility"] = data['Log Returns'].rolling(window=window_days_1).std()
    data[f"{window_days_2}-Day Volatility"] = data['Log Returns'].rolling(window=window_days_2).std()
    data["volatility_spread"] = data[f"{window_days_1}-Day Volatility"] - data[f"{window_days_2}-Day Volatility"]
    data["cum_mean_vol_spread"] = data["volatility_spread"].expanding().mean()
    data["cum_std_vol_spread"] = data["volatility_spread"].expanding().std()
    data["+1 cum_std_vol_spread"] = data["cum_mean_vol_spread"] + data["cum_std_vol_spread"]
    data["-1 cum_std_vol_spread"] = data["cum_mean_vol_spread"] - data["cum_std_vol_spread"]
    data["+2 cum_std_vol_spread"] = data["cum_mean_vol_spread"] + 2*data["cum_std_vol_spread"]
    data["-2 cum_std_vol_spread"] = data["cum_mean_vol_spread"] - 2*data["cum_std_vol_spread"]

    return data