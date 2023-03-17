from trading.testers.rules_testing import RuleTesting, rule_validation
from trading.func_brokers import get_assets
from trading import Asset

import numpy as np
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
import plotly.express as xp

np.random.seed(1)

assets = np.random.choice( list( get_assets()["binance"].keys() ), 40 )

def sell_column(asset, target ):
    asset.df["sell"] = False
    true_values = asset.df[ asset.df["buy"] == True ].index.tolist()
    close = asset.df["close"]

    for i in true_values:
        close_price = close[i]
        close_aux = close[ i: ]
        close_aux = ( close_aux / close_price ) - 1
        pct_index = close_aux[ close_aux > target ]

        if len(pct_index) == 0:
            # Este mensaje no importa por el momento, solo nos indica que como no hay mejor, yo no
            # nos moveremos a los siguientes puesto
            # que al no cerrarse esta orden, no podemos abrir ni cerrar las demas.
            # raise Exception( "Testing did not prove a better return." )
            break
        
        pct_index = pct_index.index[0]

        try:
            asset.df.loc[ pct_index, "sell" ] = True
        except Exception as e:
            raise Exception( f"{e}. Got {type(pct_index)} from {pct_index}." )
    
    return asset

class NAsset(Asset):
    def ema_several(self, length, period):
        v = (self.ema(length) < self.df["close"]).rolling(period).sum()
        return v
    
    def ema_slope_several(self, length, slope, period):
        v = (self.ema_slope(length, slope) > 0).rolling(period).sum()
        return v
    
    def rsi_smooth_slope_several(self, length, smoth, slope, period):
        v = (self.rsi_smoth_slope(length, smoth, slope) > 0).rolling(period).sum()
        return v

    def rsi_diff(self, length, smoth):
        return self.rsi(length=length) - self.rsi_smoth( length=length, smoth=smoth )

    def rolling_fractals(self, lenght, period, shift = False, order = "buy"):
        v = self.william_fractals(period, shift = shift, order = order)
        return v.rolling( lenght ).sum()

def single_asset(symbol):

    asset = NAsset(
        symbol=symbol,
        start = datetime(2023,1,14),
        end = datetime(2023,1,15, 10),
        frequency = "5min",
        broker = "binance",
        fiat = "USDT",
        from_ = "db"
    )

    # rt = RuleTesting( asset, rules = {"buy":""}, target = 0.002  )

    asset.df[ "buy" ] = asset.william_fractals( period = 3, shift = True )
    # asset.df["sell"] = asset.rolling_fractals( lenght=3, period=3, shift = True, order = "sell" ) > 0

    asset = sell_column( asset, 0.002 )

    results = rule_validation(asset)
    print(results)


def seasonality(symbol):
    asset = Asset(
        symbol,
        start = datetime(2023,3,10),
        end = datetime(2023,3,16, 12), # At least 3000 records
        frequency="3min",
        broker = "binance",
        fiat = "USDT",
        source="db"
    )

    if asset is None or len(asset.df) == 0:
        return []

    asset.df["trend"] = asset.ema(60)
    asset.df["trend_res"] = asset.df["close"] - asset.df["trend"]
    asset.df["season"] = asset.sma( 30, target = "trend_res" )
    asset.df["season_res"] = asset.df["trend_res"] - asset.df["season"]

    seasonal = asset.df[["season"]].dropna()

    # sampling rate
    sr = len(seasonal)
    # sampling interval
    ts = 1/sr
    t = np.arange(0,1,ts)

    # r = round(seasonal["season"].std(), ndigits=2)
    r = seasonal["season"].std()
    
    reg = []
    for i in range(4, 34, 1):
        y = np.sin(np.pi*i*t) * r

        seasonal["sin"] = y

        error  = np.linalg.norm( seasonal["season"] - seasonal["sin"] )

        reg.append([ i, error ])

    reg = pd.DataFrame(reg, columns = ["freq", "error"])
    i = reg[ reg[ "error" ] == reg["error"].min() ]["freq"].iloc[0]
    y = np.sin(np.pi*i*t)*r

    zeros = np.zeros(len(asset.df) - len(y))
    asset.df[ "sin" ] = zeros.tolist() + y.tolist()
    asset.df["buy"] = asset.df["sin"] == asset.df["sin"].min()

    asset = sell_column( asset, 0.002 )

    df = rule_validation(asset)

    ret = df["acc"].iloc[-1] if len(df) > 0 else 0
    min_ = df["returns"].min() if len(df) > 0 else 0

    return [ symbol, i, ret , min_ , r]


def main():
    symbol = assets[0]
    single_asset( symbol ) 

    
if __name__ == "__main__":
    # main()
    results = [ seasonality(symbol) for symbol in assets ]

    results = pd.DataFrame(results, columns = ["symbol", "freq", "acc", "max drawdown", "std"]).dropna()

    print(results["acc"].mean())
    print(results["max drawdown"].min())