from trading.testers.rules_testing import RuleTesting, rule_validation
from trading.func_brokers import get_assets
from trading import Asset

import numpy as np
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import multiprocessing as mp

np.random.seed(1)

assets = np.random.choice( list( get_assets()["binance"].keys() ), 10 )

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


def main():
    symbol = assets[0]
    single_asset( symbol )

    
if __name__ == "__main__":
    main()