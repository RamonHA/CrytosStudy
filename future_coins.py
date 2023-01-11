# Sample code to detect which Future coins to use

from trading.assets.binance import Binance
from trading import Asset
from trading.func_brokers import historic_download
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
from trading.func_aux import pretty_time

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]

trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if t[-4:] == "USDT"]

st = time.time()

yes_yes_yes = []
yes_yes = []
yes = []
growth = []

first_rule = []
second_rule = []
third_rule = []
forth_rule = []

L = 5

for s, f in trading_pairs:

    asset = Asset(
        symbol=s,
        fiat = f,
        frequency= f"{L}min",
        end = datetime.now(),
        start = datetime.now() - timedelta(seconds= 60*L*150 ),
        from_ = "ext_api",
        broker="binance"
    )

    if asset.df is None or len(asset.df) == 0: continue

    asset.df["rsi"] = asset.rsi_smoth(14, 14)
    asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
    asset.df["ema_slope"] = asset.ema_slope(40, 2)
    asset.df["ema"] = (asset.ema(40) < asset.df["close"]).rolling(2).sum()
    asset.df["growth"] = asset.df["close"].pct_change( 20 )
    asset.df["rsi_smoth_slope"] = asset.rsi_smoth_slope( 7,7,3 )
    asset.df["changes"] = asset.df["close"].pct_change()
    asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(2,4)
    
    d = asset.df.iloc[-1].to_dict()

    growth.append( {"symbol": asset.symbol, "return": d["growth"]} )

    if d["ema"] == 2:
        # changes = asset.df.iloc[-10:]["changes"].mean()
        pos_changes = asset.df[ asset.df["changes"] > 0 ].iloc[-10:]["changes"].mean()
        arr = {"symbol": asset.symbol, "return": pos_changes}
        if d["ema_slope"] > 0 and d["rsi"] > 40 and d["oneside_gaussian_filter_slope"] > 0:

            if d["rsi_smoth_slope"] > 0:
                if d["buy_wf"]:
                    first_rule.append(arr)
                    continue
                    
                second_rule.append( arr )
                continue
        
            third_rule.append(arr)
            continue
        
        forth_rule.append(arr)
    
def myFunc(e):
  return e['return']

first_rule.sort(key = myFunc, reverse=True)
second_rule.sort(key = myFunc, reverse=True)
third_rule.sort(key = myFunc, reverse=True)
forth_rule.sort(key = myFunc, reverse=True)
growth.sort(key = myFunc, reverse=True)

print("\n")

print(first_rule, (len(first_rule)/len(trading_pairs)) )
print("\n")

print(second_rule, (len(second_rule)/len(trading_pairs)) )
print("\n")

print(third_rule, (len(third_rule)/len(trading_pairs)) )
print("\n")

print(forth_rule, (len(forth_rule)/len(trading_pairs)) )
print("\n")

print("Greatest growth: \n", growth[:5])

print("\n")

print(time.time() - st)
