from trading import Asset
from datetime import date
import pandas as pd
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

def new(symbol, **kwargs):

    start = kwargs.get("start", date(2022,6,1))
    end = kwargs.get("end", date(2022, 11, 1) )

    asset = Asset(
        symbol=symbol,
        fiat = "USDT",
        start = start,
        end = end,
        frequency = "1d",
        broker = "binance",
        from_ = "ext_api"
    )

    return asset

def test(asset):
    buy = False
    price_bought = None
    price_sold = None
    returns = []
    for i in range(len(asset.df)):
        if not buy and asset.df.iloc[i]["buy"] :
            buy = True
            price_bought = asset.df.iloc[i]["close"]
        
        elif buy and asset.df.iloc[i]["sell"]:
            buy = False
            price_sold = asset.df.iloc[i]["close"]
                
            returns.append( [ asset.df.index[i] , price_bought, price_sold ] )

    df = pd.DataFrame(returns, columns = ["date", "bought", "sold"] )
    df.set_index("date", inplace = True)
    df["returns"] = ((df["sold"] / df["bought"]) - 1).round(3)
    df[ "acc" ] = (df["sold"] / df["bought"]).cumprod().round(3)

    return df

def plot_strategy(asset, results = None):
    df = copy(asset.df)

    df["buy"] = df["buy"]*df["close"]
    df["sell"] = df["sell"]*df["close"]

    df["buy"] = df["buy"].replace( 0, np.nan )
    df["sell"] = df["sell"].replace( 0, np.nan )

    df[ [ "close" ] ].plot( figsize = (12,4) )
    plt.scatter( x = df.index,  y = df["buy"] , color = "g")
    plt.scatter( x = df.index,  y = df["sell"] , color = "r")

    if results is not None:
        plt.plot(  results["acc"] , color = "m")

def normalize(df, cols):

    for col in cols:
        df[col] = ( df[col] - df[col].min() ) / ( df[col].max() - df[col].min() )

    return df

def features(asset, clf = True):
 
    ori_cols = asset.df.drop(columns = ["volume"]).columns

    for i in [20, 40, 60, 80]:
        asset.df[ f"ema_{i}"] = asset.ema(i)
        asset.df[ f"roc_{i}" ] = asset.roc(i)

        for j in range(2, 12, 3):
            asset.df[ f"ema_{i}_slope_{j}" ] = asset.df[ f"ema_{i}" ].pct_change( j )
        
        for c in ["close", "high", "volume"]:
            asset.df["std{}_{}".format(c, i)] = asset.df[c].rolling(i).std()

    for i in [7, 14, 21]:
        asset.df[ f"rsi_{i}"] = asset.rsi_smoth(i, 2)
        
        for j in range(2,7, 2):
            asset.df[ f"rsi_{i}_slope_{j}" ] = asset.df[ f"rsi_{i}" ].pct_change( j )
    
    for i in [2,3,4,5,6]:
        asset.df[f"momentum_{i}"] = asset.momentum(i)
        asset.df[f"momentum_ema_{i}"] = asset.momentum(i, target = "ema_20")
        asset.df[f"momentum_rsi_{i}"] = asset.momentum(i, target = "rsi_7")

    asset.df["hl"] = asset.df["high"] - asset.df["low"]
    asset.df["ho"] = asset.df["high"] - asset.df["open"]
    asset.df["lo"] = asset.df["low"] - asset.df["open"]
    asset.df["cl"] = asset.df["close"] - asset.df["low"]
    asset.df["ch"] = asset.df["close"] - asset.df["high"]

    asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
    for i in [2,3,4]:
        for j in [2,3,4]:
            asset.df[f"oneside_gaussian_filter_slope_{i}_{j}"] = asset.oneside_gaussian_filter_slope(i,j)

    asset.df["obv"] = asset.obv()

    for i in [20, 40, 60]:
        s, r = asset.support_resistance(i)
        asset.df[ f"support_{i}" ] = ( s / asset.df["close"] ) - 1
        asset.df[ f"resistance_{i}" ] = ( r / asset.df["close"] ) - 1

    # Normalization
    n_cols = list( set(asset.df.columns) - set(ori_cols) )
    
    asset.df = normalize(asset.df, cols = n_cols)

    asset.df["engulfing"] = asset.engulfing()
    asset.df["william_buy"] = asset.william_fractals(2, order = "buy").apply(lambda x : 1 if x == True else 0).rolling(5).sum()
    asset.df["william_sell"] = asset.william_fractals(2, order = "sell").apply(lambda x : 1 if x == True else 0).rolling(5).sum()

    if clf:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1).apply(lambda x: 1 if x > 0 else 0)
    else:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1)

    asset.df.drop(columns = ori_cols, inplace = True)

    return asset

