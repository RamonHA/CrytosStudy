from trading.testers.rules_testing import RulesGenerator, RuleTesting
from trading import Asset
from datetime import date, datetime
import time
import json
import matplotlib.pyplot as plt
import numpy as np

st = time.time()

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

def plot_buy(asset):
    asset.df["buy1"] = asset.df["buy"] * asset.df["close"]
    asset.df["buy1"] = asset.df["buy1"].replace(0, np.nan)

    asset.df[ "close" ].plot()
    plt.scatter(y = asset.df[["buy1"]], x = asset.df.index, color = "r")

asset = Asset(
    "HOT",
    start = datetime( 2023, 3, 27, 0 ),
    end = datetime.now(),
    frequency="3min",
    fiat = "USDT",
    broker = "binance",
    source = "ext_api"
)

def sandr(asset):
    # El problema de esta es que casi nunca entraba, pero cuando entraba si las cerraba
    # l = 9 if asset.ema(27).rolling(20).std().iloc[-1] <= 0.7421 else 18
    # _, asset.df["resistance"] = asset.support_resistance(l)
    # asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
    # asset.df["rsi"] = asset.rsi_smoth_slope(30, 4, 7) > 0.00223 
    # asset.df["sma"] = asset.sma_slope(44, 12) > (-0.00625)

    # After 19/03/2023 meta aplication
    # se detiene esta estrategia por varias entradas erroneas. 21/3/2023
    asset.df["ema_std"] = asset.ema(43).rolling(19).std()
    # max_std = asset.df["ema_std"].max()
    # max_std = asset.df["ema_std"].max()
    l = 9 if asset.df["ema_std"].iloc[-1] <= 0.35 else 19
    _, asset.df["resistance"] = asset.support_resistance(l)
    asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
    
    # Last change: 223/03/23
    # Error con CTK
    asset.df["rsi_smoth"] = asset.rsi_smoth( 21, 10 ).rolling( 10 ).std() > 1.2 # < 0.7
    
    asset.df[ "rsi_thr" ] = ( asset.rsi(7) >= 71 ).rolling(17).sum() == 0

    # Last Change: 23/03/23
    # COTI Liquidation
    asset.df["rsi_slope"] = asset.rsi(10)
    asset.df["rsi_slope"] = asset.ema_slope(10, 3, target = "rsi_slope") > 0
    
    asset.df["sma"] = asset.ema_slope(30, 4) > (0) # -0.00625

    return asset.df[ [ "resistance", "rsi_smoth", "rsi_thr", "rsi_slope", "sma" ] ].all(axis = 1)


asset.df["buy"] = sandr(asset)

# asset.df["rsi_slope"] = asset.rsi_smoth_slope(27, 20, 4) > (5.75/1000)
# asset.df["rsi_slope2"] = asset.rsi_smoth_slope(28, 8, 3) > (-4/1000)
# asset.df["sma"] = asset.sma_slope( 33, 20 ) > ( 0 )
# # asset.df["ema"] = asset.ema_slope( 36, 14 ) > (1.5/1000)
# asset.df["dema"] = asset.dema(12).pct_change(11) > (5/1000)
# asset.df["hull_twma"] = asset.hull_twma(7).pct_change(5) > (-4/1000)

# asset.df["roc"] = asset.roc( 15 ).pct_change(12) > (7/1000)

# # asset.df["vwap"] = asset.vwap( 22 ).pct_change(11) > (5/1000)

# # asset.df["ema_std"] = asset.ema(26).rolling(8).std() > (-1/1000)

# asset.df["buy"] = asset.df[["rsi_slope", "rsi_slope2", "sma", "dema", "hull_twma", ]].all(axis = 1)

r = RuleTesting( asset, rules = {"buy":"buy == True"} , target = 0.0015)
r.run()

# rules = [ 
#     # "rsi_smooth_slope_several == 2",
#     # "ema < close",
#     # "ema_slope > 0",
#     "rsi_smoth < {}",
#     # "rsi_smoth_slope > 0",
#     # "william_fractals == True",
#     # "oneside_gaussian_filter_slope > 0",
#     # "supertrend > 0"
#     # "engulfing == 1"
# ]

# universe = [ 
#     ( 50, 91, 5 ) # 20, 30, 40, 50
# ]

# columns = { 
#     # "rsi_smooth_slope_several":[ [7, 14], [7, 14], [3], [2] ],
#     # "ema":(10, 61, 10), # 30, 60, 90, 120
#     # "ema_slope":[ (10, 61, 10), (2, 5) ],
#     "rsi_smoth":[ [ 7, 9, 11, 14, 21], [ 3, 7, 9, 11, 14] ], # 7, 14, 21   
#     # "rsi_smoth_slope":[  [7, 9, 11, 14], [ 7, 9, 11, 14], (2, 5)  ], # 7-2, 7-3, 7-4, 14-2, 14-3, 14-4
#     "william_fractals":[ [2, 3], [True] ],
#     # "oneside_gaussian_filter_slope":[(2, 5), (2,6)],
#     # "supertrend":[ [5, 7,10,15, 20], [2,3,4] ]
# }

# # 4 * 4 * 3 * 6 

# rg = RulesGenerator( asset=asset, rules=rules, universe=universe , target = [0.001, 0.0015, 0.002, 0.0025, 0.003], and_or = "and", columns=columns)

# rg.run()

# results = rg.results.sort_values(by = "acc", ascending = False)

# print(results.head())

# for i in results.iloc[:10].index:
#     print( rg.rules_obj[i].asset.params, results.loc[i]["rules"] )

# # resume = {
# #     "rules":rules,
# #     "universe":universe,
# #     "columns":columns,
# #     "results":results.to_dict()
# # }

# # s = "-".join( list(columns.keys()) )

# # with open( f"results/ta_testing/{s}.json", "w" ) as fp:
# #     json.dump( resume, fp )

# # results.iloc[:10].to_csv( "fractals_rsismoth.csv" )

# print("\n\n")
# print( time.time() - st )