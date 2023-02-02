from trading.testers.rules_testing import RulesGenerator
from trading import Asset
from datetime import date
import time
import json

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


asset = NAsset(
    "LTC",
    start = date(2022,1,1),
    end = date(2022,10,1),
    frequency="1d",
    fiat = "usdt",
    broker = "binance",
    from_ = "ext_api"
)

rules = [ 
    # "rsi_smooth_slope_several == 2",
    # "ema < close",
    # "ema_slope > 0",
    "rsi_smoth < {}",
    # "rsi_smoth_slope > 0",
    "william_fractals == True",
    # "oneside_gaussian_filter_slope > 0",
    # "supertrend > 0"
    # "engulfing == 1"
]

universe = [ 
    ( 50, 91, 5 ) # 20, 30, 40, 50
]

columns = { 
    # "rsi_smooth_slope_several":[ [7, 14], [7, 14], [3], [2] ],
    # "ema":(10, 61, 10), # 30, 60, 90, 120
    # "ema_slope":[ (10, 61, 10), (2, 5) ],
    "rsi_smoth":[ [ 7, 9, 11, 14, 21], [ 3, 7, 9, 11, 14] ], # 7, 14, 21   
    # "rsi_smoth_slope":[  [7, 9, 11, 14], [ 7, 9, 11, 14], (2, 5)  ], # 7-2, 7-3, 7-4, 14-2, 14-3, 14-4
    "william_fractals":[ [2, 3], [True] ],
    # "oneside_gaussian_filter_slope":[(2, 5), (2,6)],
    # "supertrend":[ [5, 7,10,15, 20], [2,3,4] ]
}

# 4 * 4 * 3 * 6 

rg = RulesGenerator( asset=asset, rules=rules, universe=universe , target = [0.001, 0.0015, 0.002, 0.0025, 0.003], and_or = "and", columns=columns)

rg.run()

results = rg.results.sort_values(by = "acc", ascending = False)

print(results.head())

for i in results.iloc[:10].index:
    print( rg.rules_obj[i].asset.params, results.loc[i]["rules"] )

# resume = {
#     "rules":rules,
#     "universe":universe,
#     "columns":columns,
#     "results":results.to_dict()
# }

# s = "-".join( list(columns.keys()) )

# with open( f"results/ta_testing/{s}.json", "w" ) as fp:
#     json.dump( resume, fp )

# results.iloc[:10].to_csv( "fractals_rsismoth.csv" )

print("\n\n")
print( time.time() - st )