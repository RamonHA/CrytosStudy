import warnings 
warnings.filterwarnings("ignore")

from trading.processes import Simulation
from datetime import date

from filters import *
# from classifiers import *
import pandas as pd
import json
import multiprocessing as mp
from trading.func_aux import pretty_time, timing
import time
    
def single_exec(freq, tests_time, periods, length, rsi, smoth, slope):

    print( freq, tests_time, periods , length, rsi, smoth, slope)

    def rsislopes(asset):
        # r = asset.rsi_smoth( 14,14 ).iloc[-1]

        # if r > limit:
        #     return None

        v = asset.rsi_smoth_slope( rsi, smoth, slope ).iloc[-1]

        return v if v > 0 else None

    # def emaslope(asset):
    #     v = asset.ema_slope( ema, slope ).iloc[-1]
    #     return v if v > 0 else None

    # def rsislope_ema(asset):
    #     v = asset.rsi_smoth_slope( 11,7,3 ).iloc[-1]

    #     if v < 0: return None

    #     r = asset.ema_slope( 10, 2 ).iloc[-1]

    #     if r < 0: return None

    #     return v 

    def momentum(asset):
        asset.df["ema"] = asset.ema( length )
        m = asset.momentum(periods, target = "ema")
        return m.iloc[ -1 ]


    sim = Simulation(
        broker = "binance",
        fiat = "usdt",
        end = date(2022,12,4),
        simulations=60,
        parallel=False,
        commission=0,
        verbose = 2
    )

    sim.analyze(
        frequency=freq, # 1d
        test_time=tests_time, # 1
        analysis= {
             # "William_RSIVariants_EMASlope":{
            #     "function":william_and_rsi_variants,
            #     "time":150,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":3
            # },
            # "Slopes":{
            #     "function":slopes,
            #     "time":200*24,
            #     "type":"filter",
            # },
            # "SimpleEMA":{
            #     "frequency":"2h"
            #     "function":simple_ema,
            #     "time":100,
            #     "type":"filter",
            #     "filter":"highest", "lowest", "positive", "all"
            #     "filter_qty":3, # 0.1
            # }
            "LowestEMAMomentum_{}_{}".format(periods, length):{
                "function":momentum,
                "time":50,
                "type":"filter",
                "filter":"lowest",
                "filter_qty":0.3
            },
            "RSISlopes_{}_{}_{}".format(rsi, smoth, slope):{
                "function":rsislopes,
                "time":80, # Antes 70
                "type":"filter",
                "filter":"highest",
                "filter_qty":0.7
            }
            # "EMASlope_{}_{}".format(ema, slope):{
            #     "function":emaslope,
            #     "time":150,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":0.7
            # }
            # "RSISlope_EMASlope":{
            #     "function":rsislope_ema,
            #     "time":150,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":0.7
            # }
        },
        run = True,
    )
    
    for r, o in [ ("efficientfrontier", "minvol"), ("efficientsemivariance", "minsemivariance") ]:
        for t in [ 60, 100]:
            sim.optimize(
                balance_time = t,
                risk = r,
                objective = o,
                run = True
            )

    df_indv = sim.results_compilation()

    if len(df_indv) == 0:
        return pd.DataFrame()

    df_indv["frequency"] = freq
    df_indv["periods"] = periods

    return df_indv

@timing
def main():
    params_list = []
    for freq, tests_time in [  ("6h", 4) ]: # ("4h", 6), ("1d", 1)
        for periods in range(2, 5):
            for length in [ 3, 7, 14 ]:
            # for ema in range(10, 61, 10):
            #     for slope in range(2, 5):
            # for pct in [.1, .2, .3, .4, .5]:
                for rsi in [ 11]:
                    for smoth in [ 7, 11]:
                        for slope in [2,3]:
                            params_list.append( (freq, tests_time, periods, length, rsi, smoth, slope) )

    print(f"A total of {len(params_list)} params are being tested.")
    print(params_list)

    with mp.Pool( 4 ) as pool:
        resulting_dataframes = pool.starmap( single_exec, params_list )

    try:
        df = pd.concat( resulting_dataframes, axis = 0 )

        df.to_csv("lowestemamomentum_rsismothslope.csv")

        df = df.sort_values(by = "acc", ascending=False).reset_index(drop = True)

        print(df.head())
        print("\n")
        for i in range(5):
            print(df["route"].iloc[i])

    except:
        r = { t.to_dict() for t in resulting_dataframes }

        with open("lowestemamomentum_rsismothslope.json", "w") as fp:
            json.dump( r, fp )
       

if __name__ == "__main__":
    main()
    
