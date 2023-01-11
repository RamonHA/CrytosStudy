import warnings 
warnings.filterwarnings("ignore")

from datetime import date
import pickle

from filters import *
from functions import *
# from classifiers import *

import pandas as pd
import json
import numpy as np
import multiprocessing as mp
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from trading.processes import Simulation
from trading.variables.params_grid import RF_C_GRID,RF_R_GRID, DT_R_GRID
from trading.func_brokers import get_assets
from trading.grid_search.brute_force import BruteGridSearch
from trading.func_aux import timing

def rf(asset):

    GRID =  {'n_estimators': [200, 500],
        'criterion': ['squared_error', 'absolute_error']}

    asset = features(asset, clf = False)

    bt = BruteGridSearch( asset.df, regr=RandomForestRegressor(), parameters=GRID , error_ascending= True)
    bt.test(parallel = False)
    ppd = bt.predict(one = True)

    return ppd

with open( f"Models/RFC_40Assets_BruteFroce", 'rb') as fp:
    RFC_MODEL = pickle.load(fp)

def rf_pretrain(asset):

    asset = features( asset, clf = True )

    df = asset.df.drop(columns = ["target"]).iloc[-1:].replace( [ np.nan, np.inf, -np.inf ], 0 )

    p = RFC_MODEL.predict( df )

    return True if p[-1] == 1 else False

def single_exec():

    # print( periods )

    def momentum(asset):
        m = asset.momentum(4)

        v = asset.rsi_smoth_slope( 11,7,3 ).iloc[-1]

        if v < 0: return None

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
        frequency="1d",
        test_time=1,
        analysis= {

            # "LowestMomentum_{}".format(periods):{
            #     "function":momentum,
            #     "time":50,
            #     "type":"filter",
            #     "filter":"lowest",
            #     "filter_qty":0.3,
            #     "frequency":"6h"
            # },

            "LowestMomentum2_RSISmonthSlopeSlope":{
                "function":momentum,
                "time":50,
                "type":"filter",
                "filter":"lowest",
                "filter_qty":0.3,
                "frequency":"6h"
            },
   
            # "RFregrBruteForce_asFilter":{
            #     "function":rf,
            #     "time":300,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":0.5
            # },
            "RFCPretrained_BruteForce":{
                "function":rf_pretrain,
                "time":50,
                "type":"filter",
            },
        },
        run = True,
    )
    
    for r, o in [ ("efficientfrontier", "minvol"), ("efficientsemivariance", "minsemivariance") ]:
        for t in [ 40, 60]:
            sim.optimize(
                balance_time = t,
                risk = r,
                objective = o,
                run = True
            )

    df_indv = sim.results_compilation()

    if len(df_indv) == 0:
        return pd.DataFrame()

    # df_indv["periods"] = periods

    return df_indv

@timing
def main():
    # params_list = []
    # for periods in range(3, 5):
    #     params_list.append( (periods) )

    # print(f"A total of {len(params_list)} params are being tested.")
    # print(params_list)

    # with mp.Pool( 4 ) as pool:
    #     resulting_dataframes = pool.map( single_exec, params_list )

    df = single_exec()

    try:
        # df = pd.concat( resulting_dataframes, axis = 0 )

        df.to_csv("lowermomentum_rsismothslope_rfcpretrained_bruteforce.csv")

        df = df.sort_values(by = "acc", ascending=False).reset_index(drop = True)

        print(df.head())
        print("\n")
        l = 5 if len(df) >= 5 else len(df)
        for i in range(l):
            print(df["route"].iloc[i])

    except:
        resulting_dataframes = {}
        r = { t.to_dict() for t in resulting_dataframes }

        with open("lowermomentum_rsismothslope_rfcpretrained_bruteforce.json", "w") as fp:
            json.dump( r, fp )
       

if __name__ == "__main__":
    main()
    
