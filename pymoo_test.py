import warnings
warnings.filterwarnings("ignore")

from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.core.problem import StarmapParallelization

from pymoo.termination import get_termination

from pymoo.optimize import minimize

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import IntegerRandomSampling

import numpy as np
from copy import deepcopy
from datetime import date, datetime
import pandas as pd
import json
from multiprocessing.pool import ThreadPool
import os

from trading.testers.rules_testing import rule_validation
from trading import Asset
from trading.func_brokers import get_assets
from trading.func_aux import timing

np.random.seed(1)

def sell_column(asset, target ):
    asset.df["sell"] = False
    true_values = asset.df[ asset.df["buy"] == True ].index.tolist()
    close = asset.df["close"]

    if len(true_values) == 0:
        return asset

    last_value_index = true_values[-1]

    for i in true_values:
        close_price = close[i] # bought price
        close_aux = close[ i: ]
        close_aux = ( close_aux / close_price ) - 1
        pct_index = close_aux[ close_aux >= target ]

        if len(pct_index) == 0:
            # Esta entrada nos indica que como no hay mejor, entonces el retorno de la estrategia 
            # estara ligada al ultrimo precio de nuestro tabal
            # Sera el ultimo precio de la tabla lo que sera considerado como la orden de cierre.
            asset.df.loc[ -1, "sell" ] = True
            # raise Exception( "Testing did not prove a better return." )
            break
        
        pct_index = pct_index.index[0]

        try:
            asset.df.loc[ pct_index, "sell" ] = True
        except Exception as e:
            raise Exception( f"{e}. Got {type(pct_index)} from {pct_index}." )
    
    return asset

class TATunning(ElementwiseProblem):

    def __init__(self, assets, dias, **kwargs):
        self.assets = assets
        self.dias = dias
        super().__init__(n_var=kwargs["n_var"],
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=kwargs["xl"], # np.array([7,3, 40]),
                         xu=kwargs["xu"], #np.array([21,14, 90])
                         vtype=int
        )

    def my_obj_func(self, asset, x):
        """ 
            Rule selection:
                1:  
        """
        asset = deepcopy(asset)

        cols_to_use = []

        # Anterior test
        # if (x[1] >= x[3]) or ( x[10] >= x[11] ):
        #     return np.inf

        # if x[0]:
        #     cols_to_use.append("rsi")
        #     asset.df["rsi1"] = asset.rsi_smoth(x[1], x[2])
        #     asset.df["rsi2"] = asset.rsi_smoth(x[3], x[4])
        #     asset.df["rsi"] = (asset.df["rsi1"] > asset.df["rsi2"]).astype(int).diff().rolling(2).sum() > 0

        #     if x[16]:
        #         cols_to_use.append("rsi2")
        #         asset.df["rsi2"] = asset.df["rsi2"] < x[17]

        # if x[5]:
        #     cols_to_use.append("rsi_slope")
        #     asset.df["rsi_slope"] = asset.rsi_smoth_slope(x[6], x[7], x[8]) > 0

        # if x[9]:
        #     cols_to_use.append("ema")
        #     asset.df["ema1"] = asset.ema(x[10])
        #     asset.df["ema2"] = asset.ema(x[11])
        #     asset.df["ema"] = (asset.df["ema1"] > asset.df["ema2"]).astype(int).diff().rolling(2).sum() > 0
        
        # if x[12]:
        #     cols_to_use.append("rsi_thr")
        #     asset.df["rsi_thr"] = (asset.rsi( x[13] ) > x[14]).rolling(x[15]).sum() == 0

        # if x[21]:
        #     cols_to_use.append("buy_wf")
        #     asset.df["buy_wf"] = asset.william_fractals(3, shift=True)
        
        # # if x[14]:
        # #     cols_to_use.append("sell_wf")
        # #     asset.df["sell_wf"] = asset.william_fractals(3, shift=True, order = "sell").rolling(3).sum() == 0
        
        # # if x[15]:
        # #     cols_to_use.append("rsi")
        # #     asset.df["rsi"] = asset.rsi_smoth(x[0], x[1]) < x[2]
        
        # if x[18]:
        #     cols_to_use.append("ema_slope")
        #     asset.df["ema_slope"] = asset.ema_slope( x[19], x[20] ) > 0

        # if x[22]:
        #     cols_to_use.append("rsi_std")
        #     asset.df[ "rsi_std" ] = asset.rsi_smoth(x[23],x[24]).rolling(x[25]).std()
        
        # # if x[17]:
        # #     cols_to_use.append("rsi_slope")
        # #     asset.df["rsi_slope"] = asset.rsi_smoth_slope( x[6], x[7], x[8] ) > x[9]
        
        # # if x[18]:
        # #     cols_to_use.append("oneside_gaussian_filter_slope")
        # #     asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(x[10],x[11]) > x[12]
        
        # # if x[19]:
        # #     cols_to_use.append("engulfing_buy")
        # #     asset.df["engulfing_buy"] = asset.engulfing() == 1
        
        # # if x[20]:
        # #     cols_to_use.append("engulfing_sell")
        # #     asset.df["engulfing_sell"] = asset.engulfing() != -1

        # # if x[21]:
        # #     cols_to_use.append("dema_sma")
        # #     asset.df["dema_sma"] = asset.dema( x[22] ) > asset.sma( x[23] )

        # if (x[0] >= x[4]):
        #     return np.inf

        # std_series = asset.ema(x[1]).rolling(x[2]).std()
        # l = x[0] if std_series.iloc[-1] <= (x[3]/100) else x[4]
        # _, asset.df["resistance"] = asset.support_resistance(l)
        # asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
        # cols_to_use.append("resistance")
        
        # i = 5
        # if x[i]:
        #     asset.df["rsi"] = asset.rsi_smoth_slope(x[i+1], x[i+2], x[i+3]) > (x[i+4]/1000)
        #     cols_to_use.append("rsi")
        
        # i = 10
        # if x[i]:
        #     asset.df["sma"] = asset.sma_slope(x[i+1], x[i+2]) > (x[i+3]/1000)
        #     cols_to_use.append("sma")

        # Slopes
        i = 0
        if x[i]:
            asset.df["rsi"] = asset.rsi_smoth_slope(x[i+1], x[i+2], x[i+3]).between(x[i+4]/1000, x[i+5]/1000)
            cols_to_use.append("rsi")
            i += 6
        
        if x[i]:
            asset.df["rsi2"] = asset.rsi_smoth_slope(x[i+1], x[i+2], x[i+3]).between(x[i+4]/1000, x[i+5]/1000)
            cols_to_use.append("rsi2")
            i += 6

        if x[i]:
            asset.df["sma"] =   asset.sma_slope(x[i+1], x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("sma")
            i += 5
        
        if x[i]:
            asset.df["ema"] =  asset.ema_slope(x[i+1], x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("ema")
            i += 5
        
        if x[i]:
            asset.df["dema"] =  asset.dema(x[i+1]).pct_change(x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("dema")
            i += 5
        
        if x[i]:
            asset.df["hull_twma"] =  asset.hull_twma(x[i+1]).pct_change(x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("hull_twma")
            i += 5

        if x[i]:
            asset.df["cci"] =  asset.cci_slope(x[i+1], x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("cci")
            i += 5

        if x[i]:
            asset.df["roc"] = asset.roc(x[i+1]).pct_change(x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("roc")
            i += 5
        
        if x[i]:
            asset.df["vwap"] = asset.vwap(x[i+1]).pct_change(x[i+2]).between(x[i+3]/1000, x[i+4]/1000)
            cols_to_use.append("vwap")
            i += 5

        if x[i]:
            cols_to_use.append("ema_std")
            asset.df[ "ema_std" ] =  asset.ema(x[i+1]).rolling(x[i+2]).std().between(x[i+3]/1000, x[i+4]/1000)
            i += 5
        
        if x[i]:
            cols_to_use.append("rsi_std")
            asset.df[ "rsi_std" ] =  asset.rsi_smoth(x[i+1], x[i+2]).rolling(x[i+3]).std().between(x[i+4]/1000, x[i+5]/1000)
            i += 6


        if len(cols_to_use) == 0:
            return np.inf

        df = asset.df[ cols_to_use ]

        asset.df[ "buy" ] = df.all(axis = 1)

        # Clean buy columns
        # Get only the first buy column that pops out
        asset.df["extra_buy"] = asset.df["buy"] != asset.df["buy"].shift(1)
        asset.df["buy"] = asset.df["buy"] & asset.df["extra_buy"]

        asset = sell_column(asset, target = 0.0015)
        
        df = rule_validation(asset)
        
        if len(df) >= self.dias*10: # 10 TRANSACCIONES por dia

            acc = df["acc"].iloc[-1]
            if acc == 0:
                acc = np.inf
            else:
                acc = 1 / acc
        
        elif len(df) >= self.dias:
            """ Se divide entre la cantidad de dias para penalizarlo por las pocas entradas que tuvo """
            acc = df["acc"].iloc[-1]
            if acc == 0:
                acc = np.inf
            else:
                acc = 1 / ( 1 + ((acc - 1) / self.dias) )
        
        elif len(df) > 0:
            """ Se divide entre la cantidad de dias por 10 para penalizarlo por las pocas entradas que tuvo """
            acc = df["acc"].iloc[-1]
            if acc == 0:
                acc = np.inf
            else:
                acc = 1 / ( 1 + ((acc - 1) / (self.dias*10)) )
        else:
            acc = np.inf
        
        # print(acc)
        return acc

    def _evaluate(self, x, out, *args, **kwargs):
        
        def rounding(i, x):
            return round(x)
            # if i not in [5, 9, 12]:
            #     return round(x)
            # else:
            #     return x

        x = [ rounding(i, xx) for i, xx in enumerate(x) ]

        f = [ self.my_obj_func( asset, x ) for asset in self.assets ]    
        out["F"] = [ np.min( f )] # change for support and resistance

def prep_asset( symbl ):
    asset = Asset(
        symbl,
        start = date(2023,3,1),
        end = date(2023,3,10),
        frequency="3min",
        fiat = "USDT",
        broker = "binance",
        source = "db"
    )

    return asset

@timing
def main(symbols, algorithm_name):

    print(f"------ {algorithm_name} ------")

    gen = 10

    assets = [ prep_asset(i) for i in symbols ]

    # initialize the thread pool and create the runner
    n_threads = 5
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = TATunning(
        assets = assets,         
        dias = 9, # de cuantos dias se tiene registro - 1                                       
        # n_var = 26,# 24,
        # xl = [0, 5, 2, 5, 2, 0, 5, 2, 2, 0, 4, 4, 0, 5, 40, 4, 0, 40, 0, 4, 2, 0, 0, 5, 2, 2],# [ 5,3, 35, 10, 2, -1, 7, 3, 2, -1,2 ,2 , -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10],
        # xu = [1, 36, 20, 36, 20, 1, 36, 30, 15, 1, 50, 50, 1, 36, 85, 30, 1, 85, 1, 50, 15, 1, 1, 36, 15, 20],# [ 28, 14, 90 ,120, 5, 1, 28, 14, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100]

        n_var = 58,# 24,
        # xl = [5, 5, 5, 1, 3, 0, 5, 2, 2, -10, 0, 5, 2, -10],
        # xu = [25, 50, 30, 150, 20, 1, 30, 15, 7, 10, 1, 50, 15, 10],

        xl = [0, 3, 2, 2, -10]*2 + [0, 3, 2, -10]*8 + [0, 3, 2, 3, -10],
        xu = [1, 30, 15, 8, 10]*2 + [1, 30, 15, 10]*8 + [1, 30, 15, 21, 10],
        elementwise_evaluation=True,
        elementwise_runner=runner,
    )

    if algorithm_name == "DE":
        algorithm = DE(
            pop_size=100,
            sampling= IntegerRandomSampling(), # LHS(),
            variant="DE/best/2/bin",
            CR=0.2,
            dither="vector",
            jitter=False
        )

    elif algorithm_name == "NSGA2":
        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
    
    elif algorithm_name == "GA":
        """ https://pymoo.org/customization/discrete.html """
        algorithm = GA(
            pop_size=100,
            eliminate_duplicates=True
        )
    
    elif algorithm_name == "CMAES":
        algorithm = CMAES(x0=np.random.random(problem.n_var))
    
    elif algorithm_name == "PSO":
        algorithm = PSO()

    termination = get_termination("n_gen", gen)

    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

    X = res.X
    F = res.F

    def extract(x):
        x = x[0]
        if isinstance( x, float ):
            return x
        
        return extract( x )

    best_acc = extract( F )

    print("This is acc ", 1/best_acc)
    
    X = X.tolist()

    if isinstance( X[0], list ):
        print(X[0])
    
    else:
        print(X)
    

    data = {
            "acc": 1 / best_acc,
            "param":X
        }
    

    path = f"results/metaheuristics/SupporResistance_RSIslope_EMAslope_{algorithm_name}_{gen}.json"

    if os.path.isfile(path):
        with open( path, "r" ) as fp:
            past_data = json.load( fp )
    else:
        with open( path, "w" ) as fp:
            json.dump( data, fp )

    if past_data["acc"] < data["acc"]:
        with open( path, "w" ) as fp:
            json.dump( data, fp )
    
    pool.close()


if __name__ == "__main__":

    symbols = np.random.choice( list(get_assets()["binance"].keys()) , 20 )

    for algorithm in ["DE"]: # Si "DE", "NSGA2", "GA", "PSO"  #No "CMAES",
        main( symbols, algorithm )
    