import warnings
warnings.filterwarnings("ignore")

from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.termination import get_termination

from pymoo.optimize import minimize

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS

import numpy as np
from copy import deepcopy
from datetime import date, datetime
import pandas as pd
import json

from trading.testers.rules_testing import rule_validation
from trading import Asset
from trading.func_brokers import get_assets

np.random.seed(1)

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

class TATunning(ElementwiseProblem):

    def __init__(self, assets, **kwargs):
        self.assets = assets
        super().__init__(n_var=kwargs["n_var"],
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=kwargs["xl"], # np.array([7,3, 40]),
                         xu=kwargs["xu"], #np.array([21,14, 90])
        )

    def my_obj_func(self, asset, x):
        """ 
            Rule selection:
                1:  
        """
        asset = deepcopy(asset)

        cols_to_use = []

        if x[13]:
            cols_to_use.append("buy_wf")
            asset.df["buy_wf"] = asset.william_fractals(3, shift=True)
        
        if x[14]:
            cols_to_use.append("sell_wf")
            asset.df["sell_wf"] = asset.william_fractals(3, shift=True, order = "sell").rolling(3).sum() == 0
        
        if x[15]:
            cols_to_use.append("rsi")
            asset.df["rsi"] = asset.rsi_smoth(x[0], x[1]) < x[2]
        
        if x[16]:
            cols_to_use.append("ema_slope")
            asset.df["ema_slope"] = asset.ema_slope( x[3], x[4] ) > x[5]
        
        if x[17]:
            cols_to_use.append("rsi_slope")
            asset.df["rsi_slope"] = asset.rsi_smoth_slope( x[6], x[7], x[8] ) > x[9]
        
        if x[18]:
            cols_to_use.append("oneside_gaussian_filter_slope")
            asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(x[10],x[11]) > x[12]
        
        if x[19]:
            cols_to_use.append("engulfing_buy")
            asset.df["engulfing_buy"] = asset.engulfing() == 1
        
        if x[20]:
            cols_to_use.append("engulfing_sell")
            asset.df["engulfing_sell"] = asset.engulfing() != -1

        if x[21]:
            cols_to_use.append("dema_sma")
            asset.df["dema_sma"] = asset.dema( x[22] ) > asset.sma( x[23] )

        if len(cols_to_use) == 0:
            return np.inf

        df = asset.df[ cols_to_use ]

        asset.df[ "buy" ] = df.all(axis = 1)

        asset = sell_column(asset, target = 0.002)
        
        df = rule_validation(asset)
        
        if len(df) > 0:
            acc = df["acc"].iloc[-1]
            if acc == 0:
                acc = np.inf
            else:
                acc = 1 / acc
        else:
            acc = np.inf
        
        # print(acc)
        return acc

    def _evaluate(self, x, out, *args, **kwargs):
        
        def rounding(i, x):
            if i not in [5, 9]:
                return round(x)
            else:
                return x

        x = [ rounding(i, xx) for i, xx in enumerate(x) ]

        f = [ self.my_obj_func( asset, x ) for asset in self.assets ]    
        out["F"] = [ np.median( f )]

def prep_asset( symbl ):
    asset = Asset(
        symbl,
        start = date(2023,1,22),
        end = date(2023,1,24),
        frequency="3min",
        fiat = "USDT",
        broker = "binance",
        from_ = "db"
    )

    return asset

def main(symbols, algorithm):

    print(f"------ {algorithm} ------")

    gen = 120

    assets = [ prep_asset(i) for i in symbols ]

    problem = TATunning(
        assets = assets,                                                         
        n_var = 21,
        xl = [ 5,3, 35, 10, 2, -1, 7, 3, 2, -1,2 ,2 , -1, 0, 0, 0, 0, 0, 0, 0, 0],
        xu = [ 28, 14, 90 ,120, 5, 1, 28, 14, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    if algorithm == "DE":
        algorithm = DE(
            pop_size=100,
            sampling=LHS(),
            variant="DE/best/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )

    elif algorithm == "NSGA2":
        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
    
    elif algorithm == "GA":
        algorithm = GA(
            pop_size=100,
            eliminate_duplicates=True
        )
    
    elif algorithm == "CMAES":
        algorithm = CMAES(x0=np.random.random(problem.n_var))
    
    elif algorithm == "PSO":
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

    with open( f"{algorithm}_{gen}.json", "w" ) as fp:
        json.dump( data, fp )
        

if __name__ == "__main__":

    symbols = np.random.choice( list(get_assets()["binance"].keys()) , 20 )

    for algorithm in ["PSO"]: # Si "DE", "NSGA2", "GA",  #No "CMAES",
        main( symbols, algorithm )
    