# Simulate but without using the Simulation class as that can be the reason for it to be innefficient

import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import json

from trading.func_brokers import get_assets
from trading import Asset
from trading.func_aux import timing
from trading.optimization import Optimization


from metaheuristics import *

def get_asset(symbol):
    asset = Asset(
        symbol = symbol,
        fiat = "USDT",
        broker = "binance",
        frequency = "1d",
        start = date(2019,1,1),
        end = date(2023,5,1),
        source="db"
    )

    if asset.df is None or len(asset.df) < 10:
        return None
    
    asset.df.index = pd.to_datetime( asset.df.index )

    # asset.df = asset.df[ asset.df.index.map(lambda x : x.day == 1) ]

    # Insert macros

    return asset

@timing
def main():
    gen = 15

    
    predictions = { 
        symbol:metaheuristic_integer_batch( 
            inst = get_asset(symbol), 
            gen = gen, 
            verbose = False ,
            train_size=0.8,
            for_real=False
        ) for symbol in get_assets()["binance"] 
        }

    with open( f"results/batch/DE_{gen}.json", "w" ) as fp:
        json.dump( predictions, fp )

def read_from_batch(gen):

    with open( f"results/batch/DE_{gen}.json", "r" ) as fp:
        data = json.load(fp)

    df = pd.DataFrame.from_dict(data, orient = "index")
    cols = len(df.columns)
    cols = [ date(2023,5,1) - relativedelta(days=d) for d in range(cols) ]
    df.columns = cols    
    df = df.T
    df.sort_index(inplace = True)

    with_nan = df.isna().all(axis = 0)
    with_nan = with_nan[with_nan].index
    df.drop(columns = with_nan, inplace = True)

    with_nan = df.isna().sum(axis = 1)
    with_nan = with_nan[ with_nan > len(df.columns) // 2 ].index
    df.drop(index = with_nan, inplace = True)    

    return df 


def optimize(df):
    # Ploteando contra ETH nos percatamos que 

    extra_opt = Optimization(
            assets = df.columns.to_list(),
            start = df.index[-1] - relativedelta(days = 300),
            end = df.index[-1],
            frequency="1d",
            exp_returns = "",
            risk = "efficientsemivariance",
            objective="efficientreturn",
            broker = "binance",
            fiat = "USDT",
            source = "db",
            verbose = 2
        )   
    
    extra_opt.df.index = pd.to_datetime( extra_opt.df.index )

    value = 2000
    data = []

    for end_date in df.index:

        start_date = end_date - relativedelta(days = 60)

        sel_assets = df.loc[end_date].dropna()
        sel_assets = sel_assets[ sel_assets > 0 ]

        opt = Optimization(
            assets = sel_assets.index.to_list(),
            start = start_date,
            end = end_date,
            frequency="1d",
            exp_returns = sel_assets,
            risk = "1/n",
            objective="MaxRet",
            broker = "binance",
            fiat = "USDT",
            source = "db",
            verbose = 0
        )   

        allocation, qty, pct = opt.optimize( 
            value, 
            time = 1, 
            limits = (0,1) 
        )

        if allocation is None:
            print(end_date)
            continue

        aux_df = extra_opt.df.loc[ end_date: ]

        if len(aux_df) < 2:
            print("The End")
        else:
            real_prices = aux_df.pct_change().iloc[1]
            real_prices = (pd.Series(pct) * real_prices).dropna()
            total_return = real_prices.sum()

            data.append([
                value, total_return
            ])

            value *= (1 + total_return)

    data = pd.DataFrame(data, columns = ["value", "return"])


    return data

def optimize_manual(df):



    import riskfolio 
    from riskfolio.AuxFunctions import weights_discretizetion    

    opt = Optimization(
            assets = df.columns.to_list(),
            start = df.index[-1] - relativedelta(days = 300),
            end = df.index[-1],
            frequency="1d",
            exp_returns = "",
            risk = "efficientsemivariance",
            objective="efficientreturn",
            broker = "binance",
            fiat = "USDT",
            source = "db",
            verbose = 2
        )   
    
    opt.df.index = pd.to_datetime( opt.df.index )
    
    for end_date in df.index:

        start_date = end_date - relativedelta(days = 90)

        sel_assets = df.loc[end_date].dropna()
        sel_assets = sel_assets[ sel_assets > 0 ]

        aux_df = opt.df.loc[start_date:end_date][ sel_assets.index.to_list() ]

        latest_price = aux_df.iloc[-1]

        port = riskfolio.Portfolio(returns = aux_df.pct_change().dropna())

        port.assets_stats()

        port.mu = sel_assets
        
        hist = True # Use historical scenarios for risk measures that depend on scenarios
        rf = 0 # Risk free rate

        l = 2 # Risk aversion factor, only useful when obj is 'Utility'
                # Es el factor de cuanto un inversionista es capaz de aceptar riesgo

        w = port.optimization(
            model="Classic", 
            rm="MSV", 
            obj="Sharpe", 
            rf=rf, 
            l=l, 
            hist=hist
        )

        if w is None:
            raise Exception( "The problem doesn't have a solution with actual input parameters" )

        allocation = weights_discretizetion(w, latest_price, 2000)

        allocation = { i:v for i,v in allocation[0].to_dict().items() if v > 0 }

        qty = { i:( v*latest_price[i] ) for i,v in allocation.items() }

        total_money = sum(qty.values())
        pct = { i:(v/total_money) for i,v in qty.items() }

        aux_df = opt.df.loc[ end_date: ]

        if len(aux_df) < 2:
            print("The End")
        else:
            real_prices = aux_df.pct_change().iloc[1]
            real_prices = (pd.Series(pct) * real_prices).dropna()

            total_return = real_prices.sum()

        break

if __name__ == "__main__":
    # main()
    pass