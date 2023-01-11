import warnings
warnings.filterwarnings("ignore")

from datetime import date
from trading.features import pct
from trading.processes import Simulation
from trading.strategy import Strategy
import collections
from trading.grid_search.brute_force import BruteGridSearch
from trading.variables.params_grid import RF_C_GRID, SVM_R_GRID
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from copy import deepcopy, copy
import matplotlib.pyplot as plt


def prediction_split(df):
    train = df.iloc[:-1]
    test = df.iloc[-1:]
    return train, test

def engulfing(asset):
    en = asset.engulfing()

    return True if en.iloc[-1] == -1 else False

def william_frac(asset):

    sell, _ = asset.william_fractals(3)

    return sell.iloc[-3]

def william_frac_buy(asset):
    _, buy = asset.william_fractals(3)

    return buy.iloc[-3]

def distsim(asset):
    st = Strategy( asset=asset, tas = "trend_oneparam" )

    st.TREND_FIRST_PARAM = ( 3, 30, 3 )

    r = st.value( target=[ 1 ], verbose = 0 )

    r = r[ r["result"] < 0.05 ]

    if r.empty:
        return None
    
    r.sort_values( by = "result", ascending=True , inplace = True)

    l = 10 if len(r) > 10 else int(len(r) / 2)

    r = r.iloc[ :10 ]

    v = [ ]
    last_v = st.asset.df.iloc[-1].to_dict()

    for i, row in r.iterrows():
        v.append( row["range_down"][0] < last_v[ row["col"] ] < row["range_down"][1] )
    
    counter = collections.Counter(v)

    return (counter[False] > counter[True]) and (counter[False] > len( v ) // 2)

def rfc(asset):
    ori_cols = asset.df.columns

    asset.df["rsi"] = asset.rsi(14)

    df_aux = pct( asset.df, cols = ["close", "volume", "rsi"], lags = 7 , shift=True)

    df_aux["target"] = df_aux["close"].pct_change(1).shift(-1).apply(lambda x: 1 if x > 0 else 0)

    df_aux.drop(columns = ori_cols, inplace = True)

    # df_aux.dropna(axis=0, inplace = True)

    df_aux = df_aux.round(3)

    params_grid = RF_C_GRID

    params_grid["n_estimators"] = range( 10, 300, 30 )
    # params_grid["max_features"] = ["sqrt", "log2", 3,5,7, 0.5, 0.25]
    # params_grid["max_depth"] = [3,4, None]

    bgs = BruteGridSearch(
        df = df_aux,
        regr = RandomForestClassifier(),
        parameters = params_grid,
        error = "precision",
        error_ascending=False,
        train_test_split=0.9
    )

    bgs.test(pos_label = 1)

    r = bgs.predict( one = True )

    return True if r == 1 else False

def normalize(df, cols):

    for col in cols:
        df[col] = ( df[col] - df[col].min() ) / ( df[col].max() - df[col].min() )

    return df

def svm(asset):

    ori_cols = asset.df.columns
    
    asset.df["rsi"] = asset.rsi(14)
    asset.df["cci"] = asset.cci(14)
    # asset.df["vpt"] = asset.vpt()
    # asset.df["support"], asset.df["resistance"] = asset.support_resistance( 14 )

    asset.df["hl"] = asset.df["high"] - asset.df["low"]
    asset.df["ho"] = asset.df["high"] - asset.df["open"]
    asset.df["lo"] = asset.df["low"] - asset.df["open"]
    asset.df["cl"] = asset.df["close"] - asset.df["low"]
    asset.df["ch"] = asset.df["close"] - asset.df["high"]

    for i in [7, 14, 21]:
        for c in ["close", "high", "volume"]:
            asset.df["std{}_{}".format(c, i)] = asset.df[c].rolling(i).std()

    n_cols = list( set(asset.df.columns) - set(ori_cols) )

    df_aux = pct( asset.df, cols = ["close", "volume", "rsi", "cci"], lags = 7 , shift=True)

    df_aux = normalize(df_aux, cols = n_cols)

    df_aux["target"] = df_aux["close"].pct_change(1).shift(-1).apply(lambda x: 1 if x > 0 else 0)

    df_aux.drop(columns = ori_cols, inplace = True)

    # df_aux.dropna(axis=0, inplace = True)

    bgs = BruteGridSearch(
        df = df_aux,
        regr = SVR(),
        parameters = SVM_R_GRID,
        error = "rmse",
        error_ascending=True,
        train_test_split=0.9
    )

    bgs.test()

    r = bgs.predict( one = True )

    return r

def slope_sort(asset):
    return asset.sma(8).iloc[-1]

def func_1h(asset):

    asset.df["sma"] = asset.sma(23).pct_change(2)
    if asset.df["sma"].iloc[-1] < 0: return False

    asset.df["ema"] = asset.ema(23).pct_change(2)
    if asset.df["ema"].iloc[-1] < 0: return False

    asset.df = asset.transform( asset.df, "12h" )

    if not distsim(deepcopy(asset)): return False

    return slope_sort(deepcopy(asset))

def func_6h(asset):
    if engulfing(deepcopy(asset)): return False
    if william_frac(deepcopy(asset)): return False
    return True

def func_1d(asset):
    
    if not rfc(deepcopy(asset)): return None
    return svm(deepcopy(asset))

if __name__ == "__main__":

    sim = Simulation(
        broker = "binance",
        fiat = "usdt",
        commission = 0.001,
        end = date(2022,10,1),
        simulations=60,
        parallel=True,
        verbose = 1
    )

    sim.analyze(
        frequency="1d",
        test_time=1,
        run = False,
        cpus = 4,
        analysis={
            # "WilliamFractalSelector":{
            #     "function":william_frac_buy,
            #     "time":150,
            #     "type":"filter"
            # }
            "SMA_EMA_DistSim_SMASlopeSort":{
                "function":func_1d,
                "time":150,
                "type":"filter",
                "frequency":"1h",
                "filter":"highest",
                "filter_qty":0.3
            },
            "Engulfing_WilliamFrac":{
                "function":func_6h,
                "time":150,
                "type":"filter",
                "frequency":"6h",
            },
            "RFclfHalving_SVM":{
                "function":func_1d,
                "time":200,
                "type":"prediction"
            }
        }
    )

    d = [
        ("efficientfrontier", "minvol"),
        ("efficientsemivariance", "minsemivariance"),
        ("efficientcvar", "mincvar"),
        ("efficientcdar", "mincdar")
    ]

    for r, o in d:#, "mad", "msv", "flpm", "slpm", "cvar", "evar", "wr", "mdd", "add", "cdar", "edar", "uci" ]:
        
        # for t in [0.005, 0.01, 0.02]:
        sim.optimize(
            balance_time=33,
            exp_return=True,
            # value = 1,
            risk = r,
            objective = o,
            # target_return = t,
            run = False,
            filter = "positive",
            # filter_qty = 10
        )

    results = sim.results_compilation()

    df = sim.behaviour( results.loc[ 0, "route" ] )

    df[ "acc" ].plot()
    plt.show()

