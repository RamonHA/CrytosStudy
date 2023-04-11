""" Make a parameter tunning strategy, trygin to improve 
    a clasification model to determine whether a certain level 
    of percentage of return can be obtained.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.feature_selection import RFECV

from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

from trading.testers.rules_testing import RuleTesting
from trading import Asset
from trading.func_aux import min_max

from functions import features

import warnings
warnings.filterwarnings("ignore")

def get_asset(**kwargs):

    asset = Asset(
        symbol = kwargs.get("symbol", "LTC"),
        start = datetime.now() - timedelta(days = 6),
        end = datetime.now(),
        frequency="3min",
        fiat = "USDT",
        broker = "binance",
        source = "ext_api"
    )

    return asset

def attributes(asset):
    """  
        Attributes based on feature_selection application
    """
    asset.df[ f"ema_90_18" ] = asset.ema_slope( 90, 18  ).apply(lambda x: round(x, 4))
    asset.df[ f"sma_90_18" ] = asset.sma_slope( 90, 18  ).apply(lambda x: round(x, 4))

    # for i in [10, 30, 90]:
    #     slopes = [1] + [ int(i/10), int(i/5) ]
    #     for s in slopes:
    #         asset.df[ f"ema_{i}_{s}" ] = asset.ema_slope( i, s  ).apply(lambda x: round(x, 4))
    #         asset.df[ f"sma_{i}_{s}" ] = asset.sma_slope( i, s  ).apply(lambda x: round(x, 4))

    asset.df["trend_res"] = asset.df["close"] - asset.ema(40)
    asset.df["season"] = asset.sma( 20, target = "trend_res" )
    # asset.df["season_res"] = asset.df["trend_res"] - asset.df["season"]

    seasonal = asset.df[["season"]].dropna()

    # sampling rate
    sr = len(seasonal)
    # sampling interval
    ts = 1/sr
    t = np.arange(0,1,ts)

    # r = round(seasonal["season"].std(), ndigits=2)
    r = seasonal["season"].std()
    
    reg = []
    for i in range(8, 30, 1):
        y = np.sin(np.pi*i*t) * r

        if len(y) != len(seasonal):
            continue

        seasonal["sin"] = y

        error  = np.linalg.norm( seasonal["season"] - seasonal["sin"] )

        reg.append([ i, error ])

    if len(reg) == 0:
        print(f"  symbol {asset.symbol} no reg")
        return False

    reg = pd.DataFrame(reg, columns = ["freq", "error"])
    i = reg[ reg[ "error" ] == reg["error"].min() ]["freq"].iloc[0]
    y = np.sin(np.pi*i*t)*r

    zeros = np.zeros(len(asset.df) - len(y))
    asset.df[ "sin" ] = zeros.tolist() + y.tolist()

    asset.df["mean"] = asset.ema(5)
    asset.df["resistance"], asset.df["support"] = asset.support_resistance(10, support = 'mean', resistance = "mean")
    # asset.df["rel_sr"] = (asset.df["mean"] - asset.df["support"]) / asset.df["resistance"]

    asset.df[ f"roc_21" ] = asset.roc(21)
    asset.df[f"rsi_14_smoth_14"] = asset.rsi(14).rolling(14).mean()

    # for i in [7, 14, 21]:
    #     asset.df[ f"roc_{i}" ] = asset.roc(i)
    #     asset.df[f"rsi_{i}"] = asset.rsi(i)
    #     asset.df[f"rsi_{i}_std"] = asset.df[f"rsi_{i}"].rolling(10).std() 
    #     for k in [7, 10, 14]:
    #         asset.df[f"rsi_{i}_smoth_{k}"] = asset.df[f"rsi_{i}"].rolling(k).mean()
    #         for j in [3, 6, 9]:
    #             asset.df[f"rsi_{i}_smoth_{k}_slope_{j}"] = asset.df[f"rsi_{i}_smoth_{k}"].pct_change(j)

    # asset.df["ho"] = (asset.df["high"] / asset.df["open"]) - 1
    # asset.df["hl"] = (asset.df["high"] / asset.df["low"]) - 1
    # asset.df["lo"] = (asset.df["low"] / asset.df["open"]) - 1
    # asset.df["cl"] = (asset.df["close"] / asset.df["low"]) - 1
    # asset.df["ch"] = (asset.df["close"] / asset.df["high"]) - 1

    asset.df["obv"] = asset.obv()

    return asset

def prep_target(asset, pct = 0.0015, leverage = 20, stop_loss = 0.5, window = 20):
    """  
        Fix prep asset to just consider a 20 period window in front of buy sell
    """
    df = asset.df.copy()

    real_stop_loss = (1/leverage)*stop_loss
    close = df["close"]
    df["target"] = False

    for index in df.index:
        fulfillment = False
        possible_close = close.loc[index:]

        if len(possible_close) > window:
            possible_close = possible_close.iloc[:window]

        price = close[index]
        sell_price = price * (1 + pct)
        stop_limit_price = price * ( 1 - real_stop_loss )

        sell_index = possible_close[ possible_close >= sell_price ]
        stop_limit_index = possible_close[ possible_close <= stop_limit_price ]

        if len(sell_index) == 0:
            fulfillment = False
        
        else:
            if len(stop_limit_index) == 0:
                fulfillment = True
            else:
                if sell_index.index[0] > stop_limit_index.index[0]: # if stop limit is first
                    fulfillment = False
                else:
                    fulfillment = True

        df.loc[ index, "target" ] = fulfillment

    return df

def shuffle_data(df):

    df.reset_index(drop = True, inplace = True)

    return shuffle(df)

def balance_dataset(df):
    qty_per_class = df['target'].value_counts().to_frame().sort_values(by = "target", ascending = True)
    
    class_true = df[df['target'] == 1]
    class_false = df[df['target'] == 0]

    if qty_per_class.index[0]:
        class_false = class_false.sample(qty_per_class["target"].iloc[0])
    else:
        class_true = class_true.sample(qty_per_class["target"].iloc[0])

    test_under = pd.concat([class_true, class_false], axis=0)

    return test_under

def feature_selection(df):

    selector = RFECV(RandomForestClassifier(), step=1, cv=5)
    X = df.drop(columns = ["target"])
    y = df[["target"]]
    selector = selector.fit(X, y)
    cols = [i for i in selector.support_*X.columns if i != "" ]
    print(cols)

def cross_validation(df):

    parameters = {
        "n_estimators": [50, 100, 150],
        "criterion":["gini", "entropy"]
    }

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv = round(1 / split_ratio))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    print(res_cv)

def simulation():
    for symbol in ["LTC", "ETH", "BTC"]:
        st = time.time()
        asset = get_asset(symbol = symbol)
        asset = attributes(asset) # attributes(asset)
        df = prep_target(asset)
        
        df.drop(columns = ["open", "low", "high", "close"], inplace = True)
        df.dropna(inplace = True)
        
        last_row = df.iloc[-1:]
        df = df.iloc[:-1]

        df = balance_dataset(df)
    
        if df.empty:
            raise Exception("DF is empty")
        
        print(f"{symbol}: {df.shape}")
    
        split_ratio = 30/len(df)
        split_ratio = 0.25 if split_ratio < 0.25 else split_ratio

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns = ["target"]), 
            df[["target"]], 
            test_size=split_ratio, 
            random_state=42
        )

        parameters = {
            "n_estimators": [50, 100, 150],
            "criterion":["gini", "entropy"]
        }

        clf = GridSearchCV(RandomForestClassifier(), parameters, cv = round(1 / split_ratio))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
        # res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
        print(f"Precision for {symbol} is {precision}")
        pred = clf.predict(last_row.drop(columns = ["target"]))
        print("Best params: ")
        print(clf.best_params_)

        print(f"Prediction for {symbol} is: ")
        print(pred)

        print(time.time() - st)
        print("\n")

def rfc(df):
    print(" -- RANDOM FOREST CLASSIFIER --")
    split_ratio = 30/len(df)
    split_ratio = 0.25 if split_ratio < 0.25 else split_ratio

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns = ["target"]), 
        df[["target"]], 
        test_size=split_ratio, 
        random_state=42
    )
    st = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    
    print(res)
    ttime = time.time() - st
    print(ttime)

    # st = time.time()

    # parameters = {
    #     "n_estimators": [50, 100, 150],
    #     "criterion":["gini", "entropy"]
    # }

    # clf = GridSearchCV(RandomForestClassifier(), parameters, cv = round(1 / split_ratio))
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    # res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    # print(res_cv)
    # ttime_cv = time.time() - st
    # print(ttime_cv)

def svc(df):
    print("\n -- SUPPORT VECTOR MACHINE --")
    split_ratio = 30/len(df)
    split_ratio = 0.25 if split_ratio < 0.25 else split_ratio

    df = min_max(df, exception=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns = ["target"]), 
        df[["target"]], 
        test_size=split_ratio, 
        random_state=42
    )

    st = time.time()
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    
    print(res)
    ttime = time.time() - st
    print(ttime)

    st = time.time()

    parameters = {
        "C": [0.5, 1, 1.5],
        "kernel":["linear", "poly", "rbf", "sigmoid"]
    }

    clf = GridSearchCV(SVC(), parameters, cv = round(1 / split_ratio))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    print(res_cv)
    ttime_cv = time.time() - st
    print(ttime_cv)

def knn(df):
    print("\n -- KNEAREST KNEIGHBORS --")
    split_ratio = 30/len(df)
    split_ratio = 0.25 if split_ratio < 0.25 else split_ratio

    df = min_max(df, exception=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns = ["target"]), 
        df[["target"]], 
        test_size=split_ratio, 
        random_state=42
    )

    st = time.time()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    
    print(res)
    ttime = time.time() - st
    print(ttime)

    st = time.time()

    parameters = {
        "n_neighbors": [3,5,7],
        "weights":["uniform", "distance"]
    }

    clf = GridSearchCV(KNeighborsClassifier(), parameters, cv = round(1 / split_ratio))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    print(res_cv)
    ttime_cv = time.time() - st
    print(ttime_cv)

def naive(df):
    print("\n -- NAIVE BAYES --")
    split_ratio = 30/len(df)
    split_ratio = 0.25 if split_ratio < 0.25 else split_ratio

    df = min_max(df, exception=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns = ["target"]), 
        df[["target"]], 
        test_size=split_ratio, 
        random_state=42
    )

    st = time.time()
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    res = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    
    print(res)
    ttime = time.time() - st
    print(ttime)

    # st = time.time()

    # parameters = {
    #     "n_estimators": [50, 100, 150],
    #     "criterion":["gini", "entropy"]
    # }

    # clf = GridSearchCV(RandomForestClassifier(), parameters, cv = round(1 / split_ratio))
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    # res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    # print(res_cv)
    # ttime_cv = time.time() - st
    # print(ttime_cv)

if __name__ == "__main__":
    asset = get_asset()
    asset = attributes(asset) # attributes(asset)
    df = prep_target(asset)
    
    df.drop(columns = ["open", "low", "high", "close"], inplace = True)
    df.dropna(inplace = True)

    df = shuffle_data(df)
    
    df = balance_dataset(df)
    
    if df.empty:
        raise Exception("DF is empty")
    
    rfc(df.copy())
    svc(df.copy())
    knn(df.copy())
    naive(df.copy()) # Worst of them all



    