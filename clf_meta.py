""" Make a parameter tunning strategy, trygin to improve 
    a clasification model to determine whether a certain level 
    of percentage of return can be obtained.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

from trading.testers.rules_testing import RuleTesting
from trading import Asset

from functions import features

import warnings
warnings.filterwarnings("ignore")

def get_asset(**kwargs):

    asset = Asset(
        symbol = kwargs.get("symbol", "LTC"),
        start = datetime.now() - timedelta(days = 4),
        end = datetime.now(),
        frequency="3min",
        fiat = "USDT",
        broker = "binance",
        source = "ext_api"
    )

    return asset

def attributes(asset):

    for i in [10, 30, 90]:
        asset.df[ f"ema_{i}" ] = asset.ema_slope( i, int(i/10)  ).apply(lambda x: round(x, 4))
        asset.df[ f"sma_{i}" ] = asset.sma_slope( i, int(i/10)  ).apply(lambda x: round(x, 4))

    asset.df["mean"] = asset.ema(5)
    asset.df["resistance"], asset.df["support"] = asset.support_resistance(10, support = 'mean', resistance = "mean")
    asset.df["rel_sr"] = (asset.df["mean"] - asset.df["support"]) / asset.df["resistance"]

    for i in [7, 14, 21]:
        asset.df[f"rsi_{i}"] = asset.rsi(i)
        asset.df[f"rsi_{i}_std"] = asset.df[f"rsi_{i}"].rolling(10).std() 
        for k in [7, 10, 14]:
            asset.df[f"rsi_{i}_smoth_{k}"] = asset.df[f"rsi_{i}"].rolling(k).mean()
            for j in [3, 6, 9]:
                asset.df[f"rsi_{i}_smoth_{k}_slope_{j}"] = asset.df[f"rsi_{i}_smoth_{k}"].pct_change(j)



    return asset

def prep_target(asset, pct = 0.0015, leverage = 20, stop_loss = 0.5):
    df = asset.df.copy()

    real_stop_loss = (1/leverage)*stop_loss
    close = df["close"]
    df["target"] = False

    for index in df.index:
        fulfillment = False
        possible_close = close.loc[index:]
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

    

if __name__ == "__main__":
    asset = get_asset()
    asset = attributes(asset) # attributes(asset)
    df = prep_target(asset)
    
    df.drop(columns = ["open", "low", "high", "close"], inplace = True)
    df.dropna(inplace = True)
    
    df = balance_dataset(df)
    
    if df.empty:
        raise Exception("DF is empty")
    
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

    st = time.time()

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
    ttime_cv = time.time() - st
    print(ttime_cv)
