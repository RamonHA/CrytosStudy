import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, ParameterGrid

from trading import Asset
from trading.assets.binance import Binance
from trading.func_aux import timing

import pandas as pd
import math
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

def get_asset( symbol , period = 2000, end = datetime.now()):

    return Asset(
        symbol,
        broker = "binance",
        fiat = "USDT",

        start = end - relativedelta(minutes=period*5),
        end =  end,

        frequency = "5min",

        source = "ext_api"
    )

def features_extraction(asset, train_size = 0.8, shuffle = True):
    x = asset.df.drop(columns = ["target"]).to_numpy()
    y = asset.df["target"].to_numpy().reshape(-1, 1)

    if train_size != 1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=42, shuffle=shuffle)
        # train_size = int( len(x) * train_size )
        # x_train, x_test = x[ :train_size ], x[ train_size: ]
        # y_train, y_test = y[ :train_size ], y[ train_size: ]
    else:
        x_train = x[ :-1 ]
        y_train = y[:-1]
        x_test, y_test = [], []

    return x_train, y_train, x_test, y_test

def balance_dataset_sequiential(df):
    return df[ df[ "target" ] != df["target"].shift(-1) ]

def balance_dataset_randomly(df):
    qty_per_class = df['target'].value_counts().to_frame().sort_values(by = "target", ascending = True)
    
    class_true = df[df['target'] == 1]
    class_false = df[df['target'] == 0]

    if qty_per_class.index[0]:
        class_false = class_false.sample(qty_per_class["target"].iloc[0])
    else:
        class_true = class_true.sample(qty_per_class["target"].iloc[0])

    test_under = pd.concat([class_true, class_false], axis=0)

    return test_under

def prep_target(asset, pct = 0.0005, leverage = 20, stop_loss = 0.5, window = 30):
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

def normalize(df, cols):

    for col in cols:
        df[col] = ( df[col] - df[col].min() ) / ( df[col].max() - df[col].min() )

    return df

def features(asset, clf = True, drop = True, shift = True, target = True):
 
    ori_cols = asset.df.drop(columns = ["volume"]).columns

    for i in range( 1, 6 ):
        asset.df[ f"shift_{i}" ] = asset.df["close"].pct_change( 1 ).shift( i )
        asset.df[f"close_{i}"] = asset.df["close"].pct_change( i )

    for i in [20, 40, 60, 80]:
        asset.df[ f"ema_{i}"] = asset.ema(i)
        asset.df[ f"roc_{i}" ] = asset.roc(i)

        for j in range(2, 12, 3):
            asset.df[ f"ema_{i}_slope_{j}" ] = asset.df[ f"ema_{i}" ].pct_change( j ) 
        
        for c in ["close", "high", "volume"]:
            asset.df["std{}_{}".format(c, i)] = asset.df[c].rolling(i).std()

    for i in [7, 14, 21]:
        asset.df[ f"rsi_{i}"] = asset.rsi_smoth(i, 2)
        
        for j in range(2,7, 2):
            asset.df[ f"rsi_{i}_slope_{j}" ] = asset.df[ f"rsi_{i}" ].pct_change( j )
    
    for i in [2,3,4,5,6]:
        asset.df[f"momentum_{i}"] = asset.momentum(i)
        asset.df[f"momentum_ema_{i}"] = asset.momentum(i, target = "ema_20")
        asset.df[f"momentum_rsi_{i}"] = asset.momentum(i, target = "rsi_7")

    asset.df["hl"] = asset.df["high"] - asset.df["low"]
    asset.df["ho"] = asset.df["high"] - asset.df["open"]
    asset.df["lo"] = asset.df["low"] - asset.df["open"]
    asset.df["cl"] = asset.df["close"] - asset.df["low"]
    asset.df["ch"] = asset.df["close"] - asset.df["high"]

    asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
    for i in [2,3,4]:
        for j in [2,3,4]:
            asset.df[f"oneside_gaussian_filter_slope_{i}_{j}"] = asset.oneside_gaussian_filter_slope(i,j)

    asset.df["obv"] = asset.obv()

    for i in [20, 40, 60]:
        s, r = asset.support_resistance(i)
        asset.df[ f"support_{i}" ] = ( s / asset.df["close"] ) - 1
        asset.df[ f"resistance_{i}" ] = ( r / asset.df["close"] ) - 1

    # Normalization
    n_cols = list( set(asset.df.columns) - set(ori_cols) )
    
    asset.df = normalize(asset.df, cols = n_cols)

    asset.df["engulfing"] = asset.engulfing()
    asset.df["william_buy"] = asset.william_fractals(2, order = "buy").apply(lambda x : 1 if x == True else 0).rolling(5).sum()
    asset.df["william_sell"] = asset.william_fractals(2, order = "sell").apply(lambda x : 1 if x == True else 0).rolling(5).sum()

    if target:
        if clf:
            asset.df["target"] = asset.df["close"].pct_change().shift(-1 if shift else 0).apply(lambda x: 1 if x > 0 else 0)
        else:
            asset.df["target"] = asset.df["close"].pct_change().shift(-1 if shift else 0)
    else:
        ori_cols = list( set(ori_cols) - set(["target"]) )

    if drop:
        asset.df.drop(columns = ori_cols, inplace = True)

    return asset

def pipeline(symbol, reg, error = "precision", period = 3000, balance_type = "sequential", shuffle = False, train_size = 0.8):

    asset = get_asset( symbol, period=period, end=datetime.now() )

    if asset.df is None or len(asset.df) == 0:
        print("No info from source")
        return None
    
    asset.df = prep_target( asset, pct = 0.0005, window=6 )

    asset = features( asset, clf=False, drop = True , target=False)

    if train_size == 1:
        validation = asset.df.iloc[-1:].drop(columns = ["target"])

        if validation.isna().any().any():
            print(f"{symbol} has NA in validation set")
            return None
        
        validation = validation.to_numpy().astype('float32')

    asset.df = asset.df.replace( [np.inf, -np.inf], np.nan ).dropna()

    if len(asset.df) == 0:
        print("Remove all nas")
        return None

    asset.df = {
        "random":balance_dataset_randomly,
        "sequential":balance_dataset_sequiential
    }[ balance_type ](asset.df)

    x_train, y_train, x_test, y_test = features_extraction(asset, shuffle = shuffle, train_size=train_size)

    reg.fit( x_train, y_train )

    # if train_size != 1:
    #     pred = reg.predict(x_test)
    # else:
    #     pred = reg.predict( validation )

    # if isinstance(error, str):
    #     error = {
    #         "precision":precision_score
    #     }[error]( y_test, pred )
    
    # return error, pred

    reg.save( filepath = f"results/ann/{symbol}" )
    
class ANN():
    def __init__(
            self, 
            epochs,
            batch_size,
            n_hidden,
            output_classes = 2,
            validation_split = 0.1,
            verbose = 0, 
            **kwargs
        ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_hidden = n_hidden
        self.validation_split = validation_split

        for i, v in kwargs.items(): setattr( self, i, v )

    def neural_model(self):
        self.model = tf.keras.models.Sequential()

        self.model.add(
            keras.layers.Dense(
                self.n_hidden,
                input_shape = (self.features,),
                name = "Dense-Layer-1",
                activation="relu"
            )
        )

        self.model.add(
            keras.layers.Dense(
                self.n_hidden,
                name = "Dense-Layer-2",
                activation="relu"
            )
        )

        # Softmax layer for categorical prediction
        self.model.add(
            keras.layers.Dense(
                1,
                name = "Final",
                activation="sigmoid"
            )
        )

        self.model.compile(
            loss = "binary_crossentropy",
            metrics = [tf.keras.metrics.Precision(thresholds=0.5)] # ["accuracy"]#
        )

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.instances, self.features = X.shape

        self.neural_model()

        self.history = self.model.fit(
            self.X,
            self.y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            # validation_split=self.validation_split
        )

    def predict(self, X):

        pred = self.model.predict(X)

        return [ round(i[0]) for i in pred ]

    def save(self, filepath):
        self.model.save( filepath= filepath)

@timing
def main():
    bi = Binance()

    futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

    trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]
    bad = ["USDCUSDT"]

    trading_pairs = [  t[:-4] for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

    trading_pairs = trading_pairs[2:10] 
    print(trading_pairs)

    with open("trading_pairs.txt", "w") as output:
        output.write(str(trading_pairs))

    for symbol in trading_pairs:
        print(symbol)
        pipeline( 
            symbol=symbol,
            period=4000,
            reg = ANN( epochs=300, batch_size=8, n_hidden=32 ) ,
            train_size=1,   
            error = None
        )

if __name__ == "__main__":
    main()