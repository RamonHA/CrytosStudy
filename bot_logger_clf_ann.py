# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import tensorflow as tf

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import multiprocessing as mp
from copy import copy
import numpy as np

import numpy, scipy.optimize

import socket
from urllib3.exceptions import NewConnectionError, MaxRetryError, ConnectionError

import logging
from pathlib import Path

from registro import futures

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 5
PCT = 1.0005
SHARE = .05
LEVERAGE = 20
STOP_LIMIT_PCT = 0.5

BOT_COUNTER = 0


bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]
bad = ["USDCUSDT"]

trading_pairs = [ (t[:-4],) for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

trading_pairs = trading_pairs[:10] 

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

def analyze_single(symbol, scale = 0.8, mode = "optimize", reg = "poly", forecasting = True):
    asset = Asset(
            symbol=symbol,
            fiat = "USDT",
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*100 ),
            source = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    try:
        model = tf.keras.models.load_model(f"results/ann/{symbol}")
    except Exception as e:
        print( f"Exception with {symbol}. Exception: \n{e}" )
        return None

    asset = features( asset, clf=False, drop = True , target = False)

    validation = asset.df.iloc[-1:]# .drop(columns = ["target"])

    if validation.isna().any().any():
        print(f"{symbol} has NA in validation set")
        return None

    validation = validation.to_numpy().astype('float32')

    pred = model.predict( validation )

    return pred[0]

# @timing
def analyze():
    print("Analyze")
    
    print(f"Trading pairs: {len(trading_pairs)}")

    with mp.Pool( mp.cpu_count() ) as pool:
        assets = pool.starmap(
            analyze_single,
            trading_pairs
        )
    
    assets = [ [symbol[0], r] for symbol ,r in zip( trading_pairs, assets) if r is not None ]

    df = pd.DataFrame(assets, columns = ["symbol", "prediction"])
    df.sort_values(by = "prediction", ascending=False, inplace = True)

    df = df[ ( df["prediction"] > 0 )] # (~df["pred"]) &

    if len(df) == 0:
        return []
    
    return df["symbol"].values[:5]

# @timing
def set_orders(symbol):
    logging.info(f"Set order for: {symbol}")

    bi = Binance(symbol="")

    symbol = "{}USDT".format(symbol)
    pct = PCT
    share = SHARE
    leverage = LEVERAGE

    max_leverage = [i for i in bi.client.futures_leverage_bracket() if symbol in i["symbol"]][0]["brackets"][0]["initialLeverage"]
    leverage = leverage if max_leverage >= leverage else max_leverage
    logging.info(f"Set order leverage: {leverage}")

    balance = float([ i["balance"] for i in bi.client.futures_account_balance() if i["asset"] == "USDT"][0])
    price = bi.client.futures_symbol_ticker(symbol = symbol).get("price", False)
    
    if not price:
        print(f"Algo malo con el symbolo: {symbol}")
        return None

    price = float(price)

    ticker_info = bi.client.get_symbol_info(symbol)

    qty_rouding = len(str(float([i["stepSize"] for i in ticker_info["filters"] if i["filterType"] == "LOT_SIZE"][0])).split(".")[-1])
    qty = balance*leverage*share / price

    try:
        bi.client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print(f"Exception : {e}")
        print(e.__dict__)
        print(max_leverage)
        return None

    try:
        bi.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
    except Exception as e:
        print(f"Exception : {e}")
        print(e.__dict__)

    def set_buy_order(symbol, qty, qty_rouding):

        try:
            
            qty = round( qty, qty_rouding   )

            orderBuy = bi.client.futures_create_order(
                symbol = symbol,
                type = "MARKET",
                # timeInForce ="GTC",
                side = "BUY",
                quantity = qty,
            )
        except Exception as e:
            if e.code == -1111:
                print("Redo buy order")
                print(f"Quantity rounding: {qty_rouding}", end = "\t")
                qty_rouding -= 1
                print(f"New Quantity rounding: {qty_rouding}")
                if qty_rouding < 0:
                    return None
                return set_buy_order(symbol, qty, qty_rouding)
            else:
                print( f"No order for {symbol}. Exception: {e}")
                print(type(e), e, e.__dict__)
                if hasattr(e, "code"):
                    print(e.code)
                return None

        return orderBuy

    orderBuy = set_buy_order(symbol, qty, qty_rouding)
    if orderBuy is None:
        print(f"Error with buy order for {symbol} due rounding")
        return None

    logging.info( orderBuy )
    
    # Check buy order

    print("Buy order done!")
    futures(balance)
    time.sleep(3)

    df_trades = pd.DataFrame(bi.client.futures_account_trades())
    df_trades = df_trades[ df_trades["orderId"] == orderBuy["orderId"] ]

    if len(df_trades) == 0:
        print(f"No orders with id { orderBuy['orderId'] }")
        return None

    df_trades["qty"] = df_trades["qty"].astype(float)
    df_trades["price"] = df_trades["price"].astype(float)

    qty = df_trades["qty"].iloc[-1]
    real_price_bought = df_trades["price"].iloc[-1]

    if len(df_trades) > 1:
       real_price_bought = df_trades["price"].max() 
       qty = df_trades["qty"].sum()
    
    price_rounding = len(str(real_price_bought).split(".")[-1])
    price_sell = real_price_bought*pct

    def set_sell_order(symbol, price_sell, qty, price_rounding):
        try:
            price_sell = round(price_sell, price_rounding)
            # stop_price = round( stop_price, price_rounding )

            orderSell = bi.client.futures_create_order(
                    symbol = symbol,
                    type = "LIMIT", 
                    timeInForce ="GTC",
                    side = "SELL",
                    price = price_sell, 
                    quantity = qty,
                    # stopPrice = stop_price ,
                )
                
        except Exception as e:
            if e.code == -1111:
                print("Redo sell order")
                price_rounding -= 1
                if price_rounding < 0:
                    return None
                return set_sell_order(symbol, price_sell, qty, price_rounding)
            
            else:
                print(type(e), e, e.__dict__)
                return None

        return orderSell

    orderSell = set_sell_order(symbol, price_sell, qty, price_rounding)
    if orderSell is None:
        print(f"Error with sell order for {symbol} due rounding")
        return None
    print("Sell order done!")
    logging.info( orderSell )

    # Check sell order

    orderSell["leverage"] = leverage
    orderSell["balance"] = balance

    print(orderSell)

    return orderSell

def check_market():

    asset = Asset(
            symbol="DOGE",
            fiat = "USDT",
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*150 ),
            from_ = "ext_api",
            broker="binance"
        )

    return asset

def wait(orderSell):
    
    for i in range(4):
        bi = Binance(symbol="")

        try:
            df_trades = pd.DataFrame(bi.client.futures_account_trades(  ))
            break
        except (socket.gaierror, NewConnectionError, MaxRetryError, ConnectionError) as e:
            # print(e, e.__dict__)
            # raise Exception(e)
            print( f"Exception error, iteration {i}")

        if i == 3:
            raise Exception(e)

    df_trades = df_trades[ df_trades["orderId"] == orderSell["orderId"] ]

    if len(df_trades) > 0:
        return True

    return False

def main():
    symbols = analyze()
    
    if len(symbols) == 0:
        print("No order is going to be sent")
        return 0

    symbol = symbols[0]

    bad_symbols = ["SC", "RAY"]
    def checker(symbol, counter):
        
        counter += 1

        if symbol in bad_symbols:
            if len(symbols) == 1:
                print("No good symbol to run.")
                return None

            if counter > 3: # 3 porque solo hay dos simbolos malos
                print("Counter reach 3")    
                return None
            
            return checker( symbols[counter] , counter= counter)
        
        return symbol
    
    symbol = checker(symbol, counter = 0)

    if symbol is None:
        return 1

    print(f"Symbol {symbol} is going to be process")

    orderSell = set_orders(symbol)

    if orderSell is None:
        print("No order was fullfill")
        return 2

    return orderSell

def bot():
    global BOT_COUNTER
    BOT_COUNTER += 1

    logging.info( f"Bot counter: {BOT_COUNTER}" )

    orderSell = main()

    if not isinstance(orderSell, dict):
        print(f"Waiting {L} minutes to analyze new positions")
        time.sleep( 60*L )
        bot()

    # Wait for order to fill
    for i in range(4):

        try:
            bi = Binance(account = "futures")
            while not bi.wait(orderSell):
                print("Waiting another minute!")
                time.sleep( 60*L )

            break

        except (socket.gaierror, NewConnectionError, MaxRetryError, ConnectionError) as e:
            # print(e, e.__dict__)
            # raise Exception(e)
            print( f"Exception error, iteration {i}")
            print("Waiting another minute!")
            time.sleep( 60*1 )

        if i == 3:
            raise Exception(e)
        
    print("Order fill!\n\n")

    bot()

def get_orders():
    bi = Binance(symbol="")
    df_trades = pd.DataFrame(bi.client.futures_account_trades())
    df_trades["time"] = df_trades["time"].apply(lambda x : datetime.fromtimestamp( x/1000 ))

    return df_trades

if __name__ == "__main__":


    logging.basicConfig(
        filename= f"logs/{Path(__file__).stem}_{date.today()}.log", 
        level=logging.INFO,
        format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    )

    logging.info(f'Interval: {L}')
    logging.info(f'PCT: {PCT}')
    logging.info(f'Share: {SHARE}')
    logging.info(f'Leverage: {LEVERAGE}')
    # logging.info(f'Assets: { ",".join( trading_pairs ) }')

    bot()
    # get_orders()
    # main()
    
    # historic_download( 
    #     broker = "binance", 
    #     fiat = "USDT", 
    #     frequency= "1min",
    #     start = (datetime.today() - relativedelta(days = 1)).date(),
    #     from_ = "ext_api",
    #     verbose = True
    # )