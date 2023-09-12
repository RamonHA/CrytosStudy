# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

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

trading_pairs = [ t[:-4] for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

trading_pairs = trading_pairs[:20] 

def features_extraction(asset, train_size = 0.8):
    x = asset.df.drop(columns = ["target"]).to_numpy()
    y = asset.df["target"].to_numpy().reshape(-1, 1)

    if train_size != 1:
        train_size = int( len(x) * train_size )
        x_train, x_test = x[ :train_size ], x[ train_size: ]
        y_train, y_test = y[ :train_size ], y[ train_size: ]
    else:
        x_train = x[ :-1 ]
        y_train = y[:-1]
        x_test, y_test = [], []

    return x_train, y_train, x_test, y_test

def linear_reg(asset, forecasting = False):
    x,y = features_extraction(asset)

    lr = LinearRegression()
    lr.fit( X=x, y=y )

    if forecasting:
        forecasting = lr.predict( (x[-1] + 1).reshape(-1, 1) )
        forecasting = forecasting[0][0]
        forecasting = ( forecasting / y[-1][0] ) - 1

    pred = lr.predict(x)

    return y, pred, { "coef":lr.coef_[0][0], "forecasting":forecasting }

def poly_reg(asset, forecasting = False):
    x,y = features_extraction(asset)

    degree=3
    polyreg = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    polyreg.fit(x,y)

    if forecasting:
        forecasting = polyreg.predict( (x[-1] + 1).reshape(-1, 1) )
        forecasting = forecasting[0][0]
        forecasting = ( forecasting / y[-1][0] ) - 1
    
    pred = polyreg.predict(x)

    return y, pred, {"forecasting":forecasting}

def fit_sin_optimize(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''    
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    # return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}   

    season = fitfunc(tt) 

    return season

def fit_sin_brute(flat_season, scale = 0.8):
    # sampling rate
    sr = len(flat_season)
    # sampling interval
    ts = 1/sr
    t = np.arange(0,1,ts)

    r = np.std(flat_season)

    reg = []
    for i in range(4, 100, 2):
        y = np.sin(np.pi*i*t) * r

        error  = np.linalg.norm( flat_season - y )

        reg.append([ i, error ])


    reg = pd.DataFrame(reg, columns = ["freq", "error"])

    i = reg[ reg[ "error" ] == reg["error"].min() ]["freq"].iloc[0]
    y_season = np.sin(np.pi*i*t) * r * scale

    return y_season

def fit_sin(asset, mode = "brute", reg = "linear", forecasting = False):
    y, y_pred, params = {
        "linear":linear_reg,
        "poly":poly_reg
    }[ reg ]( asset, forecasting=forecasting )

    y_season = y - y_pred

    flat_season = [i[0] for i in y_season]

    if mode == "brute":
        fit_season = fit_sin_brute( flat_season )
    
    elif mode == "optimize":
        x = np.array(range(len(asset.df)))
        fit_season = fit_sin_optimize( x, flat_season )
    
    
    return y_season, fit_season, params

def get_pred(asset, scale = 0.8, mode = "brute", reg = "linear", forecasting = False):
    y_season, fit_season, params = fit_sin(asset, mode = mode, reg = reg, forecasting=forecasting)

    std = np.std( fit_season ) * scale
    min = np.min( fit_season )
    max = np.max( fit_season )
    last = fit_season[-1]
    point = ( last - min ) / ( max - min)

    r = {
        "pred":(-std) > last,
        "std":std,
        "change":(fit_season[-1]/fit_season[-2]) -1,
        "point":point,
        "slope":params.get("coef", 0),
        "forecasting":params.get("forecasting", 0)
    }

    return r

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

def analyze_single(symbol, scale = 0.8, mode = "optimize", reg = "poly", forecasting = True):
    asset = Asset(
            symbol=symbol,
            fiat = "USDT",
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*1000 ),
            source = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    asset.df = prep_target( asset, pct = 0.0005, window=10 )
    
    asset = features( asset, clf=False, drop = True , target = False)

    validation = asset.df.iloc[-1:].drop(columns = ["target"])

    if validation.isna().any().any():
        print(f"{symbol} has NA in validation set")
        return None

    asset.df = asset.df.replace( [np.inf, -np.inf], np.nan ).dropna()

    if len(asset.df) == 0:
        return None

    x_train, y_train, x_test, y_test = features_extraction(asset, train_size=1)

    reg = SVC( kernel="poly", degree=2, C=10 )

    reg.fit( x_train, y_train )

    pred = reg.predict( validation )

    if pred[0]:
        return asset.df[ "ema_40_slope_2" ].iloc[-1]
    
    return False

# @timing
def analyze():
    print("Analyze")
    
    print(f"Trading pairs: {len(trading_pairs)}")

    with mp.Pool( mp.cpu_count() ) as pool:
        assets = pool.starmap(
            analyze_single,
            trading_pairs
        )
    
    assets = [ [symbol, r] for symbol,r in zip( trading_pairs, assets) if r is not None ]

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
    logging.info(f'Assets: { ",".join( trading_pairs ) }')

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