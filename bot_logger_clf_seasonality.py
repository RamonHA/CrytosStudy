# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import multiprocessing as mp
from copy import copy
import numpy as np

import numpy, scipy.optimize


import logging
from pathlib import Path

from registro import futures

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 3
PCT = 1.0012
SHARE = .07
LEVERAGE = 55
STOP_LIMIT_PCT = 0.5

BOT_COUNTER = 0


bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]
bad = ["USDCUSDT"]

trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

trading_pairs = trading_pairs[:30] 

def features_extraction(asset):
    x = np.array(range(len(asset.df))).reshape(-1, 1)
    y = asset.df["close"].to_numpy().reshape(-1, 1)
    return x, y

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

def analyze_single(symbol, scale = 0.8, mode = "optimize", reg = "poly", forecasting = True):
    asset = Asset(
            symbol=symbol,
            fiat = "USDT",
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*1500 ),
            source = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    if (asset.df["close"].pct_change() == 0).iloc[-5:].all():
        return None
    
    rsi = asset.rsi_smoth_slope(15,15, 2)
    # asset.df["rsi"] = asset.rsi(15)
    # rsi = asset.ema_slope( 15, 2, target="rsi" )

    if all(rsi.iloc[-3:] < 0):
        return None

    r = get_pred( asset , scale = scale, mode = mode, reg = reg, forecasting=forecasting)
    r["symbol"] = symbol

    return r

# @timing
def analyze():
    print("Analyze")
    
    print(f"Trading pairs: {len(trading_pairs)}")

    with mp.Pool( mp.cpu_count() ) as pool:
        assets = pool.starmap(
            analyze_single,
            [ symbol for symbol in trading_pairs ]   
        )
    
    assets = [ r for r in assets if r is not None ]

    df = pd.DataFrame(assets)
    df.sort_values(by = "forecasting", ascending=False, inplace = True)

    df = df[ ( df["forecasting"] > 0 )] # (~df["pred"]) &

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
    bi = Binance(symbol="")
    try:
        df_trades = pd.DataFrame(bi.client.futures_account_trades())
    except Exception as e:
        print(e, e.__dict__)
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
    bi = Binance(account = "futures")
    while not bi.wait(orderSell):
        print("Waiting another minute!")
        time.sleep( 60*1 )

    print("Order fill!\n\n")

    bot()

def get_orders():
    bi = Binance(symbol="")
    df_trades = pd.DataFrame(bi.client.futures_account_trades())
    df_trades["time"] = df_trades["time"].apply(lambda x : datetime.fromtimestamp( x/1000 ))

    return df_trades

if __name__ != "__main__":


    logging.basicConfig(
        filename= f"logs/{Path(__file__).stem}_{date.today()}.log", 
        level=logging.INFO,
        format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    )

    logging.info(f'Interval: {L}')
    logging.info(f'PCT: {PCT}')
    logging.info(f'Share: {SHARE}')
    logging.info(f'Leverage: {LEVERAGE}')
    logging.info(f'Assets: { ",".join([ i+j for i,j in trading_pairs ]) }')

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