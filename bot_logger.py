# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset
from trading.func_aux import timing
from trading.func_brokers import historic_download

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import multiprocessing as mp
from copy import copy
import numpy as np

import logging
from pathlib import Path

from registro import futures

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 3
PCT = 1.0015
SHARE = .03
LEVERAGE = 25

BOT_COUNTER = 0


bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]

bad = ["USDCUSDT"]

trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

class Error():
    pass

def analyze_single(s, f):
    asset = Asset(
            symbol=s,
            fiat = f,
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*300 ),
            source = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    return asset

def slopes_strategy(asset):
    asset.df["rsi_slope"] = asset.rsi_smoth_slope(27, 20, 4) > (5.75/1000)
    asset.df["rsi_slope2"] = asset.rsi_smoth_slope(28, 8, 3) > (-4/1000)
    asset.df["sma"] = asset.sma_slope( 33, 20 ) > ( 0 )
    asset.df["dema"] = asset.dema(12).pct_change(11) > (5/1000)
    asset.df["hull_twma"] = asset.hull_twma(7).pct_change(5) > (-4/1000)
    asset.df["roc"] = asset.roc( 15 ).pct_change(12) > (7/1000)
    asset.df[ "rsi_thr" ] = ( asset.rsi(10) >= 70 ).rolling(20).sum() == 0
    asset.df["slopes"] = asset.df[["rsi_slope", "rsi_slope2", "sma", "dema", "hull_twma", "rsi_thr"]].all(axis = 1)

    return asset.df["slopes"]

def sandr(asset):
    # El problema de esta es que casi nunca entraba, pero cuando entraba si las cerraba
    # l = 9 if asset.ema(27).rolling(20).std().iloc[-1] <= 0.7421 else 18
    # _, asset.df["resistance"] = asset.support_resistance(l)
    # asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
    # asset.df["rsi"] = asset.rsi_smoth_slope(30, 4, 7) > 0.00223 
    # asset.df["sma"] = asset.sma_slope(44, 12) > (-0.00625)

    # After 19/03/2023 meta aplication
    # se detiene esta estrategia por varias entradas erroneas. 21/3/2023
    asset.df["ema_std"] = asset.ema(43).rolling(19).std()
    # max_std = asset.df["ema_std"].max()
    # max_std = asset.df["ema_std"].max()
    l = 9 if asset.df["ema_std"].iloc[-1] <= 0.35 else 19
    _, asset.df["resistance"] = asset.support_resistance(l)
    asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
    asset.df["rsi_smoth"] = asset.rsi_smoth( 21, 10 ).rolling( 10 ).std() > 0.6 # < 0.7
    asset.df[ "rsi_thr" ] = ( asset.rsi(7) >= 71 ).rolling(17).sum() == 0

    # Last Change: 23/03/23
    # COTI Liquidation
    asset.df["rsi_slope"] = asset.rsi(10)
    asset.df["rsi_slope"] = asset.ema_slope(10, 3, target = "rsi_slope") > 0
    
    asset.df["sma"] = asset.ema_slope(30, 4) > (0) # -0.00625

    return asset.df[ [ "resistance", "rsi_smoth", "rsi_thr", "rsi_slope", "sma" ] ].all(axis = 1)


def analysis(asset):
    """  
        Last update: 7/3/2023

        Based on results of 100gen pymoo_test
    """

    asset.df["trend"] = asset.ema(50)
    asset.df["trend_res"] = asset.df["close"] - asset.df["trend"]
    asset.df["season"] = asset.sma( 25, target = "trend_res" )
    asset.df["season_res"] = asset.df["trend_res"] - asset.df["season"]

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
    asset.df["buy"] = asset.df["sin"] == asset.df["sin"].min()

    # Support and resistance
    asset.df["sandr"] = sandr( asset )

    # Add slopes strategy 21/03/23
    # Error con RNDR
    # asset.df["slopes"] = slopes_strategy(asset)

    d = asset.df.iloc[-1].to_dict()

    seasonality = d["buy"]
    if seasonality:
        logging.info( f"Asset {asset.symbol} fills seasonality rule." )
        logging.info( str(d) )
    
    # slopes = d["slopes"]
    # if slopes:
    #     logging.info( f"Asset {asset.symbol} fills slope rule." )
    #     logging.info( str(d) )
    
    sandr_ = d["sandr"]
    if sandr_:
        logging.info( f"Asset {asset.symbol} fills support and resistance rule." )
        logging.info( str(d) )

    return ( seasonality  or sandr_)

# @timing
def analyze():
    print("Analyze")

    first_rule = []
    second_rule = []

    assets = []

    def myFunc(e):
        return e['return']

    with mp.Pool( mp.cpu_count() ) as pool:
        assets = pool.starmap(
            analyze_single,
            [ (s,f) for s,f in trading_pairs ]   
        )
    
    assets = [ { "asset":asset } for asset in assets if asset is not None ]

    first_rule = [ {"symbol": s["asset"].symbol, "return": s["asset"].momentum(3).iloc[-1]} for s in assets if analysis(s["asset"]) ]

    second_rule = copy(first_rule)
        
    first_rule.sort(key = myFunc, reverse=False)
    second_rule.sort(key = myFunc, reverse=True)

    return first_rule, second_rule

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

            orderSell = bi.client.futures_create_order(
                    symbol = symbol,
                    type = "LIMIT", 
                    timeInForce ="GTC",
                    side = "SELL",
                    price = price_sell, 
                    quantity = qty,
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
    
    sold_price = float(orderSell["price"])
    bougth_price = sold_price / PCT
    symbol = orderSell["symbol"]
    actual_price = float( bi.client.get_symbol_ticker(symbol = symbol)["price"] )

    qty = float(orderSell["origQty"])
    usdt_bought = sold_price * qty
    real_leverage = usdt_bought / ( orderSell["balance"]*SHARE )
    real_leverage = round(real_leverage)

    pct_min = 1 / real_leverage

    if (( actual_price / bougth_price ) - 1) < -( pct_min*0.5 ):
        newOrderSell = bi.client.futures_create_order(
                symbol = symbol,
                type = "MARKET",
                # timeInForce ="GTC",
                side = "SELL",
                quantity = qty,
            )
        
        time.sleep(3)
        
        return True

    return False

def main():
    f, s = analyze()
    
    if len(f) == 0:
        # if len(s) == 0:
        #     print("No order is going to be sent")

        #     return 0
        # orders = s
        print("No order is going to be sent")
        return 0
    else:
        orders = f # + s
    
        # print("Orders")
        # print(f)
        # print(s)
        # print("\n")
        # return 0
    
    # logging.info( f"Strategy selected assets: { ','.join( [ i['symbol'] for i in orders ] ) }" )

    symbol = orders[0]["symbol"]

    bad_symbols = ["SC", "RAY"]
    def checker(symbol, counter):
        
        counter += 1

        if symbol in bad_symbols:
            if len(orders) == 1:
                print("No good symbol to run.")
                return None

            if counter > 3: # 3 porque solo hay dos simbolos malos
                print("Counter reach 3")    
                return None
            
            return checker( orders[1]["symbol"] , counter= counter)
        
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

    # total_time = time.time() - st
    # start_time = datetime.now() - relativedelta(seconds= total_time + ( 60*5 ) )

    # historic_download( 
    #     broker = "binance", 
    #     fiat = "usdt", 
    #     frequency= "1min",
    #     start = start_time.date(),
    #     from_ = "ext_api",
    #     verbose = False
    # )
    # print("\n")

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