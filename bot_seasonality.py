# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset
from trading.func_aux import timing
from trading.func_brokers import historic_download

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import multiprocessing as mp
from copy import copy
import numpy as np

from registro import futures

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 3
PCT = 1.0015
SHARE = .03
LEVERAGE = 20

bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]

trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if t[-4:] == "USDT"]

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

def analysis(asset):
    """  
        Last update: 7/3/2023

        Based on results of 100gen pymoo_test
    """

    asset.df["trend"] = asset.ema(60)
    asset.df["trend_res"] = asset.df["close"] - asset.df["trend"]
    asset.df["season"] = asset.sma( 30, target = "trend_res" )
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


    l = 9 if asset.ema(27).rolling(20).std().iloc[-1] <= 0.7421 else 18
    _, asset.df["resistance"] = asset.support_resistance(l)
    asset.df["resistance"] = (asset.df["resistance"] == asset.df["close"]) | (asset.df["resistance"] == asset.df["low"])
    asset.df["rsi"] = asset.rsi_smoth_slope(30, 4, 7) > 0.00223 
    asset.df["sma"] = asset.sma_slope(44, 12) > (-0.00625)

    d = asset.df.iloc[-1].to_dict()

    return (
        d["buy"] or (
            d["resistance"] and 
            d["rsi"] and 
            d["sma"]
        )
    )

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
    bi = Binance(symbol="")

    symbol = "{}USDT".format(symbol)
    pct = PCT
    share = SHARE
    leverage = LEVERAGE

    max_leverage = [i for i in bi.client.futures_leverage_bracket() if symbol in i["symbol"]][0]["brackets"][0]["initialLeverage"]
    leverage = leverage if max_leverage >= leverage else max_leverage

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
    
    symbol = orders[0]["symbol"]

    bad_symbols = ["SC", "RAY"]
    def checker(symbol):
        
        if symbol in bad_symbols:
            if len(orders) == 1:
                print("No good symbol to run.")
                return None

            return checker( orders[1]["symbol"] )
        
        return symbol
    
    symbol = checker(symbol)
    if symbol is None:
        return 1

    print(f"Symbol {symbol} is going to be process")

    orderSell = set_orders(symbol)

    if orderSell is None:
        print("No order was fullfill")
        return 2

    return orderSell

def bot():
        
    orderSell = main()

    if not isinstance(orderSell, dict):
        print(f"Waiting {L} minutes to analyze new positions")
        time.sleep( 60*L )
        bot()

    # Wait for order to fill
    bi = Binance()
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