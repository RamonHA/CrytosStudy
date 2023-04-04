# Sample code to detect which Future coins to use

import warnings
warnings.filterwarnings("ignore")

from trading.assets.binance import Binance
from trading import Asset
from trading.func_aux import timing
from trading.func_brokers import historic_download

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

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
PCT = 1.0012
SHARE = .05
LEVERAGE = 25
STOP_LIMIT_PCT = 0.5

BOT_COUNTER = 0


bi = Binance(symbol="")

# futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

# trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]

# bad = ["USDCUSDT"]

# trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if (t[-4:] == "USDT" and t not in bad)]

trading_pairs = [ ("BTC", "USDT"), ("ETH", "USDT"), ("LTC", "USDT"), ("DOGE", "USDT") ]

class Error():
    pass

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

def clf_test(asset):
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

    cv_split = round(1 / split_ratio)
    if cv_split == 1:
        return False, 0

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

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv = cv_split)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_pred, labels = [0, 1])
    # res_cv = pd.DataFrame([precision,recall,fscore,support], index = ["precision","recall","fscore","support"])
    pred = clf.predict(last_row.drop(columns = ["target"]))
    logging.info(f"{asset.symbol}\tPrecision: {precision}\tBest params: ")
    logging.info(clf.best_params_)

    return pred[0], precision[-1]

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
    
    pred, precision = clf_test(asset)

    if pred:
        logging.info( f"Asset {asset.symbol} fills clf rule." )

    return pred

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

def calculate_stop_price(bougth_price, leverage, pct_limit):

    pct_min = pct_limit*( 1 / leverage)/100

    return bougth_price*(1 - pct_min)

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

    # stop_price = calculate_stop_price(real_price_bought, leverage,  STOP_LIMIT_PCT )
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