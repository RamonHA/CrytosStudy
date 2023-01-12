# Sample code to detect which Future coins to use

from trading.assets.binance import Binance
from trading import Asset
from trading.func_aux import timing
from trading.func_brokers import historic_download

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import multiprocessing as mp

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 5

bi = Binance(symbol="")

futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols

trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]

trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if t[-4:] == "USDT"]

def analyze_single(s, f):
    asset = Asset(
            symbol=s,
            fiat = f,
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*150 ),
            from_ = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    return asset

# @timing
def analyze():
    print("Analyze")
    growth = []

    first_rule = []
    second_rule = []
    third_rule = []
    forth_rule = []

    assets = []
    buy_order = []

    def myFunc(e):
        return e['return']

    with mp.Pool( mp.cpu_count() // 2 ) as pool:
        assets = pool.starmap(
            analyze_single,
            [ (s,f) for s,f in trading_pairs ]   
        )
    
    assets = [ { "asset":asset, "return":asset.momentum(3).iloc[-1] } for asset in assets if asset is not None ]

    # for s, f in trading_pairs:

    #     asset = Asset(
    #         symbol=s,
    #         fiat = f,
    #         frequency= f"{L}min",
    #         end = datetime.now(),
    #         start = datetime.now() - timedelta(seconds= 60*L*150 ),
    #         from_ = "ext_api",
    #         broker="binance"
    #     )

    #     if asset.df is None or len(asset.df) == 0: continue

    #     assets.append( {"asset": asset, "return": asset.momentum(3).iloc[-1] } )

    assets.sort(key = myFunc, reverse=True)

    l = len(assets)

    assets = assets[ :int( 0.3*l ) ]

    for s in assets:

        asset = s["asset"]

        asset.df["rsi"] = asset.rsi_smoth(14, 14)
        asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
        asset.df["ema_slope"] = asset.ema_slope(40, 2)
        asset.df["ema"] = (asset.ema(40) < asset.df["close"]).rolling(4).sum()
        asset.df["growth"] = asset.df["close"].pct_change( 20 )
        asset.df["rsi_smoth_slope"] = asset.rsi_smoth_slope( 7,7,3 )
        asset.df["changes"] = asset.df["close"].pct_change()
        asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(2,4)
        
        d = asset.df.iloc[-1].to_dict()

        growth.append( {"symbol": asset.symbol, "return": d["growth"]} )

        if d["ema"] == 4:
            # changes = asset.df.iloc[-10:]["changes"].mean()
            pos_changes = asset.df[ asset.df["changes"] > 0 ].iloc[-10:]["changes"].mean()
            arr = {"symbol": asset.symbol, "return": pos_changes}
            if d["ema_slope"] > 0 and d["rsi"] > 40 and d["oneside_gaussian_filter_slope"] > 0:

                if d["rsi_smoth_slope"] > 0:
                    if d["buy_wf"]:
                        first_rule.append(arr)
                        buy_order.append(arr)
                        continue
                        
                    second_rule.append( arr )
                    buy_order.append(arr)
                    continue
            
                third_rule.append(arr)
                buy_order.append(arr)
                continue
            
            forth_rule.append(arr)
            buy_order.append(arr)
        
    first_rule.sort(key = myFunc, reverse=True)
    second_rule.sort(key = myFunc, reverse=True)
    third_rule.sort(key = myFunc, reverse=True)
    forth_rule.sort(key = myFunc, reverse=True)
    growth.sort(key = myFunc, reverse=True)

    # print("\n")

    # print(first_rule, (len(first_rule)/len(trading_pairs)) )
    # print("\n")

    # print(second_rule, (len(second_rule)/len(trading_pairs)) )
    # print("\n")

    # print(third_rule, (len(third_rule)/len(trading_pairs)) )
    # print("\n")

    # print(forth_rule, (len(forth_rule)/len(trading_pairs)) )
    # print("\n")

    # print("Greatest growth: \n", growth[:5])

    # print("\n")

    return first_rule, second_rule

# @timing
def set_orders(symbol):
    bi = Binance(symbol="")

    symbol = "{}USDT".format(symbol)
    pct = 1.002
    share = .25
    leverage = 25

    max_leverage = [i for i in bi.client.futures_leverage_bracket() if symbol in i["symbol"]][0]["brackets"][0]["initialLeverage"]
    leverage = leverage if max_leverage >= leverage else max_leverage

    balance = float([ i["balance"] for i in bi.client.futures_account_balance() if i["asset"] == "USDT"][0])
    price = bi.client.futures_symbol_ticker(symbol = symbol)["price"]
    price = float(price)

    ticker_info = bi.client.get_symbol_info(symbol)

    qty_rouding = len(str(float([i["stepSize"] for i in ticker_info["filters"] if i["filterType"] == "LOT_SIZE"][0])).split(".")[-1])
    qty = balance*leverage*share / price

    bi.client.futures_change_leverage(symbol=symbol, leverage=leverage)

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
                qty_rouding -= 1
                orderBuy = set_buy_order(symbol, qty, qty_rouding)

        return orderBuy

    orderBuy = set_buy_order(symbol, qty, qty_rouding)
    print("Buy order done!")
    time.sleep(3)

    df_trades = pd.DataFrame(bi.client.futures_account_trades())
    df_trades = df_trades[ df_trades["orderId"] == orderBuy["orderId"] ]

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
                orderSell = set_sell_order(symbol, price_sell, qty, price_rounding)

        return orderSell

    orderSell = set_sell_order(symbol, price_sell, qty, price_rounding)
    print("Sell order done!")

    print(orderSell)

    return orderSell

def wait(orderSell):
    bi = Binance(symbol="")
    df_trades = pd.DataFrame(bi.client.futures_account_trades())

    df_trades = df_trades[ df_trades["orderId"] == orderSell["orderId"] ]

    if len(df_trades) > 0:
        return True
        
    return False

def main():
    f, s = analyze()
    
    orders = f
    if len(f) == 0:
        if len(s) == 0:
            print("No order is going to be sent")
            return 
        orders = s
    
    symbol = orders[0]["symbol"]

    print(f"Symbol {symbol} is going to be process")

    return set_orders( symbol )

def bot():
        
    orderSell = main()

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

if __name__ == "__main__":
    bot()
    # main()
    
    # historic_download( 
    #     broker = "binance", 
    #     fiat = "USDT", 
    #     frequency= "1min",
    #     start = (datetime.today() - relativedelta(days = 1)).date(),
    #     from_ = "ext_api",
    #     verbose = True
    # )