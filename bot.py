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

from registro import futures

# Create only when the code is going to start to automatically run every N minutes
# historic_download( "binance", "usdt", "1min", "" )

L = 1
PCT = 1.0015
SHARE = .04
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

    with mp.Pool( mp.cpu_count() ) as pool:
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

    # assets.sort(key = myFunc, reverse=True)

    # l = len(assets)

    # assets = assets[ :int( 0.3*l ) ]

    for s in assets:

        asset = s["asset"]
        asset.df["growth"] = asset.df["close"].pct_change( 50 )
        asset.df["changes"] = asset.df["close"].pct_change()

        # asset.df["buy_wf"] = asset.william_fractals(3, shift=True)
        # asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(3,2) > 0
        
        asset.df["rsi"] = (asset.rsi( 7 ) > 64).rolling(14).sum()
        # asset.df["rsi_smoth"] = (asset.rsi_smoth(7, 5) > 67).rolling(14).sum()
        # asset.df["rsi_slope"] = asset.df["rsi_smoth"].pct_change(periods = 3)

        asset.df["rsi_slope"] = asset.rsi_smoth(7, 16).pct_change(periods = 2)
        
        # asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
        
        asset.df["ema_slope"] = asset.ema_slope(15, 3)
        asset.df["ema_slope_smoth"] = asset.sma( 3, target = "ema_slope" )
        asset.df["ema_slope_slope"] = asset.df["ema_slope_smoth"].diff()

        # asset.df["ema"] = (asset.ema(90) < asset.df["close"]).rolling(3).sum()
        
        # # asset.df["rsi_smoth_slope"] = asset.rsi_smoth_slope( 7,7,3 )
        # # asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(2,4)
        
        asset.df["sell"] = asset.william_fractals(3, shift=True, order = "sell").rolling(3).sum()

        # asset.df["support"], asset.df["resistance"] = asset.support_resistance( 15 , support="close", resistance = "open")
        # asset.df["sc"] = (asset.df["support"] == asset.df["close"]).rolling(3).sum()

        d = asset.df.iloc[-1].to_dict()

        # growth.append( {"symbol": asset.symbol, "return": d["growth"]} )

        # if d["buy_wf"] and d["sell"] == 0 and d["rsi"] < 70:
        # if d["buy_wf"] and d["oneside_gaussian_filter_slope"] and d["rsi"] == 0 and d["rsi_smoth"] == 0 and d["ema_slope"] > 0 and d["rsi_slope"] > 0 and d["sell"] == 0 and d["ema"] == 3 and d["growth"] < 0.03 and d["sc"] == 0:
        if  d["rsi"] == 0 and d["ema_slope"] > 0 and d["ema_slope_slope"] > 0 and d["rsi_slope"] > 0 and d["sell"] == 0 and d["growth"] < 0.03: #and d["sc"] == 0:
            # pos_changes = asset.df[ asset.df["changes"] > 0 ].iloc[-10:]["changes"].mean()
            arr = {"symbol": asset.symbol, "return": asset.momentum(3).iloc[-1]}
            first_rule.append(arr)


        # if d["ema"] == 2 and d["sell"] == 0:
        #     # changes = asset.df.iloc[-10:]["changes"].mean()
        #     pos_changes = asset.df[ asset.df["changes"] > 0 ].iloc[-10:]["changes"].mean()
        #     arr = {"symbol": asset.symbol, "return": pos_changes}
        #     if d["ema_slope"] > 0 and d["rsi"] > 40 and d["oneside_gaussian_filter_slope"] > 0:

        #         if d["rsi_smoth_slope"] > 0:
        #             if d["buy_wf"]:
        #                 first_rule.append(arr)
        #                 buy_order.append(arr)
        #                 continue
                        
        #             second_rule.append( arr )
        #             buy_order.append(arr)
        #             continue
            
        #         third_rule.append(arr)
        #         buy_order.append(arr)
        #         continue
            
        #     forth_rule.append(arr)
        #     buy_order.append(arr)
        
    first_rule.sort(key = myFunc, reverse=False)

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
    price = bi.client.futures_symbol_ticker(symbol = symbol)["price"]
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
            symbol="BTC",
            fiat = "USDT",
            frequency= f"{L}min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*L*150 ),
            from_ = "ext_api",
            broker="binance"
        )

    return

def wait(orderSell):
    try:
        bi = Binance(symbol="")

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