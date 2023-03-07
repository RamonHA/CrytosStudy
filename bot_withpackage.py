""" Test the Bot strategy using the Trading package """

from trading.processes import Bot
from trading.assets.binance import Binance
import time
import pandas as pd

L = 3
PCT = .0015
SHARE = .05
LEVERAGE = 50

def func(asset):
    asset.df["rsi_main"] = asset.rsi(7)
    asset.df["rsi1"] = asset.ema(2, target = "rsi_main" ) # asset.rsi_smoth(7, 3)
    asset.df["rsi2"] = asset.ema(7, target = "rsi_main") # asset.rsi_smoth(7, 7)

    asset.df["rsi2_slope"] = asset.rsi_smoth_slope(10, 10, 3) # asset.df["rsi2"].pct_change(3)

    asset.df["ema1"] = asset.ema(8)
    asset.df["ema2"] = asset.ema(16)

    asset.df["rsi"] = (asset.df["rsi1"] > asset.df["rsi2"]).astype(int).diff().rolling(2).sum()
    asset.df["ema"] = (asset.df["ema1"] > asset.df["ema2"]).astype(int).diff().rolling(2).sum()

    asset.df["rsi_thr"] = (asset.rsi( 7 ) > 67).rolling(20).sum()

    asset.df[ "rsi_std" ] = asset.rsi_smoth(10,10).rolling(18).std()

    d = asset.df.iloc[-1].to_dict()

    if  (
        d["rsi"] > 0 and        # rsi fast above slow
        d["ema"] > 0 and        # ema fast above slow
        d["rsi_thr"] == 0 and   # rsi max point
        d["rsi2"] < 55 and      # rsi min point
        d["rsi2_slope"] > 0 and # rsi slope,
        d["rsi_std"] > 2      # Ensure volatility. Low volatility refers to steady price or side-trend
        ):

        return asset.momentum(3).iloc[-1]
    
    return None


if __name__ == "__main__":

    # Get available Future pairs
    bi = Binance( #-> Asset
        fiat = "USDT",
        account = "futures"
    )

    trading_pairs = bi.trading_pairs()

    # Set bot main config
    bot = Bot(
        name = "TestWithModule",
        broker = "binance",
        fiat = "USDT",
        verbose = 2,
        assets = trading_pairs,
        account="futures",
        account_config = {
            "leverage": LEVERAGE
        }
    )

    # Set analyzis config
    bot.analyze(
        frequency=f"{L}min",
        run = True,
        analysis={
            "FilterTA":{
                "type":"filter",
                "function":func,
                "time":60*150
            }
        }
    )

    # How to choose
    if len(bot.results) == 0:
        print("It is empty")
    
    # bot.optimize() -> {Dictionary with selection assets and qty}

    bot.choose(
        value = bi.wallet_balance()*LEVERAGE*SHARE, # In orther to calculate exactly the amount to buy
        filter = "highest",
        filter_qty = 1,
    )

    bot.buy()

    time.sleep(3)

    ################################################################
    df_trades = pd.DataFrame(bi.client.futures_account_trades())
    df_trades["qty"] = df_trades["qty"].astype(float)
    df_trades["price"] = df_trades["price"].astype(float)

    sell_prices = {}
    for symbol, orderBuy in bot.buy_orders:
        df_trades = df_trades[ df_trades["orderId"] == orderBuy["orderId"] ]

        if len(df_trades) == 0:
            print(f"No orders with id { orderBuy['orderId'] }")
            continue

        qty = df_trades["qty"].iloc[-1]
        real_price_bought = df_trades["price"].iloc[-1]

        if len(df_trades) > 1:
            real_price_bought = df_trades["price"].max() 
            qty = df_trades["qty"].sum()
            
        price_rounding = len(str(real_price_bought).split(".")[-1])
        price_sell = real_price_bought*PCT
    ################################################################

    orderSell = bot.sell(
        sell_price = PCT, # can also be a dictionary
    )

    bot.wait(  
        # stop_limit = .01,
        # stop_limit_leverage_scale = True
    )

    


