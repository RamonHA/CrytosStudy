""" Test the Bot strategy using the Trading package """

from trading.processes import Bot
from trading.assets.binance import Binance

L = 3
PCT = 1.0015
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
    bi = Binance()
    futures_exchange_info = bi.client.futures_exchange_info()  # request info on all futures symbols
    trading_pairs = [info['symbol'] for info in futures_exchange_info['symbols']]
    trading_pairs = [ ( t[:-4], t[-4:] ) for t in trading_pairs if t[-4:] == "USDT"]

    # Set bot main config
    bot = Bot(
        name = "TestWithModule",
        broker = "binance",
        fiat = "USDT",
        verbose = 2,
        assets = trading_pairs,
        account="future"
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

    # selectedCrypto = {"BTC":2, "ETH":1.8}
    selectedCrypto = bot.filter(
        bot.results, 
        filter = "highest", 
        filter_qty = 1
    )

    orderBuy = bot.buy()

    orderSell = bot.sell()

    bot.wait(  )


