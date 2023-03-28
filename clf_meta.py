""" Make a parameter tunning strategy, trygin to improve 
    a clasification model to determine whether a certain level 
    of percentage of return can be obtained.
"""

from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

from trading.testers.rules_testing import RuleTesting
from trading import Asset

def get_asset(**kwargs):

    asset = Asset(
        symbol = kwargs.get("symbol", "LTC"),
        start = datetime( 2023, 3, 27, 0 ),
        end = datetime.now(),
        frequency="3min",
        fiat = "USDT",
        broker = "binance",
        source = "ext_api"
    )

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




