from trading import Asset
from trading.func_aux import get_assets

from datetime import datetime, timedelta
import plotly.express as xp


def analyze_single(s, f):
    asset = Asset(
            symbol=s,
            fiat = f,
            frequency= f"1min",
            end = datetime.now(),
            start = datetime.now() - timedelta(seconds= 60*3*200 ),
            source = "ext_api",
            broker="binance"
        )

    if asset.df is None or len(asset.df) == 0: 
        return None

    return asset

def strategy(symbol):

    asset = analyze_single(symbol, "USDT")

    if asset is None: return None

    asset.df["market"] = asset.sma(100)

    if asset.df["market"].pct_change(10).iloc[-1] < 0:
        return asset


if __name__ == "__main__":
    assets = { symbol:strategy(symbol) for symbol in get_assets()["binance"] }
    assets = { i:v for i,v in assets.items() if v }

    print(len(get_assets()["binance"]), len(assets))

    for i in range(5):
        if i >= len(assets): break
        symbol = list(assets.keys())[i]
        df = assets[ symbol ].df
        xp.line( df[["close", "market"]], title= symbol).show()




