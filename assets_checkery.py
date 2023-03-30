from trading import Asset

from datetime import datetime


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

