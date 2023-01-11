import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from trading.variables.params_grid import RF_C_GRID

def slopes(asset):

    asset.df[ f"ema_1h" ] = asset.ema_slope( 24, 3 )
    asset.df[ f"ema_6h" ] = asset.ema_slope( 12, 3 )
    asset.df[ f"ema_12h" ] = asset.ema_slope( 10, 3 )

    asset.df.reset_index(inplace = True)
    asset.df["date"] = pd.to_datetime( asset.df["date"] )
    asset.df = asset.df.resample( "1D", on = "date" ).agg({
        "open":"first", 
        "low":"min",
        "high":"max",
        "close":"last",
        "volume":"sum",
        "ema_1h":"last",
        "ema_6h":"last",
        "ema_12h":"last",
    })

    asset.df[ "ema_1d" ] = asset.ema_slope( 7, 3 )
    asset.df[ "ema_1w" ] = asset.ema_slope( 28, 3 )
    asset.df[ "ema_1m" ] = asset.ema_slope( 90, 3 )

    asset.df = asset.df.dropna()

    asset.df["target"] = asset.df["close"].pct_change().shift(-1).apply( lambda x : 1 if x > 0 else 0)

    asset.df = asset.df[ [ "ema_1h", "ema_6h", "ema_12h", "ema_1d", "ema_1w", "ema_1m", "target" ] ]

    train = asset.df.iloc[:-1]
    test = asset.df.iloc[-1:]

    rf = RandomForestClassifier()
    rf.fit( train.drop(columns = ["target"]) , train[["target"]])

    return float(rf.predict(  test.drop(columns = ["target"]) )[0])

def cv_slopes(asset):

    asset.df[ f"ema_1h" ] = asset.ema_slope( 24, 3 )
    asset.df[ f"ema_6h" ] = asset.ema_slope( 12, 3 )
    asset.df[ f"ema_12h" ] = asset.ema_slope( 10, 3 )

    asset.df.reset_index(inplace = True)
    asset.df["date"] = pd.to_datetime( asset.df["date"] )
    asset.df = asset.df.resample( "1D", on = "date" ).agg({
        "open":"first", 
        "low":"min",
        "high":"max",
        "close":"last",
        "volume":"sum",
        "ema_1h":"last",
        "ema_6h":"last",
        "ema_12h":"last",
    })

    asset.df[ "ema_1d" ] = asset.ema_slope( 7, 3 )
    asset.df[ "ema_1w" ] = asset.ema_slope( 28, 3 )
    asset.df[ "ema_1m" ] = asset.ema_slope( 90, 3 )

    asset.df = asset.df.dropna()

    asset.df["target"] = asset.df["close"].pct_change().shift(-1).apply( lambda x : 1 if x > 0 else 0)

    asset.df = asset.df[ [ "ema_1h", "ema_6h", "ema_12h", "ema_1d", "ema_1w", "ema_1m", "target" ] ]

    train = asset.df.iloc[:-1]
    test = asset.df.iloc[-1:]

    rf = RandomForestClassifier()
    rf.fit( train.drop(columns = ["target"]) , train[["target"]])

    return float(rf.predict(  test.drop(columns = ["target"]) )[0])