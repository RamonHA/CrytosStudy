
def simple_ema(asset):
    asset.df["ema"] = asset.ema(90) 
    asset.df["order"] = asset.df["ema"] < asset.df["close"]

    if asset.df["order"].iloc[-2:].all():
        return asset.df["ema"].iloc[-1]
    
    return False

def slopes(asset):

    asset.df[ f"ema_1h" ] = asset.ema_slope( 24, 3 )
    asset.df[ f"ema_6h" ] = asset.ema_slope( 12, 3 )

    asset.df["emas"] = ( asset.df["close"] > asset.df["ema_1h"] ) & ( asset.df["close"] > asset.df["ema_6h"] )

    asset.df["rsi"] = asset.rsi_smoth(14, 14)

def william_and_rsi_variants(asset):
    """  """

    asset.df["rsi"] = asset.rsi_smoth(14, 14) > 50
    asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
    asset.df["emaslope"] = asset.ema_slope(90, 3)
    asset.df["ema_slope"] = asset.df["emaslope"] > 0
    asset.df["rsi_smoth_slope"] = asset.rsi_smoth_slope( 7,7,3 ) > 0
    asset.df["oneside_gaussian_filter_slope"] = asset.oneside_gaussian_filter_slope(2,4) > 0

    asset.df["order"] =  asset.df["ema_slope"] & asset.df["buy_wf"] & asset.df["rsi"] & asset.df["rsi_smoth_slope"] & asset.df["oneside_gaussian_filter_slope"]

    if asset.df.iloc[-2:]["order"].any():
        return asset.df.iloc[-1]["emaslope"]

    return False

