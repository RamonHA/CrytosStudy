import pandas as pd
from datetime import datetime

from trading.assets.binance import Binance
from trading.func_aux import PWD

def  futures(value = None):

    bi = Binance()

    if value is None:
        value = [ i["balance"] for i in bi.client.futures_account_balance() if i["asset"] == "USDT"][0]

    registro = [ datetime.today().strftime( "%Y-%m-%d %H:%M" ), value ]
    print(registro)
    df  = pd.DataFrame([registro])

    df.to_csv( PWD("binance/futures.csv") , index = False, header=False, mode = "a")

def simple_earn():
    """ Date, symbol, total amount, est APR, Est Total Value (in BTC and Mex) """
    registro = [ datetime.today().strftime( "%Y-%m-%d %H:%M" ), "cake", 2.56486025, 6.2, 0.0005004, 218.91  ]
    df  = pd.DataFrame([registro])

    df.to_csv( PWD("binance/staking.csv") , index = False, header=False, mode = "a")

if __name__ == "__main__":
    # futures()
    simple_earn()