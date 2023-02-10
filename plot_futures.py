import pandas as pd
from trading.func_aux import PWD
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv( PWD("binance/futures.csv") , header = None)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime( df["date"] )
    df.set_index("date", inplace = True)

    df.plot()
    plt.show()


if __name__ == "__main__":
    main()