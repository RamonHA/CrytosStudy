import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as xp

files = [i for i in os.listdir() if "csv" in i]

bench = "momentum_results_best.csv"

bench_df = pd.read_csv(bench)
bench_df = bench_df.sort_values(by = "acc", ascending=False).reset_index(drop=True)
bench_dfb = pd.read_csv( bench_df["route"].iloc[0] + "/resume.csv" )

file = "lowestmomentum_rsismothslope.csv"
df = pd.read_csv(file)
df = df.sort_values(by = "acc", ascending=False).reset_index(drop=True)
dfb = pd.read_csv( df["route"].iloc[0] + "/resume.csv" )

bench_dfb["acc"].plot(label = "Bench")
dfb["acc"].plot(label = "New")
plt.legend()


def separate(df):

    df[[ "momentum",  "period", "rsi", "length", "smoth", "slope"]] = df["route"].apply(lambda x: pd.Series(x.split("/")[1] ))[0].str.split("_", expand = True)

    return df