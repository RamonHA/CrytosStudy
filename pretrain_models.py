from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trading.variables.params_grid import RF_C_GRID,RF_R_GRID, DT_R_GRID
from trading.func_brokers import get_assets
from trading.grid_search.brute_force import BruteGridSearch
from trading.func_aux import timing

import numpy as np
from copy import deepcopy
import pickle
import json

from functions import *

CLF = True
QTY = 40
CREATE = True

GRID =  {'n_estimators': [ 100, 200, 500],
 'criterion': ['gini', 'entropy']}

MODEL = RandomForestClassifier() if CLF else RandomForestRegressor()

@timing
def main():
    if CREATE:
        binance = np.random.choice( list(get_assets()["binance"].keys()) , QTY )
    else:
        with open( f"Models/assets.json", "r" ) as fp:
            binance = json.load( fp )["assets"]

    df = pd.DataFrame()

    for symbol in binance:
        asset = new(symbol, start = date(2010,1,1), end = date(2022,10,4))

        if asset.df is None or (0 in asset.df.shape):
            continue

        rasset = features( deepcopy(asset), clf = True )

        df = pd.concat([ df, rasset.df ], axis = 0)
    
    print( "Shape of table: ", df.shape )

    bt = BruteGridSearch( df, regr=MODEL, parameters=GRID , error_ascending= False, error = "precision")
    bt.test(parallel = True, cpus = 4, pos_label = 1)
    ppd = bt.predict(one = False)

    with open( f"Models/RF{ 'C' if CLF else 'R' }_{QTY}Assets_BruteFroce", 'wb') as fp:
        pickle.dump(bt.regr, fp )
    
    with open( f"Models/assets.json", "w" ) as fp:
        json.dump( {"assets":binance.tolist()}, fp )

if __name__ == "__main__":
    main()
    