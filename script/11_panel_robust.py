import os
import pickle
import sys

import numpy as np
import pandas as pd

from garchm import Data, QGRID, two_step

MARKETS = ['SPX', 'IXIC', 'DJI', 'RUT', 'GSPTSE', 'BVSP', 'MXX', 'FTSE', 'GDAXI', 'FCHI', 'AEX', 'SSMI', 'IBEX',
           'N225', 'HSI', 'AXJO', 'KS11', 'BSESN', 'TWII', 'STI']


def grid(r, var):
    ts = two_step(Data(r), var, qs=(), qgrid=QGRID)
    return np.array([[ts['lr_grid'][q] for q in QGRID], [ts['lk_grid'][q] for q in QGRID]])


if __name__ == '__main__':
    path = 'results/panel_robust.pkl'
    res = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    df = pd.read_csv('../data/returns.csv', parse_dates=['date'])
    for var in sys.argv[1:] or ['garch', 'gjr', 'egarch']:
        for m in MARKETS:
            s = df[['date', m]].dropna()
            r, year = s[m].values, s.date.dt.year.values
            for sample, keep in [('full', year > 0), ('drop2020', year != 2020)]:
                if (var, m, sample) in res or (sample == 'full' and var != 'garch'):
                    continue
                res[(var, m, sample)] = grid(r[keep], var)
                pickle.dump(res, open(path, 'wb'))
            print(var, m, flush=True)
