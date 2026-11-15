import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from garchm import Data, NO_KAPPA, QGRID, VAR, filt, fit_frozen, simulate, spec, starts_from, two_step

MARKETS = ['SPX', 'IXIC', 'DJI', 'RUT', 'GSPTSE', 'BVSP', 'MXX', 'FTSE', 'GDAXI', 'FCHI', 'AEX', 'SSMI', 'IBEX',
           'N225', 'HSI', 'AXJO', 'KS11', 'BSESN', 'TWII', 'STI']
SEED = 20261115
BURN = 500
CHUNK = 20


def null_models(var):
    df = pd.read_csv('../data/returns.csv', parse_dates=['date']).set_index('date')
    Z = pd.DataFrame(index=df.index, columns=MARKETS, dtype=float)
    out = {}
    for m in MARKETS:
        r = df[m].dropna()
        d = Data(r.values)
        ts = two_step(d, var, qs=(), qgrid=QGRID)
        null = ts['sign']
        z = filt(d, null['p'], NO_KAPPA, VAR[var][1], null['frozen'], want=('path',))['z']
        z = (z - z[1:].mean()) / z[1:].std()
        Z.loc[r.index[1:], m] = z[1:]
        out[m] = dict(p=null['p'], T=d.T, lr_grid=ts['lr_grid'], lk_grid=ts['lk_grid'])
    return out, Z.values


def market_grid(p, var, z):
    d = Data(simulate(z, p, vtype=VAR[var][1])[0][BURN:])
    sp1 = spec('mlag', var)
    s1 = fit_frozen(sp1, d, None, starts_from(sp1, d, p, n_rand=1, seed=0))
    s1['z'] = filt(d, s1['p'], NO_KAPPA, VAR[var][1], want=('path',))['z']
    ts = two_step(d, var, qs=(), qgrid=QGRID, s1=s1, light=True)
    return np.array([[ts['lr_grid'][q] for q in QGRID], [ts['lk_grid'][q] for q in QGRID]])


def draw(b, nulls, Z, var):
    # dates are resampled jointly, so returns of different markets on the same day stay together
    rng = np.random.default_rng(SEED + b)
    n = max(v['T'] for v in nulls.values()) + BURN
    rows = Z[rng.integers(0, len(Z), n)]
    out = np.full((len(MARKETS), 2, len(QGRID)), np.nan)
    for i, m in enumerate(MARKETS):
        z = rows[:, i].copy()
        miss = np.isnan(z)
        pool = Z[~np.isnan(Z[:, i]), i]
        z[miss] = rng.choice(pool, miss.sum())
        try:
            out[i] = market_grid(nulls[m]['p'], var, z[:nulls[m]['T'] + BURN])
        except Exception:
            pass
    return out


if __name__ == '__main__':
    var, B = sys.argv[1], int(sys.argv[2])
    path = f'results/panel_boot_{var}.pkl'
    if os.path.exists(path):
        res = pickle.load(open(path, 'rb'))
    else:
        nulls, Z = null_models(var)
        res = dict(nulls=nulls, Z=Z, draws={})
    t0 = time.time()
    todo = [b for b in range(B) if b not in res['draws']]
    for k in range(0, len(todo), CHUNK):
        batch = todo[k:k + CHUNK]
        for b, g in zip(batch, Parallel(n_jobs=-1)(delayed(draw)(b, res['nulls'], res['Z'], var) for b in batch)):
            res['draws'][b] = g
        os.makedirs('results', exist_ok=True)
        pickle.dump(res, open(path, 'wb'))
        print(var, len(res['draws']), f'{(time.time() - t0) / 60:.1f} min', flush=True)
