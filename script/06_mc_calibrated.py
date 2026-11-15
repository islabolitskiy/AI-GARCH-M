import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from garchm import (Data, IX, NO_KAPPA, NP, QGRID, filt, fit_frozen, inference, se, simulate, spec,
                    starts_from, two_step)

BURN = 500


def design(cell):
    est = pickle.load(open('results/estimates_SPX.pkl', 'rb'))['fam']['garch']
    if cell == 'orig':
        p = np.zeros(NP)
        p[[IX[n] for n in ('mu', 'l1', 'l2', 'lk', 'om', 'a', 'b')]] = [0.0, 0.1, 0.1, 0.5, 0.05, 0.05, 0.85]
        return p, 0.025, 5000, None
    z = est['mlag']['path']['z'][1:]
    pool = (z - z.mean()) / z.std()
    p = est['ai05' if cell in ('null05', 'alt05') else 'ai025']['p'].copy()
    if cell == 'null05':
        p[IX['lk']] = 0.0
    return p, (0.025 if cell == 'alt025' else 0.05), len(z) + 1, pool


def replication(b, p, q, T, pool, k_true):
    rng = np.random.default_rng(777000 + b)
    z = rng.choice(pool, T + BURN) if pool is not None else rng.standard_normal(T + BURN)
    r, _, _ = simulate(z, p, kappa=k_true, vtype=0)
    d = Data(r[BURN:])
    try:
        sp1 = spec('mlag', 'garch')
        s1 = fit_frozen(sp1, d, None, starts_from(sp1, d, p, n_rand=1, seed=b))
        s1['z'] = filt(d, s1['p'], NO_KAPPA, 0, want=('path',))['z']
        ts = two_step(d, 'garch', qs=(q,), qgrid=QGRID, s1=s1, light=True, seed=b)
    except Exception:
        return None
    z, f0 = ts['z'], ts['sign']
    I = (z < 0).astype(float)
    spA = spec('ai', 'garch')
    out = dict(b=b, sup=max(ts['lr_grid'].values()))
    for tag, k in (('evt', ts['kap'][q]), ('empirical', float(np.quantile(z[1:], q))), ('true', k_true)):
        if tag == 'evt':
            fA = ts['ai'][q]
        else:
            fA = fit_frozen(spA, d, (I, (z < k).astype(float)), starts_from(spA, d, f0['p'], n_rand=0), kappa=k)
        inference(fA, d)
        out[f'lk_{tag}'], out[f'se_{tag}'] = fA['p'][IX['lk']], se(fA, 'lk')
        out[f'LR_{tag}'], out[f'kap_{tag}'] = 2 * (f0['nll'] - fA['nll']), k
    return out


if __name__ == '__main__':
    cell, R = sys.argv[1], int(sys.argv[2])
    p, q, T, pool = design(cell)
    k_true = float(np.quantile(pool, q)) if pool is not None else float(norm.ppf(q))
    rows = Parallel(n_jobs=-1)(delayed(replication)(b, p, q, T, pool, k_true) for b in range(R))
    rows = [x for x in rows if x is not None]
    os.makedirs('results', exist_ok=True)
    pickle.dump(dict(rows=rows, p=p, q=q, T=T, k_true=k_true), open(f'results/mc_{cell}.pkl', 'wb'))
    print(cell, len(rows))
