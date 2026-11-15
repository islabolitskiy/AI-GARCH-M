import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

from garchm import (Data, IX, NO_KAPPA, QGRID, VAR, filt, fit_frozen, returns, simulate, spec, starts_from,
                    two_step)

QS = (0.01, 0.025, 0.05, 0.10)
SEED = 20261114
BURN = 500


def draw(b, p_null, var, T, pool):
    rng = np.random.default_rng(SEED + b)
    r, _, _ = simulate(rng.choice(pool, size=T + BURN, replace=True), p_null, vtype=VAR[var][1])
    d = Data(r[BURN:])
    try:
        sp1 = spec('mlag', var)
        s1 = fit_frozen(sp1, d, None, starts_from(sp1, d, p_null, n_rand=1, seed=b))
        s1['z'] = filt(d, s1['p'], NO_KAPPA, VAR[var][1], want=('path',))['z']
        ts = two_step(d, var, qs=QS, qgrid=QGRID, s1=s1, light=True, seed=b)
    except Exception:
        return None
    row = dict(b=b, sup=max(ts['lr_grid'].values()), l2=ts['sign']['p'][IX['l2']])
    for q in QS:
        row[f'lr_{q}'] = ts['lr'][q]
        row[f'lk_{q}'] = ts['ai'][q]['p'][IX['lk']]
    return row


if __name__ == '__main__':
    index, var, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = Data(returns(index).r.values)
    ts = two_step(d, var, qs=QS, qgrid=QGRID)
    null = ts['sign']
    z = filt(d, null['p'], NO_KAPPA, VAR[var][1], null['frozen'], want=('path',))['z'][1:]
    pool = (z - z.mean()) / z.std()

    rows = Parallel(n_jobs=-1)(delayed(draw)(b, null['p'], var, d.T, pool) for b in range(B))
    rows = [x for x in rows if x is not None]
    obs = dict(lr=ts['lr'], lr_grid=ts['lr_grid'], sup=max(ts['lr_grid'].values()), kap=ts['kap'])
    os.makedirs('results', exist_ok=True)
    pickle.dump(dict(obs=obs, rows=rows), open(f'results/boot_{index}_{var}.pkl', 'wb'))

    sup = np.array([x['sup'] for x in rows])
    print(index, var, len(rows), f"sup-LR={obs['sup']:.2f}", f"p={(1 + (sup >= obs['sup']).sum()) / (len(sup) + 1):.3f}")
