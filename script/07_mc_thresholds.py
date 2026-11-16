import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, t as student_t

from garchm import Data, IX, NO_KAPPA, NP, filt, fit_frozen, kappa_q, simulate, spec, starts_from

QS = [0.01, 0.025, 0.05]
METHODS = [('empirical', 'empirical', {}), ('evt', 'evt', {}), ('evt_b63', 'evt', dict(block=63)),
           ('gpd', 'gpd', {}), ('hill', 'hill', {}), ('student', 'student', {}),
           ('evt_u05', 'evt', dict(u_q=0.05)), ('evt_u15', 'evt', dict(u_q=0.15))]
P = np.zeros(NP)
P[[IX[n] for n in ('mu', 'l1', 'om', 'a', 'b')]] = [0.0, 0.1, 0.05, 0.05, 0.85]
SC = np.sqrt(5 / 3)


def replication(b, dist, T):
    rng = np.random.default_rng(9000 + b + 100000 * (dist == 't5') + 7 * T)
    z = rng.standard_normal(T + 500) if dist == 'normal' else rng.standard_t(5, T + 500) / SC
    r, _, _ = simulate(z, P)
    d = Data(r[500:])
    try:
        sp = spec('mlag', 'garch')
        f = fit_frozen(sp, d, None, starts_from(sp, d, P, n_rand=1, seed=b))
        zr = filt(d, f['p'], NO_KAPPA, 0, want=('path',))['z'][1:]
        out = dict(b=b)
        for name, method, kw in METHODS:
            if name in ('evt_b63', 'evt_u05', 'evt_u15') and T < 2500:
                continue
            k = kappa_q(zr, QS, method=method, **kw)
            for q in QS:
                out[f'{name}_{q}'] = k[q]
        return out
    except Exception:
        return None


if __name__ == '__main__':
    dist, T, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    true = {q: norm.ppf(q) if dist == 'normal' else student_t.ppf(q, 5) / SC for q in QS}
    rows = [x for x in Parallel(n_jobs=-1)(delayed(replication)(b, dist, T) for b in range(R)) if x is not None]
    os.makedirs('results', exist_ok=True)
    pickle.dump(dict(rows=rows, true=true, dist=dist, T=T), open(f'results/mc_thr_{dist}_{T}.pkl', 'wb'))
    print(dist, T, len(rows))
