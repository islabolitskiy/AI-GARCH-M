import os
import pickle

import numpy as np
from scipy.optimize import minimize

from garchm import Data, IX, NP, filt, fit, returns, spec, two_step

END = '2026-05-08'
KAPPA = -1.7727065009982415   # threshold of the submitted version (q = 5%)


def profile(sp, d, p, step=0.0005, n=4):
    """Log-likelihood along lambda_2 around p."""
    out = []
    for h in np.arange(-n, n + 1) * step:
        q = p.copy()
        q[IX['l2']] += h
        out.append((h, -filt(d, q, KAPPA, sp.vtype)['nll']))
    return np.array(out)


def search(sp, d, x0, seed, n=6):
    """Powell and Nelder-Mead from perturbed starting points on the discontinuous likelihood."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in sp.bounds(d)])
    hi = np.array([b[1] for b in sp.bounds(d)])

    def f(x):
        if np.any(x < lo) or np.any(x > hi):
            return 1e12
        p = np.zeros(NP)
        p[sp.idx] = x
        v = filt(d, p, KAPPA, sp.vtype)['nll']
        return v if np.isfinite(v) else 1e12

    best = []
    for i in range(n + 1):
        x = x0 if i == 0 else x0 * (1 + rng.normal(0, 0.15, sp.k)) + rng.normal(0, 0.01, sp.k)
        res = minimize(f, x, method='Powell', options=dict(maxiter=20000, xtol=1e-7, ftol=1e-12))
        res = minimize(f, res.x, method='Nelder-Mead', options=dict(maxiter=40000, xatol=1e-8, fatol=1e-10, adaptive=True))
        if res.fun < 1e11:
            p = np.zeros(NP)
            p[sp.idx] = res.x
            best.append((-res.fun, p[IX['l2']], p[IX['lk']], *res.x))
    return best


if __name__ == '__main__':
    d = Data(returns('SPX', END).r.values)
    out = {}
    for name in ('sign', 'ai'):
        sp = spec(name, 'garch')
        runs = []
        for s in range(20):
            f = fit(sp, d, kappa=KAPPA, n_rand=0 if s == 0 else 1, seed=s,
                    polish=(s % 2 == 0), fixedpoint=(s % 3 != 0))
            runs.append((f['logL'], f['p'][IX['l2']], f['p'][IX['lk']], *f['p'][sp.idx]))
            if s == 0:
                x0 = f['p'][sp.idx]
        runs += search(sp, d, x0, seed=0)
        out[name] = np.array(runs)
        print(name, out[name][:, 0].min().round(2), out[name][:, 0].max().round(2))
        top = np.zeros(NP)
        top[sp.idx] = out[name][out[name][:, 0].argmax(), 3:]
        out[f'profile_{name}'] = profile(sp, d, top)
    ts = two_step(d, 'garch')
    out['two_step'] = dict(sign=ts['sign']['logL'], ai=ts['ai'][0.05]['logL'], lr=ts['lr'][0.05],
                           lk=ts['ai'][0.05]['p'][IX['lk']])
    out['submitted'] = dict(sign=-9098.38, ai=-9096.45, lr=3.87, lk=0.107)
    os.makedirs('results', exist_ok=True)
    pickle.dump(out, open('results/replication.pkl', 'wb'))
    print(out['two_step'])
