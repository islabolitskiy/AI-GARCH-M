import os
import pickle
import sys

import numpy as np

from garchm import Data, NO_KAPPA, VAR, filt, fit, fit_frozen, kappa_q, returns, spec, starts_from

MODELS = [(v, m) for v in ('garch', 'gjr', 'egarch') for m in ('const', 'mlag', 'sign', 'ai05', 'ai025')]
START, STEP = '2010-01-01', 250


def forecast(r, s, e, var, kind, warm):
    vt = VAR[var][1]
    dE, dF = Data(r[:s]), Data(r[:e])
    dF.s2init = dE.s2init
    if kind in ('const', 'mlag'):
        f = fit(spec(kind, var), dE, warm=warm, n_rand=2, polish=False)
        return filt(dF, f['p'], NO_KAPPA, vt, want=('path',)), f['p']

    s1 = fit(spec('mlag', var), dE, warm=warm['s1'] if warm else None, n_rand=2, polish=False)
    zE = filt(dE, s1['p'], NO_KAPPA, vt, want=('path',))['z']
    z = filt(dF, s1['p'], NO_KAPPA, vt, want=('path',))['z']
    I = (z < 0).astype(float)
    if kind == 'sign':
        k, It, sp = NO_KAPPA, np.zeros(dF.T), spec('sign', var)
    else:
        q = {'ai05': 0.05, 'ai025': 0.025}[kind]
        k = kappa_q(zE[1:], [q])[q]
        It, sp = (z < k).astype(float), spec('ai', var)
    f = fit_frozen(sp, dE, (I[:s], It[:s]), starts_from(sp, dE, warm['m'] if warm else s1['p'], n_rand=1), kappa=k)
    return filt(dF, f['p'], k, vt, (I, It), want=('path',)), {'m': f['p'], 's1': s1['p']}


if __name__ == '__main__':
    index = sys.argv[1]
    df = returns(index)
    r = df.r.values
    s0 = int(np.argmax(df.date.values >= np.datetime64(START)))
    T = len(r)
    F = {m: dict(m=np.full(T, np.nan), s2=np.full(T, np.nan)) for m in MODELS}
    warm = {m: None for m in MODELS}
    for s in range(s0, T, STEP):
        e = min(s + STEP, T)
        for m in MODELS:
            path, warm[m] = forecast(r, s, e, m[0], m[1], warm[m])
            F[m]['m'][s:e] = r[s:e] - path['e'][s:e]
            F[m]['s2'][s:e] = path['s2'][s:e]
    hist = np.r_[np.nan, np.cumsum(r)[:-1] / np.arange(1, T)]
    os.makedirs('results', exist_ok=True)
    pickle.dump(dict(F=F, r=r, s0=s0, hist=hist, dates=df.date.values), open(f'results/oos_{index}.pkl', 'wb'))
    print(index, 'done')
