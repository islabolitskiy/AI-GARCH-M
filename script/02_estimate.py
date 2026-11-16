import os
import pickle
import sys

import numpy as np
from scipy.stats import chi2, kurtosis, norm, skew

from garchm import (Data, IX, VAR, fit, fit_frozen, inference, returns, score_test, spec, starts_from,
                    two_step, wald)

QS = (0.01, 0.025, 0.05, 0.10)


def ljung_box(x, lags):
    x = x - x.mean()
    n = len(x)
    ac = np.array([(x[k:] @ x[:-k]) / (x @ x) for k in range(1, lags + 1)])
    Q = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lags + 1)))
    return Q, chi2.sf(Q, lags)


def engle_ng(z):
    y, zl = z[1:] ** 2, z[:-1]
    neg = (zl < 0).astype(float)
    X = np.column_stack([np.ones_like(y), neg, neg * zl, (1 - neg) * zl])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    t = b / np.sqrt(np.diag(XtXi @ (X.T * e ** 2) @ X @ XtXi))
    lm = len(y) * (1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))
    return dict(sign_t=t[1], neg_size_t=t[2], pos_size_t=t[3], joint=lm, joint_p=chi2.sf(lm, 3))


def diagnostics(res):
    z, e = res['path']['z'][1:], res['path']['e'][1:]
    return dict(skew=skew(z), exkurt=kurtosis(z), LBz10=ljung_box(z, 10), LBz20=ljung_box(z, 20),
                LBz2_10=ljung_box(z ** 2, 10), LBz2_20=ljung_box(z ** 2, 20), EN=engle_ng(z),
                p_neg=(z < 0).mean(), eps_mean=e.mean())


def encompassing(d, var, z, kappa, extra):
    I = (z < 0).astype(float)
    base = inference(fit(spec('mlag', var, extra), d, n_rand=3), d)
    s0 = spec('sign', var, extra)
    sg = inference(fit_frozen(s0, d, (I, np.zeros(d.T)), starts_from(s0, d, base['p'])), d)
    sA = spec('ai', var, extra)
    ai = inference(fit_frozen(sA, d, (I, (z < kappa).astype(float)), starts_from(sA, d, sg['p']), kappa), d)
    return dict(base=base, sign=sg, ai=ai, LR_l2=2 * (base['nll'] - sg['nll']), LR_lk=2 * (sg['nll'] - ai['nll']),
                W_l2=wald(sg, ['l2']), W_lk=wald(ai, ['lk']), W_both=wald(ai, ['l2', 'lk']))


def estimate(index, end=None):
    df = returns(index, end)
    r = df.r.values
    d = Data(r)
    out = dict(index=index, T=d.T, start=str(df.date.min().date()), end=str(df.date.max().date()),
               desc=dict(mean=r.mean(), sd=r.std(ddof=1), skew=skew(r), exkurt=kurtosis(r)),
               fam={}, two={})
    for var in VAR:
        fam = {k: inference(fit(spec(k, var), d), d) for k in ('const', 'mlag', 'mcon')}
        ts = two_step(d, var, qs=QS)
        z, I = ts['z'], (ts['z'] < 0).astype(float)
        fam['sign'] = inference(ts['sign'], d)
        for q in QS:
            inference(ts['ai'][q], d)
        fam['ai05'], fam['ai025'] = ts['ai'][0.05], ts['ai'][0.025]

        tests = {}
        for q in QS:
            fA = ts['ai'][q]
            W, pW = wald(fA, ['lk'])
            LM, pLM = score_test(d, ts['sign']['p'], fA['spec'], fA['frozen'], fA['kappa'])
            tests[q] = dict(LR=ts['lr'][q], pLR=chi2.sf(ts['lr'][q], 1), W=W, pW=pW, LM=LM, pLM=pLM,
                            lk=fA['p'][IX['lk']], kap=ts['kap'][q], n_tail=int((z[1:-1] < ts['kap'][q]).sum()),
                            aic=fA['aic'])

        alt = {}
        sA = spec('ai', var)
        for name, k in (('empirical', float(np.quantile(z[1:], 0.05))), ('gauss', float(norm.ppf(0.05)))):
            f = inference(fit_frozen(sA, d, (I, (z < k).astype(float)), starts_from(sA, d, ts['sign']['p']), k), d)
            alt[name] = dict(kap=k, lk=f['p'][IX['lk']], LR=2 * (ts['sign']['nll'] - f['nll']))

        enc = {'phi': encompassing(d, var, z, ts['kap'][0.05], ['phi']),
               'phipsi': encompassing(d, var, z, ts['kap'][0.05], ['phi', 'psi'])}
        if var != 'garch':
            enc['l0phipsi'] = encompassing(d, var, z, ts['kap'][0.05], ['l0', 'phi', 'psi'])

        out['fam'][var] = fam
        out['two'][var] = dict(kap=ts['kap'], tests=tests, lr_l2=2 * (ts['s1']['nll'] - ts['sign']['nll']),
                               thresholds=alt, enc=enc,
                               diag={'mlag': diagnostics(fam['mlag']), 'ai05': diagnostics(fam['ai05'])})
        print(index, var, 'done', flush=True)
    return out


if __name__ == '__main__':
    index = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs('results', exist_ok=True)
    res = estimate(index, end)
    pickle.dump(res, open(f'results/estimates_{index}.pkl', 'wb'))
