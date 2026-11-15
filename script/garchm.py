import ctypes
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, norm, genextreme, genpareto, t as student_t

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libcore.so'))
_D = ctypes.POINTER(ctypes.c_double)
_lib.filt.restype = ctypes.c_double
_lib.filt.argtypes = [_D, ctypes.c_int, _D, ctypes.c_double, ctypes.c_int] + [_D] * 10 + [ctypes.c_double]
_lib.sim.restype = None
_lib.sim.argtypes = [_D, ctypes.c_int, _D, ctypes.c_double, ctypes.c_int, ctypes.c_double, _D, _D, _D]
_NULL = ctypes.cast(None, _D)

NAMES = ['mu', 'l0', 'l1', 'l2', 'lk', 'phi', 'psi', 'om', 'a', 'g', 'b']
NP = len(NAMES)
IX = {n: i for i, n in enumerate(NAMES)}
VAR = {'garch': (['om', 'a', 'b'], 0), 'gjr': (['om', 'a', 'g', 'b'], 0), 'egarch': (['om', 'a', 'g', 'b'], 1)}
MEAN = {'const': ['mu'], 'mlag': ['mu', 'l1'], 'mcon': ['mu', 'l0'],
        'sign': ['mu', 'l1', 'l2'], 'ai': ['mu', 'l1', 'l2', 'lk']}
SCALE = {'mu': 0.01, 'l0': 0.02, 'l1': 0.02, 'l2': 0.02, 'lk': 0.05, 'phi': 0.02, 'psi': 0.01,
         'om': 0.005, 'a': 0.01, 'g': 0.02, 'b': 0.005}
QGRID = np.round(np.arange(0.01, 0.1001, 0.005), 4)
NO_KAPPA = -99.0


def _ptr(a):
    return a.ctypes.data_as(_D) if a is not None else _NULL


class Data:
    def __init__(self, r):
        self.r = np.ascontiguousarray(r, float)
        self.T = len(self.r)
        self.s2init = float(np.var(self.r))


def filt(d, p, kappa=NO_KAPPA, vtype=0, frozen=None, want=('nll',), fsg=None):
    p = np.ascontiguousarray(p, float)
    T = d.T
    g = np.empty(NP) if 'grad' in want else None
    S = np.empty((T, NP)) if 'scores' in want else None
    path = 'path' in want
    s2, e, ll, I, It = (np.empty(T) for _ in range(5)) if path else (None,) * 5
    fI = fIt = None
    if frozen is not None:
        fI = np.ascontiguousarray(frozen[0], float)
        fIt = np.ascontiguousarray(frozen[1], float)
    fsg = np.ascontiguousarray(fsg, float) if fsg is not None else None
    nll = _lib.filt(_ptr(d.r), T, _ptr(p), float(kappa), int(vtype), _ptr(fI), _ptr(fIt), _ptr(fsg),
                    _ptr(g), _ptr(S), _ptr(s2), _ptr(e), _ptr(ll), _ptr(I), _ptr(It), d.s2init)
    out = dict(nll=nll)
    if g is not None:
        out['grad'] = g
    if S is not None:
        out['scores'] = S
    if path:
        out.update(s2=s2, e=e, ll=ll, I=I, It=It, z=e / np.sqrt(s2))
    return out


def simulate(z, p, kappa=NO_KAPPA, vtype=0, s2init=None):
    z = np.ascontiguousarray(z, float)
    p = np.ascontiguousarray(p, float)
    n = len(z)
    if s2init is None:
        if vtype == 0:
            s2init = p[IX['om']] / max(1 - p[IX['a']] - 0.5 * p[IX['g']] - p[IX['b']], 1e-3)
        else:
            s2init = np.exp(p[IX['om']] / (1 - p[IX['b']]))
    r, s2, e = np.empty(n), np.empty(n), np.empty(n)
    _lib.sim(_ptr(z), n, _ptr(p), float(kappa), int(vtype), float(s2init), _ptr(r), _ptr(s2), _ptr(e))
    return r, s2, e


class Spec:
    def __init__(self, name, free, vtype=0):
        self.name, self.free, self.vtype = name, list(free), vtype
        self.idx = np.array([IX[f] for f in self.free])
        self.k = len(self.free)

    def bounds(self, d):
        b = []
        for f in self.free:
            if f == 'mu':
                b.append((-1, 1))
            elif f == 'om':
                b.append((1e-7, 10 * d.s2init) if self.vtype == 0 else (-3, 3))
            elif f == 'a':
                b.append((0.0, 0.6) if self.vtype == 0 else (-1, 1))
            elif f == 'g':
                b.append((0.0, 0.8) if self.vtype == 0 else (-1, 1))
            elif f == 'b':
                b.append((0.0, 0.9999))
            elif f == 'phi':
                b.append((-0.9, 0.9))
            else:
                b.append((-5, 5))
        return b

    def default(self, d):
        p = np.zeros(NP)
        p[IX['mu']] = np.mean(d.r)
        if self.vtype == 0:
            p[IX['om']], p[IX['a']], p[IX['b']] = 0.02 * d.s2init, 0.08, 0.88
            if 'g' in self.free:
                p[IX['a']], p[IX['g']] = 0.02, 0.12
        else:
            p[IX['om']], p[IX['a']], p[IX['g']], p[IX['b']] = 0.0, 0.15, -0.1, 0.97
        return p


def spec(mean, var, extra=()):
    V, vt = VAR[var]
    return Spec(mean, MEAN[mean] + list(extra) + V, vt)


# ------------------------------------------------------------------ optimisation

def _projg(g, x, lo, hi):
    g = g.copy()
    g[(x <= lo + 1e-7) & (g > 0)] = 0
    g[(x >= hi - 1e-7) & (g < 0)] = 0
    return float(np.max(np.abs(g))) if len(g) else 0.0


def _in_bounds(sp, d, x):
    return all(lo <= v <= hi for v, (lo, hi) in zip(x, sp.bounds(d)))


def _scaled_min(f, x0, bnds, sc, maxiter=5000):
    # L-BFGS-B in scaled coordinates; smaller scales are tried if the first step overshoots
    best = None
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    for shrink in (1.0, 0.1, 0.01):
        s_ = sc * shrink
        fb = lambda u, s_=s_: (lambda v: (v[0] if v[0] < 1e8 else 1e12, v[1] * s_))(f(u * s_))
        bu = [(l / q, h / q) for (l, h), q in zip(bnds, s_)]
        res = minimize(fb, np.asarray(x0) / s_, jac=True, method='L-BFGS-B', bounds=bu,
                       options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-7, maxcor=20))
        res.x = res.x * s_
        if best is None or res.fun < best.fun:
            best = res
        if res.nit > 1 or _projg(f(res.x)[1] * sc, res.x, lo, hi) < 1e-3:
            break
    return best


def _objective(sp, d, kappa, frozen):
    def f(x):
        p = np.zeros(NP)
        p[sp.idx] = x
        o = filt(d, p, kappa, sp.vtype, frozen, want=('nll', 'grad'))
        if not np.isfinite(o['nll']):
            return 1e12, np.zeros(sp.k)
        return o['nll'], o['grad'][sp.idx]
    return f


def _lbfgs(sp, d, kappa, x0, frozen=None):
    bnds = sp.bounds(d)
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    sc = np.array([SCALE[n] for n in sp.free])
    return _scaled_min(_objective(sp, d, kappa, frozen), np.clip(x0, lo + 1e-9, hi - 1e-9), bnds, sc).x


def _nll(sp, d, kappa, x):
    p = np.zeros(NP)
    p[sp.idx] = x
    return filt(d, p, kappa, sp.vtype)['nll']


def _result(sp, d, x, nll, kappa, frozen=None):
    p = np.zeros(NP)
    p[sp.idx] = x
    return dict(spec=sp, p=p, nll=nll, logL=-nll, k=sp.k, kappa=kappa, frozen=frozen,
                aic=2 * sp.k + 2 * nll, bic=sp.k * np.log(d.T) + 2 * nll)


def fit(sp, d, kappa=NO_KAPPA, warm=None, n_rand=4, seed=0, polish=True, fixedpoint=False):
    """ML with indicators recomputed from the model's own residuals."""
    base = sp.default(d)
    if warm is not None:
        warm = np.asarray(warm, float)
        bw = base.copy()
        for f in sp.free:
            if warm[IX[f]] != 0 or f == 'mu':
                bw[IX[f]] = warm[IX[f]]
        starts = [bw[sp.idx], base[sp.idx]]
    else:
        starts = [base[sp.idx]]
    rng = np.random.default_rng(seed)
    for _ in range(n_rand):
        s0 = starts[0]
        starts.append(s0 + rng.normal(0, 0.1, len(s0)) * np.maximum(np.abs(s0), 0.02))

    best_x, best_v = None, np.inf
    for x0 in starts:
        try:
            x = _lbfgs(sp, d, kappa, x0)
        except Exception:
            continue
        v = _nll(sp, d, kappa, x)
        if fixedpoint:
            for _ in range(8):
                p = np.zeros(NP)
                p[sp.idx] = x
                o = filt(d, p, kappa, sp.vtype, want=('path',))
                xf = _lbfgs(sp, d, kappa, x, frozen=(o['I'], o['It']))
                vf = _nll(sp, d, kappa, xf)
                if vf >= v - 1e-9:
                    break
                x, v = xf, vf
        if v < best_v:
            best_x, best_v = x, v

    if polish and best_x is not None:
        f = lambda x: _nll(sp, d, kappa, x) if _in_bounds(sp, d, x) else 1e12
        res = minimize(f, best_x, method='Nelder-Mead',
                       options=dict(maxiter=4000 * sp.k, xatol=1e-9, fatol=1e-11, adaptive=True))
        if res.fun < best_v - 1e-9:
            best_x, best_v = res.x, res.fun
            x = _lbfgs(sp, d, kappa, best_x)
            v = _nll(sp, d, kappa, x)
            if v < best_v:
                best_x, best_v = x, v
    return _result(sp, d, best_x, best_v, kappa)


def fit_frozen(sp, d, frozen, starts, kappa=NO_KAPPA):
    """ML with state indicators held fixed (smooth likelihood)."""
    bnds = sp.bounds(d)
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    sc = np.array([SCALE[n] for n in sp.free])
    f = _objective(sp, d, kappa, frozen)
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lo + 1e-9, hi - 1e-9)
        try:
            res = _scaled_min(f, x0, bnds, sc)
        except Exception:
            continue
        if res.nit <= 1 and _projg(f(res.x)[1] * sc, res.x, lo, hi) > 1e-2:
            for j in range(3):
                x1 = np.clip(res.x + np.random.default_rng(j).normal(0, 0.3, sp.k) * sc, lo + 1e-9, hi - 1e-9)
                r3 = _scaled_min(f, x1, bnds, sc)
                if r3.fun < res.fun:
                    res = r3
                if r3.nit > 1:
                    break
        if best is None or res.fun < best.fun:
            best = res
    return _result(sp, d, best.x, best.fun, kappa, frozen)


def starts_from(sp, d, p_warm, n_rand=2, seed=0):
    p_warm = np.asarray(p_warm, float)
    s = [p_warm[sp.idx]]
    dflt = sp.default(d)
    dw = dflt.copy()
    for f in ('om', 'a', 'g', 'b', 'mu'):
        if p_warm[IX[f]] != 0 or f == 'mu':
            dw[IX[f]] = p_warm[IX[f]]
    s.append(dw[sp.idx])
    rng = np.random.default_rng(seed)
    for _ in range(n_rand):
        s.append(s[0] + rng.normal(0, 0.05, sp.k) * np.maximum(np.abs(s[0]), 0.02))
    return s


# ------------------------------------------------------------------ inference

def _scores(sp, d, p, kappa, frozen, fsg):
    return filt(d, p, kappa, sp.vtype, frozen, want=('scores',), fsg=fsg)['scores'][1:, sp.idx]


def _hessian(sp, d, p, kappa, frozen, fsg, h=1e-5):
    H = np.zeros((sp.k, sp.k))
    for j, i in enumerate(sp.idx):
        hp = h * max(1.0, abs(p[i]))
        pp, pm = p.copy(), p.copy()
        pp[i] += hp
        pm[i] -= hp
        gp = filt(d, pp, kappa, sp.vtype, frozen, want=('grad',), fsg=fsg)['grad'][sp.idx]
        gm = filt(d, pm, kappa, sp.vtype, frozen, want=('grad',), fsg=fsg)['grad'][sp.idx]
        H[:, j] = (gp - gm) / (2 * hp)
    return 0.5 * (H + H.T)


def inference(res, d, nw_lags=None):
    """Sandwich (Bollerslev-Wooldridge) and Newey-West covariances, indicators held at estimates."""
    sp, p, kappa = res['spec'], res['p'], res['kappa']
    path = filt(d, p, kappa, sp.vtype, res['frozen'], want=('path',))
    frozen = res['frozen'] if res['frozen'] is not None else (path['I'], path['It'])
    fsg = np.sign(path['z']) if sp.vtype == 1 else None
    S = _scores(sp, d, p, kappa, frozen, fsg)
    H = _hessian(sp, d, p, kappa, frozen, fsg)
    Hi = np.linalg.inv(H)
    B = S.T @ S
    T = S.shape[0]
    if nw_lags is None:
        nw_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Bn = B.copy()
    for l in range(1, nw_lags + 1):
        G = S[l:].T @ S[:-l]
        Bn += (1 - l / (nw_lags + 1)) * (G + G.T)
    res['cov_bw'] = Hi @ B @ Hi
    res['cov_nw'] = Hi @ Bn @ Hi
    res['se_bw'] = np.sqrt(np.maximum(np.diag(res['cov_bw']), 0))
    res['se_nw'] = np.sqrt(np.maximum(np.diag(res['cov_nw']), 0))
    res['pval'] = 2 * norm.sf(np.abs(p[sp.idx] / res['se_bw']))
    res['nw_lags'] = nw_lags
    res['path'] = path
    return res


def se(res, name):
    return res['se_bw'][res['spec'].free.index(name)]


def wald(res, names):
    ii = [res['spec'].free.index(n) for n in names]
    b = res['p'][res['spec'].idx][ii]
    W = float(b @ np.linalg.solve(res['cov_bw'][np.ix_(ii, ii)], b))
    return W, float(chi2.sf(W, len(ii)))


def score_test(d, p_null, sp_full, frozen, kappa, names=('lk',)):
    fsg = np.sign(filt(d, p_null, kappa, sp_full.vtype, frozen, want=('path',))['z']) if sp_full.vtype == 1 else None
    S = _scores(sp_full, d, p_null, kappa, frozen, fsg)
    H = _hessian(sp_full, d, p_null, kappa, frozen, fsg)
    g, Hi = S.sum(0), np.linalg.inv(H)
    R = np.zeros((len(names), sp_full.k))
    for i, n in enumerate(names):
        R[i, sp_full.free.index(n)] = 1
    a = R @ Hi @ g
    LM = float(a @ np.linalg.solve(R @ Hi @ (S.T @ S) @ Hi @ R.T, a))
    return LM, float(chi2.sf(LM, len(names)))


# ------------------------------------------------------------------ threshold

def _gev_shape(z, block):
    x = -np.asarray(z)
    nb = len(x) // block
    c, _, _ = genextreme.fit(x[:nb * block].reshape(nb, block).max(1))
    return -c


def _gpd_scale(exc, xi):
    n = len(exc)
    if abs(xi) < 1e-8:
        return float(exc.mean())
    lo = 1e-6 if xi >= 0 else max(-xi * exc.max() + 1e-6, 1e-6)
    hi = max(exc.std() * 100, lo * 1e3)

    def nll(s):
        y = 1 + xi * exc / s
        return np.inf if (y <= 0).any() else n * np.log(s) + (1 + 1 / xi) * np.log(y).sum()

    return float(minimize_scalar(nll, bounds=(lo, hi), method='bounded', options=dict(xatol=1e-8)).x)


def kappa_q(z, qs, method='evt', u_q=0.10, block=21):
    """Left-tail quantiles of standardized residuals.
    evt: GPD above the (1-u_q) quantile of -z with shape fixed at the GEV block-maxima estimate."""
    z = np.asarray(z)
    if method == 'empirical':
        return {q: float(np.quantile(z, q)) for q in qs}
    if method == 'student':
        df, loc, sc = student_t.fit(z)
        return {q: float(student_t.ppf(q, df, loc, sc)) for q in qs}
    x = -z
    T = len(z)
    u = float(np.quantile(x, 1 - u_q))
    exc = x[x > u] - u
    n_u = len(exc)
    if method == 'hill':
        xs = np.sort(x)[::-1]
        xs = xs[xs > 0]
        k = min(n_u, len(xs) - 1)
        gam = np.mean(np.log(xs[:k])) - np.log(xs[k])
        return {q: -float(xs[k] * (k / (q * T)) ** gam) if q <= u_q else float(np.quantile(z, q)) for q in qs}
    if method == 'evt':
        xi = _gev_shape(z, block)
        sc = _gpd_scale(exc, xi)
    elif method == 'gpd':
        xi, _, sc = genpareto.fit(exc, floc=0)
    out = {}
    for q in qs:
        if q > u_q + 1e-12:
            out[q] = float(np.quantile(z, q))
        elif abs(xi) < 1e-8:
            out[q] = -float(u + sc * np.log(n_u / (q * T)))
        else:
            out[q] = -float(u + sc / xi * ((q * T / n_u) ** (-xi) - 1))
    return out


# ------------------------------------------------------------------ two-step estimator

def step1(d, var, extra=(), n_rand=4, seed=0):
    f = fit(spec('mlag', var, extra), d, n_rand=n_rand, seed=seed)
    f['z'] = filt(d, f['p'], NO_KAPPA, VAR[var][1], want=('path',))['z']
    return f


def two_step(d, var, qs=(0.01, 0.025, 0.05, 0.10), extra=(), qgrid=None, s1=None, light=False, seed=0):
    """Step 1: restricted model, standardized residuals z. Step 2: sign-asymmetric and AI-GARCH-M
    with indicators 1{z<0}, 1{z<kappa} fixed; LR of the extreme-loss term at each tail level."""
    if s1 is None:
        s1 = step1(d, var, extra, n_rand=1 if light else 4, seed=seed)
    z = s1['z']
    I = (z < 0).astype(float)
    nr = 0 if light else 2
    sp0 = spec('sign', var, extra)
    f0 = fit_frozen(sp0, d, (I, np.zeros(d.T)), starts_from(sp0, d, s1['p'], nr, seed))
    out = dict(s1=s1, sign=f0, z=z, ai={}, lr={}, lr_grid={}, lk_grid={})
    out['kap'] = kappa_q(z[1:], list(qs)) if qs else {}
    spA = spec('ai', var, extra)
    for q in qs:
        k = out['kap'][q]
        fA = fit_frozen(spA, d, (I, (z < k).astype(float)), starts_from(spA, d, f0['p'], nr, seed), kappa=k)
        out['ai'][q] = fA
        out['lr'][q] = max(0.0, 2 * (f0['nll'] - fA['nll']))
    for q in (qgrid if qgrid is not None else []):
        k = float(np.quantile(z[1:], q))
        fA = fit_frozen(spA, d, (I, (z < k).astype(float)), starts_from(spA, d, f0['p'], 0, seed), kappa=k)
        out['lr_grid'][q] = max(0.0, 2 * (f0['nll'] - fA['nll']))
        out['lk_grid'][q] = fA['p'][IX['lk']]
    return out


def returns(index, end=None, path='../data/returns.csv'):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['date'])
    if end is not None:
        df = df[df.date <= end]
    return df[['date', index]].dropna().rename(columns={index: 'r'}).reset_index(drop=True)
