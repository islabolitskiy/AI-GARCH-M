import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from garchm import IX

INDICES = ['SPX', 'IXIC', 'DJI']
VARS = ['garch', 'gjr', 'egarch']
NAME = {'SPX': 'S\\&P 500', 'IXIC': 'Nasdaq', 'DJI': 'Dow Jones'}
VNAME = {'garch': 'GARCH', 'gjr': 'GJR', 'egarch': 'EGARCH'}
PAR = {'mu': '\\mu', 'l0': '\\lambda_0', 'l1': '\\lambda_1', 'l2': '\\lambda_2', 'lk': '\\lambda_\\kappa',
       'phi': '\\phi', 'psi': '\\psi', 'om': '\\omega', 'a': '\\alpha', 'g': '\\gamma', 'b': '\\beta'}
ENC_FULL = {'garch': 'phipsi', 'gjr': 'l0phipsi', 'egarch': 'l0phipsi'}


def load(path):
    return pickle.load(open(path, 'rb'))


def stars(p):
    return '^{***}' if p < .01 else '^{**}' if p < .05 else '^{*}' if p < .1 else ''


def coef(res, name, dec=3):
    i = res['spec'].free.index(name)
    b, s = res['p'][IX[name]], res['se_bw'][i]
    return f'${b:.{dec}f}{stars(2 * norm.sf(abs(b / s)))}$', f'$({s:.{dec}f})$'


def boot_p(index, var):
    B = load(f'results/boot_{index}_{var}.pkl')
    sup = np.array([r['sup'] for r in B['rows']])
    return B['obs']['sup'], (1 + (sup >= B['obs']['sup']).sum()) / (len(sup) + 1), np.quantile(sup, 0.95), B


def table(env, caption, colspec, head, rows, note, label=None):
    L = [f'\\begin{{{env}}}[t]\\centering\\small', f'\\caption{{{caption}}}' + (f'\\label{{{label}}}' if label else ''),
         '\\begin{threeparttable}', f'\\begin{{tabular}}{{{colspec}}}\\toprule'] + head + rows + \
        ['\\bottomrule\\end{tabular}', f'\\begin{{tablenotes}}\\footnotesize\\item {note}\\end{{tablenotes}}',
         '\\end{threeparttable}', f'\\end{{{env}}}']
    return '\n'.join(L)


def supp_table(caption, colspec, head, rows, note):
    return '\n'.join(['\\begin{table}[htbp]\\centering\\small', f'\\caption{{{caption}}}',
                      f'\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{{colspec}}}\\toprule'] + head + rows +
                     ['\\bottomrule\\end{tabular}}', f'\\par\\footnotesize {note}', '\\end{table}'])


# ---------------------------------------------------------------- main text

def table1(E):
    rows = ['\\multicolumn{7}{l}{\\textit{Panel A. Fit}} \\\\']
    for lab, k in [('No premium', 'const'), ('GARCH-M, $\\lambda_1\\sigma_{t-1}^2$', 'mlag'),
                   ('GARCH-M, $\\lambda_0\\sigma_t^2$', 'mcon'), ('Sign-asymmetric ($\\lambda_\\kappa=0$)', 'sign'),
                   ('AI-GARCH-M, $q=2.5\\%$', 'ai025'), ('AI-GARCH-M, $q=5\\%$', 'ai05')]:
        rows.append(lab + ' & ' + ' & '.join(f"${E['fam'][v][k]['logL']:.2f}$ & ${E['fam'][v][k]['aic']:.2f}$"
                                             for v in VARS) + ' \\\\')
    rows.append('\\midrule \\multicolumn{7}{l}{\\textit{Panel B. Estimates (sandwich standard errors)}} \\\\')
    for lab, k, n in [('$\\lambda_1$, sign-asymmetric', 'sign', 'l1'), ('$\\lambda_2$, sign-asymmetric', 'sign', 'l2'),
                      ('$\\lambda_\\kappa$, $q=2.5\\%$', 'ai025', 'lk'), ('$\\lambda_\\kappa$, $q=5\\%$', 'ai05', 'lk'),
                      ('$\\lambda_0$, GARCH-M ($\\sigma_t^2$)', 'mcon', 'l0')]:
        rows.append(lab + ' & ' + ' & '.join(' & '.join(coef(E['fam'][v][k], n)) for v in VARS) + ' \\\\')
    pers = []
    for v in VARS:
        p = E['fam'][v]['sign']['p']
        x = p[IX['b']] if v == 'egarch' else p[IX['a']] + p[IX['g']] / 2 + p[IX['b']]
        pers.append(f'\\multicolumn{{2}}{{c}}{{${x:.3f}$}}')
    rows.append('Persistence, sign-asymmetric & ' + ' & '.join(pers) + ' \\\\')
    head = ['& \\multicolumn{2}{c}{GARCH} & \\multicolumn{2}{c}{GJR} & \\multicolumn{2}{c}{EGARCH} \\\\ '
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
            'Mean equation & $\\log L$ & AIC & $\\log L$ & AIC & $\\log L$ & AIC \\\\ \\midrule']
    note = ('Daily log returns (\\%), 4 January 2000--30 October 2026, $T=6747$. Gaussian QML; in the '
            'sign-asymmetric and AI-GARCH-M models the state indicators are fixed at their step-1 values (Section 3). '
            'Persistence is $\\alpha+\\gamma/2+\\beta$ for GARCH and GJR and $\\beta$ for EGARCH. '
            '$^{***}$, $^{**}$, $^{*}$: significance at 1\\%, 5\\%, 10\\%.')
    return table('table*', 'S\\&P 500: model comparison and price-of-variance estimates', 'lrrrrrr', head, rows, note, 'tab:sp')


def table2(EE):
    rows = []
    for index in INDICES:
        E = EE[index]
        for j, v in enumerate(VARS):
            two = E['two'][v]
            b2, s2 = coef(E['fam'][v]['sign'], 'l2')
            k1, sk1 = coef(E['fam'][v]['ai025'], 'lk')
            k2, sk2 = coef(E['fam'][v]['ai05'], 'lk')
            sup, p, _, _ = boot_p(index, v)
            full = two['enc'][ENC_FULL[v]]
            rows.append(f"{NAME[index] if j == 0 else ''} & {VNAME[v]} & {b2} & ${two['lr_l2']:.2f}$ & {k1} & {k2} & "
                        f"${sup:.2f}$ & ${p:.3f}$ & ${two['enc']['phi']['W_l2'][1]:.3f}$ & ${full['W_l2'][1]:.3f}$ & "
                        f"${full['W_lk'][1]:.3f}$ \\\\")
            rows.append(f' & & {s2} & & {sk1} & {sk2} & & & & & \\\\')
        if index != INDICES[-1]:
            rows.append('\\midrule')
    head = ['& & \\multicolumn{2}{c}{Downside term} & \\multicolumn{4}{c}{Extreme-loss term} & '
            '\\multicolumn{3}{c}{Encompassing (3), Wald $p$} \\\\',
            '\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}',
            'Index & Variance & $\\hat\\lambda_2$ & LR & $\\hat\\lambda_\\kappa^{2.5\\%}$ & $\\hat\\lambda_\\kappa^{5\\%}$ & '
            'sup-LR & $p_{\\text{boot}}$ & $\\lambda_2\\mid\\phi$ & $\\lambda_2$ & $\\lambda_\\kappa$ \\\\ \\midrule']
    note = ('$\\hat\\lambda_2$: sign-asymmetric model; LR: test of $\\lambda_2=0$ against the GARCH-M with lagged variance '
            '($\\chi^2_1$ 5\\% critical value 3.84). $\\hat\\lambda_\\kappa^{q}$: AI-GARCH-M at tail level $q$; sandwich '
            'standard errors in parentheses. sup-LR: maximum of $\\mathrm{LR}(q)$ over $q\\in\\{1\\%,1.5\\%,\\dots,10\\%\\}$; '
            '$p_{\\text{boot}}$: null-imposed residual bootstrap rerunning both steps ($B=999$ for the S\\&P 500, 499 otherwise). '
            'Last three columns: robust Wald $p$-values in the encompassing mean equation (3), for $\\lambda_2$ when only the '
            'reversal term $\\phi\\varepsilon_{t-1}$ is added, and for $\\lambda_2$ and $\\lambda_\\kappa$ ($q=5\\%$) when '
            '$\\lambda_0\\sigma_t^2$, $\\phi\\varepsilon_{t-1}$ and $\\psi\\varepsilon_{t-1}^2$ are all added (under GARCH, '
            '$\\sigma_t^2$ is a linear combination of $1$, $\\varepsilon_{t-1}^2$ and $\\sigma_{t-1}^2$, so $\\lambda_0$ is dropped).')
    return table('table*', 'Downside and extreme-loss terms across indices and variance equations', 'llrrrrrrrrr',
                 head, rows, note, 'tab:tests')


def nw_var(x, L=5):
    x = x - x.mean()
    v = x @ x / len(x)
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / len(x)
    return v


def clark_west(y, m0, m1):
    f = (y - m0) ** 2 - ((y - m1) ** 2 - (m0 - m1) ** 2)
    return norm.sf(f.mean() / np.sqrt(nw_var(f) / len(f)))


def diebold_mariano(l1, l0):
    x = l1 - l0
    return x.mean() / np.sqrt(nw_var(x) / len(x))


def oos_stats(index):
    O = load(f'results/oos_{index}.pkl')
    F, r, s0, hist = O['F'], O['r'], O['s0'], O['hist']
    sl = slice(s0, len(r))
    y, h = r[sl], hist[sl]
    ex = pd.to_datetime(O['dates'][sl]).year != 2020
    qlike = lambda m: np.log(F[m]['s2'][sl]) + y ** 2 / F[m]['s2'][sl]
    logscore = lambda m: -0.5 * (np.log(2 * np.pi) + np.log(F[m]['s2'][sl]) + (y - F[m]['m'][sl]) ** 2 / F[m]['s2'][sl])
    q0 = qlike(('gjr', 'const'))
    out = {}
    for v in VARS:
        for k, nest in (('mlag', None), ('sign', 'mlag'), ('ai05', 'sign')):
            m = F[(v, k)]['m'][sl]
            out[(v, k)] = dict(R2=1 - ((y - m) ** 2).sum() / ((y - h) ** 2).sum(),
                               R2x=1 - ((y - m)[ex] ** 2).sum() / ((y - h)[ex] ** 2).sum(),
                               p_hist=clark_west(y, h, m),
                               p_nest=clark_west(y, F[(v, nest)]['m'][sl], m) if nest else np.nan,
                               dql=(qlike((v, k)) - q0).mean(), dm=diebold_mariano(qlike((v, k)), q0),
                               ls=logscore((v, k)).mean())
        if v != 'gjr':
            out[(v, 'var')] = dict(dql=(qlike((v, 'const')) - q0).mean(), dm=diebold_mariano(qlike((v, 'const')), q0))
    out['n'], out['start'] = len(y), str(pd.Timestamp(O['dates'][s0]).date())
    return out


def tab_b9(OO):
    rows = []
    for index in INDICES:
        S = OO[index]
        for j, (k, lab) in enumerate([('mlag', 'GARCH-M'), ('sign', 'Sign-asymmetric'), ('ai05', 'AI-GARCH-M, $q=5\\%$')]):
            x = S[('gjr', k)]
            nest = '---' if np.isnan(x['p_nest']) else f"${x['p_nest']:.3f}$"
            rows.append(f"{NAME[index] if j == 0 else ''} & {lab} & ${100 * x['R2']:.2f}$ & ${100 * x['R2x']:.2f}$ & "
                        f"${x['p_hist']:.3f}$ & {nest} & ${100 * x['dql']:.3f}$ & ${x['dm']:.2f}$ \\\\")
        for v in ('garch', 'egarch'):
            x = S[(v, 'var')]
            rows.append(f" & \\textit{{{VNAME[v]} variance}} & & & & & ${100 * x['dql']:.3f}$ & ${x['dm']:.2f}$ \\\\")
        if index != INDICES[-1]:
            rows.append('\\midrule')
    head = ['& & \\multicolumn{4}{c}{Returns} & \\multicolumn{2}{c}{Variance} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-8}',
            'Index & Mean equation & $R^2_{OS}$ & excl.\\ 2020 & $p_{CW}^{\\,\\bar r}$ & $p_{CW}^{\\,\\text{nest}}$ & '
            '$\\Delta$QLIKE & DM \\\\ \\midrule']
    note = (f"Expanding window, parameters re-estimated every 250 trading days, {OO['SPX']['n']} forecasts from 4 January 2010. "
            '$R^2_{OS}$ (\\%): out-of-sample $R^2$ against the prevailing historical mean \\citep{campbell2008predicting}, '
            'for all days and excluding 2020. $p_{CW}$: one-sided \\citet{clark2007approximately} $p$-values against the '
            'prevailing mean ($\\bar r$) and against the nested model (GARCH-M for the sign-asymmetric model, the '
            'sign-asymmetric model for AI-GARCH-M). $\\Delta$QLIKE: mean QLIKE loss ($\\log\\sigma_t^2+r_t^2/\\sigma_t^2$) '
            'relative to GJR with constant mean, $\\times100$; DM: Diebold--Mariano $t$-statistic (Newey--West, 5 lags), '
            'negative values favour the row model. The last two rows of each panel compare variance equations with a constant mean.')
    return supp_table('Out-of-sample forecasts for the U.S. indices, 2010--2026 (GJR variance)', 'llrrrrrr', head, rows, note)


def pooled_cw(v, a, b, markets=None):
    cols = []
    for m in markets or MARKETS:
        O = load(f'results/oos_{m}.pkl')
        F, r, s0 = O['F'], O['r'], O['s0']
        sl = slice(s0, len(r))
        y = r[sl]
        m0 = O['hist'][sl] if a is None else F[(v, a)]['m'][sl]
        m1 = F[(v, b)]['m'][sl]
        f = ((y - m0) ** 2 - ((y - m1) ** 2 - (m0 - m1) ** 2)) / y.var()
        cols.append(pd.Series(f, index=pd.to_datetime(O['dates'][sl]), name=m))
    x = pd.concat(cols, axis=1, sort=True).mean(axis=1).values
    return norm.sf(x.mean() / np.sqrt(nw_var(x) / len(x)))


def oos_rows(OO, v):
    rows = []
    for m in MARKETS:
        S = OO[m]
        g, x, a = S[(v, 'mlag')], S[(v, 'sign')], S[(v, 'ai05')]
        dm = f"${S[('garch', 'var')]['dm']:.2f}$" if v == 'gjr' else ''
        rows.append(f"{MNAME[m]} & ${100 * g['R2']:.2f}$ & ${100 * x['R2']:.2f}$ & ${100 * x['R2x']:.2f}$ & "
                    f"${x['p_hist']:.3f}$ & ${x['p_nest']:.3f}$ & ${100 * a['R2']:.2f}$ & ${a['p_nest']:.3f}$" +
                    (f" & {dm}" if v == 'gjr' else '') + ' \\\\')
    R2 = np.array([OO[m][(v, 'sign')]['R2'] for m in MARKETS])
    cnt = lambda k, key: int((np.array([OO[m][(v, k)][key] for m in MARKETS]) < 0.05).sum())
    rows.append('\\midrule')
    rows.append(f"Positive $R^2_{{OS}}$ / $p<0.05$ & & {int((R2 > 0).sum())} & & {cnt('sign', 'p_hist')} & {cnt('sign', 'p_nest')} & & "
                f"{cnt('ai05', 'p_nest')}" + (f" & {sum(OO[m][('garch', 'var')]['dm'] > 1.96 for m in MARKETS)}" if v == 'gjr' else '') + ' \\\\')
    rows.append(f"Pooled Clark--West $p$ & & & & ${pooled_cw(v, None, 'sign'):.3f}$ & ${pooled_cw(v, 'mlag', 'sign'):.3f}$ & & "
                f"${pooled_cw(v, 'sign', 'ai05'):.3f}$" + (' & ' if v == 'gjr' else '') + ' \\\\')
    ex = [m for m in MARKETS if m not in ('SPX', 'IXIC', 'DJI', 'RUT')]
    rows.append(f"\\quad excluding U.S. & & & & ${pooled_cw(v, None, 'sign', ex):.3f}$ & ${pooled_cw(v, 'mlag', 'sign', ex):.3f}$ & & "
                f"${pooled_cw(v, 'sign', 'ai05', ex):.3f}$" + (' & ' if v == 'gjr' else '') + ' \\\\')
    return rows


OOS_HEAD = ('Index & GARCH-M & $R^2_{OS}$ & excl.\\ 2020 & $p_{CW}^{\\,\\bar r}$ & $p_{CW}^{\\,\\text{nest}}$ & '
            '$R^2_{OS}$ & $p_{CW}^{\\,\\text{nest}}$')
OOS_NOTE = ('Expanding window from January 2010, parameters re-estimated every 250 trading days (4106--4307 forecasts). '
            '$R^2_{OS}$ (\\%): out-of-sample $R^2$ against the prevailing historical mean \\citep{campbell2008predicting}, '
            'for all days and excluding 2020 for the sign-asymmetric model. $p_{CW}$: one-sided \\citet{clark2007approximately} '
            '$p$-values against the prevailing mean ($\\bar r$) and against the nested model (GARCH-M for the sign-asymmetric '
            'model, the sign-asymmetric model for AI-GARCH-M). Pooled Clark--West: the adjusted loss differentials, scaled by '
            'the variance of returns, are averaged across indices on each date and tested with Newey--West standard errors; '
            'the last row excludes the four U.S. indices.')


def table4(OO):
    head = ['& & \\multicolumn{4}{c}{Sign-asymmetric} & \\multicolumn{2}{c}{AI-GARCH-M, $q=5\\%$} & Variance \\\\ '
            '\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}',
            OOS_HEAD + ' & DM \\\\ \\midrule']
    note = (OOS_NOTE + ' DM: Diebold--Mariano $t$-statistic (Newey--West, 5 lags) for the QLIKE loss of GARCH relative to '
            'GJR, both with constant mean, positive values favour GJR; the count row gives the number of indices with $t>1.96$.')
    tex = table('table*', 'Out-of-sample one-day-ahead forecasts, 2010--2026 (GJR variance)', 'lrrrrrrrr',
                head, oos_rows(OO, 'gjr'), note, 'tab:oos')
    return tex.replace('\\small', '\\scriptsize')


def tab_b10(OO):
    out = []
    for v in ('garch', 'egarch'):
        head = ['& & \\multicolumn{4}{c}{Sign-asymmetric} & \\multicolumn{2}{c}{AI-GARCH-M, $q=5\\%$} \\\\ '
                '\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}', OOS_HEAD + ' \\\\ \\midrule']
        out.append(supp_table(f'Out-of-sample forecasts for the 20 indices, {VNAME[v]} variance', 'lrrrrrrr',
                              head, oos_rows(OO, v), OOS_NOTE))
    return '\n\n'.join(out)


# ---------------------------------------------------------------- supplement

def tab_b1(E):
    cols = [(v, k) for v in VARS for k in ('sign', 'ai05')]
    rows = []
    for n in ['mu', 'l1', 'l2', 'lk', 'om', 'a', 'g', 'b']:
        r1, r2, r3 = [f'${PAR[n]}$'], [''], ['']
        for v, k in cols:
            f = E['fam'][v][k]
            if n in f['spec'].free:
                i = f['spec'].free.index(n)
                b, s, s2 = f['p'][IX[n]], f['se_bw'][i], f['se_nw'][i]
                r1.append(f'${b:.4f}{stars(2 * norm.sf(abs(b / s)))}$')
                r2.append(f'$({s:.4f})$')
                r3.append(f'$[{s2:.4f}]$')
            else:
                r1.append('---'); r2.append(''); r3.append('')
        rows += [' & '.join(x) + ' \\\\' for x in (r1, r2, r3)]
    rows.append('\\midrule $\\log L$ & ' + ' & '.join(f"${E['fam'][v][k]['logL']:.2f}$" for v, k in cols) + ' \\\\')
    rows.append('$\\hat\\kappa$ & ' + ' & '.join('---' if k == 'sign' else f"${E['two'][v]['kap'][0.05]:.3f}$"
                                                for v, k in cols) + ' \\\\')
    head = [' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{VNAME[v]}}}' for v in VARS) + ' \\\\',
            ' & ' + ' & '.join('Sign' if k == 'sign' else 'AI, $q=5\\%$' for _, k in cols) + ' \\\\ \\midrule']
    note = (f"Bollerslev--Wooldridge sandwich standard errors in parentheses, Newey--West (Bartlett kernel, "
            f"{E['fam']['garch']['sign']['nw_lags']} lags) in brackets; both hold the state indicators at their step-1 values. "
            'EGARCH: $\\log\\sigma_t^2=\\omega+\\beta\\log\\sigma_{t-1}^2+\\alpha(|z_{t-1}|-\\mathbb{E}|z|)+\\gamma z_{t-1}$. '
            'Stars refer to the sandwich standard errors.')
    return supp_table('S\\&P 500: full coefficient estimates', 'l' + 'c' * len(cols), head, rows, note)


def tab_b2(EE):
    rows = []
    for index in INDICES:
        for v in VARS:
            t = EE[index]['two'][v]['tests']
            B = load(f'results/boot_{index}_{v}.pkl')
            for j, q in enumerate([0.01, 0.025, 0.05, 0.10]):
                x = t[q]
                lrb = np.array([r[f'lr_{q}'] for r in B['rows']])
                pb = (1 + (lrb >= B['obs']['lr'][q]).sum()) / (len(lrb) + 1)
                rows.append(f"{NAME[index] if (v == 'garch' and j == 0) else ''} & {VNAME[v] if j == 0 else ''} & "
                            f"{100 * q:g}\\% & ${x['kap']:.3f}$ & {x['n_tail']} & ${x['lk']:.3f}$ & ${x['LR']:.2f}$ & "
                            f"${x['pLR']:.3f}$ & ${pb:.3f}$ & ${x['pW']:.3f}$ & ${x['pLM']:.3f}$ \\\\")
        rows.append('\\midrule')
    rows.pop()
    head = ['Index & Var. & $q$ & $\\hat\\kappa_q$ & $N_q$ & $\\hat\\lambda_\\kappa$ & LR & $p_{\\chi^2}$ & '
            '$p_{\\text{boot}}$ & $p_{W}$ & $p_{LM}$ \\\\ \\midrule']
    note = ('$\\hat\\kappa_q$: EVT threshold (Section 3); $N_q$: number of days on which the extreme-loss indicator enters the '
            'mean equation. LR: AI-GARCH-M against the sign-asymmetric model. $p_{\\chi^2}$: $\\chi^2_1$ reference; '
            '$p_{\\text{boot}}$: pointwise null-imposed residual bootstrap rerunning both steps; $p_W$, $p_{LM}$: robust Wald '
            'and score tests based on the sandwich covariance.')
    return supp_table('Pointwise tests of $\\lambda_\\kappa=0$ at fixed tail levels', 'llrrrrrrrrr', head, rows, note)


def tab_b3(EE):
    rows = []
    for index in INDICES:
        for j, v in enumerate(VARS):
            t = EE[index]['two'][v]
            x, e, g = t['tests'][0.05], t['thresholds']['empirical'], t['thresholds']['gauss']
            rows.append(f"{NAME[index] if j == 0 else ''} & {VNAME[v]} & ${x['kap']:.3f}$ & ${x['lk']:.3f}$ & ${x['LR']:.2f}$ & "
                        f"${e['kap']:.3f}$ & ${e['lk']:.3f}$ & ${e['LR']:.2f}$ & ${g['lk']:.3f}$ & ${g['LR']:.2f}$ \\\\")
    head = ['& & \\multicolumn{3}{c}{EVT} & \\multicolumn{3}{c}{Empirical quantile} & \\multicolumn{2}{c}{Gaussian, $-1.645$} \\\\',
            '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}',
            'Index & Var. & $\\hat\\kappa$ & $\\hat\\lambda_\\kappa$ & LR & $\\hat\\kappa$ & $\\hat\\lambda_\\kappa$ & LR & '
            '$\\hat\\lambda_\\kappa$ & LR \\\\ \\midrule']
    note = ('EVT: generalized Pareto fit to the 10\\% largest values of $-\\hat z_t$ with shape fixed at the GEV estimate from '
            '21-day block maxima; empirical: 5\\% quantile of $\\hat z_t$; Gaussian: $\\Phi^{-1}(0.05)$. LR against the '
            'sign-asymmetric model.')
    return supp_table('Alternative thresholds at $q=5\\%$', 'llrrrrrrrr', head, rows, note)


def tab_b4(EE):
    rows = []
    for index in INDICES:
        for v in VARS:
            for j, (k, lab) in enumerate([('mlag', 'GARCH-M'), ('ai05', 'AI, $q=5\\%$')]):
                d = EE[index]['two'][v]['diag'][k]
                rows.append(f"{NAME[index] if (v == 'garch' and j == 0) else ''} & {VNAME[v] if j == 0 else ''} & {lab} & "
                            f"${d['skew']:.2f}$ & ${d['exkurt']:.2f}$ & ${d['LBz10'][1]:.3f}$ & ${d['LBz20'][1]:.3f}$ & "
                            f"${d['LBz2_20'][1]:.3f}$ & ${d['EN']['sign_t']:.2f}$ & ${d['EN']['joint_p']:.3f}$ \\\\")
        rows.append('\\midrule')
    rows.pop()
    head = ['Index & Var. & Model & Skew. & Ex.kurt. & $Q_{z}(10)$ & $Q_{z}(20)$ & $Q_{z^2}(20)$ & Sign bias & '
            'Joint EN \\\\ \\midrule']
    note = ('Standardized residuals $\\hat z_t$. $Q$: Ljung--Box $p$-values. Sign bias: robust $t$-statistic on '
            '$\\mathbb{I}\\{\\hat z_{t-1}<0\\}$ in the \\citet{engle1993measuring} regression of $\\hat z_t^2$; Joint EN: '
            '$p$-value of the joint sign- and size-bias test.')
    return supp_table('Residual diagnostics', 'lllrrrrrrr', head, rows, note)


def tab_b5(S):
    rows = []
    for index in INDICES:
        for v in ('garch', 'gjr'):
            for j, per in enumerate(['2000-2012', '2013-2026']):
                o = S[(index, v, per)]
                rows.append(f"{NAME[index] if (v == 'garch' and j == 0) else ''} & {VNAME[v] if j == 0 else ''} & "
                            f"{per.replace('-', '--')} & ${o['l2']:.3f}$ $({o['se_l2']:.3f})$ & ${o['LR_l2']:.2f}$ & "
                            f"${o['lk0.025']:.3f}$ $({o['se_lk0.025']:.3f})$ & ${o['LR0.025']:.2f}$ & "
                            f"${o['lk0.05']:.3f}$ $({o['se_lk0.05']:.3f})$ & ${o['LR0.05']:.2f}$ \\\\")
        rows.append('\\midrule')
    for v in ('garch', 'gjr'):
        D = [o for k, o in S.items() if k[0] == 'SPX' and k[1] == v and k[2].startswith('drop')]
        rng = lambda key, dec: f"${min(o[key] for o in D):.{dec}f}$--${max(o[key] for o in D):.{dec}f}$"
        rows.append(f"\\multicolumn{{3}}{{l}}{{S\\&P 500, {VNAME[v]}, leave one year out}} & {rng('l2', 3)} & "
                    f"{rng('LR_l2', 1)} & {rng('lk0.025', 3)} & {rng('LR0.025', 1)} & {rng('lk0.05', 3)} & {rng('LR0.05', 1)} \\\\")
    head = ['Index & Var. & Period & $\\hat\\lambda_2$ & LR($\\lambda_2$) & $\\hat\\lambda_\\kappa^{2.5\\%}$ & LR & '
            '$\\hat\\lambda_\\kappa^{5\\%}$ & LR \\\\ \\midrule']
    note = ('Two-step estimates on each subsample (sandwich standard errors in parentheses). The last rows give the range over '
            '27 re-estimations that each drop one calendar year (returns before and after the dropped year are concatenated).')
    return supp_table('Subsamples', 'lllrrrrrr', head, rows, note)


def tab_b6(E):
    tags = {'garch': ['phi', 'phipsi'], 'gjr': ['phi', 'l0phipsi'], 'egarch': ['phi', 'l0phipsi']}
    lab = {'phi': '$+\\phi$', 'phipsi': '$+\\phi,\\psi$', 'l0phipsi': '$+\\lambda_0,\\phi,\\psi$'}
    cols = [(v, t) for v in VARS for t in tags[v]]
    rows = []
    for n in ['l0', 'l1', 'l2', 'lk', 'phi', 'psi']:
        r1, r2 = [f'${PAR[n]}$'], ['']
        for v, t in cols:
            f = E['two'][v]['enc'][t]['ai']
            if n in f['spec'].free:
                b, s = coef(f, n)
                r1.append(b); r2.append(s)
            else:
                r1.append('---'); r2.append('')
        rows += [' & '.join(r1) + ' \\\\', ' & '.join(r2) + ' \\\\']
    for name, key in (('LR($\\lambda_2$)', 'LR_l2'), ('LR($\\lambda_\\kappa$)', 'LR_lk')):
        rows.append(name + ' & ' + ' & '.join(f"${E['two'][v]['enc'][t][key]:.2f}$" for v, t in cols) + ' \\\\')
    rows.append('Wald $p$($\\lambda_2,\\lambda_\\kappa$) & ' +
                ' & '.join(f"${E['two'][v]['enc'][t]['W_both'][1]:.3f}$" for v, t in cols) + ' \\\\')
    head = [' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{VNAME[v]}}}' for v in VARS) + ' \\\\',
            ' & ' + ' & '.join(lab[t] for _, t in cols) + ' \\\\ \\midrule']
    note = ('Mean equation $r_t=\\mu+\\lambda_0\\sigma_t^2+(\\lambda_1+\\lambda_2I_{t-1}+\\lambda_\\kappa\\tilde I_{t-1})'
            '\\sigma_{t-1}^2+\\phi\\varepsilon_{t-1}+\\psi\\varepsilon_{t-1}^2+\\varepsilon_t$, $q=5\\%$. Sandwich standard errors in '
            'parentheses. LR($\\lambda_2$): sign-asymmetric against the same equation without indicators; LR($\\lambda_\\kappa$): '
            'AI-GARCH-M against sign-asymmetric. Under GARCH, $\\sigma_t^2$ is an exact linear combination of $1$, '
            '$\\varepsilon_{t-1}^2$ and $\\sigma_{t-1}^2$, so $\\lambda_0$ is not separately identified and is omitted.')
    return supp_table('S\\&P 500: encompassing mean equation (3)', 'l' + 'c' * len(cols), head, rows, note)


def tab_c1():
    cv = np.quantile([r['sup'] for r in load('results/boot_SPX_garch.pkl')['rows']], 0.95)
    label = {'null05': '$\\lambda_\\kappa=0$, $q^*=5\\%$', 'alt05': '$\\lambda_\\kappa=0.088$, $q^*=5\\%$',
             'alt025': '$\\lambda_\\kappa=0.183$, $q^*=2.5\\%$', 'orig': 'Original design$^{a}$'}
    rows = []
    for cell in ['null05', 'alt05', 'alt025', 'orig']:
        M = load(f'results/mc_{cell}.pkl')
        df = pd.DataFrame(M['rows'])
        lt = M['p'][IX['lk']]
        if cell == 'alt05':
            label[cell] = f'$\\lambda_\\kappa={lt:.3f}$, $q^*=5\\%$'
        if cell == 'alt025':
            label[cell] = f'$\\lambda_\\kappa={lt:.3f}$, $q^*=2.5\\%$'
        for j, (tag, name) in enumerate([('evt', 'EVT'), ('empirical', 'empirical'), ('true', 'true')]):
            lk, s, LR = df[f'lk_{tag}'], df[f'se_{tag}'], df[f'LR_{tag}']
            sup = f'${np.mean(df["sup"] > cv):.3f}$' if (j == 0 and cell != 'orig') else ''
            rows.append(f"{label[cell] if j == 0 else ''} & {name} & ${lk.mean():.3f}$ & ${lk.std():.3f}$ & "
                        f"${np.sqrt(((lk - lt) ** 2).mean()):.3f}$ & ${s.mean():.3f}$ & ${np.mean(LR > 3.841):.3f}$ & "
                        f"${np.mean((lk / s) ** 2 > 3.841):.3f}$ & {sup} \\\\")
        rows.append('\\midrule')
    rows.pop()
    head = ['Design & Threshold & Mean $\\hat\\lambda_\\kappa$ & SD & RMSE & Mean SE & Rej.\\ LR & Rej.\\ Wald & '
            'Rej.\\ sup-LR \\\\ \\midrule']
    note = ('500 replications (300 for the original design). DGP: the AI-GARCH-M model at the S\\&P 500 GARCH estimates, '
            'innovations resampled from the standardized S\\&P residuals; the true threshold is the $q^*$-quantile of the '
            'innovation pool. Each replication reruns step 1, the threshold estimation and step 2 at $q=q^*$. SE: sandwich '
            'standard error. Rejection rates at the nominal 5\\% level: LR and Wald with $\\chi^2_1$ critical values; sup-LR '
            f'over $q\\in\\{{1\\%,\\dots,10\\%\\}}$ with the 95\\% bootstrap critical value from the S\\&P 500 GARCH null '
            f'bootstrap ({cv:.2f}). $^{{a}}$Design of the original submission: $\\mu=0$, $\\lambda_1=\\lambda_2=0.1$, '
            '$\\lambda_\\kappa=0.5$, $\\omega=0.05$, $\\alpha=0.05$, $\\beta=0.85$ ($\\mathbb{E}[\\sigma_t^2]=0.5$), Gaussian '
            'innovations, $q^*=2.5\\%$, $T=5000$.')
    return supp_table('Calibrated Monte Carlo (S\\&P 500, GARCH variance, $T=6747$)', 'llrrrrrrr', head, rows, note)


def tab_c2():
    methods = [('empirical', 'Empirical'), ('evt', 'EVT (ours)'), ('gpd', 'GPD'), ('hill', 'Hill'),
               ('student', 'Student-$t$'), ('evt_b63', 'EVT, 63-day'), ('evt_u05', 'EVT, $u$=5\\%'),
               ('evt_u15', 'EVT, $u$=15\\%')]
    rows = []
    for dist in ['normal', 't5']:
        for q in [0.01, 0.025, 0.05]:
            for j, T in enumerate([1000, 2500, 5000, 20000]):
                M = load(f'results/mc_thr_{dist}_{T}.pkl')
                df = pd.DataFrame(M['rows'])
                cells = []
                for m, _ in methods:
                    c = f'{m}_{q}'
                    if c in df:
                        e = df[c] - M['true'][q]
                        cells.append(f'${np.sqrt((e ** 2).mean()):.3f}$ $({e.mean():+.3f})$')
                    else:
                        cells.append('---')
                first = ('Normal' if dist == 'normal' else '$t_5$') if (q == 0.01 and j == 0) else ''
                qlab = '%g\\%%' % (100 * q) if j == 0 else ''
                rows.append(f'{first} & {qlab} & {T} & ' + ' & '.join(cells) + ' \\\\')
            rows.append('\\midrule')
    rows.pop()
    head = ['Innov. & $q$ & $T$ & ' + ' & '.join(m[1] for m in methods) + ' \\\\ \\midrule']
    note = ('300 replications per cell. DGP: GARCH-M ($\\mu=0$, $\\lambda_1=0.1$, $\\omega=0.05$, $\\alpha=0.05$, '
            '$\\beta=0.85$); thresholds estimated from the standardized residuals of the fitted model. Empirical: sample '
            '$q$-quantile. EVT (ours): GPD fitted to the 10\\% largest values of $-\\hat z_t$ with shape fixed at the GEV '
            'estimate from 21-day block maxima. GPD: unconstrained fit \\citep{mcneil2000estimation}. Hill: Hill/Weissman '
            'estimator with the same 10\\% of order statistics. Student-$t$: quantile of a location-scale Student-$t$ fitted '
            'by ML. The last three columns vary the block length and the exceedance cutoff $u$.')
    return supp_table('Threshold estimators: RMSE (bias in parentheses) of $\\hat\\kappa_q$', 'lll' + 'r' * len(methods),
                      head, rows, note)


def tab_d1(R):
    s, a, sub, two = R['sign'], R['ai'], R['submitted'], R['two_step']
    rows = [f"Submitted version & ${sub['sign']:.2f}$ & ${sub['ai']:.2f}$ & ${sub['lr']:.2f}$ & ${sub['lk']:.3f}$ \\\\",
            f"Self-referential indicators, lowest optimum found & ${s[:, 0].min():.2f}$ & ${a[:, 0].min():.2f}$ & & \\\\",
            f"Self-referential indicators, highest optimum found & ${s[:, 0].max():.2f}$ & ${a[:, 0].max():.2f}$ & "
            f"${2 * (a[:, 0].max() - s[:, 0].max()):.2f}$ & ${a[a[:, 0].argmax(), 2]:.3f}$ \\\\",
            f"Two-step, indicators fixed at step-1 values & ${two['sign']:.2f}$ & ${two['ai']:.2f}$ & ${two['lr']:.2f}$ & "
            f"${two['lk']:.3f}$ \\\\"]
    head = ['Estimator & $\\log L$ sign-asym. & $\\log L$ AI ($q=5\\%$) & LR & $\\hat\\lambda_\\kappa$ \\\\ \\midrule']
    note = ('With indicators recomputed from the residuals of the model being estimated, the quasi-likelihood is '
            'discontinuous in the parameters, and different starting values and optimizers (L-BFGS-B with and without '
            'indicator fixed-point iterations, Powell, Nelder--Mead) stop at different points. Fixing the indicators at '
            'their step-1 values makes the likelihood smooth.')
    return supp_table('Submitted results, self-referential indicators and the two-step estimator (S\\&P 500, GARCH, '
                      'sample to 8 May 2026)', 'lrrrr', head, rows, note)


# ---------------------------------------------------------------- international sample

MARKETS = ['SPX', 'IXIC', 'DJI', 'RUT', 'GSPTSE', 'BVSP', 'MXX', 'FTSE', 'GDAXI', 'FCHI', 'AEX', 'SSMI', 'IBEX',
           'N225', 'HSI', 'AXJO', 'KS11', 'BSESN', 'TWII', 'STI']
MNAME = {'SPX': 'S\\&P 500', 'IXIC': 'Nasdaq', 'DJI': 'Dow Jones', 'RUT': 'Russell 2000', 'GSPTSE': 'S\\&P/TSX',
         'BVSP': 'Bovespa', 'MXX': 'IPC', 'FTSE': 'FTSE 100', 'GDAXI': 'DAX', 'FCHI': 'CAC 40', 'AEX': 'AEX',
         'SSMI': 'SMI', 'IBEX': 'IBEX 35', 'N225': 'Nikkei 225', 'HSI': 'Hang Seng', 'AXJO': 'S\\&P/ASX 200',
         'KS11': 'KOSPI', 'BSESN': 'BSE Sensex', 'TWII': 'TAIEX', 'STI': 'Straits Times'}


def panel_boot(var):
    P = load(f'results/panel_boot_{var}.pkl')
    D = np.array([P['draws'][b] for b in sorted(P['draws'])])
    G = D[:, :, 0, :] if D.ndim == 4 else D
    qgrid = sorted(P['nulls'][MARKETS[0]]['lr_grid'])
    obs = np.array([[P['nulls'][m]['lr_grid'][q] for q in qgrid] for m in MARKETS])
    ok = ~np.isnan(G).any(axis=(1, 2))
    G = G[ok]
    sup_b, sup_o = G.max(2), obs.max(1)
    p_m = (1 + (sup_b >= sup_o).sum(0)) / (len(G) + 1)
    pool_b, pool_o = G.sum(1).max(1), obs.sum(0).max()
    p_pool = (1 + (pool_b >= pool_o).sum()) / (len(G) + 1)
    out = dict(sup=sup_o, p=p_m, pool=pool_o, p_pool=p_pool, B=len(G), q_pool=qgrid[int(obs.sum(0).argmax())],
               cv_pool=np.quantile(pool_b, 0.95), qgrid=qgrid, obs=obs)
    if D.ndim == 4 and 'lk_grid' in P['nulls'][MARKETS[0]]:
        K = D[ok][:, :, 1, :]
        ko = np.array([[P['nulls'][m]['lk_grid'][q] for q in qgrid] for m in MARKETS])
        n = len(MARKETS)
        sg_b = (np.sign(K) * np.sqrt(np.maximum(G, 0))).sum(1).max(1) / np.sqrt(n)
        sg_o = (np.sign(ko) * np.sqrt(np.maximum(obs, 0))).sum(0) / np.sqrt(n)
        out.update(sgn=sg_o.max(), q_sgn=qgrid[int(sg_o.argmax())],
                   p_sgn=(1 + (sg_b >= sg_o.max()).sum()) / (len(G) + 1), lk=ko,
                   npos=int((ko[:, int(obs.sum(0).argmax())] > 0).sum()))
    return out


def table_panel(EP):
    PB = {v: panel_boot(v) for v in VARS}
    rows, d2 = [], []
    for i, m in enumerate(MARKETS):
        E = EP[m]
        f, a = E['fam']['gjr']['sign'], E['fam']['gjr']['ai025']
        e = E['two']['gjr']['enc']['phi']
        b2, s2 = coef(f, 'l2')
        bk, sk = coef(a, 'lk')
        bphi = e['sign']['p'][IX['l2']]
        ch = a['p'][IX['l2']] - f['p'][IX['l2']]
        d2.append(ch)
        sups = ' & '.join(f"${PB[v]['sup'][i]:.2f}$ $({PB[v]['p'][i]:.3f})$" for v in VARS)
        rows.append(f"{MNAME[m]} & {b2} {s2} & ${E['two']['gjr']['lr_l2']:.2f}$ & "
                    f"${bphi:.3f}{stars(e['W_l2'][1])}$ & ${ch:+.3f}$ & {bk} {sk} & {sups} \\\\")
    rows.append('\\midrule')
    l2 = np.array([EP[m]['fam']['gjr']['sign']['p'][IX['l2']] for m in MARKETS])
    l2p = np.array([EP[m]['two']['gjr']['enc']['phi']['sign']['p'][IX['l2']] for m in MARKETS])
    pp = np.array([EP[m]['two']['gjr']['enc']['phi']['W_l2'][1] for m in MARKETS])
    lk = np.array([EP[m]['fam']['gjr']['ai025']['p'][IX['lk']] for m in MARKETS])
    sig = sum(EP[m]['two']['gjr']['lr_l2'] > 3.841 and l2[i] > 0 for i, m in enumerate(MARKETS))
    rows.append(f"Mean & ${l2.mean():.3f}$ & & ${l2p.mean():.3f}$ & ${np.mean(d2):+.3f}$ & ${lk.mean():.3f}$ & & & \\\\")
    rows.append(f"Positive (significant) & {int((l2 > 0).sum())} ({sig}) & & {int((l2p > 0).sum())} ({int(((l2p > 0) & (pp < 0.05)).sum())}) & & "
                f"{int((lk > 0).sum())} & " + ' & '.join(f"{int((PB[v]['p'] < 0.05).sum())}" for v in VARS) + ' \\\\')
    rows.append('Pooled sup-LR & & & & & & ' + ' & '.join(f"${PB[v]['pool']:.2f}$ $({PB[v]['p_pool']:.3f})$" for v in VARS) + ' \\\\')
    rows.append('Tail level of the maximum & & & & & & ' + ' & '.join(f"${100 * PB[v]['q_pool']:g}\\%$" for v in VARS) + ' \\\\')
    head = ['& \\multicolumn{5}{c}{GJR variance} & \\multicolumn{3}{c}{sup-LR ($p_{\\text{boot}}$)} \\\\',
            '\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}',
            'Index & $\\hat\\lambda_2$ & LR & $\\hat\\lambda_2\\mid\\phi$ & $\\Delta\\hat\\lambda_2$ & $\\hat\\lambda_\\kappa^{2.5\\%}$ & GARCH & GJR & EGARCH \\\\ \\midrule']
    note = (f"$\\hat\\lambda_2$: sign-asymmetric model, sandwich standard errors in parentheses; LR: test of $\\lambda_2=0$; "
            f"$\\hat\\lambda_2\\mid\\phi$: the same coefficient when the reversal term $\\phi\\varepsilon_{{t-1}}$ is added to the mean equation, "
            f"with stars from the robust Wald test; $\\Delta\\hat\\lambda_2$: change of $\\hat\\lambda_2$ when the extreme-loss term "
            f"($q=2.5\\%$) is added; $\\hat\\lambda_\\kappa^{{2.5\\%}}$: AI-GARCH-M. sup-LR over $q\\in\\{{1\\%,1.5\\%,\\dots,10\\%\\}}$ "
            f"with $p$-values from a joint null-imposed bootstrap that resamples whole dates across markets ({PB['garch']['B']}, "
            f"{PB['gjr']['B']} and {PB['egarch']['B']} draws). Pooled sup-LR: maximum over $q$ of the sum of $\\mathrm{{LR}}(q)$ "
            "across the 20 markets. In the row Positive (significant) the sup-LR columns give the number of indices with $p<0.05$. "
            "$^{*}$, $^{**}$, $^{***}$: significance at 10\\%, 5\\% and 1\\%.")
    tex = table('table*', 'International evidence: 20 stock indices, 2000--2026', 'lrrrrrrrr', head, rows, note, 'tab:panel')
    return tex.replace('\\small', '\\scriptsize\\setlength{\\tabcolsep}{2.5pt}'), PB


def tab_b7(EP):
    rows = []
    for m in MARKETS:
        E = EP[m]
        cells = []
        for v in ('garch', 'egarch'):
            b2, _ = coef(E['fam'][v]['sign'], 'l2')
            bk, _ = coef(E['fam'][v]['ai025'], 'lk')
            cells += [b2, f"${E['two'][v]['lr_l2']:.2f}$", bk, f"${E['two'][v]['tests'][0.025]['LR']:.2f}$",
                      f"${E['two'][v]['enc']['phi']['W_l2'][1]:.3f}$"]
        rows.append(f"{MNAME[m]} & " + ' & '.join(cells) + ' \\\\')
    head = ['& \\multicolumn{5}{c}{GARCH} & \\multicolumn{5}{c}{EGARCH} \\\\ \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}',
            'Index & $\\hat\\lambda_2$ & LR & $\\hat\\lambda_\\kappa^{2.5\\%}$ & LR & $p(\\lambda_2\\mid\\phi)$ & '
            '$\\hat\\lambda_2$ & LR & $\\hat\\lambda_\\kappa^{2.5\\%}$ & LR & $p(\\lambda_2\\mid\\phi)$ \\\\ \\midrule']
    note = ('Two-step estimates. LR statistics for $\\lambda_2$ (against the GARCH-M with lagged variance) and for $\\lambda_\\kappa$ '
            'at $q=2.5\\%$ (against the sign-asymmetric model). $p(\\lambda_2\\mid\\phi)$: robust Wald $p$-value of $\\lambda_2$ when the '
            'reversal term $\\phi\\varepsilon_{t-1}$ is added to the mean equation.')
    return supp_table('International sample under GARCH and EGARCH variance', 'l' + 'r' * 10, head, rows, note)


def tab_b8(PB):
    R = load('results/panel_robust.pkl')
    qs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]
    qgrid = [round(q, 4) for q in PB['garch']['qgrid']]
    cols = []
    for v in VARS:
        for sample in ('full', 'drop2020'):
            if sample == 'full':
                lr = PB[v]['obs']
                lk = PB[v]['lk'] if 'lk' in PB[v] else np.array([R[(v, m, 'full')][1] for m in MARKETS])
            else:
                A = np.array([R[(v, m, sample)] for m in MARKETS])
                lr, lk = A[:, 0], A[:, 1]
            cols.append((lr, lk))
    rows = []
    for q in qs:
        j = qgrid.index(round(q, 4))
        rows.append(f"${100 * q:g}\\%$ & " + ' & '.join(f"${lr[:, j].sum():.1f}$ $({int((lk[:, j] > 0).sum())})$" for lr, lk in cols) + ' \\\\')
    rows.append('\\midrule')
    rows.append('Maximum over $q$ & ' + ' & '.join(f"${lr.sum(0).max():.1f}$" for lr, lk in cols) + ' \\\\')
    rows.append('95\\% critical value & ' + ' & '.join(f"${PB[v]['cv_pool']:.1f}$ & " for v in VARS)[:-3] + ' \\\\')
    head = ['& \\multicolumn{2}{c}{GARCH} & \\multicolumn{2}{c}{GJR} & \\multicolumn{2}{c}{EGARCH} \\\\ '
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
            '$q$ & Full & Without 2020 & Full & Without 2020 & Full & Without 2020 \\\\ \\midrule']
    note = ('Sum of $\\mathrm{LR}(q)$ for the extreme-loss term across the 20 indices and, in parentheses, the number of indices '
            'with $\\hat\\lambda_\\kappa>0$. ``Without 2020\'\' re-estimates every model with calendar year 2020 removed. '
            'The critical value is the 95\\% quantile of the pooled sup-LR statistic under the joint bootstrap of the full sample.')
    return supp_table('International sample: pooled LR statistics by tail level', 'lrrrrrr', head, rows, note)


# ---------------------------------------------------------------- numbers quoted in the text

def numbers(EE, OO, S, R):
    L = []
    E = EE['SPX']
    for v in ('garch', 'gjr'):
        f = E['fam'][v]['ai05']
        p, z = f['p'], f['path']['z'][1:]
        pm, cm = (z < 0).mean(), np.mean(z ** 2 * (z < 0))
        m = p[IX['om']] / (1 - p[IX['a']] - p[IX['g']] * cm - p[IX['b']])
        I, It = f['frozen']
        s2 = f['path']['s2']
        tail = s2[np.where(It[:-1] == 1)[0]]
        L.append(f"SPX {v} AI(5%): m={m:.3f}; implied E[r]={p[IX['mu']] + m * (p[IX['l1']] + p[IX['l2']] * pm + p[IX['lk']] * 0.05):.4f} "
                 f"(sample {E['desc']['mean']:.4f}); mu={p[IX['mu']]:.4f}, baseline={m * p[IX['l1']]:.4f}, "
                 f"downside={m * p[IX['l2']] * pm:.4f}, extreme={m * p[IX['lk']] * 0.05:.4f}; mean residual="
                 f"{f['path']['e'][1:].mean():.4f}; median variance on tail days={np.median(tail):.2f}; "
                 f"next-day effect: l2={p[IX['l2']] * np.median(tail):.3f}, lk={p[IX['lk']] * np.median(tail):.3f}")
    for index in INDICES:
        for v in VARS:
            sup, p, cv, B = boot_p(index, v)
            size = np.mean([r['lr_0.05'] > 3.841 for r in B['rows']])
            L.append(f'{index} {v}: sup-LR={sup:.2f}, p={p:.3f}, cv95={cv:.2f}, null rejection of pointwise LR(5%)={size:.3f}')
    for v in ('garch', 'gjr'):
        D = [o for k, o in S.items() if k[0] == 'SPX' and k[1] == v and k[2].startswith('drop')]
        yrs = [k[2][4:] for k in S if k[0] == 'SPX' and k[1] == v and k[2].startswith('drop')]
        lr = [o['LR0.05'] for o in D]
        L.append(f'SPX {v} leave-one-year-out: LR(5%) min {min(lr):.2f} (drop {yrs[int(np.argmin(lr))]}), max {max(lr):.2f}')
    for index in INDICES:
        for k in ('mlag', 'sign', 'ai05'):
            x = OO[index][('gjr', k)]
            L.append(f"OOS {index} gjr {k}: R2={100 * x['R2']:.2f}% excl2020={100 * x['R2x']:.2f}% p_hist={x['p_hist']:.3f} "
                     f"p_nest={x['p_nest']:.3f} logscore={x['ls']:.4f}")
    pr = R['profile_ai']
    L.append('Likelihood along lambda_2 around the best AI optimum (step, logL): ' +
             ', '.join(f'{h:+.4f}:{v:.2f}' for h, v in pr))
    a, s = R['ai'], R['sign']
    L.append(f"Replication: LR range {R['submitted']['lr']:.2f} to {2 * (a[:, 0].max() - s[:, 0].max()):.2f}; "
             f"gap to best: sign {s[:, 0].max() - R['submitted']['sign']:.2f}, AI {a[:, 0].max() - R['submitted']['ai']:.2f}")
    return '\n'.join(L)


def panel_numbers(EP, PB):
    import pandas as pd
    df = pd.read_csv('../data/returns.csv')
    ac = np.array([np.corrcoef(df[m].dropna().values[1:], df[m].dropna().values[:-1])[0, 1] for m in MARKETS])
    L = []
    for v in VARS:
        l2 = np.array([EP[m]['fam'][v]['sign']['p'][IX['l2']] for m in MARKETS])
        l2p = np.array([EP[m]['two'][v]['enc']['phi']['sign']['p'][IX['l2']] for m in MARKETS])
        pp = np.array([EP[m]['two'][v]['enc']['phi']['W_l2'][1] for m in MARKETS])
        lr = np.array([EP[m]['two'][v]['lr_l2'] for m in MARKETS])
        d = np.array([EP[m]['fam'][v]['ai025']['p'][IX['l2']] for m in MARKETS]) - l2
        sig = lr > 3.841
        P = PB[v]
        L.append(f"Panel {v}: l2>0 {int((l2 > 0).sum())}, sig+ {int((sig & (l2 > 0)).sum())}, sig- {int((sig & (l2 < 0)).sum())}; "
                 f"corr(l2, AC1)={np.corrcoef(l2, ac)[0, 1]:.2f}; l2|phi>0 {int((l2p > 0).sum())}, sig5% {int(((l2p > 0) & (pp < .05)).sum())}, "
                 f"mean {l2.mean():.3f}->{l2p.mean():.3f}; |dl2| mean {np.abs(d).mean():.4f}, max among sig+ {np.abs(d[sig & (l2 > 0)]).max():.4f}; "
                 f"single sup p<0.05: {[MARKETS[i] for i in np.where(P['p'] < .05)[0]]}; pooled {P['pool']:.2f} p={P['p_pool']:.3f} "
                 f"q={P['q_pool']} cv95={P['cv_pool']:.2f} B={P['B']}"
                 + (f"; signed {P['sgn']:.2f} p={P['p_sgn']:.3f} q={P['q_sgn']} npos@qpool {P['npos']}" if 'sgn' in P else ''))
    L.append('AC1: ' + ', '.join(f'{m} {a:.3f}' for m, a in zip(MARKETS, ac)))
    return '\n'.join(L)


if __name__ == '__main__':
    out = '../paper/tables'
    os.makedirs(out, exist_ok=True)
    EE = {i: load(f'results/estimates_{i}.pkl') for i in INDICES}
    OO = {m: oos_stats(m) for m in MARKETS}
    S = load('results/subsamples.pkl')
    R = load('results/replication.pkl')
    EP = {m: load(f'results/estimates_{m}.pkl') for m in MARKETS}
    tab_p, PB = table_panel(EP)
    T = dict(tab1=table1(EE['SPX']), tab2=table2(EE), tab3=tab_p, tab4=table4(OO),
             tabB1=tab_b1(EE['SPX']), tabB2=tab_b2(EE), tabB3=tab_b3(EE), tabB4=tab_b4(EE), tabB5=tab_b5(S),
             tabB6=tab_b6(EE['SPX']), tabB7=tab_b7(EP), tabB8=tab_b8(PB), tabB9=tab_b9(OO),
             tabC1=tab_c1(), tabC2=tab_c2(), tabD1=tab_d1(R))
    T['tabB10'], T['tabB11'] = tab_b10(OO).split('\n\n')
    for name, tex in T.items():
        open(f'{out}/{name}.tex', 'w').write(tex + '\n')
    txt = numbers(EE, OO, S, R) + '\n' + panel_numbers(EP, PB) + '\n'
    open('results/numbers.txt', 'w').write(txt)
    print(txt)
