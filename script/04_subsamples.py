import os
import pickle

from garchm import Data, IX, inference, returns, se, two_step


def estimate(r, var):
    d = Data(r)
    ts = two_step(d, var, qs=(0.025, 0.05))
    sg = inference(ts['sign'], d)
    out = dict(T=d.T, l2=sg['p'][IX['l2']], se_l2=se(sg, 'l2'), LR_l2=2 * (ts['s1']['nll'] - sg['nll']))
    for q in (0.025, 0.05):
        f = inference(ts['ai'][q], d)
        out[f'lk{q}'], out[f'se_lk{q}'], out[f'LR{q}'] = f['p'][IX['lk']], se(f, 'lk'), ts['lr'][q]
    return out


if __name__ == '__main__':
    res = {}
    for index in ['SPX', 'IXIC', 'DJI']:
        df = returns(index)
        year = df.date.dt.year
        for var in ['garch', 'gjr']:
            res[(index, var, '2000-2012')] = estimate(df.r[year <= 2012].values, var)
            res[(index, var, '2013-2026')] = estimate(df.r[year >= 2013].values, var)
            if index == 'SPX':
                for y in range(2000, 2027):
                    res[(index, var, f'drop{y}')] = estimate(df.r[year != y].values, var)
        print(index, 'done', flush=True)
    os.makedirs('results', exist_ok=True)
    pickle.dump(res, open('results/subsamples.pkl', 'wb'))
