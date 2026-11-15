import glob
import os
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else '../data/raw'
names = {'GSPC': 'SPX'}

series = {}
for f in sorted(glob.glob(os.path.join(src, '*.xlsx'))):
    ticker = os.path.basename(f)[:-5]
    px = pd.read_excel(f)
    px.columns = [c.split('.')[-1] for c in px.columns]
    px['Date'] = pd.to_datetime(px['Date']).dt.normalize()
    # Yahoo dates some Friday closes of the ASX 200 as the following Sunday
    if ticker == 'AXJO':
        sun = px['Date'].dt.weekday == 6
        px.loc[sun, 'Date'] -= pd.Timedelta(days=2)
    px = px.dropna(subset=['Close']).drop_duplicates('Date', keep='last').sort_values('Date')
    c = px.set_index('Date')['Close']
    series[names.get(ticker, ticker)] = 100 * np.log(c).diff().dropna()

r = pd.DataFrame(series)
r.index.name = 'date'
r = r[['SPX', 'IXIC', 'DJI'] + sorted(c for c in r.columns if c not in ('SPX', 'IXIC', 'DJI'))]
r.to_csv('../data/returns.csv')
print(r.notna().sum().to_string())
