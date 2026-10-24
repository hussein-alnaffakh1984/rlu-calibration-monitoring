"""Step 6. Effect of test volume on PBRTQC: streams thinned at random to 50 % and 25 % (20 replicates each).
Parameters fixed at the main (winsorised) choice; limits re-derived on each thinned year-1 stream.
Biases of +/-20 % and +/-30 % persist for 60 days; detection = first alarm within 60 days (clean start points).
Output: volume.csv
"""
import json
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import PBRTQC, CUT, load_data, pbrtqc_stream, simulate_bias, out

D = load_data(); CH = json.load(open(out('pbrtqc_chosen.json')))


def job(a):
    t, frac, rep = a
    x = pbrtqc_stream(D, t)
    if frac < 1:
        x = x[np.random.default_rng(rep).random(len(x)) < frac].reset_index(drop=True)
    tune = x[x.run < CUT]; h = len(tune); per_day = len(x) / max(1, (x.run.max() - x.run.min()).days)
    rows = []
    for kind in ['MA', 'MM', 'EWMA', 'CUSUM']:
        try:
            det = PBRTQC(kind, CH[f'{t}|winsor|{kind}']).fit_transform(tune).calibrate(tune)
        except ValueError:      # too few year-1 results for this block size after thinning
            rows += [dict(test=t, frac=frac, rep=rep, per_day=per_day, algo=kind, bias=b, Ped=np.nan, MNPed=np.nan, days=np.nan,
                          n_sim=0, eligible=0) for b in [-30, -20, 20, 30]]
            continue
        sm = simulate_bias(det, x, h, len(x), [-30, -20, 20, 30], 60, seed=1000 + rep, horizon_days=60)
        for r in sm.to_dict('records'):
            rows.append(dict(test=t, frac=frac, rep=rep, per_day=per_day, algo=kind, **r))
    return rows


if __name__ == '__main__':
    jobs = [(t, 1.0, 0) for t in ['FERR', 'TSH']] + [(t, f, s) for t in ['FERR', 'TSH'] for f in [0.5, 0.25] for s in range(20)]
    with Pool(2) as p:
        res = sum(p.map(job, jobs), [])
    df = pd.DataFrame(res); df.to_csv(out('volume.csv'), index=False)
    print(df.groupby(['test', 'frac', 'algo']).agg(per_day=('per_day', 'mean'), Ped=('Ped', 'mean'), days=('days', 'median'),
                                                   eligible=('eligible', 'median')).round(2).to_string())
