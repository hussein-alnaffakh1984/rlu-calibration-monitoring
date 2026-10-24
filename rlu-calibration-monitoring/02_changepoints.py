"""Step 2. Locate changes in the RLU-result relation (putative calibration or reagent-lot changes).

Outputs
  changepoints.json        breakpoints (indices into each analyte's curve stream) for merge thresholds 3/5/8/10 %
  events.csv               one row per change point (primary 5 % threshold): curve shift, signal shift,
                           shift at low/median/high signal, sex-specific curve shift, testing gap
  segments.csv             one row per segment: curve level, residual SD
  validity.json            gap analysis, alternative segmentation, cross-analyte coincidence, merge-threshold counts
"""
import json
import numpy as np, pandas as pd, ruptures as rpt
from scipy import stats
from common import (ANALYTES, WIN, PEN, load_data, curve_stream, pelt, segment_levels, cubic_fit, cubic_eval,
                    hl, block_ci, out)

THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
GAP_DAYS, GAP_WIN = 2.0, 5


def merge(x, b, min_shift):
    b = list(b)
    while len(b) > 1:
        diff = np.abs(np.diff(segment_levels(x, b))); k = int(np.argmin(diff))
        if diff[k] >= np.log1p(min_shift):
            break
        b = b[:k] + b[k + 1:]
    return b


def gap_near(gaps, i):
    """True if any inter-result gap >= GAP_DAYS occurs within GAP_WIN results of index i.
    gaps[j] = time between result j-1 and result j."""
    lo, hi = max(1, i - GAP_WIN), min(len(gaps) - 1, i + GAP_WIN)
    return bool((gaps[lo:hi + 1] >= GAP_DAYS).any())


if __name__ == '__main__':
    d = load_data(); rng = np.random.default_rng(2026)
    cps, ev_rows, seg_rows, val = {}, [], [], {'gap': {}, 'alt': {}, 'thresholds': {}}
    streams = {}
    for t in ANALYTES:
        x = curve_stream(d, t); streams[t] = x
        raw = pelt(x, PEN)
        cps[t] = {f'{int(100 * m)}': merge(x, raw, m) for m in THRESHOLDS}
        val['thresholds'][t] = {k: len(v) - 1 for k, v in cps[t].items()}
        b = cps[t]['5']
        lr, lv = np.log(x.RLU.values), np.log(x.value.values); mu, sd = lr.mean(), lr.std()
        ref = np.log(np.median(x.RLU.values)); day = x.run.dt.date.values
        gaps = np.r_[np.nan, np.diff(x.run.values).astype('timedelta64[s]').astype(float) / 86400]
        # segments
        s0 = 0
        for k, e in enumerate(b):
            beta = cubic_fit(lr[s0:e], lv[s0:e], mu, sd)
            res = lv[s0:e] - cubic_eval(beta, lr[s0:e], mu, sd)
            seg_rows.append(dict(test=t, segment=k + 1, start=x.run.iloc[s0], end=x.run.iloc[e - 1], n=e - s0,
                                 level_at_median_rlu=float(np.exp(cubic_eval(beta, ref, mu, sd))), resid_sd=float(res.std()),
                                 nF=int((x.sex.values[s0:e] == 'F').sum()), nM=int((x.sex.values[s0:e] == 'M').sum())))
            s0 = e
        # change points
        starts = [0] + b[:-1]
        for k in range(1, len(b)):
            ie = b[k - 1]; a0 = max(starts[k - 1], ie - WIN); b1 = min(b[k], ie + WIN)
            A, B = np.arange(a0, ie), np.arange(ie, b1)
            fA, fB = cubic_fit(lr[A], lv[A], mu, sd), cubic_fit(lr[B], lv[B], mu, sd)
            r = dict(test=t, event=k, time=x.run.iloc[ie], index=ie, n_before=len(A), n_after=len(B),
                     curve_shift=100 * np.expm1(cubic_eval(fB, ref, mu, sd) - cubic_eval(fA, ref, mu, sd)))
            q = np.percentile(lr[a0:b1], [10, 50, 90]); sh = 100 * np.expm1(cubic_eval(fB, q, mu, sd) - cubic_eval(fA, q, mu, sd))
            r.update(shift_P10=sh[0], shift_P50=sh[1], shift_P90=sh[2], shift_range=sh.max() - sh.min())
            for sx in ['F', 'M']:
                a_, b_ = A[x.sex.values[A] == sx], B[x.sex.values[B] == sx]
                r[f'curve_shift_{sx}'] = (100 * np.expm1(cubic_eval(cubic_fit(lr[b_], lv[b_], mu, sd), ref, mu, sd)
                                                         - cubic_eval(cubic_fit(lr[a_], lv[a_], mu, sd), ref, mu, sd))
                                          if len(a_) >= 8 and len(b_) >= 8 else np.nan)
            ss = hl(lr[A], lr[B]); lo, hi = block_ci(lr[A], lr[B], day[A], day[B], rng)
            r.update(signal_shift=100 * np.expm1(ss), signal_lo=100 * np.expm1(lo), signal_hi=100 * np.expm1(hi))
            r['gap_before'] = gaps[ie]; r['gap_near'] = gap_near(gaps, ie)
            ev_rows.append(r)
        # testing-gap base rate: share of positions with a >= 2-day gap within +/- GAP_WIN results
        pos = np.arange(1, len(x))
        base = np.mean([gap_near(gaps, i) for i in pos])
        e_t = [r for r in ev_rows if r['test'] == t]
        k_ = sum(r['gap_near'] for r in e_t)
        val['gap'][t] = dict(events=len(e_t), with_gap=int(k_), base_rate=float(base),
                             p=float(stats.binomtest(int(k_), len(e_t), base, alternative='greater').pvalue) if e_t else np.nan,
                             immediately_before=int(sum(r['gap_before'] >= GAP_DAYS for r in e_t)))
        # alternative segmentation: binary segmentation, L1 cost, residuals from the whole-period curve
        f = cubic_fit(lr, lv, mu, sd); res = lv - cubic_eval(f, lr, mu, sd)
        sig = 1.4826 * np.median(np.abs(np.diff(res))) / np.sqrt(2)
        bk = rpt.Binseg(model='l1', min_size=30, jump=1).fit(res.reshape(-1, 1)).predict(pen=3 * sig * np.log(len(res)))[:-1]
        val['alt'][t] = dict(events=len(b) - 1, alternative=len(bk), matched=int(sum(any(abs(c - r) <= 10 for c in bk) for r in b[:-1])))
        print(t, val['thresholds'][t], val['gap'][t], val['alt'][t], flush=True)
    ev = pd.DataFrame(ev_rows); ev.to_csv(out('events.csv'), index=False)
    pd.DataFrame(seg_rows).to_csv(out('segments.csv'), index=False)
    allk = sum(v['with_gap'] for v in val['gap'].values())
    # cross-analyte coincidence of change-point times (pairs from different analytes within 7 days)
    def pairs(tt):
        c = 0; ks = list(tt)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                A_ = np.asarray(tt[ks[i]], 'datetime64[s]').astype(float); B_ = np.asarray(tt[ks[j]], 'datetime64[s]').astype(float)
                c += int((np.abs(A_[:, None] - B_[None, :]) / 86400 <= 7).sum())
        return c
    obs_t = {t: ev[ev.test == t].time.values for t in ANALYTES}; obs = pairs(obs_t); null = []
    for _ in range(5000):
        null.append(pairs({t: streams[t].run.values[rng.choice(np.arange(30, len(streams[t]) - 30), len(obs_t[t]), replace=False)]
                           for t in ANALYTES}))
    null = np.array(null)
    val['coincidence'] = dict(observed=obs, expected=float(null.mean()), p=float((null >= obs).mean()))
    json.dump({t: v for t, v in cps.items()}, open(out('changepoints.json'), 'w'))
    json.dump(val, open(out('validity.json'), 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(ev.drop(columns=['index']).round(1).to_string(index=False))
    print('events', len(ev), 'median |curve shift|', ev.curve_shift.abs().median().round(1),
          'range', ev.curve_shift.min().round(1), ev.curve_shift.max().round(1))
    print('coincidence', val['coincidence'])
