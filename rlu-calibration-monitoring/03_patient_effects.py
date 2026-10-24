"""Step 3. How far did each change in the RLU-result relation reach patient results?

Patient shift: Hodges-Lehmann shift of log results (all numeric, uncensored results) in up to 150 results
before vs after each change point (windows stop at neighbouring change points); day-level block-bootstrap
95% CI; two-sided day-level permutation p (10,000 permutations); Benjamini-Hochberg q across all change points.
Category: q < 0.05 'shift detected'; CI within +/-10 % 'compensated'; otherwise 'indeterminate'.

Transmission (primary analytes FERR, TSH, VD-T II): curve shift vs patient shift (Spearman, sign test,
OLS slope with bootstrap CI, Theil-Sen, per-analyte slopes, model with analyte intercepts and interaction test).
Decomposition: patient shift ~ curve shift + beta * signal shift, with beta = local d log(result)/d log(RLU).
Sensitivity: windows of 50/100 results, placebo cut points, merge thresholds 3/8/10 %, TSH Nov-Dec 2024 excluded.
Classification change between calibrations at decision limits for change points with a detected shift.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from common import (ANALYTES, PRIMARY, WIN, load_data, curve_stream, cubic_fit, cubic_eval, local_slope,
                    hl, block_ci, perm_p, out)
import warnings
warnings.filterwarnings('ignore')

LIMITS = {'FERR': [('<', 15), ('<', 30)], 'TSH': [('>', 5.1), ('<', 0.35)], 'VD-T II': [('<', 20), ('<', 12)]}
TSH_REVERSALS = ('2024-11-17', '2024-12-11')


def patient_stream(d, t):
    x = d[(d.test == t) & (d.censor == '') & d.value.notna() & (d.value > 0)]
    return x.sort_values('run').reset_index(drop=True)


def windows(x, times, W):
    idx = [int(np.searchsorted(x.run.values, np.datetime64(s))) for s in times]
    for k, ie in enumerate(idx):
        pv = idx[k - 1] if k else 0; nx = idx[k + 1] if k + 1 < len(idx) else len(x)
        yield k, np.arange(max(pv, ie - W), ie), np.arange(ie, min(nx, ie + W))


def curve_shift_for(xc, b, k):
    """log curve shift at change point k (1-based) from breakpoints b, as in step 2."""
    lr, lv = np.log(xc.RLU.values), np.log(xc.value.values); mu, sd = lr.mean(), lr.std(); ref = np.log(np.median(xc.RLU.values))
    starts = [0] + b[:-1]; ie = b[k - 1]; A = np.arange(max(starts[k - 1], ie - WIN), ie); B = np.arange(ie, min(b[k], ie + WIN))
    return cubic_eval(cubic_fit(lr[B], lv[B], mu, sd), ref, mu, sd) - cubic_eval(cubic_fit(lr[A], lv[A], mu, sd), ref, mu, sd), A, B


def transmission(c, r, rng):
    """c, r: log curve shift and log patient shift."""
    rho, p = stats.spearmanr(c, r); same = int((np.sign(c) == np.sign(r)).sum())
    slope = np.polyfit(c, r, 1)[0]
    bs = [np.polyfit(c[i], r[i], 1)[0] for i in (rng.integers(0, len(c), len(c)) for _ in range(5000))]
    ts = stats.theilslopes(r, c)
    return dict(n=len(c), spearman=rho, spearman_p=p, same_sign=same, sign_p=stats.binomtest(same, len(c), .5).pvalue,
                ols_slope=slope, ols_lo=np.percentile(bs, 2.5), ols_hi=np.percentile(bs, 97.5),
                theil_sen=ts[0], ts_lo=ts[2], ts_hi=ts[3])


if __name__ == '__main__':
    d = load_data(); rng = np.random.default_rng(31)
    ev = pd.read_csv(out('events.csv'), parse_dates=['time']); cps = json.load(open(out('changepoints.json')))
    rows, ws, placebo = [], [], []
    for t in ANALYTES:
        x = patient_stream(d, t); xc = curve_stream(d, t); e = ev[ev.test == t].reset_index(drop=True)
        day = x.run.dt.date.values; lv = np.log(x.value.values)
        for k, A, B in windows(x, e.time.values, WIN):
            ee = e.iloc[k]; s = hl(lv[A], lv[B]); lo, hi = block_ci(lv[A], lv[B], day[A], day[B], rng)
            p = perm_p(lv[A], lv[B], day[A], day[B], rng, reps=10000)
            Af, Bf = A[x.sex.values[A] == 'F'], B[x.sex.values[B] == 'F']
            a0, b1 = int(ee['index']) - int(ee.n_before), int(ee['index']) + int(ee.n_after)
            rows.append(dict(test=t, event=k + 1, time=ee.time, n_before=len(A), n_after=len(B), curve_shift=ee.curve_shift,
                             signal_shift=ee.signal_shift, beta=local_slope(xc, a0, b1),
                             shift=100 * np.expm1(s), lo=100 * np.expm1(lo), hi=100 * np.expm1(hi), p_perm=p,
                             women_shift=100 * np.expm1(hl(lv[Af], lv[Bf])) if len(Af) >= 20 and len(Bf) >= 20 else np.nan))
        for W in [50, 100]:
            for k, A, B in windows(x, e.time.values, W):
                ws.append(dict(W=W, test=t, event=k + 1, shift=100 * np.expm1(hl(lv[A], lv[B])),
                               p=perm_p(lv[A], lv[B], day[A], day[B], rng, reps=1000)))
        # placebo cut points inside segments with >= 320 patient results
        b = cps[t]['5']; edges = [0] + b
        for s0, s1 in zip(edges[:-1], edges[1:]):
            t0, t1 = xc.run.iloc[s0], xc.run.iloc[s1 - 1]
            m = np.where((x.run >= t0) & (x.run <= t1))[0]
            if len(m) < 2 * WIN + 20:
                continue
            for _ in range(30):
                c = int(rng.integers(m[0] + WIN, m[-1] - WIN)); A, B = np.arange(c - WIN, c), np.arange(c, c + WIN)
                placebo.append(perm_p(lv[A], lv[B], day[A], day[B], rng, reps=500))
        print(t, 'done', flush=True)
    r = pd.DataFrame(rows)
    r['q'] = stats.false_discovery_control(r.p_perm.values, method='bh')
    r['category'] = np.where(r.q < 0.05, 'shift detected', np.where((r.lo > -10) & (r.hi < 10), 'compensated', 'indeterminate'))
    lc, ls, lr_ = np.log1p(r.curve_shift / 100), np.log1p(r.signal_shift / 100), np.log1p(r['shift'] / 100)
    r['predicted_shift'] = 100 * np.expm1(lc + r.beta * ls)          # curve shift + beta * signal shift
    r['same_sign'] = np.sign(r.curve_shift) == np.sign(r['shift'])
    r.to_csv(out('patient_effects.csv'), index=False)
    ws = pd.DataFrame(ws)
    for W, g in ws.groupby('W'):
        ws.loc[g.index, 'q'] = stats.false_discovery_control(g.p.values, method='bh')
    ws.to_csv(out('window_sensitivity.csv'), index=False)

    S = {}
    prim = r[r.test.isin(PRIMARY)]
    c, y = np.log1p(prim.curve_shift.values / 100), np.log1p(prim['shift'].values / 100)
    S['primary'] = transmission(c, y, rng)
    v = r[r.test == 'VD']; S['VD'] = dict(n=len(v), spearman=stats.spearmanr(v.curve_shift, v['shift'])[0], same_sign=int(v.same_sign.sum()))
    S['per_analyte'] = {t: dict(n=int(len(g)), ols_slope=float(np.polyfit(np.log1p(g.curve_shift / 100), np.log1p(g['shift'] / 100), 1)[0]),
                                theil_sen=float(stats.theilslopes(np.log1p(g['shift'] / 100), np.log1p(g.curve_shift / 100))[0]))
                        for t, g in prim.groupby('test')}
    df = pd.DataFrame(dict(y=y, c=c, analyte=prim.test.values))
    m1 = smf.ols('y ~ c + C(analyte)', df).fit(); m2 = smf.ols('y ~ c * C(analyte)', df).fit()
    S['analyte_adjusted'] = dict(slope=m1.params['c'], lo=m1.conf_int().loc['c', 0], hi=m1.conf_int().loc['c', 1], p=m1.pvalues['c'],
                                 interaction_p=float(m2.compare_f_test(m1)[1]))
    # decomposition
    s_ = np.log1p(prim.signal_shift.values / 100); bs_ = prim.beta.values * s_; pred = c + bs_
    S['decomposition'] = dict(curve_vs_signal_spearman=stats.spearmanr(c, s_)[0], curve_vs_signal_p=stats.spearmanr(c, s_)[1],
                              curve_vs_beta_signal_spearman=stats.spearmanr(c, bs_)[0],
                              observed_vs_predicted_spearman=stats.spearmanr(y, pred)[0], observed_vs_predicted_p=stats.spearmanr(y, pred)[1],
                              slope_observed_on_predicted=float(np.polyfit(pred, y, 1)[0]),
                              r2_curve_only=float(np.corrcoef(c, y)[0, 1] ** 2), r2_predicted=float(np.corrcoef(pred, y)[0, 1] ** 2),
                              median_abs_error_curve_only=float(np.median(np.abs(100 * np.expm1(y) - 100 * np.expm1(c)))),
                              median_abs_error_predicted=float(np.median(np.abs(100 * np.expm1(y) - 100 * np.expm1(pred)))),
                              offset_events=int((np.sign(c) != np.sign(bs_)).sum()),
                              beta_median=float(np.median(prim.beta)))
    # sensitivity: TSH reversals excluded
    keep = ~((prim.test == 'TSH') & (prim.time >= TSH_REVERSALS[0]) & (prim.time <= TSH_REVERSALS[1] + ' 23:59'))
    S['without_tsh_reversals'] = transmission(c[keep.values], y[keep.values], rng)
    # sensitivity: merge thresholds
    S['thresholds'] = {}
    for th in ['3', '5', '8', '10']:
        cc, yy, n_all, nd = [], [], 0, 0
        for t in ANALYTES:
            x = patient_stream(d, t); xc = curve_stream(d, t); b = cps[t][th]; lv = np.log(x.value.values)
            times = xc.run.values[b[:-1]]; n_all += len(times)
            for k, A, B in windows(x, times, WIN):
                cs, _, _ = curve_shift_for(xc, b, k + 1)
                if t in PRIMARY:
                    cc.append(cs); yy.append(hl(lv[A], lv[B]))
        tr = transmission(np.array(cc), np.array(yy), rng)
        S['thresholds'][th] = dict(events=n_all, primary=tr['n'], spearman=tr['spearman'], same_sign=tr['same_sign'],
                                   ols_slope=tr['ols_slope'], theil_sen=tr['theil_sen'])
    pl = np.array(placebo); S['placebo'] = dict(n=len(pl), share_p_below_05=float((pl < 0.05).mean()))
    S['window'] = {int(W): g[g.q < 0.05][['test', 'event']].values.tolist() for W, g in ws.groupby('W')}
    S['categories'] = r.category.value_counts().to_dict()
    S['ci_width'] = dict(median=float((r.hi - r.lo).median()), q1=float((r.hi - r.lo).quantile(.25)), q3=float((r.hi - r.lo).quantile(.75)))
    # classification change between calibrations (results after a change point with a detected shift)
    imp = []
    for e in r[(r.category == 'shift detected') & r.test.isin(PRIMARY)].itertuples():
        x = d[(d.test == e.test) & d.value.notna()].sort_values('run'); nxt = r[(r.test == e.test) & (r.event == e.event + 1)]
        t1 = nxt.time.iloc[0] if len(nxt) else x.run.max() + pd.Timedelta(seconds=1)
        seg = x[(x.run >= e.time) & (x.run < t1)]; v_ = seg.value.values
        for op, Lm in LIMITS[e.test]:
            row = dict(test=e.test, time=e.time, limit=f'{op}{Lm}', n=len(seg), days=int(round((min(t1, x.run.max()) - e.time).total_seconds() / 86400)))
            for lab, sh in [('patient', e.shift), ('curve', e.curve_shift)]:
                v0 = v_ / (1 + sh / 100); a_ = (v_ < Lm) if op == '<' else (v_ > Lm); b_ = (v0 < Lm) if op == '<' else (v0 > Lm)
                row[f'changed_{lab}'] = int((a_ != b_).sum()); row[f'pct_{lab}'] = 100 * (a_ != b_).mean()
            imp.append(row)
    pd.DataFrame(imp).to_csv(out('classification_change.csv'), index=False)
    json.dump(S, open(out('patient_effects_summary.json'), 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(r.drop(columns=['p_perm']).round(3).to_string(index=False))
    print(json.dumps(S, indent=1, default=lambda o: round(float(o), 3)))
    print(pd.DataFrame(imp).round(1).to_string(index=False))
