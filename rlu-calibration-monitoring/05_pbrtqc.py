"""Step 5. Patient-based real-time QC (PBRTQC) on the same streams.

Algorithms: moving average (MA), moving median (MM), EWMA, two-sided CUSUM, on Box-Cox-transformed results
standardised with year-1 (tuning) mean and SD.
Variants: 'winsor'  - results capped at the year-1 1st/99th percentiles (primary)
          'symmetric' - as winsor, control limits symmetric about the year-1 median of the statistic
          'sex'     - as winsor, standardised separately for women and men (stratified)
          'exclude' - results outside the year-1 1st/99th percentiles removed (sensitivity)
Control limits: <= 2 alarm episodes per 1,000 year-1 results. Parameter: chosen on year-1 simulations
(mean of MNPed with undetected runs counted as 400) over biases of +/-10, 20, 30 %, among settings whose limits,
set on the first 75 % of year 1, kept the last 25 % of year 1 in the alarm state for <= 10 % of results.
Evaluation (15 Sep 2024 onward): bias-detection curves on clean start points; alarms after change points
(7 days, 30 days, before the next change point) against a chance baseline from random times.

Outputs: pbrtqc_sim.csv, pbrtqc_real.csv, pbrtqc_events.csv, pbrtqc_pooled.csv, pbrtqc_chosen.json,
         pbrtqc_limits.csv (control limits and year-1 distribution of each chosen statistic)
"""
import json, zlib
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import (PBRTQC, PBRTQC_ANALYTES, CUT, load_data, pbrtqc_stream, curve_stream, simulate_bias,
                    rlu_monitor, out)

GRID = {'MA': [5, 10, 20, 40, 80, 160, 320, 640], 'MM': [5, 10, 20, 40, 80, 160, 320, 640],
        'EWMA': [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], 'CUSUM': [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5]}
VARIANTS = {'winsor': dict(outliers='winsor', by_sex=False), 'symmetric': dict(outliers='winsor', symmetric=True),
            'sex': dict(outliers='winsor', by_sex=True), 'exclude': dict(outliers='exclude', by_sex=False)}
MAX_HOLDOUT = 0.10
MAX_EVAL_IN_ALARM = 0.50
TUNE_BIAS, EVAL_BIAS, L = [-30, -20, -10, 10, 20, 30], [-50, -30, -20, -10, 10, 20, 30, 50], 400
D = load_data()


def seed(*a):
    return zlib.crc32('|'.join(map(str, a)).encode())


def score(df):
    return float((df.MNPed.fillna(L) * df.Ped + L * (1 - df.Ped)).mean())


def fit(t, variant, kind, p):
    x = pbrtqc_stream(D, t); tune = x[x.run < CUT]
    try:
        det = PBRTQC(kind, p, **VARIANTS[variant]).fit_transform(tune).calibrate(tune)
    except ValueError:           # block or burn-in longer than the year-1 stream
        det = None
    return x, det


def holdout_in_alarm(t, variant, kind, p):
    """Stability check inside year 1: limits set on its first 75 %, share of the last 25 % in the alarm state."""
    x = pbrtqc_stream(D, t); h = int((x.run < CUT).sum()); h0 = int(0.75 * h); first = x.iloc[:h0]
    try:
        det = PBRTQC(kind, p, **VARIANTS[variant]).fit_transform(first).calibrate(first)
    except ValueError:
        return 1.0
    al = det.alarm_vector(x.value.values[:h], x.sex.values[:h])
    return float(al[h0:].mean())


def tune_job(a):
    t, variant, kind, p = a
    x, det = fit(t, variant, kind, p); h = int((x.run < CUT).sum())
    if det is None:
        return (t, variant, kind, p, np.inf, 1.0)
    # tuning simulations use year-1 data only (bias periods end before the cut)
    sm = simulate_bias(det, x.iloc[:h].reset_index(drop=True), 0, h, TUNE_BIAS, 60, seed(t, variant, kind, p))
    return (t, variant, kind, p, score(sm), holdout_in_alarm(t, variant, kind, p))


def eval_job(a):
    t, variant, kind, p = a
    x, det = fit(t, variant, kind, p); h = int((x.run < CUT).sum())
    sm = simulate_bias(det, x, h, len(x), EVAL_BIAS, 150, seed(t, variant, kind, 'eval'))
    sm['test'], sm['variant'], sm['algo'], sm['param'] = t, variant, kind, p
    al = det.alarm_vector(x.value.values, x.sex.values)
    return sm, al


def detect(ta, te, hi):
    """For each event time te[i]: any alarm raised in (te[i], hi[i]]."""
    return np.array([((ta >= a) & (ta <= b)).any() for a, b in zip(te, hi)])


if __name__ == '__main__':
    jobs = [(t, v, k, p) for t in PBRTQC_ANALYTES for v in VARIANTS for k, ps in GRID.items() for p in ps]
    with Pool(2) as pool:
        tuned = pool.map(tune_job, jobs)
    # parameter choice: lowest simulated score among settings that stay in control in the held-out last quarter of
    # year 1 (<= MAX_HOLDOUT of results in the alarm state); if none qualifies, the most stable setting
    tun = pd.DataFrame(tuned, columns=['test', 'variant', 'algo', 'param', 'score', 'holdout_in_alarm'])
    tun.to_csv(out('pbrtqc_tuning.csv'), index=False)
    best = {}
    for (t, v, k), g in tun.groupby(['test', 'variant', 'algo']):
        ok = g[(g.holdout_in_alarm <= MAX_HOLDOUT) & np.isfinite(g.score)]
        r = ok.sort_values('score').iloc[0] if len(ok) else g.sort_values(['holdout_in_alarm', 'score']).iloc[0]
        best[(t, v, k)] = (r.param.item() if hasattr(r.param, 'item') else r.param, r.score)
    best = {key: (int(p) if float(p).is_integer() and key[2] in ('MA', 'MM') else float(p), s) for key, (p, s) in best.items()}
    chosen = {f'{t}|{v}|{k}': p for (t, v, k), (p, s) in best.items()}
    json.dump(chosen, open(out('pbrtqc_chosen.json'), 'w'), indent=1)
    print('chosen', chosen, flush=True)
    with Pool(2) as pool:
        res = pool.map(eval_job, [(t, v, k, p) for (t, v, k), (p, s) in best.items()])
    sims = pd.concat([r[0] for r in res]); sims.to_csv(out('pbrtqc_sim.csv'), index=False)
    # control limits relative to the year-1 distribution of each chosen statistic (explains asymmetric detection)
    lim = []
    for (t, v, k), (p, s) in best.items():
        x, det = fit(t, v, k, p); h = int((x.run < CUT).sum())
        z, _ = det.z(x.value.values[:h], x.sex.values[:h]); st = det.statistic(z); q = np.nanpercentile(st, [2.5, 50, 97.5])
        lim.append(dict(test=t, variant=v, algo=k, param=p, lower=det.lo, upper=det.hi, median=q[1], q025=q[0], q975=q[2]))
    pd.DataFrame(lim).to_csv(out('pbrtqc_limits.csv'), index=False)
    alarms = {(r[0].test.iloc[0], r[0].variant.iloc[0], r[0].algo.iloc[0]): r[1] for r in res}

    # ------------------------------------------------ real change points (evaluation period)
    ev = pd.read_csv(out('events.csv'), parse_dates=['time']); eff = pd.read_csv(out('patient_effects.csv'), parse_dates=['time'])
    mon = json.load(open(out('monitor_params.json')))
    real, evrows, CH = [], [], {}
    rng = np.random.default_rng(5)
    for t in PBRTQC_ANALYTES:
        x = pbrtqc_stream(D, t); tsec = x.run.values.astype('datetime64[s]').astype(np.int64); ev_eval = x.run >= CUT
        e = ev[ev.test == t].sort_values('time'); te_all = e.time.values.astype('datetime64[s]').astype(np.int64)
        nxt = np.r_[te_all[1:], tsec[-1]]
        m = te_all >= np.datetime64(CUT, 's').astype(np.int64); te, tn = te_all[m], nxt[m]
        cat = eff[eff.test == t].set_index('event').category.reindex(e.event).values[m]
        pool_t = tsec[ev_eval.values & (tsec <= tsec[-1] - 30 * 86400)]
        xc = curve_stream(D, t)
        ma = rlu_monitor(np.log(xc.RLU.values), np.log(xc.value.values), mon['T'], mon['R'], mon['Kr'])
        mon_t = xc.run.values[[a[1] for a in ma]].astype('datetime64[s]').astype(np.int64)
        methods = {('RLU monitor', '-'): (mon_t, None)}
        for (tt, v, k), al in alarms.items():
            if tt == t:
                st = np.where(al & ~np.r_[False, al[:-1]])[0]
                methods[(k, v)] = (tsec[st], al)
        n_eval = int(ev_eval.sum()); n_eval_c = int((xc.run >= CUT).sum())
        for (name, v), (ta, al) in methods.items():
            ta_e = ta[ta >= np.datetime64(CUT, 's').astype(np.int64)]
            row = dict(test=t, method=name, variant=v, events=len(te), alarms_eval=len(ta_e),
                       per_1000=1000 * len(ta_e) / (n_eval_c if al is None else n_eval),
                       time_in_alarm=float(al[ev_eval.values].mean()) if al is not None else np.nan)
            hits = {}
            for lab, hi_fun in [('7d', lambda s: s + 7 * 86400), ('30d', lambda s: s + 30 * 86400), ('next', None)]:
                hi = hi_fun(te) if hi_fun else tn
                dd = detect(ta, te, hi); hits[lab] = dd
                # chance: random times, same window lengths
                lens = hi - te; ch = []
                for _ in range(2000):
                    s = rng.choice(pool_t, len(te), replace=False); ch.append(detect(ta, s, s + lens).sum())
                ch = np.array(ch); CH.setdefault((name, v, lab), {})[t] = ch
                row[f'det_{lab}'] = int(dd.sum()); row[f'chance_{lab}'] = float(ch.mean()); row[f'p_{lab}'] = float((ch >= dd.sum()).mean())
            real.append(row)
            for i in range(len(te)):
                evrows.append(dict(test=t, time=pd.Timestamp(te[i], unit='s'), category=cat[i], method=name, variant=v,
                                   **{f'hit_{k_}': bool(h_[i]) for k_, h_ in hits.items()}))
    real = pd.DataFrame(real); real.to_csv(out('pbrtqc_real.csv'), index=False)
    evr = pd.DataFrame(evrows); evr.to_csv(out('pbrtqc_events.csv'), index=False)
    # post hoc validity filter (uses evaluation data): a chart in the alarm state for more than half of the evaluation
    # period cannot raise new alarms, and a chart that raised no alarm cannot flag anything; such charts are reported
    # but left out of the pooled comparison
    real['valid'] = ~(real.time_in_alarm > MAX_EVAL_IN_ALARM) & ((real.alarms_eval > 0) | (real.method == 'RLU monitor'))
    real.to_csv(out('pbrtqc_real.csv'), index=False)
    pooled = []
    for (name, v, lab), per in CH.items():
        g = real[(real.method == name) & (real.variant == v) & real.valid]
        if not len(g):
            continue
        tot = np.sum([per[t] for t in g.test], axis=0); obs = int(g[f'det_{lab}'].sum())
        pooled.append(dict(method=name, variant=v, window=lab, analytes='/'.join(g.test), detected=obs, events=int(g.events.sum()),
                           chance=float(tot.mean()), p=float((tot >= obs).mean())))
    pooled = pd.DataFrame(pooled); pooled.to_csv(out('pbrtqc_pooled.csv'), index=False)
    pd.set_option('display.width', 250)
    print(sims.pivot_table(index=['test', 'variant', 'algo'], columns='bias', values='Ped').round(2).to_string())
    print(real.round(2).to_string(index=False)); print(pooled.round(3).to_string(index=False))
