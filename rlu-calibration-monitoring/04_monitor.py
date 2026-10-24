"""Step 4. Online RLU monitor: tuned on year 1 only, evaluated on years 2-3.

For each new result the monitor predicts log(result) from log(RLU) with a quadratic fitted to the previous
results of the current calibration (<= 200; >= 30 at start, Kr after an alarm) and alarms when R consecutive
results deviate by more than T (log scale).
Tuning: year-1 results segmented on their own; change-free stretches (segments >= 120 results, 10 trimmed at
each end); synthetic steps (results x (1+s), RLU unchanged) of +/-10 %; false alarms <= 1 per 1,000.
Evaluation: the same on a segmentation of the evaluation period alone, steps of 3-20 %; alarms on all
evaluation data (per 1,000 evaluation results of the curve stream) and agreement with the change points.

Outputs: monitor_params.json, monitor_tuning.csv, monitor_injection.csv, monitor_alarms.csv
"""
import json
import numpy as np, pandas as pd
from common import ANALYTES, CUT, PEN, MIN_SHIFT, load_data, curve_stream, pelt, segment_levels, rlu_monitor, out

K0 = 30


def change_points(x):
    b = pelt(x, PEN)
    while len(b) > 1:
        diff = np.abs(np.diff(segment_levels(x, b))); k = int(np.argmin(diff))
        if diff[k] >= np.log1p(MIN_SHIFT):
            break
        b = b[:k] + b[k + 1:]
    return b


def stretches(x):
    if len(x) < 150:
        return []
    x = x.reset_index(drop=True); res, s0 = [], 0
    for e in change_points(x):
        if e - s0 >= 120:
            i = np.arange(s0, e)[10:-10]; res.append((np.log(x.RLU.values[i]), np.log(x.value.values[i])))
        s0 = e
    return res


def inject(strs, T, R, Kr, step, nrep, rng, win=30):
    det, delays, n = 0, [], 0
    for _ in range(nrep):
        lr, lv = strs[rng.integers(len(strs))]
        if len(lr) < 100:
            continue
        s = int(rng.integers(50, len(lr) - 40)); lv2 = lv.copy(); lv2[s:] += np.log1p(step)
        hit = [a for a in rlu_monitor(lr, lv2, T, R, Kr) if s <= a[0] < s + win]; n += 1
        if hit:
            det += 1; delays.append(hit[0][1] - s + 1)
    return det / max(n, 1), (np.median(delays) if delays else np.nan), n


def false_alarms(strs, T, R, Kr):
    a = sum(len(rlu_monitor(lr, lv, T, R, Kr)) for lr, lv in strs); n = sum(max(len(lr) - K0, 0) for lr, _ in strs)
    return 1000 * a / max(n, 1)


if __name__ == '__main__':
    d = load_data(); rng = np.random.default_rng(20261024)
    streams = {t: curve_stream(d, t) for t in ANALYTES}
    tune = sum((stretches(streams[t][streams[t].run < CUT]) for t in ANALYTES), [])
    evs = {t: stretches(streams[t][streams[t].run >= CUT]) for t in ANALYTES}
    grid = []
    for T in [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]:
        for R in [2, 3, 4]:
            for Kr in [10, 30]:
                far = false_alarms(tune, T, R, Kr)
                pu = inject(tune, T, R, Kr, 0.10, 60, rng)[0]; pdn = inject(tune, T, R, Kr, -0.10, 60, rng)[0]
                grid.append(dict(T=T, R=R, Kr=Kr, FAR=far, det10=(pu + pdn) / 2))
    g = pd.DataFrame(grid); g.to_csv(out('monitor_tuning.csv'), index=False)
    best = g[g.FAR <= 1.0].sort_values(['det10', 'FAR'], ascending=[False, True]).iloc[0]
    T, R, Kr = float(best['T']), int(best['R']), int(best['Kr'])
    json.dump(dict(T=T, R=R, Kr=Kr, tuning_results=int(sum(len(s[0]) for s in tune))), open(out('monitor_params.json'), 'w'))
    print('chosen', dict(T=T, R=R, Kr=Kr), 'tuning results', sum(len(s[0]) for s in tune), flush=True)
    ev = pd.read_csv(out('events.csv'), parse_dates=['time'])
    rows, allal = [], []
    for t in ANALYTES:
        if evs[t]:
            far = false_alarms(evs[t], T, R, Kr)
            for step in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
                for sg in [1, -1]:
                    p, dl, n = inject(evs[t], T, R, Kr, sg * step, 100, rng)
                    rows.append(dict(test=t, step=sg * step, Ped=p, median_delay_results=dl, n=n, false_per_1000_stretch=far))
        x = streams[t]; al = rlu_monitor(np.log(x.RLU.values), np.log(x.value.values), T, R, Kr)
        ta = x.run.values[[a[1] for a in al]]
        et = ev[ev.test == t].time.values; e_eval = et[et >= np.datetime64(CUT)]
        in_eval = ta >= np.datetime64(CUT)
        near = np.array([((et <= a) & (a <= et + np.timedelta64(7, 'D'))).any() for a in ta], bool)
        n_eval = int((x.run >= CUT).sum())
        allal.append(dict(test=t, eval_results=n_eval, alarms_eval=int(in_eval.sum()), per_1000=1000 * in_eval.sum() / max(n_eval, 1),
                          other_alarms=int((in_eval & ~near).sum()), other_per_1000=1000 * (in_eval & ~near).sum() / max(n_eval, 1),
                          events_eval=len(e_eval), flagged_7d=int(sum(((ta >= e) & (ta <= e + np.timedelta64(7, 'D'))).any() for e in e_eval))))
    inj = pd.DataFrame(rows); inj.to_csv(out('monitor_injection.csv'), index=False)
    aa = pd.DataFrame(allal); aa.to_csv(out('monitor_alarms.csv'), index=False)
    print(inj.pivot_table(index='test', columns='step', values='Ped').round(2).to_string())
    print(inj.groupby('test').false_per_1000_stretch.first().round(2).to_dict())
    print(aa.round(2).to_string(index=False))
