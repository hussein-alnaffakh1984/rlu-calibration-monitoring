"""Step 7. Comparison with the analyser's calibration log.

Input: data/calibration_log.csv, transcribed from photographs of the CL-900i screen
Reagent > Calibration Result > History (calibration-date filter 24 Oct 2023 - 24 Oct 2026).
TSH and VD were listed with all statuses; ferritin and VD-T II with status 'Extended' only.
'Cal Failed' entries do not change the curve and are excluded.

A logged calibration is 'detected' when a change point (events.csv) falls from 1 day before to 7 days after it,
and a change point is 'matched' when a logged calibration lies in the same window before it.
The local curve shift across each logged calibration is estimated from up to 60 results on either side,
truncated at neighbouring logged calibrations; calibrations with < MIN_SIZE results on either side are not evaluable.

Outputs: calibration_log_match.csv, changepoints_vs_log.csv, calibration_log_summary.json
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from common import ANALYTES, PRIMARY, CUT, MIN_SIZE, load_data, curve_stream, window_curve_shift, rlu_monitor, out
import os

LO, HI = -1.0, 7.0
LOG_START = pd.Timestamp('2023-10-24')
d = load_data()
log = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration_log.csv'), dtype=str)
log['cal_time'] = pd.to_datetime(log.cal_time)
cal = log[log.status != 'Cal Failed'].sort_values(['test', 'cal_time']).copy()
cal['prev_lot'] = cal.groupby('test').lot.shift()
cal['new_lot'] = np.where(cal.prev_lot.isna(), np.nan, (cal.lot != cal.prev_lot).astype(float))
ev = pd.read_csv(out('events.csv'), parse_dates=['time'])
pe = pd.read_csv(out('patient_effects.csv'), parse_dates=['time'])
seg = pd.read_csv(out('segments.csv'), parse_dates=['start', 'end'])
par = json.load(open(out('monitor_params.json')))


def lag_days(t, c):
    return (t - c).dt.total_seconds() / 86400


# ------------------------------------------------------------- logged calibrations -> change points
R = []
for t in ANALYTES:
    x = curve_stream(d, t); runs = x.run.values
    c = cal[cal.test == t]; e = ev[ev.test == t]; ct = c.cal_time.values
    for _, r in c.iterrows():
        lg = (e.time - r.cal_time).dt.total_seconds() / 86400
        hit = lg[(lg >= LO) & (lg <= HI)]
        ie = int(np.searchsorted(runs, np.datetime64(r.cal_time)))
        prev = ct[ct < np.datetime64(r.cal_time)]; nxt = ct[ct > np.datetime64(r.cal_time)]
        a0 = max(0, ie - 60, int(np.searchsorted(runs, prev[-1])) if len(prev) else 0)
        b1 = min(len(x), ie + 60, int(np.searchsorted(runs, nxt[0])) if len(nxt) else len(x))
        evaluable = (ie - a0 >= MIN_SIZE) and (b1 - ie >= MIN_SIZE)
        sh = window_curve_shift(x, a0, ie, b1) if evaluable else np.nan
        icp = int(np.searchsorted(runs, np.datetime64(e.loc[hit.index[0], 'time']))) if len(hit) else -1
        R.append(dict(test=t, cal_time=r.cal_time, results_to_cp=icp - ie if len(hit) else np.nan, lot=r.lot, serial=r.serial, status=r.status,
                      new_lot=r.new_lot, n_before=ie - a0, n_after=b1 - ie, evaluable=evaluable,
                      local_shift_pct=100 * np.expm1(sh) if evaluable else np.nan,
                      detected=len(hit) > 0, lag_days=float(hit.iloc[0]) if len(hit) else np.nan,
                      change_point=str(e.loc[hit.index[0], 'time']) if len(hit) else ''))
R = pd.DataFrame(R)

# ------------------------------------------------------------- change points -> logged calibrations
C = []
for _, r in ev.iterrows():
    c = cal[cal.test == r.test]; lg = lag_days(r.time, c.cal_time); k = lg[(lg >= LO) & (lg <= HI)]
    cc = c.loc[k.index[0]] if len(k) else None
    C.append(dict(test=r.test, event=r.event, time=r.time, curve_shift=r.curve_shift, signal_shift=r.signal_shift,
                  in_log_window=r.time >= LOG_START + pd.Timedelta(days=1), matched=len(k) > 0,
                  lag_days=float(k.iloc[0]) if len(k) else np.nan,
                  cal_time=str(cc.cal_time) if cc is not None else '', lot=cc.lot if cc is not None else '',
                  new_lot=cc.new_lot if cc is not None else np.nan))
C = pd.DataFrame(C)
C = C.merge(pe[['test', 'event', 'beta', 'category']], on=['test', 'event'], how='left')

# return of reversal segments to an earlier curve: for each unmatched change point, compare the level of the
# segment it opens with the level two segments earlier (the segment before the preceding change)
S = {}
for t in ANALYTES:
    s = seg[seg.test == t].reset_index(drop=True)
    for i in range(2, len(s)):
        S[(t, s.start[i])] = 100 * (s.level_at_median_rlu[i] / s.level_at_median_rlu[i - 2] - 1)
C['level_vs_two_back_pct'] = [S.get((r.test, r.time), np.nan) for r in C.itertuples()]

# ------------------------------------------------------------- signal offset: new lot vs same lot
C['curve_log'] = np.log1p(C.curve_shift / 100)
C['signal_component_log'] = C.beta * np.log1p(C.signal_shift / 100)
C['offset_fraction'] = -C.signal_component_log / C.curve_log
P = C[C.test.isin(PRIMARY) & C.matched & C.new_lot.notna()]
nl, sl = P[P.new_lot == 1], P[P.new_lot == 0]
V = C[(C.test == 'VD') & C.matched & C.new_lot.notna()]


def grp(g):
    return dict(n=len(g), opposed=int((np.sign(g.signal_component_log) == -np.sign(g.curve_log)).sum()),
                median_offset=float(g.offset_fraction.median()) if len(g) else None,
                spearman=float(stats.spearmanr(g.curve_log, g.signal_component_log)[0]) if len(g) > 2 else None,
                dates=[f'{a} {str(b)[:10]}' for a, b in zip(g.test, g.time)])


lot = dict(primary_new_lot=grp(nl), primary_same_lot=grp(sl),
           mannwhitney_p=float(stats.mannwhitneyu(nl.offset_fraction, sl.offset_fraction, alternative='two-sided').pvalue),
           vd_new_lot=grp(V[V.new_lot == 1]), vd_same_lot=grp(V[V.new_lot == 0]),
           excluded_first_in_log=[f'{a} {str(b)[:10]}' for a, b in zip(C[C.matched & C.new_lot.isna()].test, C[C.matched & C.new_lot.isna()].time)])

# ------------------------------------------------------------- RLU monitor alarms vs logged calibrations
mon = []
for t in ANALYTES:
    x = curve_stream(d, t)
    al = rlu_monitor(np.log(x.RLU.values), np.log(x.value.values), par['T'], par['R'], par['Kr'])
    ta = pd.Series(pd.to_datetime(x.run.values[[a[1] for a in al]]))
    c = R[(R.test == t) & (R.cal_time >= CUT)]
    for _, r in c.iterrows():
        lg = (ta - r.cal_time).dt.total_seconds() / 86400
        mon.append(dict(kind='cal', test=t, evaluable=r.evaluable, big=bool(abs(r.local_shift_pct) >= 5) if r.evaluable else False,
                        flagged=bool(((lg >= LO) & (lg <= HI)).any())))
    cc = cal[cal.test == t]
    for a in ta[ta >= CUT]:
        lg = lag_days(a, cc.cal_time)
        mon.append(dict(kind='alarm', test=t, evaluable=True, big=False, flagged=bool(((lg >= LO) & (lg <= HI)).any())))
M = pd.DataFrame(mon)
mc = M[M.kind == 'cal']; ma = M[M.kind == 'alarm']
monitor = dict(eval_logged=len(mc), eval_logged_evaluable=int(mc.evaluable.sum()),
               eval_logged_ge5=int(mc.big.sum()), eval_logged_ge5_flagged=int(mc[mc.big].flagged.sum()),
               eval_logged_lt5_flagged=int(mc[mc.evaluable & ~mc.big].flagged.sum()), eval_logged_lt5=int((mc.evaluable & ~mc.big).sum()),
               eval_alarms=len(ma), eval_alarms_after_logged=int(ma.flagged.sum()),
               per_test={t: dict(cal=int(mc[mc.test == t].big.sum()), flagged=int(mc[(mc.test == t) & mc.big].flagged.sum()),
                                 alarms=int((ma.test == t).sum()), alarms_after_logged=int(ma[ma.test == t].flagged.sum()))
                         for t in ANALYTES})

# ------------------------------------------------------------- PBRTQC restricted to log-confirmed change points
pr = pd.read_csv(out('pbrtqc_real.csv'))
pev = pd.read_csv(out('pbrtqc_events.csv'), parse_dates=['time'])
pev = pev.merge(C[['test', 'time', 'matched']], on=['test', 'time'], how='left')
valid = pr[['test', 'method', 'variant', 'valid']] if 'valid' in pr.columns else None
pbr = {}
for (m, v), g in pev.groupby(['method', 'variant']):
    if v not in ('winsor', '-'):
        continue
    if valid is not None and m != 'RLU monitor':
        ok = valid[(valid.method == m) & (valid.variant == v) & valid.valid].test
        g = g[g.test.isin(ok)]
    g = g[g.matched]
    pbr[m] = dict(n=len(g), d7=int(g.hit_7d.sum()), d30=int(g.hit_30d.sum()), next=int(g.hit_next.sum()))

# ------------------------------------------------------------- chance baseline for matches
# the same number of pseudo-calibrations per analyte placed at random result times within the logged period
rng = np.random.default_rng(20261025); REPS = 2000; chance = np.zeros(REPS)
for t in ANALYTES:
    x = curve_stream(d, t); rt = x.run[(x.run >= LOG_START)].values
    k = int((cal.test == t).sum()); cps = C[(C.test == t) & C.in_log_window].time.values
    for b in range(REPS):
        pc = rng.choice(rt, k, replace=False)
        lg = (cps[:, None] - pc[None, :]) / np.timedelta64(1, 'D')
        chance[b] += ((lg >= LO) & (lg <= HI)).any(axis=1).sum()
chance_match = dict(mean=float(chance.mean()), p95=float(np.percentile(chance, 95)), max=float(chance.max()),
                    p=float((np.sum(chance >= int(C[C.in_log_window].matched.sum())) + 1) / (REPS + 1)))

# ------------------------------------------------------------- summary
inw = C[C.in_log_window]
summ = dict(
    logged_total=int(len(log)), logged_failed=int((log.status == 'Cal Failed').sum()), logged_success=int(len(R)),
    evaluable=int(R.evaluable.sum()), detected=int(R.detected.sum()),
    detected_evaluable=int(R[R.evaluable].detected.sum()),
    not_evaluable=[f'{a} {str(b)[:10]}' for a, b in zip(R[~R.evaluable].test, R[~R.evaluable].cal_time)],
    per_test={t: dict(logged=int((R.test == t).sum()), evaluable=int(R[(R.test == t)].evaluable.sum()),
                      detected=int(R[(R.test == t)].detected.sum()),
                      cps_in_window=int((inw.test == t).sum()), cps_matched=int(inw[inw.test == t].matched.sum()))
              for t in ANALYTES},
    cps_in_window=int(len(inw)), cps_matched=int(inw.matched.sum()),
    cp_at_first_result=int((R[R.detected].results_to_cp == 0).sum()), results_to_cp_max=int(R[R.detected].results_to_cp.max()),
    lag_median=float(R[R.detected].lag_days.median()), lag_max=float(R[R.detected].lag_days.max()),
    lag_min=float(R[R.detected].lag_days.min()),
    detected_abs_shift_min=float(R[R.detected].local_shift_pct.abs().min()),
    detected_abs_shift_max=float(R[R.detected].local_shift_pct.abs().max()),
    missed_evaluable_abs_shift_max=float(R[R.evaluable & ~R.detected].local_shift_pct.abs().max()),
    missed_evaluable_abs_shift_median=float(R[R.evaluable & ~R.detected].local_shift_pct.abs().median()),
    missed_evaluable=[f'{a} {str(b)[:10]} ({c:+.1f}%)' for a, b, c in
                      zip(R[R.evaluable & ~R.detected].test, R[R.evaluable & ~R.detected].cal_time, R[R.evaluable & ~R.detected].local_shift_pct)],
    unmatched=[dict(test=r.test, time=str(r.time)[:10], curve=round(r.curve_shift, 1),
                    level_vs_two_back=None if np.isnan(r.level_vs_two_back_pct) else round(r.level_vs_two_back_pct, 1),
                    category=r.category) for r in inw[~inw.matched].itertuples()],
    before_log=[f'{a} {str(b)[:10]}' for a, b in zip(C[~C.in_log_window].test, C[~C.in_log_window].time)],
    patient_shift_detected=[dict(test=r.test, time=str(r.time)[:10], matched=bool(r.matched)) for r in C[C.category == 'shift detected'].itertuples()],
    lot=lot, monitor=monitor, pbrtqc_confirmed=pbr, chance_match=chance_match)
R.to_csv(out('calibration_log_match.csv'), index=False)
C.to_csv(out('changepoints_vs_log.csv'), index=False)
json.dump(summ, open(out('calibration_log_summary.json'), 'w'), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
