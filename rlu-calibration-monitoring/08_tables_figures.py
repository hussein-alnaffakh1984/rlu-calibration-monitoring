"""Step 8. Figures (Fig1-Fig4, FigS1, FigS2) and manuscript tables (tables.json)."""
import json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from common import ANALYTES, PRIMARY, CUT, load_data, curve_stream, out

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
F_C, M_C = '#2a78d6', '#eb6834'
CA = {'FERR': '#2a78d6', 'TSH': '#eb6834', 'VD-T II': '#6250d6', 'VD': '#1baf7a'}
C = {'MA': '#2a78d6', 'MM': '#eb6834', 'EWMA': '#1baf7a', 'CUSUM': '#eda100'}
MK = {'MA': 'o', 'MM': 's', 'EWMA': '^', 'CUSUM': 'D'}
NAMES = {'FERR': 'Ferritin', 'TSH': 'TSH', 'VD': '25-OH vitamin D, VD (secondary)', 'VD-T II': '25-OH vitamin D, VD-T II'}
SHORT = {'FERR': 'Ferritin', 'TSH': 'TSH', 'VD-T II': 'VD-T II', 'VD': 'VD (secondary)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black',
                     'text.color': 'black', 'legend.labelcolor': 'black', 'font.family': 'DejaVu Sans', 'savefig.dpi': 600})


def clean(ax):
    ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


def deviation(x):
    lr, lv = np.log(x.RLU.values), np.log(x.value.values); z = (lr - lr.mean()) / lr.std()
    X = np.column_stack([np.ones_like(z), z, z ** 2, z ** 3]); b, *_ = np.linalg.lstsq(X, lv, rcond=None)
    return 100 * np.expm1(lv - X @ b)


if __name__ == '__main__':
    d = load_data()
    ev = pd.read_csv(out('events.csv'), parse_dates=['time']); seg = pd.read_csv(out('segments.csv'), parse_dates=['start', 'end'])
    eff = pd.read_csv(out('patient_effects.csv'), parse_dates=['time'])
    CAL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration_log.csv'), dtype=str)
    CAL['cal_time'] = pd.to_datetime(CAL.cal_time)
    # ---- Fig 1
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True)
    axes = axes.ravel()
    for i, (ax, t) in enumerate(zip(axes, ANALYTES)):
        x = curve_stream(d, t); x['dev'] = deviation(x)
        for sx, c, lab in [('F', F_C, 'Women'), ('M', M_C, 'Men')]:
            g = x[x.sex == sx]; ax.scatter(g.run, g.dev, s=3, color=c, alpha=.5, lw=0, label=lab, zorder=2)
        cat = eff[eff.test == t].set_index('event').category
        for e in ev[ev.test == t].itertuples():
            on = cat.get(e.event) == 'shift detected'
            ax.axvline(e.time, color=INK if on else MUTED, lw=1.4 if on else .8, ls='-' if on else (0, (3, 3)), zorder=1)
        for s in seg[seg.test == t].itertuples():
            m = x[(x.run >= s.start) & (x.run <= s.end)].dev.median(); ax.plot([s.start, s.end], [m, m], color=INK, lw=2, zorder=3)
        lg = CAL[CAL.test == t]; tr = ax.get_xaxis_transform()
        okc = lg[lg.status != 'Cal Failed'].cal_time; bad = lg[lg.status == 'Cal Failed'].cal_time
        ax.plot(okc, np.full(len(okc), 1.0), ls='none', marker='v', ms=6, color='#c0392b', transform=tr, clip_on=False, zorder=5,
                label='Logged calibration' if i == 0 else None)
        ax.plot(bad, np.full(len(bad), 0.9), ls='none', marker='x', ms=5, mew=1.2, color=MUTED, transform=tr, clip_on=False, zorder=5,
                label='Failed calibration attempt' if i == 0 else None)
        ax.set_ylabel('Deviation (%)' if i % 2 == 0 else ''); clean(ax); ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.set_title(f'{"ABCD"[i]}  {SHORT[t].replace(' (secondary)', '')}  (n = {len(x):,}; {len(ev[ev.test == t])} change points)', loc='left', fontsize=9, color='black', pad=10)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls='none', marker='o', ms=4, color=F_C, alpha=.7, label='Women'),
          Line2D([], [], ls='none', marker='o', ms=4, color=M_C, alpha=.7, label='Men'),
          Line2D([], [], ls='none', marker='v', ms=6, color='#c0392b', label='Logged calibration'),
          Line2D([], [], ls='none', marker='x', ms=5, mew=1.2, color=MUTED, label='Failed calibration attempt')]
    fig.legend(handles=hd, loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.09, hspace=0.42, wspace=0.16); fig.savefig(out('Fig1.png')); plt.close(fig)

    # ---- Fig S1: within-segment scatter (rolling robust SD of residuals from each segment's own curve)
    cps = json.load(open(out('changepoints.json')))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for t in ANALYTES:
        x = curve_stream(d, t); lr, lv = np.log(x.RLU.values), np.log(x.value.values); res = np.empty(len(x)); s0 = 0
        mu, sd = lr.mean(), lr.std()
        for e in cps[t]['5']:
            z = (lr[s0:e] - mu) / sd; X = np.column_stack([np.ones_like(z), z, z ** 2, z ** 3])
            b, *_ = np.linalg.lstsq(X, lv[s0:e], rcond=None); res[s0:e] = lv[s0:e] - X @ b; s0 = e
        rs = pd.Series(100 * res, index=x.run).rolling(60, center=True).apply(
            lambda r: 1.4826 * np.median(np.abs(r - np.median(r))), raw=True)          # robust SD (scaled MAD)
        ax.plot(rs.index, rs.values, color=CA[t], lw=1.3, label=SHORT[t])
    ax.set_ylabel('Rolling robust SD of residuals (%)\n(scaled MAD, 60 results)'); clean(ax); ax.legend(frameon=False, fontsize=8, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    fig.tight_layout(); fig.savefig(out('FigS1.png')); plt.close(fig)

    # ---- Fig 2: transmission and decomposition
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    ax = axes[0]
    for t in ANALYTES:
        g = eff[eff.test == t]
        ax.errorbar(g.curve_shift, g['shift'], yerr=[g['shift'] - g.lo, g.hi - g['shift']], fmt='o', ms=5.5, color=CA[t],
                    mfc='white' if t == 'VD' else CA[t], elinewidth=.8, alpha=.9, label=SHORT[t])
    p = eff[eff.test.isin(PRIMARY)]; xx = np.log1p(p.curve_shift / 100); yy = np.log1p(p['shift'] / 100); b = np.polyfit(xx, yy, 1)
    xs = np.linspace(-0.32, 0.45, 50)
    ax.plot(100 * np.expm1(xs), 100 * np.expm1(np.polyval(b, xs)), color=INK, lw=1.6, label=f'Fit, primary analytes (slope {b[0]:.2f})')
    ax.plot([-30, 60], [-30, 60], color=MUTED, lw=.8, ls=(0, (3, 3)), label='Full transmission')
    ax.axhline(0, color=GRID, lw=.8, zorder=0); ax.axvline(0, color=GRID, lw=.8, zorder=0)
    ax.set_xlim(-32, 125); ax.set_ylim(-45, 150); clean(ax)
    ax.set_xlabel('Curve shift at median RLU (%)'); ax.set_ylabel('Shift in patient results, % (95% CI)')
    ax.set_title('A  Curve shift and patient-result shift', loc='left', fontsize=10, color='black'); ax.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2, columnspacing=1.0)
    ax = axes[1]
    cvl = pd.read_csv(out('changepoints_vs_log.csv'))[['test', 'event', 'matched', 'new_lot', 'in_log_window']]
    ge = eff.merge(cvl, on=['test', 'event'], how='left')
    ge['cls'] = np.where(~ge.in_log_window.astype(bool), 'unk', np.where(~ge.matched.astype(bool), 'none', np.where(ge.new_lot == 1, 'new', np.where(ge.new_lot == 0, 'same', 'unk'))))
    from matplotlib.lines import Line2D
    for t in PRIMARY:
        g = ge[ge.test == t]; sig = 100 * np.expm1(g.beta * np.log1p(g.signal_shift / 100))
        for cls, mk, fill in [('new', 'o', True), ('unk', 'D', False), ('same', 's', False), ('none', 'X', False)]:
            k = (g.cls == cls).values
            if k.any():
                ax.scatter(g.curve_shift[k], sig[k], s=36, marker=mk, facecolor=CA[t] if fill else 'none', edgecolor=CA[t], lw=1.3, alpha=.85, zorder=3)
    cs = np.linspace(-0.35, 0.6, 50)       # full offset: (1 + c)(1 + signal component) = 1
    ax.plot(100 * cs, 100 * (1 / (1 + cs) - 1), color=MUTED, lw=.8, ls=(0, (3, 3)), label='Full compensation')
    ax.axhline(0, color=GRID, lw=.8, zorder=0); ax.axvline(0, color=GRID, lw=.8, zorder=0); clean(ax)
    ax.set_xlabel('Curve shift at median RLU (%)'); ax.set_ylabel('Signal component, β × RLU shift (%)')
    hd = [Line2D([], [], lw=5, color=CA[t], label=SHORT[t]) for t in PRIMARY]
    hd += [Line2D([], [], ls='none', marker='o', ms=6, mfc=MUTED, mec=MUTED, label='New lot'),
           Line2D([], [], ls='none', marker='s', ms=6, mfc='white', mec=MUTED, mew=1.3, label='Same lot'),
           Line2D([], [], ls='none', marker='X', ms=7, mfc='white', mec=MUTED, mew=1.3, label='No logged calibration'),
           Line2D([], [], ls='none', marker='D', ms=5, mfc='white', mec=MUTED, mew=1.3, label='Lot change unknown'),
           Line2D([], [], color=MUTED, lw=.8, ls=(0, (3, 3)), label='Full compensation')]
    ax.set_title('B  Curve shift and signal shift', loc='left', fontsize=10, color='black'); ax.legend(handles=hd, frameon=False, fontsize=6.5, loc='upper right')
    fig.tight_layout(); fig.savefig(out('Fig2.png')); plt.close(fig)

    # ---- Fig 3: RLU monitor, synthetic steps
    inj = pd.read_csv(out('monitor_injection.csv'))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for t in ANALYTES:
        g = inj[inj.test == t].sort_values('step')
        ax.plot(100 * g.step, 100 * g.Ped, marker='o', ms=5, lw=2, color=CA[t], label=SHORT[t].replace(' (secondary)', ''))
    ax.axvspan(-8, 8, color=GRID, alpha=.5, lw=0, zorder=0)
    ax.set_xlabel('Synthetic step in the RLU→result mapping (%)'); ax.set_ylabel('Detected within 30 results (%)')
    ax.set_xticks([-20, -15, -10, -8, -5, -3, 3, 5, 8, 10, 15, 20]); ax.tick_params(axis='x', labelsize=7)
    ax.set_ylim(-3, 103); clean(ax); ax.legend(frameon=False, fontsize=8, loc='center')
    fig.tight_layout(); fig.savefig(out('Fig3.png')); plt.close(fig)

    # ---- Fig 4: PBRTQC bias-detection curves (winsorised, primary variant)
    s = pd.read_csv(out('pbrtqc_sim.csv')); s = s[s.variant == 'winsor']
    # VD is not shown: with year-1 limits no evaluation start point had 400 alarm-free results
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for j, t in enumerate(['FERR', 'TSH']):
        for a in C:
            g = s[(s.test == t) & (s.algo == a)].sort_values('bias')
            axes[0, j].plot(g.bias, 100 * g.Ped, color=C[a], marker=MK[a], ms=5, lw=2, label=a)
            axes[1, j].plot(g.bias, g.MNPed.where(g.Ped >= .2), color=C[a], marker=MK[a], ms=5, lw=2, label=a)
        axes[0, j].set_title('AB'[j] + '  ' + {'FERR': 'Ferritin', 'TSH': 'TSH'}[t], loc='left', fontsize=10, color='black')
        axes[0, j].set_ylim(-3, 103); axes[1, j].set_xlabel('Simulated bias (%)')
        for ax in axes[:, j]:
            clean(ax); ax.set_xticks([-50, -30, -20, -10, 10, 20, 30, 50]); ax.tick_params(axis='x', labelsize=7)
    axes[0, 0].set_ylabel('Detected within 400 results (%)'); axes[1, 0].set_ylabel('Median results to detection')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout(); fig.savefig(out('Fig4.png')); plt.close(fig)

    # ---- Fig S2: volume
    v = pd.read_csv(out('volume.csv'))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex='col')
    for j, t in enumerate(['FERR', 'TSH']):
        for a in C:
            g = v[(v.test == t) & (v.algo == a)].groupby('frac').agg(pd_=('per_day', 'mean'), Ped=('Ped', 'mean'), days=('days', 'median')).sort_values('pd_')
            axes[0, j].plot(g.pd_, 100 * g.Ped, color=C[a], marker=MK[a], ms=6, lw=2, label=a)
            axes[1, j].plot(g.pd_, g.days, color=C[a], marker=MK[a], ms=6, lw=2, label=a)
        axes[0, j].set_title('AB'[j] + '  ' + {'FERR': 'Ferritin', 'TSH': 'TSH'}[t] + ' (bias ±20% and ±30%)', loc='left', fontsize=10, color='black')
        axes[1, j].set_xlabel('Results per day'); axes[0, j].set_ylim(0, 103)
        for ax in axes[:, j]:
            clean(ax)
    axes[0, 0].set_ylabel('Detected within 60 days (%)'); axes[1, 0].set_ylabel('Median days to detection')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(out('FigS2.png')); plt.close(fig)
    print('figures written to', out(''))


# ====================================================================== tables (tables.json)
from scipy import stats


def cp(k, n):
    lo = stats.beta.ppf(.025, k, n - k + 1) if k > 0 else 0; hi = stats.beta.ppf(.975, k + 1, n - k) if k < n else 1
    return f"{k}/{n} ({100 * lo:.0f}–{100 * hi:.0f})"


def tables():
    from common import NAMES as NAME
    d = load_data(); T = {}
    ev = pd.read_csv(out('events.csv'), parse_dates=['time']); eff = pd.read_csv(out('patient_effects.csv'), parse_dates=['time'])
    val = json.load(open(out('validity.json'))); S = json.load(open(out('patient_effects_summary.json')))
    mon = json.load(open(out('monitor_params.json')))
    from common import rlu_monitor
    mondet = {}
    for t in ANALYTES:
        x = curve_stream(d, t); al = rlu_monitor(np.log(x.RLU.values), np.log(x.value.values), mon['T'], mon['R'], mon['Kr'])
        ta = x.run.values[[a[1] for a in al]]
        for e in ev[ev.test == t].itertuples():
            te = np.datetime64(e.time); mondet[(t, e.event)] = bool(((ta >= te) & (ta <= te + np.timedelta64(7, 'D'))).any())
    ev = ev.merge(eff[['test', 'event', 'shift', 'lo', 'hi', 'q', 'category', 'beta']], on=['test', 'event'])
    # signal shift resolved (CI excludes 0) and its effect on results (beta x signal shift) opposite to the curve shift
    ev['opp_signal'] = ((ev.signal_lo > 0) | (ev.signal_hi < 0)) & (np.sign(ev.beta * ev.signal_shift) != np.sign(ev.curve_shift))
    # Table 1
    ri = {'FERR': 'F 12–135; M 27–375 ng/mL', 'TSH': '0.35–5.10 mIU/L', 'VD': '30–100 ng/mL', 'VD-T II': '30–100 ng/mL'}
    rows = []
    for t in ANALYTES:
        x = d[d.test == t]; xs = x[x.run >= '2025-12-01'] if t == 'VD-T II' else x
        rows.append([NAME[t], f"{x.run.min():%b %Y}–{x.run.max():%b %Y}", f"{len(x):,}", f"{100 * (x.sex == 'F').mean():.1f}",
                     f"{len(xs) / (xs.run.max() - xs.run.min()).days:.1f}",
                     f"{x.censor.isin(['<', '>']).sum()}", f"{len(curve_stream(d, t)):,}", ri[t]])
    T['t1'] = rows
    # Table 2
    rows = []
    for t in ANALYTES + ['All']:
        e = ev if t == 'All' else ev[ev.test == t]
        g = val['gap']; alt = val['alt']
        gap = (f"{sum(v['with_gap'] for v in g.values())}/{len(e)}" if t == 'All' else f"{g[t]['with_gap']}/{len(e)} ({100 * g[t]['base_rate']:.0f}%)")
        mt = sum(v['matched'] for v in alt.values()) if t == 'All' else alt[t]['matched']
        rows.append([NAME.get(t, 'All'), str(len(e)), f"{e.curve_shift.min():+.1f} to {e.curve_shift.max():+.1f}", f"{e.curve_shift.abs().median():.1f}",
                     f"{e.shift_range.median():.1f}", f"{mt}/{len(e)}", gap, f"{int(e.opp_signal.sum())}/{len(e)}",
                     f"{int((e.category == 'shift detected').sum())}/{len(e)}", f"{sum(mondet[(r.test, r.event)] for r in e.itertuples())}/{len(e)}"])
    T['t2'] = rows
    # Table 3
    ci = pd.read_csv(out('classification_change.csv'), parse_dates=['time']); rows = []
    for f in ev[ev.category == 'shift detected'].itertuples():
        c = ci[(ci.test == f.test) & (ci.time == f.time)]
        imp = '; '.join(f"{r.limit}: {r.changed_curve}–{r.changed_patient}/{r.n} ({r.pct_curve:.1f}–{r.pct_patient:.1f}%) over {r.days} days"
                        for r in c.itertuples()) if len(c) else 'Not estimated (secondary analyte)'
        rows.append([NAME[f.test], f"{f.time:%Y-%m-%d}", f"{f.curve_shift:+.1f}", f"{f.signal_shift:+.1f} ({f.signal_lo:+.1f} to {f.signal_hi:+.1f})",
                     f"{f.shift:+.1f} ({f.lo:+.1f} to {f.hi:+.1f})", f"{f.q:.3f}", imp])
    T['t3'] = rows
    # Table: comparison with the analyser's calibration log (07_calibration_log.py)
    LM = pd.read_csv(out('calibration_log_match.csv'), parse_dates=['cal_time'], dtype={'lot': str, 'serial': str})
    CV = pd.read_csv(out('changepoints_vs_log.csv'), parse_dates=['time'])
    LG = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration_log.csv'), dtype=str)
    shown_all = {'TSH', 'VD'}                       # analytes photographed with status filter ALL
    rows = []
    for t in ANALYTES + ['All']:
        lm = LM if t == 'All' else LM[LM.test == t]; cv = CV if t == 'All' else CV[CV.test == t]; cv = cv[cv.in_log_window]
        lg = LG if t == 'All' else LG[LG.test == t]
        nf = int((lg.status == 'Cal Failed').sum())
        failed = (f"{nf}" if t in shown_all else 'not shown') if t != 'All' else f"{nf}"
        det = lm[lm.detected]; ev_ = lm[lm.evaluable]; miss = ev_[~ev_.detected]
        lagtxt = (f"{det.lag_days.min():.1f}–{det.lag_days.max():.1f}" if len(det) > 1 else f"{det.lag_days.iloc[0]:.1f}") if len(det) else '–'
        um = cv[~cv.matched]; ret = int((um.level_vs_two_back_pct.abs() <= 2).sum())
        rows.append([NAME.get(t, 'All'), str(len(lm)), failed, f"{len(ev_)}", f"{len(det)}/{len(ev_)}", lagtxt,
                     f"{miss.local_shift_pct.abs().max():.1f}" if len(miss) else '–',
                     f"{det.local_shift_pct.abs().min():.1f}" if len(det) else '–',
                     f"{int(cv.matched.sum())}/{len(cv)}", f"{ret}/{len(um)}" if len(um) else '–'])
    T['tlog'] = rows
    rows = []
    for r in LM.itertuples():
        rows.append([NAME[r.test], f"{r.cal_time:%Y-%m-%d %H:%M}", str(r.lot), str(r.serial), ('new' if r.new_lot == 1 else 'same') if pd.notna(r.new_lot) else 'first listed',
                     r.status, f"{r.n_before}/{r.n_after}", f"{r.local_shift_pct:+.1f}" if r.evaluable else 'not evaluable',
                     (f"{pd.Timestamp(r.change_point):%Y-%m-%d} ({r.lag_days:.1f})") if r.detected else 'no'])
    T['s5'] = rows
    # Table 4 (primary, winsorised PBRTQC) and Table S3 (variants, pooled)
    real = pd.read_csv(out('pbrtqc_real.csv')); real['valid'] = real['valid'].fillna(True).astype(bool) if 'valid' in real else True; pooled = pd.read_csv(out('pbrtqc_pooled.csv')); evr = pd.read_csv(out('pbrtqc_events.csv'))
    ch = json.load(open(out('pbrtqc_chosen.json')))
    lab = {'MA': 'Moving average', 'MM': 'Moving median', 'EWMA': 'EWMA', 'CUSUM': 'CUSUM', 'RLU monitor': 'RLU monitor'}
    par = {'MA': 'block', 'MM': 'block', 'EWMA': 'λ', 'CUSUM': 'k'}
    mvec = {w: evr[evr.method == 'RLU monitor'].sort_values(['test', 'time'])[f'hit_{w}'].values for w in ['7d', '30d', 'next']}

    def row(m, v):
        g = real[(real.method == m) & (real.variant == v)].set_index('test'); pl = pooled[(pooled.method == m) & (pooled.variant == v)].set_index('window')
        cells = [lab[m] + ('' if v in ['-', 'winsor'] else {'sex': ', sex-standardised', 'exclude': ', outliers excluded', 'symmetric': ', symmetric limits'}[v]),
                 (f"{par[m]} = " + '/'.join(str(ch[f'{t}|{v}|{m}']) for t in ['FERR', 'TSH', 'VD'])) if m != 'RLU monitor'
                 else f"{100 * mon['T']:.0f}%, {mon['R']} results"]
        for w in ['7d', '30d', 'next']:
            if w not in pl.index:
                cells.append('not interpretable'); continue
            excl = '' if m == 'RLU monitor' or pl.loc[w, 'analytes'] == 'FERR/TSH/VD' else f" [{pl.loc[w, 'analytes'].replace('FERR', 'ferritin')}]"
            cells.append(f"{cp(int(pl.loc[w, 'detected']), int(pl.loc[w, 'events']))}; {pl.loc[w, 'chance']:.1f}; "
                         + ('<0.001' if pl.loc[w, 'p'] < 0.001 else (f"{pl.loc[w, 'p']:.4f}" if 0.045 <= pl.loc[w, 'p'] < 0.055 else (f"{pl.loc[w, 'p']:.3f}" if pl.loc[w, 'p'] < 0.1 else f"{pl.loc[w, 'p']:.2f}"))) + excl)
        cells.append('/'.join(f"{g.loc[t, 'per_1000']:.1f}" for t in ['FERR', 'TSH', 'VD']))
        cells.append('–' if m == 'RLU monitor' else '/'.join(f"{100 * g.loc[t, 'time_in_alarm']:.0f}" + ('' if g.loc[t, 'valid'] else ('ᵇ' if g.loc[t, 'alarms_eval'] == 0 else 'ᵃ'))
                                                              for t in ['FERR', 'TSH', 'VD']))
        if m == 'RLU monitor':
            cells.append('–')
        else:
            ok = evr[(evr.method == 'RLU monitor')].sort_values(['test', 'time']).test.isin(g.index[g.valid]).values
            q = evr[(evr.method == m) & (evr.variant == v)].sort_values(['test', 'time'])['hit_7d'].values
            b, c_ = int((mvec['7d'] & ~q & ok).sum()), int((~mvec['7d'] & q & ok).sum())
            pv = stats.binomtest(b, b + c_, .5).pvalue if b + c_ else 1.0
            cells.append('<0.001' if pv < 0.001 else f"{pv:.3f}")
        tj = evr[(evr.method == m) & (evr.variant == v) & (evr.category == 'shift detected')]
        cells.append('yes' if tj.hit_7d.any() else ('within 30 days' if tj.hit_30d.any() else 'no'))
        return cells
    T['t4'] = [row(m, 'winsor') for m in ['MA', 'MM', 'EWMA', 'CUSUM']] + [row('RLU monitor', '-')]
    T['s3'] = [row(m, v) for v in ['symmetric', 'sex', 'exclude'] for m in ['MA', 'MM', 'EWMA', 'CUSUM']
                if not (v == 'symmetric' and m == 'CUSUM')]          # CUSUM has one upper limit; symmetric = primary
    # Table S4: bias detection (proportion detected within 400 results) for every variant (VD only where simulated)
    sim = pd.read_csv(out('pbrtqc_sim.csv')); rows = []
    vlab = {'winsor': 'capped (primary)', 'symmetric': 'symmetric limits', 'sex': 'sex-standardised', 'exclude': 'outliers excluded'}
    for t in ['FERR', 'TSH', 'VD']:
        for v in ['winsor', 'symmetric', 'sex', 'exclude']:
            for m in ['MA', 'MM', 'EWMA', 'CUSUM']:
                if v == 'symmetric' and m == 'CUSUM':
                    continue
                g = sim[(sim.test == t) & (sim.variant == v) & (sim.algo == m)].sort_values('bias')
                if t == 'VD' and g.eligible.iloc[0] == 0:
                    continue
                cells = ['–' if pd.isna(p_) else f"{100 * p_:.0f}" for p_ in g.Ped]
                rows.append([NAME[t], vlab[v], f"{m} ({par[m]} = {ch[f'{t}|{v}|{m}']})", f"{int(g.eligible.iloc[0])}"] + cells)
    T['s4'] = rows
    # Table S2: transmission robustness
    f2 = lambda v: f"{v:.2f}"
    P = S['primary']; rows = [['All primary change points (OLS)', str(P['n']), f2(P['spearman']), f"{P['same_sign']}/{P['n']}", f"{f2(P['ols_slope'])} ({f2(P['ols_lo'])}–{f2(P['ols_hi'])})", f"{f2(P['theil_sen'])} ({f2(P['ts_lo'])}–{f2(P['ts_hi'])})"]]
    A = S['analyte_adjusted']
    rows.append([f"Analyte intercepts, common slope (different slopes, P = {f2(A['interaction_p'])})", str(P['n']), '–', '–', f"{f2(A['slope'])} ({f2(A['lo'])}–{f2(A['hi'])})", '–'])
    for t, v in S['per_analyte'].items():
        rows.append([f"{NAME[t]} only", str(v['n']), '–', '–', f2(v['ols_slope']), f2(v['theil_sen'])])
    W = S['without_tsh_reversals']
    rows.append(['TSH 17 Nov–11 Dec 2024 excluded', str(W['n']), f2(W['spearman']), f"{W['same_sign']}/{W['n']}", f"{f2(W['ols_slope'])} ({f2(W['ols_lo'])}–{f2(W['ols_hi'])})", f"{f2(W['theil_sen'])} ({f2(W['ts_lo'])}–{f2(W['ts_hi'])})"])
    for th in ['3', '8', '10']:
        v = S['thresholds'][th]
        rows.append([f"Merge threshold {th}% ({v['events']} change points in all)", str(v['primary']), f2(v['spearman']), f"{v['same_sign']}/{v['primary']}", f2(v['ols_slope']), f2(v['theil_sen'])])
    T['s2'] = rows
    # Table S1: all change points
    CVi = CV.set_index(['test', 'event'])

    def logcell(e):
        r = CVi.loc[(e.test, e.event)]
        if not r.in_log_window:
            return 'before log'
        if not r.matched:
            return 'none'
        return f"{r.lag_days:.1f} d; " + ('new lot' if r.new_lot == 1 else ('same lot' if r.new_lot == 0 else 'first listed'))
    rows = []
    for e in ev.itertuples():
        rows.append([NAME[e.test], f"{e.time:%Y-%m-%d}", f"{e.curve_shift:+.1f}", f"{e.shift_P10:+.1f} / {e.shift_P50:+.1f} / {e.shift_P90:+.1f}",
                     (f"{e.curve_shift_F:+.1f}" if pd.notna(e.curve_shift_F) else '–') + ' / ' + (f"{e.curve_shift_M:+.1f}" if pd.notna(e.curve_shift_M) else '–'),
                     f"{e.signal_shift:+.1f} ({e.signal_lo:+.1f} to {e.signal_hi:+.1f})", 'yes' if e.gap_near else 'no',
                     f"{e.shift:+.1f} ({e.lo:+.1f} to {e.hi:+.1f})", (f"{e.q:.3f}" if e.q < 0.1 else f"{e.q:.2f}"), e.category, 'yes' if mondet[(e.test, e.event)] else 'no', logcell(e)])
    T['s1'] = rows
    json.dump(T, open(out('tables.json'), 'w'), ensure_ascii=False, indent=0)
    for k, v in T.items():
        print(k); [print('  ', r) for r in v]


def asymmetry():
    """Why detection is asymmetric (primary capped charts, ferritin and TSH): limits around the year-1 median of the
    statistic, the median over alarm-free 400-result evaluation windows of the closest approach to each limit, the
    shift of the statistic's input (SD units) for a +/-20 % bias at the median result, and the share of evaluation
    results above the cap after a +20 % bias."""
    from common import PBRTQC, pbrtqc_stream
    d = load_data(); ch = json.load(open(out('pbrtqc_chosen.json'))); rows = []
    for t in ['FERR', 'TSH']:
        x = pbrtqc_stream(d, t); tune = x[x.run < CUT]; h = len(tune)
        for k in ['MA', 'MM', 'EWMA']:
            det = PBRTQC(k, ch[f'{t}|winsor|{k}']).fit_transform(tune).calibrate(tune)
            z, _ = det.z(x.value.values, x.sex.values); st = det.statistic(z)
            al = det.alarm_vector(x.value.values, x.sex.values); cs = np.r_[0, np.cumsum(al)]
            cand = np.arange(h, len(x) - 400 + 1); cand = cand[(cs[np.minimum(cand + 400, len(x))] - cs[cand - 1]) == 0]
            up = [np.nanmin(det.hi - st[c:c + 400]) for c in cand]; lo = [np.nanmin(st[c:c + 400] - det.lo) for c in cand]
            v = np.median(x.value.values[h:])
            zs = [(det._bc(np.array([v * f])) - det._bc(np.array([v])))[0] / det.sd for f in (0.8, 1.2)]
            rows.append(dict(test=t, algo=k, param=ch[f'{t}|winsor|{k}'], lower=det.lo, upper=det.hi,
                             median_year1=np.nanmedian(st[:h]), closest_lower=np.median(lo) if lo else np.nan,
                             closest_upper=np.median(up) if up else np.nan, shift_minus20=zs[0], shift_plus20=zs[1],
                             capped_after_plus20=float((x.value.values[h:] * 1.2 > det.hi_t).mean())))
    a = pd.DataFrame(rows); a.to_csv(out('pbrtqc_asymmetry.csv'), index=False); print(a.round(3).to_string(index=False))


if __name__ == '__main__':
    tables()
    asymmetry()
