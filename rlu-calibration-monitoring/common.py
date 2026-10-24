"""Shared settings and functions for the RLU / PBRTQC analysis.

Paths are taken from environment variables so the pipeline runs anywhere:
    RLU_RAW_DIR  folder with the analyser result-list exports (.xlsx)   [default: ./data/raw]
    RLU_OUT_DIR  folder for all outputs                                  [default: ./results]
"""
import os
import numpy as np
import pandas as pd
import ruptures as rpt

RAW_DIR = os.environ.get('RLU_RAW_DIR', os.path.join(os.path.dirname(__file__), 'data', 'raw'))
OUT_DIR = os.environ.get('RLU_OUT_DIR', os.path.join(os.path.dirname(__file__), 'results'))
os.makedirs(OUT_DIR, exist_ok=True)

ANALYTES = ['FERR', 'TSH', 'VD-T II', 'VD']
PRIMARY = ['FERR', 'TSH', 'VD-T II']
PBRTQC_ANALYTES = ['FERR', 'TSH', 'VD']          # VD-T II has no year-1 data
START = pd.Timestamp(os.environ.get('RLU_START', '2023-09-15'))                # first day with exports for both sexes (women's exports start here)
CUT = pd.Timestamp('2024-09-15')                  # end of year 1 (tuning period)
NAMES = {'FERR': 'Ferritin', 'TSH': 'TSH', 'VD-T II': '25-OH vitamin D, VD-T II', 'VD': '25-OH vitamin D, VD'}

# change-point settings
PEN, MIN_SIZE, MIN_SHIFT, WIN = 0.2, 30, 0.05, 150


def out(name):
    return os.path.join(OUT_DIR, name)


def load_data():
    return pd.read_pickle(out('data.pkl'))


def curve_stream(d, test):
    """Results usable for RLU-result analyses: numeric, uncensored, positive result and RLU."""
    x = d[(d.test == test) & (d.censor == '') & d.value.notna() & (d.value > 0) & (d.RLU > 0)]
    return x.sort_values('run').reset_index(drop=True)


# ---------------------------------------------------------------- curves and change points
def _std(lr):
    return lr.mean(), lr.std()


def cubic_fit(lr, lv, mu, sd):
    z = (lr - mu) / sd
    X = np.column_stack([np.ones_like(z), z, z ** 2, z ** 3])
    beta, *_ = np.linalg.lstsq(X, lv, rcond=None)
    return beta


def cubic_eval(beta, q, mu, sd):
    z = (q - mu) / sd
    return beta[0] + beta[1] * z + beta[2] * z ** 2 + beta[3] * z ** 3


def pelt(x, pen=PEN):
    lr = np.log(x.RLU.values); mu, sd = _std(lr); z = (lr - mu) / sd
    sig = np.column_stack([np.log(x.value.values), np.ones_like(z), z, z ** 2, z ** 3])
    return rpt.Pelt(model='linear', min_size=MIN_SIZE, jump=1).fit(sig).predict(pen=pen)


def segment_levels(x, bkps):
    """log concentration each segment's curve assigns to the whole-period median RLU."""
    lr = np.log(x.RLU.values); lv = np.log(x.value.values); mu, sd = _std(lr); ref = np.log(np.median(x.RLU.values))
    lev, s0 = [], 0
    for b in bkps:
        beta = cubic_fit(lr[s0:b], lv[s0:b], mu, sd)
        lev.append(cubic_eval(beta, ref, mu, sd)); s0 = b
    return np.array(lev)


def change_points(x, pen=PEN, min_shift=MIN_SHIFT):
    """PELT on the RLU-result relation, then merge neighbours differing by < min_shift.
    Returns breakpoint list (last element = len(x))."""
    b = pelt(x, pen)
    while len(b) > 1:
        lev = segment_levels(x, b)
        diff = np.abs(np.diff(lev)); k = int(np.argmin(diff))
        if diff[k] >= np.log1p(min_shift):
            break
        b = b[:k] + b[k + 1:]
    return b


def window_curve_shift(x, a0, ie, b1, sub=None):
    """Curve shift (log) at the whole-period median RLU from windows [a0, ie) and [ie, b1)."""
    lr = np.log(x.RLU.values); lv = np.log(x.value.values); mu, sd = _std(lr); ref = np.log(np.median(x.RLU.values))
    A = np.arange(a0, ie); B = np.arange(ie, b1)
    if sub is not None:
        A = A[sub[A]]; B = B[sub[B]]
    if len(A) < 8 or len(B) < 8:
        return np.nan
    return cubic_eval(cubic_fit(lr[B], lv[B], mu, sd), ref, mu, sd) - cubic_eval(cubic_fit(lr[A], lv[A], mu, sd), ref, mu, sd)


def local_slope(x, a0, b1):
    """d log(result) / d log(RLU) of the whole-window curve at the window's median RLU."""
    lr = np.log(x.RLU.values[a0:b1]); lv = np.log(x.value.values[a0:b1]); mu, sd = _std(lr)
    beta = cubic_fit(lr, lv, mu, sd); q = np.median(lr); z = (q - mu) / sd
    return (beta[1] + 2 * beta[2] * z + 3 * beta[3] * z ** 2) / sd


# ---------------------------------------------------------------- patient-level shift statistics
def hl(a, b):
    """Hodges-Lehmann shift of b relative to a."""
    return float(np.median(np.subtract.outer(b, a)))


def block_ci(a, b, da, db, rng, reps=2000):
    ua, ub = np.unique(da), np.unique(db)
    ga = {u: a[da == u] for u in ua}; gb = {u: b[db == u] for u in ub}
    bs = [hl(np.concatenate([ga[u] for u in rng.choice(ua, len(ua))]), np.concatenate([gb[u] for u in rng.choice(ub, len(ub))]))
          for _ in range(reps)]
    return np.percentile(bs, [2.5, 97.5])


def perm_p(a, b, da, db, rng, reps=2000):
    """Two-sided day-level permutation test for the Hodges-Lehmann shift."""
    obs = abs(hl(a, b)); vals = np.concatenate([a, b]); days = np.concatenate([da, db])
    ud = np.unique(days); na = len(np.unique(da)); g = {u: vals[days == u] for u in ud}
    cnt = 0
    for _ in range(reps):
        p = rng.permutation(ud)
        cnt += abs(hl(np.concatenate([g[u] for u in p[:na]]), np.concatenate([g[u] for u in p[na:]]))) >= obs
    return (cnt + 1) / (reps + 1)


# ---------------------------------------------------------------- online RLU monitor
def rlu_monitor(lr, lv, T, R, Kr, K0=30, wmax=200):
    """Returns list of (first deviating index, alarm index)."""
    alarms, i0, run, i, n = [], 0, 0, K0, len(lr)
    while i < n:
        a = max(i0, i - wmax); xx, yy = lr[a:i], lv[a:i]
        mu, sd = xx.mean(), (xx.std() or 1.0); z = (xx - mu) / sd
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(z), z, z * z]), yy, rcond=None)
        zq = (lr[i] - mu) / sd
        if abs(lv[i] - (beta[0] + beta[1] * zq + beta[2] * zq * zq)) > T:
            run += 1
            if run >= R:
                s0 = i - R + 1; alarms.append((s0, i)); i0, run = s0, 0; i = s0 + Kr; continue
        else:
            run = 0
        i += 1
    return alarms


# ---------------------------------------------------------------- PBRTQC
from scipy import stats as _st
from scipy.signal import lfilter as _lfilter


class PBRTQC:
    """Moving average/median, EWMA or two-sided CUSUM on Box-Cox-transformed, standardised results.
    outliers='winsor' caps results at the tuning 1st/99th percentiles; 'exclude' removes them.
    by_sex=True standardises each sex with its own tuning mean/SD (stratified variant).
    symmetric=True places the two control limits at equal distances from the tuning median of the statistic
    (default: limits at equal tail quantiles, which are asymmetric when the statistic is skewed)."""

    def __init__(self, kind, p, outliers='winsor', by_sex=False, symmetric=False):
        self.kind, self.p, self.outliers, self.by_sex, self.symmetric = kind, p, outliers, by_sex, symmetric

    def fit_transform(self, tune):
        v = tune.value.values
        self.lam = _st.boxcox(v)[1]
        self.lo_t, self.hi_t = np.percentile(v, [1, 99])
        y = self._bc(np.clip(v, self.lo_t, self.hi_t))
        keep = (v >= self.lo_t) & (v <= self.hi_t)
        if self.by_sex:
            self.mu = {s: y[keep & (tune.sex.values == s)].mean() for s in ['F', 'M']}
            self.sd = {s: y[keep & (tune.sex.values == s)].std() for s in ['F', 'M']}
        else:
            self.mu, self.sd = y[keep].mean(), y[keep].std()
        return self

    def _bc(self, v):
        return _st.boxcox(v, self.lam)

    def z(self, values, sexes):
        """Standardised stream and the indices (into values) that are used."""
        if self.outliers == 'exclude':
            keep = np.where((values >= self.lo_t) & (values <= self.hi_t))[0]
        else:
            keep = np.arange(len(values))
        v = np.clip(values[keep], self.lo_t, self.hi_t); y = self._bc(v)
        if self.by_sex:
            s = sexes[keep]; mu = np.where(s == 'M', self.mu['M'], self.mu['F']); sd = np.where(s == 'M', self.sd['M'], self.sd['F'])
            return (y - mu) / sd, keep
        return (y - self.mu) / self.sd, keep

    def statistic(self, z):
        k, p = self.kind, self.p
        if k == 'MA':
            c = np.cumsum(np.insert(z, 0, 0)); o = np.full(len(z), np.nan); o[p - 1:] = (c[p:] - c[:-p]) / p; return o
        if k == 'MM':
            return pd.Series(z).rolling(p).median().values
        if k == 'EWMA':
            o = _lfilter([p], [1, -(1 - p)], z); o[:int(2 / p)] = np.nan; return o
        if k == 'CUSUM':
            hi = np.empty(len(z)); lo = np.empty(len(z)); a = b = 0.0
            for i, v in enumerate(z):
                a = max(0.0, a + v - p); b = max(0.0, b - v - p); hi[i] = a; lo[i] = b
            return np.maximum(hi, lo)
        raise ValueError(k)

    def alarms(self, stat):
        return np.nan_to_num((stat < self.lo) | (stat > self.hi), nan=0).astype(bool)

    def calibrate(self, tune, far=2.0):
        """Control limits giving <= far alarm episodes per 1,000 tuning results."""
        z, _ = self.z(tune.value.values, tune.sex.values); st = self.statistic(z); valid = st[~np.isnan(st)]
        def episodes(al):
            return int((al & ~np.r_[False, al[:-1]]).sum())
        if self.kind == 'CUSUM':
            self.lo = -np.inf
            for h in np.quantile(valid, np.linspace(.9, 1, 201)):
                self.hi = h
                if episodes(self.alarms(st)) * 1000 / len(z) <= far:
                    break
        elif self.symmetric:
            med = np.median(valid); dev = np.abs(valid - med); best = (med - dev.max(), med + dev.max())
            for c in np.quantile(dev, np.linspace(1, .9, 201)):
                self.lo, self.hi = med - c, med + c
                if episodes(self.alarms(st)) * 1000 / len(z) > far:
                    break
                best = (self.lo, self.hi)
            self.lo, self.hi = best
        else:
            best = (valid.min(), valid.max())
            for q in np.linspace(0, .05, 201):
                self.lo, self.hi = np.quantile(valid, [q / 2, 1 - q / 2])
                if episodes(self.alarms(st)) * 1000 / len(z) > far:
                    break
                best = (self.lo, self.hi)
            self.lo, self.hi = best
        return self

    def alarm_vector(self, values, sexes):
        """Boolean alarm state for every position in values (excluded positions carry the last state)."""
        z, keep = self.z(values, sexes); al = self.alarms(self.statistic(z))
        if len(keep) == len(values):
            return al
        last = np.searchsorted(keep, np.arange(len(values)), side='right') - 1
        return np.where(last >= 0, al[np.clip(last, 0, None)], False)


def pbrtqc_stream(d, test):
    """All numeric results of an analyte in run order (censored results at their limit)."""
    return d[(d.test == test) & d.value.notna() & (d.value > 0)].sort_values('run').reset_index(drop=True)


def simulate_bias(det, x, i_lo, i_hi, biases, nsim, seed, L=400, horizon_days=None, base_alarm=None):
    """Bias-detection simulation on 'clean' start points.

    A multiplicative bias (1 + b) is applied to every result from start st onward, for L results or
    horizon_days days. Start points are eligible only if the unaltered stream has no alarm from st-1 to the
    end of the bias period, so every alarm in the biased stream is caused by the bias. The statistic is
    computed on the whole stream up to the end of the bias period (natural warm-up).
    Returns per bias: probability of detection (Ped), median number of affected results before detection
    (MNPed), median days to detection, and the number of eligible start points."""
    vals, sexes = x.value.values, x.sex.values
    t = x.run.values.astype('datetime64[s]').astype(np.int64)
    if base_alarm is None:
        base_alarm = det.alarm_vector(vals, sexes)
    cs = np.r_[0, np.cumsum(base_alarm)]
    if horizon_days is None:
        ends = np.minimum(np.arange(len(x)) + L, len(x))
        cand = np.arange(max(i_lo, 1), i_hi - L + 1)
    else:
        ends = np.searchsorted(t, t + horizon_days * 86400)
        cand = np.arange(max(i_lo, 1), i_hi)
        cand = cand[t[cand] + horizon_days * 86400 <= t[i_hi - 1]]
    cand = cand[(cs[ends[cand]] - cs[cand - 1]) == 0]
    r = np.random.default_rng(seed); rows = []
    for b in biases:
        k, nped, days = 0, [], []
        starts = r.choice(cand, min(nsim, len(cand)), replace=False) if len(cand) else []
        for st in starts:
            en = ends[st]; v = vals[:en].copy(); v[st:] *= (1 + b / 100)
            al = det.alarm_vector(v, sexes[:en])
            pos = np.where(al[st:])[0]
            if len(pos):
                k += 1; nped.append(pos[0] + 1); days.append((t[st + pos[0]] - t[st]) / 86400)
        n = len(starts)
        rows.append(dict(bias=b, Ped=k / n if n else np.nan, MNPed=np.median(nped) if nped else np.nan,
                         days=np.median(days) if days else np.nan, n_sim=n, eligible=len(cand)))
    return pd.DataFrame(rows)
