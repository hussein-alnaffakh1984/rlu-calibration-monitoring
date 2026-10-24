/* RLU calibration monitor — analysis core.
 * Port of the published analysis pipeline (Python: numpy / ruptures 1.1.10) to plain JavaScript.
 * Works in the browser (window.RLUCore), in a Web Worker and in Node (module.exports).
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ small numerics
  const mean = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  const stdPop = a => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / a.length); };
  function median(a) {
    const b = Float64Array.from(a).sort(); const n = b.length;
    if (!n) return NaN;
    return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
  }
  function percentile(a, p) {   // numpy 'linear'
    const b = Float64Array.from(a).sort(); const n = b.length;
    const h = (n - 1) * p / 100, lo = Math.floor(h), hi = Math.ceil(h);
    return b[lo] + (b[hi] - b[lo]) * (h - lo);
  }
  // quickselect median for large arrays (in place)
  function qmedian(arr) {
    const n = arr.length;
    const sel = k => {
      let l = 0, r = n - 1;
      while (l < r) {
        const piv = arr[(l + r) >> 1]; let i = l, j = r;
        while (i <= j) { while (arr[i] < piv) i++; while (arr[j] > piv) j--; if (i <= j) { const t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; } }
        if (k <= j) r = j; else if (k >= i) l = i; else return arr[k];
      }
      return arr[k];
    };
    if (n % 2) return sel((n - 1) / 2);
    const a = sel(n / 2 - 1); let b = Infinity; for (let i = n / 2; i < n; i++) if (arr[i] < b) b = arr[i];
    // after sel(n/2-1), elements right of position are >= a; the next order statistic is their minimum
    return (a + b) / 2;
  }

  // solve symmetric positive (semi)definite system by Gaussian elimination with partial pivoting
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-300) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }

  // least squares of y on polynomial in z (degree deg) for index list / range
  function polyFit(z, y, idx, deg) {
    const p = deg + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
    for (const i of idx) {
      const zi = z[i]; const pw = [1]; for (let k = 1; k < p; k++) pw.push(pw[k - 1] * zi);
      for (let r = 0; r < p; r++) { b[r] += pw[r] * y[i]; for (let c = 0; c < p; c++) A[r][c] += pw[r] * pw[c]; }
    }
    return solve(A, b);
  }
  const polyEval = (beta, z) => { let s = 0, pw = 1; for (let k = 0; k < beta.length; k++) { s += beta[k] * pw; pw *= z; } return s; };
  const range = (a, b) => { const r = []; for (let i = a; i < b; i++) r.push(i); return r; };

  // ------------------------------------------------------------------ parsing
  function parseResult(s) {
    s = String(s == null ? '' : s).trim();
    if (s === '' || s === 'nan') return { value: NaN, censor: '' };
    if (s[0] === '<' || s[0] === '>') return { value: parseFloat(s.slice(1)), censor: s[0] };
    const v = Number(s); return isNaN(v) ? { value: NaN, censor: '?' } : { value: v, censor: '' };
  }

  /** records: [{test, run(Date), sex, result, rlu}] → dedup, parse, sort. */
  function prepare(records, opts = {}) {
    const seen = new Set(); const out = []; let dup = 0, norun = 0;
    for (const r of records) {
      const t = r.run instanceof Date ? r.run : new Date(typeof r.run === 'number' ? r.run : NaN);
      if (!(t instanceof Date) || isNaN(t)) { norun++; continue; }
      const key = [r.key || '', r.test, t.getTime(), r.result, r.rlu, r.sex].join('|');
      if (seen.has(key)) { dup++; continue; }
      seen.add(key);
      const p = parseResult(r.result);
      out.push({ test: String(r.test).trim(), run: t, sex: r.sex || '', value: p.value, censor: p.censor, RLU: Number(r.rlu) });
    }
    let kept = out;
    if (opts.start) kept = out.filter(r => r.run >= opts.start);
    kept.sort((a, b) => a.run - b.run);
    return { records: kept, duplicates: dup, withoutRunTime: norun, beforeStart: out.length - kept.length };
  }

  const curveStream = (recs, test) => recs.filter(r => r.test === test && r.censor === '' && r.value > 0 && r.RLU > 0);
  const patientStream = (recs, test) => recs.filter(r => r.test === test && r.censor === '' && r.value > 0);

  // ------------------------------------------------------------------ change points (PELT, linear cost)
  function pelt(lv, Z, minSize, pen) {
    // Z: array of rows [1, z, z^2, z^3]; cost = RSS of OLS fit of lv on Z over [s, e)
    const n = lv.length, p = Z[0].length;
    const nsq = p * (p + 1) / 2;
    const cXX = new Float64Array((n + 1) * nsq), cXy = new Float64Array((n + 1) * p), cyy = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      const row = Z[i], o = (i + 1) * nsq, oo = i * nsq; let q = 0;
      for (let a = 0; a < p; a++) for (let b = a; b < p; b++) { cXX[o + q] = cXX[oo + q] + row[a] * row[b]; q++; }
      for (let a = 0; a < p; a++) cXy[(i + 1) * p + a] = cXy[i * p + a] + row[a] * lv[i];
      cyy[i + 1] = cyy[i] + lv[i] * lv[i];
    }
    const cost = (s, e) => {
      const A = Array.from({ length: p }, () => new Array(p)), b = new Array(p); let q = 0;
      for (let a = 0; a < p; a++) for (let c = a; c < p; c++) { const v = cXX[e * nsq + q] - cXX[s * nsq + q]; A[a][c] = v; A[c][a] = v; q++; }
      for (let a = 0; a < p; a++) b[a] = cXy[e * p + a] - cXy[s * p + a];
      const beta = solve(A, b); const yy = cyy[e] - cyy[s];
      if (!beta) return yy;
      let bt = 0; for (let a = 0; a < p; a++) bt += beta[a] * b[a];
      return Math.max(yy - bt, 0);
    };
    // ruptures Pelt._seg with jump = 1
    const parts = new Map(); parts.set(0, { bk: [], total: 0 });
    let adm = [];
    const ind = []; for (let k = minSize; k < n; k++) ind.push(k); ind.push(n);
    for (const bkp of ind) {
      adm.push(bkp - minSize);
      const subs = [];
      for (const t of adm) {
        const pt = parts.get(t); if (!pt) { subs.push(null); continue; }
        subs.push({ t, total: pt.total + cost(t, bkp) + pen, bk: pt.bk });
      }
      let best = null; for (const s of subs) if (s && (!best || s.total < best.total)) best = s;
      parts.set(bkp, { bk: best.bk.concat([bkp]), total: best.total });
      const keep = []; for (let i = 0; i < adm.length; i++) if (subs[i] && subs[i].total <= best.total + pen) keep.push(adm[i]);
      adm = keep;
    }
    return parts.get(n).bk;
  }

  function changePoints(x, { pen = 0.2, minSize = 30, minShift = 0.05 } = {}) {
    const lr = x.map(r => Math.log(r.RLU)), lv = x.map(r => Math.log(r.value));
    const mu = mean(lr), sd = stdPop(lr), z = lr.map(v => (v - mu) / sd);
    const Z = z.map(v => [1, v, v * v, v * v * v]);
    let b = pelt(lv, Z, minSize, pen);
    const ref = (Math.log(median(x.map(r => r.RLU))) - mu) / sd;
    const levels = bk => { let s = 0; return bk.map(e => { const beta = polyFit(z, lv, range(s, e), 3); s = e; return polyEval(beta, ref); }); };
    while (b.length > 1) {
      const lev = levels(b); let k = 0, dmin = Infinity;
      for (let i = 0; i + 1 < lev.length; i++) { const d = Math.abs(lev[i + 1] - lev[i]); if (d < dmin) { dmin = d; k = i; } }
      if (dmin >= Math.log1p(minShift)) break;
      b = b.slice(0, k).concat(b.slice(k + 1));
    }
    return { bkps: b, z, lv, lr, mu, sd, ref };
  }

  // ------------------------------------------------------------------ Hodges–Lehmann and resampling
  function hl(a, b) {
    const d = new Float64Array(a.length * b.length); let q = 0;
    for (let j = 0; j < b.length; j++) for (let i = 0; i < a.length; i++) d[q++] = b[j] - a[i];
    return qmedian(d);
  }
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  function blockCI(a, b, da, db, reps, seed) {
    const r = rng(seed), ga = groupBy(a, da), gb = groupBy(b, db), ka = [...ga.keys()], kb = [...gb.keys()], out = [];
    for (let k = 0; k < reps; k++) {
      const A = [], B = [];
      for (let i = 0; i < ka.length; i++) A.push(...ga.get(ka[Math.floor(r() * ka.length)]));
      for (let i = 0; i < kb.length; i++) B.push(...gb.get(kb[Math.floor(r() * kb.length)]));
      out.push(hl(A, B));
    }
    return [percentile(out, 2.5), percentile(out, 97.5)];
  }
  function groupBy(v, keys) { const m = new Map(); v.forEach((x, i) => { const k = keys[i]; if (!m.has(k)) m.set(k, []); m.get(k).push(x); }); return m; }
  const dayKey = d => d.getUTCFullYear() + '-' + (d.getUTCMonth() + 1) + '-' + d.getUTCDate();   // times are naive (stored as UTC)

  // ------------------------------------------------------------------ events
  function searchsorted(times, t) { let lo = 0, hi = times.length; while (lo < hi) { const m = (lo + hi) >> 1; if (times[m] < t) lo = m + 1; else hi = m; } return lo; }

  /** Full analysis of one analyte. Returns change points with curve, signal and patient shifts. */
  function analyse(recs, test, opts = {}) {
    const W = opts.window || 150;
    const x = curveStream(recs, test);
    if (x.length < 2 * (opts.minSize || 30)) return { test, n: x.length, error: 'too few results' };
    const cp = changePoints(x, opts);
    const { bkps, z, lv, lr, mu, sd, ref } = cp;
    const starts = [0].concat(bkps.slice(0, -1));
    // segments
    const segs = []; let s0 = 0;
    for (const e of bkps) {
      const idx = range(s0, e), beta = polyFit(z, lv, idx, 3);
      const res = idx.map(i => lv[i] - polyEval(beta, z[i]));
      segs.push({ start: x[s0].run, end: x[e - 1].run, n: e - s0, level: Math.exp(polyEval(beta, ref)), residSD: stdPop(res) });
      s0 = e;
    }
    // whole-period curve and deviations (for the chart)
    const betaAll = polyFit(z, lv, range(0, x.length), 3);
    const deviation = x.map((r, i) => 100 * Math.expm1(lv[i] - polyEval(betaAll, z[i])));
    // patient stream for patient shifts
    const ps = patientStream(recs, test), pt = ps.map(r => r.run.getTime()), plv = ps.map(r => Math.log(r.value));
    const gaps = x.map((r, i) => i ? (r.run - x[i - 1].run) / 864e5 : NaN);
    const events = [];
    const evTimes = bkps.slice(0, -1).map(i => x[i].run.getTime());
    const pidx = evTimes.map(t => searchsorted(pt, t));
    for (let k = 1; k < bkps.length; k++) {
      const ie = bkps[k - 1], a0 = Math.max(starts[k - 1], ie - W), b1 = Math.min(bkps[k], ie + W);
      const A = range(a0, ie), B = range(ie, b1);
      const fA = polyFit(z, lv, A, 3), fB = polyFit(z, lv, B, 3);
      const curve = 100 * Math.expm1(polyEval(fB, ref) - polyEval(fA, ref));
      const q = [10, 50, 90].map(p => (percentile(lr.slice(a0, b1), p) - mu) / sd);
      const sh = q.map(v => 100 * Math.expm1(polyEval(fB, v) - polyEval(fA, v)));
      const sexShift = sx => {
        const a = A.filter(i => x[i].sex === sx), b = B.filter(i => x[i].sex === sx);
        if (a.length < 8 || b.length < 8) return NaN;
        return 100 * Math.expm1(polyEval(polyFit(z, lv, b, 3), ref) - polyEval(polyFit(z, lv, a, 3), ref));
      };
      const signal = 100 * Math.expm1(hl(A.map(i => lr[i]), B.map(i => lr[i])));
      // local slope beta = d log(result) / d log(RLU) of the window curve at the window's median log RLU
      const wl = lr.slice(a0, b1), wm = mean(wl), ws = stdPop(wl), wz = wl.map(v => (v - wm) / ws);
      const bw = polyFit(wz, lv.slice(a0, b1), range(0, wl.length), 3), zq = (median(wl) - wm) / ws;
      const beta = (bw[1] + 2 * bw[2] * zq + 3 * bw[3] * zq * zq) / ws;
      // patient windows
      const pi = pidx[k - 1], pv = k > 1 ? pidx[k - 2] : 0, nx = k < pidx.length ? pidx[k] : ps.length;
      const PA = range(Math.max(pv, pi - W), pi), PB = range(pi, Math.min(nx, pi + W));
      const patient = 100 * Math.expm1(hl(PA.map(i => plv[i]), PB.map(i => plv[i])));
      let gapNear = false; for (let i = Math.max(1, ie - 5); i <= Math.min(x.length - 1, ie + 5); i++) if (gaps[i] >= 2) gapNear = true;
      events.push({
        k, time: x[ie].run, index: ie, nBefore: A.length, nAfter: B.length, curveShift: curve,
        shiftP10: sh[0], shiftP50: sh[1], shiftP90: sh[2], shiftRange: Math.max(...sh) - Math.min(...sh),
        curveShiftF: sexShift('F'), curveShiftM: sexShift('M'), signalShift: signal, beta,
        patientShift: patient, gapBefore: gaps[ie], gapNear,
        _patient: { a: PA.map(i => plv[i]), b: PB.map(i => plv[i]), da: PA.map(i => dayKey(ps[i].run)), db: PB.map(i => dayKey(ps[i].run)) },
      });
    }
    return { test, n: x.length, nPatient: ps.length, x, deviation, segments: segs, events, bkps };
  }

  /** Confidence intervals for patient shifts (day-level block bootstrap). Heavy: run in a worker. */
  function patientCIs(events, reps = 500, seed = 2026, onProgress) {
    return events.map((e, i) => {
      const p = e._patient; const ci = blockCI(p.a, p.b, p.da, p.db, reps, seed + i);
      if (onProgress) onProgress((i + 1) / events.length);
      return [100 * Math.expm1(ci[0]), 100 * Math.expm1(ci[1])];
    });
  }

  /** Two-sided day-level permutation P value for the Hodges–Lehmann shift. */
  function permP(a, b, da, db, reps = 2000, seed = 31) {
    const r = rng(seed), obs = Math.abs(hl(a, b)), vals = a.concat(b), days = da.concat(db);
    const g = groupBy(vals, days), ud = [...g.keys()], na = new Set(da).size; let cnt = 0;
    for (let k = 0; k < reps; k++) {
      for (let i = ud.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ud[i], ud[j]] = [ud[j], ud[i]]; }
      const A = [], B = [];
      for (let i = 0; i < ud.length; i++) (i < na ? A : B).push(...g.get(ud[i]));
      if (Math.abs(hl(A, B)) >= obs) cnt++;
    }
    return (cnt + 1) / (reps + 1);
  }
  /** Benjamini–Hochberg adjusted P values. */
  function bh(p) {
    const n = p.length, o = p.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), q = new Array(n); let m = 1;
    for (let k = n - 1; k >= 0; k--) { m = Math.min(m, o[k][0] * n / (k + 1)); q[o[k][1]] = m; }
    return q;
  }
  /** Patient-shift inference for a list of events: bootstrap CI + permutation P (heavy; use a worker). */
  function patientStats(events, { reps = 500, perms = 2000, seed = 2026 } = {}, onProgress) {
    return events.map((e, i) => {
      const p = e._patient;
      const ci = blockCI(p.a, p.b, p.da, p.db, reps, seed + i), pv = permP(p.a, p.b, p.da, p.db, perms, seed + 1000 + i);
      if (onProgress) onProgress((i + 1) / events.length);
      return { lo: 100 * Math.expm1(ci[0]), hi: 100 * Math.expm1(ci[1]), p: pv };
    });
  }

  // ------------------------------------------------------------------ online RLU monitor
  /** trace (optional array of length n) receives each result's log deviation from the monitor's prediction. */
  function monitor(lr, lv, T = 0.08, R = 4, Kr = 10, K0 = 30, wmax = 200, trace = null) {
    const alarms = []; let i0 = 0, run = 0, i = K0; const n = lr.length;
    while (i < n) {
      const a = Math.max(i0, i - wmax), xs = lr.slice(a, i), ys = lv.slice(a, i);
      const mu = mean(xs); let sd = stdPop(xs); if (!sd) sd = 1;
      const zz = xs.map(v => (v - mu) / sd), beta = polyFit(zz, ys, range(0, zz.length), 2);
      const zq = (lr[i] - mu) / sd;
      const dev = beta ? lv[i] - polyEval(beta, zq) : 0;
      if (trace) trace[i] = dev;
      if (Math.abs(dev) > T) {
        run++;
        if (run >= R) { const s0 = i - R + 1; alarms.push({ first: s0, raised: i }); i0 = s0; run = 0; i = s0 + Kr; continue; }
      } else run = 0;
      i++;
    }
    return alarms;
  }

  const api = { polyFit, polyEval, parseResult, prepare, curveStream, patientStream, changePoints, analyse, patientCIs, patientStats, permP, bh, monitor, hl, median, percentile, rng };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.RLUCore = api;
})(typeof self !== 'undefined' ? self : this);
