const C = require('./rlu-core.js'); const fs = require('fs');
const lines = fs.readFileSync(require('path').join(__dirname, '..', 'results', 'data_deidentified.csv'), 'utf8').trim().split('\n');
const hdr = lines[0].split(','); const recs = [];
for (const l of lines.slice(1)) { const f = l.split(','); const o = {}; hdr.forEach((h, i) => o[h] = f[i]);
  recs.push({ test: o.test, run: new Date(o.run.replace(' ', 'T') + 'Z'), sex: o.sex, result: o.result, rlu: o.RLU }); }
const P = C.prepare(recs, { start: new Date('2023-09-15T00:00:00Z') });
console.log('records', P.records.length, 'dups', P.duplicates, 'before', P.beforeStart);
const py = fs.readFileSync(require('path').join(__dirname, '..', 'results', 'events.csv'), 'utf8').trim().split('\n'); const ph = py[0].split(',');
const pyev = py.slice(1).map(l => { const f = l.split(','); const o = {}; ph.forEach((h, i) => o[h] = f[i]); return o; });
const pe = fs.readFileSync(require('path').join(__dirname, '..', 'results', 'patient_effects.csv'), 'utf8').trim().split('\n'); const peh = pe[0].split(',');
const pyp = pe.slice(1).map(l => { const f = l.split(','); const o = {}; peh.forEach((h, i) => o[h] = f[i]); return o; });
let maxd = {}; const upd = (k, v) => maxd[k] = Math.max(maxd[k] || 0, Math.abs(v));
for (const t of ['FERR', 'TSH', 'VD-T II', 'VD']) {
  const t0 = Date.now(); const r = C.analyse(P.records, t); const ms = Date.now() - t0;
  const pe_ = pyev.filter(e => e.test === t), pp = pyp.filter(e => e.test === t);
  console.log(t, 'n', r.n, 'events js', r.events.length, 'py', pe_.length, ms + 'ms');
  r.events.forEach((e, i) => { const q = pe_[i], qp = pp[i]; if (!q) return;
    if (+q.index !== e.index) console.log('  index mismatch', e.index, q.index);
    upd('curve', e.curveShift - +q.curve_shift); upd('signal', e.signalShift - +q.signal_shift); upd('range', e.shiftRange - +q.shift_range);
    upd('patient', e.patientShift - +qp.shift); upd('beta', e.beta - +qp.beta);
    if (q.curve_shift_F) upd('sexF', e.curveShiftF - +q.curve_shift_F); });
  const lr = r.x.map(o => Math.log(o.RLU)), lv = r.x.map(o => Math.log(o.value));
  const t1 = Date.now(); const al = C.monitor(lr, lv); console.log('  monitor alarms', al.length, (Date.now() - t1) + 'ms');
}
console.log('max abs diff', maxd);
