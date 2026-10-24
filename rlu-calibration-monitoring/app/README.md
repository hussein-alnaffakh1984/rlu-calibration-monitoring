# RLU Calibration Monitor

Author: Hussein Ali Hussein Al-Naffakh, University of Alkafeel.

Static web tool (one `index.html`, no server) that applies the method of the paper
*Automated change-point detection in raw immunoassay signals identifies calibration changes: validation against analyser records and comparison with patient-based quality control*.

## Deploy
Drag the folder (or `index.html`) onto https://app.netlify.com/drop. Any static host works.

## Use
1. Add the analyser result-list exports (.xlsx/.csv), or fill the Excel/CSV template offered on the page. Needed columns: analyte (Chemistry), Result, RLU, Run Date.
   Optional: Sample ID (for duplicate removal), Sex. Files named `men_…` / `women_…` record sex.
2. Pick an analyte. The page shows the current calibration state, five charts (deviation over time, calibration curves,
   curve vs patient shift, signal compensation, monitor trace), change points (curve, signal and patient shifts,
   patient-shift CI and BH-adjusted permutation q) and the online RLU monitor alarms. CSV export for both tables and
   an HTML report with all charts and tables.
3. Settings: start date (default: first day with results for both sexes), PELT penalty, minimum segment,
   merge threshold, window, monitor threshold / run length / re-learning size. Defaults = the study's.

All computation runs in the browser; files are never uploaded.

## Verification
`rlu-core.js` is the analysis core used by the page. `verify_against_python.js` compares it with the published
Python pipeline (Node): change points, curve/signal/patient shifts and monitor alarms agree to < 1e-11 on the study data
(34/34 change points; 35/35 monitor alarms). An automated browser test (46 checks) confirms that every chart, table,
status line and export changes with the uploaded data and the settings. Bootstrap CIs and permutation P use a different random generator, so they
differ slightly between the two implementations.
