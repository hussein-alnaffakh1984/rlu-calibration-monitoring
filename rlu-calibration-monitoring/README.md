# Automated detection of immunoassay calibration changes from raw signals (RLU) — analysis code

Code for the paper *Automated change-point detection in raw immunoassay signals identifies calibration changes: validation against analyser records and comparison with patient-based quality control*.
One pipeline regenerates every number, table and figure.

## Data
The patient-level data are **not** in this repository yet: release of the de-identified data set
(`data/data_deidentified.csv`: analyte, run time, sex, result, RLU, unit; no sample numbers, patient identifiers,
ages or ordering dates) awaits the approval of the laboratory and the ethics committee.
To run the pipeline, put the analyser result-list exports in `data/raw/` named `men_*.xlsx` / `women_*.xlsx`,
or place `data_deidentified.csv` in `data/`. The analysis starts on 15 Sep 2023 (`START` in `common.py`).

`data/calibration_log.csv` — the analyser's calibration history (screen Reagent > Calibration Result > History,
calibration dates 24 Oct 2023 – 24 Oct 2026), transcribed from photographs: `test`, `lot`, `serial` (reagent pack),
`flag`, `status` (`Extended`, `Calibrated` or `Cal Failed`), `cal_time`. TSH and VD were listed with all statuses;
ferritin and VD-T II with status `Extended` only. It was obtained after the change-point settings had been fixed.


## Run
```
pip install -r requirements.txt
./run_all.sh     # 1-2 h on two cores; uses data/raw/*.xlsx, or data/data_deidentified.csv when data/raw is empty
```
Paths: `RLU_RAW_DIR` (default `data/raw`), `RLU_OUT_DIR` (default `results`).
Start date: `RLU_START` (default `2023-09-15`). The sensitivity analysis of PBRTQC parameter selection in the paper reruns `01_prepare.py` and `05_pbrtqc.py` with `RLU_START=2023-07-20` and a separate `RLU_OUT_DIR` (copy `events.csv` and `patient_effects.csv` into it first).

| Step | Script | Main outputs |
|---|---|---|
| 1 | `01_prepare.py` | `data.pkl`, `data_deidentified.csv` |
| 2 | `02_changepoints.py` | `events.csv`, `segments.csv`, `changepoints.json`, `validity.json` |
| 3 | `03_patient_effects.py` | `patient_effects.csv`, `patient_effects_summary.json`, `window_sensitivity.csv`, `classification_change.csv` |
| 4 | `04_monitor.py` | `monitor_params.json`, `monitor_injection.csv`, `monitor_alarms.csv` |
| 5 | `05_pbrtqc.py` | `pbrtqc_sim.csv`, `pbrtqc_real.csv`, `pbrtqc_pooled.csv`, `pbrtqc_events.csv`, `pbrtqc_limits.csv`, `pbrtqc_tuning.csv` |
| 6 | `06_volume.py` | `volume.csv` |
| 7 | `07_calibration_log.py` | `calibration_log_match.csv`, `changepoints_vs_log.csv`, `calibration_log_summary.json` |
| 8 | `08_tables_figures.py` | `Fig1.png`–`Fig4.png`, `FigS1.png`, `FigS2.png`, `tables.json`, `pbrtqc_asymmetry.csv` |

Shared functions are in `common.py`. All random processes are seeded.

## Methods in brief
* **Change points.** log(result) modelled as a cubic in standardised log(RLU); PELT (linear cost, minimum segment 30,
  penalty 0.2); adjacent segments merged while their curves differ by < 5 % at the median RLU (3/8/10 % as sensitivity).
* **Patient effects.** Hodges–Lehmann shift of log results within ±150 results; day-level block bootstrap CI;
  day-level permutation p (10,000); Benjamini–Hochberg.
* **RLU monitor.** Quadratic RLU→result prediction from recent results; alarm after R consecutive deviations > T; tuned on year 1.
* **Calibration history.** A logged calibration is located when a change point falls 1 day before to 7 days after it;
  curve shift across each calibration from ±60 results; lot classed against the preceding logged calibration.
* **PBRTQC.** MA, MM, EWMA, CUSUM on Box-Cox, winsorised (primary), sex-standardised or outlier-excluded results;
  limits ≤ 2 alarm episodes/1,000 year-1 results; bias-detection curves on clean start points.

## Software
Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, ruptures 1.1.10, statsmodels 0.15, matplotlib 3.11.

## Web tool
`app/` holds the companion web tool (single static `index.html`; see `app/README.md`). After running the pipeline,
`node app/verify_against_python.js` checks that the JavaScript core reproduces the Python change points.

## Author and licence
Hussein Ali Hussein Al-Naffakh, University of Alkafeel (hussein.alnaffakh@alkafeel.edu.iq). Code released under the MIT licence (see `LICENSE`).
