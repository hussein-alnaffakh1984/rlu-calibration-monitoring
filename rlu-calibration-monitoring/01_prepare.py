"""Step 1. Combine the analyser result-list exports and write a de-identified data set.

Input : RLU_RAW_DIR/{men,women}_*.xlsx  (Mindray CL-900i result-list exports, one file per sex and year)
Output: data_deidentified.csv  (analyte, run time, sex, reported result, RLU; no sample or patient identifiers)
        data.pkl               (the same with parsed result and censoring flag, from START onward, when exports
                               exist for both sexes)
If no raw files are present the step starts from data/data_deidentified.csv.
"""
import glob, os
import numpy as np, pandas as pd
from common import RAW_DIR, START, out

KEEP = ['Sample ID', 'Sample Type', 'Ordering Date', 'Chemistry', 'Result', 'RLU', 'Unit', 'Ref Range', 'Run Date']


def parse(s):
    s = str(s).strip()
    if s in ('', 'nan'):
        return np.nan, ''
    if s[0] in '<>':
        return float(s[1:]), s[0]
    try:
        return float(s), ''
    except ValueError:
        return np.nan, '?'


def from_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_DIR, '*.xlsx'))):
        name = os.path.basename(f)
        sex = 'F' if name.startswith('women') else 'M' if name.startswith('men') else None
        if sex is None:
            raise ValueError(f'file name must start with men_ or women_: {name}')
        x = pd.read_excel(f)[KEEP]; x['sex'] = sex; frames.append(x)
    raw = pd.concat(frames, ignore_index=True); n0 = len(raw)
    raw = raw.drop_duplicates(subset=KEEP + ['sex'])          # records exported twice
    raw['run'] = pd.to_datetime(raw['Run Date'], errors='coerce')
    n_norun = int(raw.run.isna().sum())
    raw = raw[raw.run.notna()]
    deid = pd.DataFrame({'test': raw['Chemistry'].values, 'run': raw.run.values, 'sex': raw.sex.values,
                         'result': raw['Result'].astype(str).values, 'RLU': raw['RLU'].values, 'unit': raw['Unit'].values})
    deid = deid.sort_values('run', kind='stable').reset_index(drop=True)
    deid.to_csv(out('data_deidentified.csv'), index=False)
    print(f'records read {n0}; duplicates removed {n0 - len(raw) - n_norun}; without run time {n_norun}; kept {len(deid)}')
    return deid


if __name__ == '__main__':
    if glob.glob(os.path.join(RAW_DIR, '*.xlsx')):
        deid = from_raw()
    else:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data_deidentified.csv')
        deid = pd.read_csv(src if os.path.exists(src) else out('data_deidentified.csv'), parse_dates=['run'],
                           dtype={'result': str})
    pv = deid.result.map(parse)
    deid['value'] = [p[0] for p in pv]; deid['censor'] = [p[1] for p in pv]
    n_before = int((deid.run < START).sum())
    deid = deid[deid.run >= START].reset_index(drop=True)      # both sexes over the same period
    print(f'results before {START:%d %b %Y} (men only) excluded: {n_before}; analysed records: {len(deid)}')
    deid.to_pickle(out('data.pkl'))
    print('run range', deid.run.min(), deid.run.max())
    print('first result by sex:', deid.groupby('sex').run.min().to_dict())
    print(deid.groupby(['test', 'sex']).size().unstack().to_string())
