import pandas as pd
from pathlib import Path
import numpy as np

val_calib_path = Path('data2/06_hyperparameter_tuning/val_calib.parquet')
df = pd.read_parquet(val_calib_path)

# Sample: failure==1인 디스크 하나 살펴보기
sample_disk = df[df['failure'] == 1]['serial_number'].iloc[0]
sample_data = df[df['serial_number'] == sample_disk].sort_values('date')
print(f'Sample disk: {sample_disk}')
print(f'  Total rows: {len(sample_data)}')
print(f'  failure==1 rows: {(sample_data["failure"] == 1).sum()}')

# failure 패턴 전체 보기
failure_arr = sample_data['failure'].values
date_arr = sample_data['date'].values

# 연속된 failure==1 구간 찾기
idx = 0
segments = []
while idx < len(failure_arr):
    if failure_arr[idx] == 1:
        start = idx
        while idx < len(failure_arr) and failure_arr[idx] == 1:
            idx += 1
        segments.append((start, idx-1, date_arr[start], date_arr[idx-1]))
    else:
        idx += 1

print(f'\nFailure segments (연속된 failure==1):')
for seg_idx, (start, end, start_date, end_date) in enumerate(segments):
    print(f'  Segment {seg_idx+1}: rows [{start}:{end+1}], dates {start_date} ~ {end_date}')

print(f'\nAll rows around failure segment:')
# failure==1 구간 중앙값 주변 출력
if segments:
    start, end = segments[-1][0], segments[-1][1]
    view_start = max(0, start - 10)
    view_end = min(len(sample_data), end + 10)
    view_data = sample_data.iloc[view_start:view_end][['date', 'failure']]
    for i, (idx, row) in enumerate(view_data.iterrows()):
        marker = '>>>' if view_start + i >= start and view_start + i <= end else '   '
        print(f'{marker} Row {view_start + i:3d}: {row["date"].date()} failure={int(row["failure"])}')
