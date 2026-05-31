import pandas as pd
from pathlib import Path

val_calib_path = Path('data2/06_hyperparameter_tuning/val_calib.parquet')
df = pd.read_parquet(val_calib_path)

print('=== val_calib 데이터 분석 ===')
print(f'Total rows: {len(df):,}')
print(f'Columns: {df.columns.tolist()}')
print(f'\nFailure distribution:')
print(df['failure'].value_counts().sort_index())
print(f'\nSerial numbers: {df["serial_number"].nunique():,} unique disks')
print(f'\nDate range:')
print(f'  Min: {df["date"].min()}')
print(f'  Max: {df["date"].max()}')

# 각 디스크별로 failure 패턴 분석
print(f'\n=== Disk-level failure pattern ===')
disk_failure_stats = df.groupby('serial_number')['failure'].agg(['sum', 'count', 'min', 'max'])
print(f'Disks with failure==1: {(disk_failure_stats["max"] == 1).sum():,}')
print(f'Disks with all failure==0: {(disk_failure_stats["max"] == 0).sum():,}')

print(f'\nDisk failure row count distribution (disks with failure==1 only):')
failed_disks = disk_failure_stats[disk_failure_stats["max"] == 1]
print(f'  Mean failure rows per disk: {failed_disks["sum"].mean():.1f}')
print(f'  Min: {failed_disks["sum"].min():.0f}')
print(f'  Max: {failed_disks["sum"].max():.0f}')
print(f'  Median: {failed_disks["sum"].median():.0f}')

# Sample: failure==1인 디스크 하나 살펴보기
sample_disk = df[df['failure'] == 1]['serial_number'].iloc[0]
sample_data = df[df['serial_number'] == sample_disk].sort_values('date')
print(f'\nSample disk: {sample_disk}')
print(f'  Total rows: {len(sample_data)}')
print(f'  failure==1 rows: {(sample_data["failure"] == 1).sum()}')
print(f'  Failure pattern (first 20 rows):')
print(sample_data[['date', 'failure']].head(20).to_string(index=False))
