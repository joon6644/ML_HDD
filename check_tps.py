import pandas as pd
import numpy as np

df = pd.read_parquet('data2/06_hyperparameter_tuning/val_calib.parquet')
y_prob = np.load('results/predictions/val_calib_probs.npy')
df['prob'] = y_prob

df['_base'] = df['serial_number'].str.replace(r'_\d+$', '', regex=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['_base', 'date'])

failed_bases = df[df['failure'] == 1]['_base'].unique()
df_f = df[df['_base'].isin(failed_bases)]

print('Failed bases:', len(failed_bases))

early_alarms = 0
valid_alarms = 0
no_alarms = 0

T = 0.7053

for b, grp in df_f.groupby('_base'):
    probs = grp['prob'].values
    failures = grp['failure'].values
    alarm_idx = np.where(probs >= T)[0]
    
    if len(alarm_idx) > 0:
        first_alarm = alarm_idx[0]
        if failures[first_alarm] == 0:
            early_alarms += 1
        else:
            valid_alarms += 1
    else:
        no_alarms += 1

print(f"Total Failed Disks: {len(failed_bases)}")
print(f"Early Alarms (failure==0): {early_alarms}")
print(f"Valid Alarms (failure==1): {valid_alarms}")
print(f"No Alarms: {no_alarms}")
