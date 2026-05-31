import pandas as pd
import numpy as np

df = pd.read_parquet('data2/06_hyperparameter_tuning/val_calib.parquet')
y_prob = np.load('results/predictions/val_calib_probs.npy')
df['prob'] = y_prob

df['_base'] = df['serial_number'].str.replace(r'_\d+$', '', regex=True)

# Max prob per base
max_probs = df.groupby('_base')['prob'].max().reset_index()

# Is failed per base
is_failed = df.groupby('_base')['failure'].max().reset_index()

merged = pd.merge(max_probs, is_failed, on='_base')

failed_probs = merged[merged['failure'] == 1]['prob'].values
normal_probs = merged[merged['failure'] == 0]['prob'].values

print("Failed Disks Max Probabilities (percentiles):")
print(np.percentile(failed_probs, [0, 25, 50, 75, 90, 95, 99, 100]))

print("\nNormal Disks Max Probabilities (percentiles):")
print(np.percentile(normal_probs, [0, 25, 50, 75, 90, 95, 99, 100]))

print("\nMean Max Prob:")
print(f"Failed: {np.mean(failed_probs):.4f}")
print(f"Normal: {np.mean(normal_probs):.4f}")
