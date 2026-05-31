import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import auc
from src.eval_core import prepare_disk_level_data, run_disk_level_grid_search

model_dir = 'results/models/seed_42'
with open(f'{model_dir}/feature_cols.json') as f:
    feature_cols = json.load(f)

models = []
for i in range(5):
    models.append(joblib.load(f'{model_dir}/subset_0{i}.pkl'))

def evaluate_dataset(path, name):
    print(f"\nEvaluating {name}...")
    df = pd.read_parquet(path)
    X = df[feature_cols].values.astype(np.float32)
    y_true = df['failure'].values
    
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    
    df_eval = pd.DataFrame({
        'serial_number': df['serial_number'],
        'date': df['date'],
        'failure': y_true
    })
    
    disks_data, n_failed, n_normal = prepare_disk_level_data(df_eval, probs)
    thresholds = np.linspace(0.001, 0.999, 100)
    
    df_grid = run_disk_level_grid_search(
        disks_data, thresholds, [1], n_failed, n_normal,
        log_dir=None, window_size=0, horizon=30
    )
    
    df_sorted = df_grid.sort_values(by="recall").copy()
    df_sorted["precision"] = df_sorted["tps"] / (df_sorted["tps"] + df_sorted["fps"] + 1e-8)
    df_sorted["precision"] = df_sorted["precision"].fillna(1.0)
    prauc = float(auc(df_sorted["recall"].values, df_sorted["precision"].values))
    
    print(f"  Failed Disks: {n_failed}, Normal Disks: {n_normal}")
    print(f"  Disk Rolling PR-AUC: {prauc:.5f}")

    # Check max probabilities distribution
    max_probs_failed = [np.max(d['probs']) for d in disks_data if d['is_failed']]
    max_probs_normal = [np.max(d['probs']) for d in disks_data if not d['is_failed']]
    print(f"  Mean max prob (Failed): {np.mean(max_probs_failed):.4f}")
    print(f"  Mean max prob (Normal): {np.mean(max_probs_normal):.4f}")

evaluate_dataset('data2/06_hyperparameter_tuning/val_tune.parquet', 'val_tune')
evaluate_dataset('data2/06_hyperparameter_tuning/val_calib.parquet', 'val_calib')
