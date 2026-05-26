import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

import config.final_eval_config as cfg
from src.eval_core import load_saved_ensemble, prepare_disk_level_data, evaluate_detailed_disk_point

artifacts = load_saved_ensemble(cfg, require_threshold=True)
BEST_T = getattr(cfg, "MANUAL_BEST_T", None)
if BEST_T is None: BEST_T = artifacts["threshold"]
BEST_N = getattr(cfg, "MANUAL_BEST_N", None)
if BEST_N is None: BEST_N = artifacts["min_alarms"]
ALARM_WINDOW = getattr(cfg, "ALARM_WINDOW", artifacts.get("window_size", 14))

df_test = pd.read_parquet(cfg.TEST_PATH)
y_prob = np.load(os.path.join(cfg.MODEL_SAVE_DIR, "test_with_failure_date_probs.npy"))

disks_data, n_failed_disks, n_normal_disks = prepare_disk_level_data(
    df_test, y_prob, target_col=cfg.TARGET_COL, serial_col=cfg.SERIAL_COL, date_col=cfg.DATE_COL
)

df_best_disk_orig = []
for disk in disks_data:
    y_pred = (disk['probs'] >= BEST_T).astype(int)
    rolling_alarms = pd.Series(y_pred).rolling(window=ALARM_WINDOW, min_periods=1).sum().values
    is_alarmed = int((rolling_alarms >= BEST_N).any())
    df_best_disk_orig.append({'is_failed': disk['is_failed'], 'is_alarmed': is_alarmed})
df_orig = pd.DataFrame(df_best_disk_orig)
tp_orig = ((df_orig['is_failed'] == 1) & (df_orig['is_alarmed'] == 1)).sum()

df_best_disk_masked = []
for disk in disks_data:
    if disk['is_failed'] == 1:
        dates = pd.Series(pd.to_datetime(disk['dates']))
        last_date = dates.iloc[-1]
        days_to_fail = (last_date - dates).dt.days
        
        y_pred = (disk['probs'] >= BEST_T).astype(int)
        rolling_alarms = pd.Series(y_pred).rolling(window=ALARM_WINDOW, min_periods=1).sum().values
        
        # Is there any alarm within the horizon?
        is_alarmed = int(((rolling_alarms >= BEST_N) & (days_to_fail <= horizon)).any())
    else:
        y_pred = (disk['probs'] >= BEST_T).astype(int)
        rolling_alarms = pd.Series(y_pred).rolling(window=ALARM_WINDOW, min_periods=1).sum().values
        is_alarmed = int((rolling_alarms >= BEST_N).any())

    df_best_disk_masked.append({'is_failed': disk['is_failed'], 'is_alarmed': is_alarmed})
df_masked = pd.DataFrame(df_best_disk_masked)
tp_masked = ((df_masked['is_failed'] == 1) & (df_masked['is_alarmed'] == 1)).sum()

print(f"T={BEST_T}, N={BEST_N}, W={ALARM_WINDOW}")
print(f"Orig TP: {tp_orig}, Masked TP: {tp_masked}")
