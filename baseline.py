import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import auc
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가 (src 모듈을 불러오기 위함)
ROOT = Path(os.path.abspath(__file__)).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_core import prepare_disk_level_data, run_disk_level_grid_search, evaluate_detailed_disk_point

# 1. 설정 및 경로
train_path = ROOT / "data2" / "03_splitting" / "train_raw.parquet"
val_calib_path = ROOT / "data2" / "03_splitting" / "val_calib_raw.parquet"
test_path = ROOT / "data2" / "03_splitting" / "test_raw.parquet"

features = [
    'smart_5_raw', 'smart_184_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw',
    'Timeout_5s', 'Timeout_Total', 'seek_error_count', 'smart_9_raw', 'smart_183_raw',
    'smart_189_raw', 'smart_190_raw', 'smart_191_raw', 'smart_194_raw', 'smart_199_raw',
    'smart_241_raw', 'smart_242_raw', 'Total_Reads', 'total_seeks'
]
target = 'failure'

use_cols = ['serial_number', 'date', target] + features

# 2. 데이터 로드 및 전처리
print("Loading data...")
df_train = pd.read_parquet(train_path, columns=use_cols)
df_calib = pd.read_parquet(val_calib_path, columns=use_cols)
df_test = pd.read_parquet(test_path, columns=use_cols)

# eval_core 내부적으로 datetime 변환이 필요할 수 있으므로 수행
for df in [df_train, df_calib, df_test]:
    df['date'] = pd.to_datetime(df['date'])

# 3. 모델 학습 (행 단위)
print("Training LightGBM model...")
X_train = df_train[features]
y_train = df_train[target]

clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

del df_train, X_train, y_train

# 4. Calibration 데이터로 임계값 도출 (06e 노트북과 완벽히 동일한 함수 활용)
print("Tuning threshold on calibration data at disk level (with Horizon 30 days)...")
calib_probs = clf.predict_proba(df_calib[features])[:, 1]
disks_data_calib, n_failed_calib, n_normal_calib = prepare_disk_level_data(
    df_calib, calib_probs, target_col=target, serial_col='serial_number', date_col='date'
)

thresholds_grid = np.linspace(0.001, 0.999, 1000)
df_grid_calib = run_disk_level_grid_search(
    disks_data_calib,
    thresholds_grid,
    [1],  # min_alarms_list = [1]
    n_failed_calib,
    n_normal_calib,
    log_dir=None,
    window_size=0,
    horizon=30
)

# FPR 1% 이하 기준 임계값 탐색
sub_calib = df_grid_calib[df_grid_calib['far'] <= 0.010]
if len(sub_calib) > 0:
    pt_calib = sub_calib.sort_values('recall', ascending=False).iloc[0]
else:
    pt_calib = df_grid_calib.sort_values('far').iloc[0]

optimal_threshold = pt_calib['threshold']
print(f"Optimal Threshold (from calib): {optimal_threshold:.4f} (Calib FPR: {pt_calib['far']:.4f})")

# 5. 테스트 데이터 평가 (06e 노트북과 동일한 방식으로 PR-AUC 및 지표 산출)
print("Predicting and evaluating on test data at disk level...")
test_probs = clf.predict_proba(df_test[features])[:, 1]
disks_data_test, n_failed_test, n_normal_test = prepare_disk_level_data(
    df_test, test_probs, target_col=target, serial_col='serial_number', date_col='date'
)

df_grid_test = run_disk_level_grid_search(
    disks_data_test,
    thresholds_grid,
    [1],
    n_failed_test,
    n_normal_test,
    log_dir=None,
    window_size=0,
    horizon=30
)

# 06e 노트북과 완전히 동일한 방식으로 PR-AUC 산출
df_sorted = df_grid_test.sort_values(by="recall").copy()
df_sorted["precision"] = df_sorted["tps"] / (df_sorted["tps"] + df_sorted["fps"] + 1e-8)
df_sorted["precision"] = df_sorted["precision"].fillna(1.0)
entity_prauc = float(auc(df_sorted["recall"].values, df_sorted["precision"].values))

print(f"Test Disk-level PR-AUC (Horizon 30D): {entity_prauc:.5f}")

# 6. 상세 평가 지점 도출
stats = evaluate_detailed_disk_point(
    disks_data_test,
    optimal_threshold,
    1,
    n_failed_test,
    n_normal_test,
    far_cap=None,
    window_size=0,
    horizon=30
)

test_fpr = stats['far']
test_precision = stats['precision']
test_recall = stats['recall']
mean_lead_time = stats['lead_time']

# 중앙값 직접 산출
lead_times = []
for disk in disks_data_test:
    if disk['is_failed'] == 1:
        probs = disk['probs']
        y_pred = (probs >= optimal_threshold).astype(int)
        if y_pred.sum() > 0:
            trigger_idx = np.where(y_pred == 1)[0][0]
            trigger_date = pd.to_datetime(disk['dates'][trigger_idx])
            last_date = pd.to_datetime(disk['dates'][-1])
            lead_time = (last_date - trigger_date).days
            if lead_time <= 30:
                lead_times.append(lead_time)

median_lead_time = np.median(lead_times) if lead_times else 0.0

print("-" * 40)
print(f"[ Test Evaluation Results (Tuned on Calib FPR <= 1%, Horizon=30D) ]")
print(f"Applied Threshold : {optimal_threshold:.4f}")
print(f"Test Actual FPR   : {test_fpr:.4f}")
print(f"Test Precision    : {test_precision:.4f}")
print(f"Test Recall       : {test_recall:.4f}")
print(f"Mean Lead Time    : {mean_lead_time:.1f} days")
print(f"Median Lead Time  : {median_lead_time:.1f} days")
print("-" * 40)
