import pandas as pd
from pathlib import Path
import numpy as np

# 07a 그리드 서치 로직 분석

val_calib_path = Path('data2/06_hyperparameter_tuning/val_calib.parquet')
df = pd.read_parquet(val_calib_path)

# 07a의 groupby 방식
df_copy = df.copy()
df_copy["_base"] = df_copy["serial_number"].str.replace(r'_\d+$', '', regex=True)

disks = []
for base_sn, grp in df_copy.groupby("_base", sort=False):
    disks.append({
        "is_failed": int(grp["failure"].max()),
        "probs": grp["failure"].values,  # 실제로는 y_prob이지만 여기서는 failure로 대체
    })

n_failed = sum(d["is_failed"] for d in disks)
n_normal = len(disks) - n_failed

print("=== 07a의 집계 단위 분석 ===")
print(f"디스크 총 개수: {len(disks):,}개")
print(f"  고장 디스크 (n_failed): {n_failed:,}개")
print(f"  정상 디스크 (n_normal): {n_normal:,}개")

print(f"\nTP/FP 계산 단위:")
print(f"  - TP = (고장 디스크 중 조건 만족 디스크 개수) → 디스크 단위")
print(f"  - FP = (정상 디스크 중 조건 만족 디스크 개수) → 디스크 단위")
print(f"  - Recall = TP / n_failed = (디스크) / (디스크) → 디스크 기준 지표")
print(f"  - FAR = FP / n_normal = (디스크) / (디스크) → 디스크 기준 지표")

print(f"\n결론: 07a는 {len(disks):,}개 디스크 기준으로 평가")
print(f"(각 그룹은 베이스 serial_number 기준으로 묶인 디스크 엔티티)")
