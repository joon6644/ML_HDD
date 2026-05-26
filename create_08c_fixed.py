import json
import copy

with open('notebooks/08b_final_evaluation_labeled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

# keep cells 0, 1, 2 (the data loading parts)
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if "3. 최적 운영점 기준 리드타임 심층 분석" in src or "3. 다중 임계값 및 Horizon별 성능 비교" in src or "3. 임계값 및 리드타임 심층 분석" in src:
        break
    if "stats_orig = evaluate_detailed_disk_point" in src:
        src = src.replace("window_size=0", "window_size=0")
    new_cells.append(c)

table_markdown = """## 3. 다중 임계값 및 Horizon별 성능 비교
과거 모델에서 사용하던 Alarm Window 및 N-Condition 로직을 제거하고, 단일 알람(1-Condition) 정책 기반으로 4가지 주요 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해 예측 기간(Horizon) 10일, 20일, 30일 기준의 성능 지표를 산출합니다.
"""

table_code = """import pandas as pd
import numpy as np
from src.eval_core import evaluate_detailed_disk_point

thresholds = [0.9820, 0.9720, 0.9590, 0.9401]
horizons = [10, 20, 30]

results = []

for t in thresholds:
    row = {'Threshold': f"{t:.4f}"}
    
    stats_30 = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=30)
    row['Disk FAR'] = f"{stats_30['far']*100:.2f} %"
    
    for h in horizons:
        stats = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=h)
        row[f'Disk Recall ({h}일)'] = f"{stats['recall']*100:.2f}%"
        row[f'Precision ({h}일)'] = f"{stats['precision']*100:.2f}%"
        
    results.append(row)

df_perf = pd.DataFrame(results)
columns_order = ['Threshold', 'Disk FAR', 'Disk Recall (10일)', 'Precision (10일)', 
                 'Disk Recall (20일)', 'Precision (20일)', 'Disk Recall (30일)', 'Precision (30일)']
df_perf = df_perf[columns_order]

from IPython.display import display, HTML
display(HTML("<h3>🚀 성능 평가 지표 (다중 임계값 x Horizon)</h3>"))
display(df_perf)
"""

leadtime_markdown = """## 4. 각 임계값 및 리드타임(Horizon)별 심층 분석
각 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해, Horizon 10일, 20일, 30일 이내에 발생한 
True Positive(고장 탐지 디스크)의 **리드타임(Lead Time)** 누적 탐지율 및 분포를 산출합니다.
"""

leadtime_code = """from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt

for t in thresholds:
    display(HTML(f"<hr><h2 style='color:#2c3e50'>🎯 임계값 T = {t:.4f} 심층 분석</h2>"))
    
    for horizon in horizons:
        lead_times = []
        for d in disks_data:
            if d['is_failed'] == 1:
                probs = d['probs']
                alarms = (probs >= t).astype(int)
                idx = np.where(alarms >= 1)[0]
                if len(idx) > 0:
                    lt = len(probs) - 1 - idx[0]
                    if lt <= horizon:
                        lead_times.append(lt)
        
        lead_times = np.array(lead_times)
        total_failed = n_failed_disks
        total_tp = len(lead_times)
        
        print(f"============================================================")
        print(f"📊 리드타임(Lead Time) 심층 기술통계 (Horizon: {horizon}일 기준)")
        print(f"============================================================")
        print(f"  - 전체 고장 디스크 수 : {total_failed:,} 개")
        print(f"  - {horizon}일 이내 탐지 성공(TP): {total_tp:,} 개 (Recall: {total_tp/total_failed*100:.2f}%)")
        print(f"")
        if total_tp > 0:
            print(f"  [기술통계 요약]")
            print(f"    • 평균(Mean)   : {np.mean(lead_times):.2f} 일")
            print(f"    • 표준편차(Std): {np.std(lead_times):.2f} 일")
            print(f"    • 최소(Min)    : {np.min(lead_times)} 일")
            print(f"    • 10% 백분위수 : {np.percentile(lead_times, 10):.1f} 일")
            print(f"    • 25% 백분위수 : {np.percentile(lead_times, 25):.1f} 일")
            print(f"    • 중앙값(Med)  : {np.median(lead_times):.1f} 일")
            print(f"    • 75% 백분위수 : {np.percentile(lead_times, 75):.1f} 일")
            print(f"    • 90% 백분위수 : {np.percentile(lead_times, 90):.1f} 일")
            print(f"    • 최대(Max)    : {np.max(lead_times)} 일")
            print(f"")
            print(f"============================================================")
            print(f"📈 구간별 누적 탐지 비율 (Cumulative Detection Rate)")
            print(f"============================================================")
            
            days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30]
            days_thresholds = [d for d in days_thresholds if d <= horizon]
            
            print(f"  리드타임 구간 | 탐지 디스크 수 | 탐지 건(TP) 중 비율 | 전체 고장 대비 비율")
            print(f"  ------------------------------------------------------------")
            for d_th in days_thresholds:
                detected_within = np.sum(lead_times <= d_th)
                ratio_tp = detected_within / total_tp * 100
                ratio_all = detected_within / total_failed * 100
                print(f"   <= {d_th:2d}일 전   | {detected_within:10,} 개 | {ratio_tp:15.1f}% | {ratio_all:16.2f}%")
            print(f"============================================================\\n")
"""

new_cells.append({"cell_type": "markdown", "metadata": {}, "source": [table_markdown]})
new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [table_code]})
new_cells.append({"cell_type": "markdown", "metadata": {}, "source": [leadtime_markdown]})
new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [leadtime_code]})

nb['cells'] = new_cells
nb['cells'][0]['source'] = ["# 08c. 다중 임계값 성능 표 및 리드타임 심층 분석\n"]

with open('notebooks/08c_final_evaluation_labeled.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Created 08c_final_evaluation_labeled.ipynb successfully")
