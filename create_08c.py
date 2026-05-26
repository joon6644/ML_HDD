import json
import copy

with open('notebooks/08b_final_evaluation_labeled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell index where "3. 최적 운영점 기준 리드타임 심층 분석" starts
# Or we can just append to the end. But the user said: "08b_final_evaluation.ipynb 이거 최종 평가 시 과거의 잔재인 윈도우 기간과 n이 적용돼있는거같은데 빼줘. 그리고 0.9820, 0.9720, 0.9590, 0.9401 이 네가지 임계값으로 threshold Disk FAR ... 이런 표 만들게 해줘. 전에 있던 차트나 그런거는 유지하고. 그 내용들 08c로 만들어줘. 그리고 리드타임 10 20 30 으로 나눠서 분석해줘 각 임계값에 대해서"

new_cells = []

# keep cells 0, 1, 2 (the data loading parts)
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if "3. 최적 운영점 기준 리드타임 심층 분석" in src or "3. 임계값 및 리드타임 심층 분석" in src:
        break
    if "stats_orig = evaluate_detailed_disk_point" in src:
        # replace window_size logic
        src = src.replace("window_size=0", "window_size=0") # just in case
    new_cells.append(c)

# We will now add a new section 3 for the table and section 4 for lead time analysis.

table_markdown = """## 3. 다중 임계값 및 Horizon별 성능 비교
과거 모델에서 사용하던 Alarm Window 및 N-Condition 로직을 제거하고,
단일 알람(1-Condition) 정책 기반으로 4가지 주요 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해
예측 기간(Horizon) 10일, 20일, 30일 기준의 성능 지표를 산출합니다.
"""

table_code = """import pandas as pd
import numpy as np
from src.eval_core import evaluate_detailed_disk_point

thresholds = [0.9820, 0.9720, 0.9590, 0.9401]
horizons = [10, 20, 30]

results = []

for t in thresholds:
    row = {'Threshold': f"{t:.4f}"}
    
    # FAR is independent of horizon, but we evaluate it anyway. We can just take it from horizon 30.
    stats_30 = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=30)
    row['Disk FAR'] = f"{stats_30['far_disk']*100:.2f}%"
    
    for h in horizons:
        stats = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=h)
        row[f'Disk Recall ({h}일)'] = f"{stats['recall_disk']*100:.2f}%"
        row[f'Precision ({h}일)'] = f"{stats['precision_disk']*100:.2f}%"
        
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
True Positive(고장 탐지 디스크)의 **리드타임(Lead Time)** 분포(Mean, 25th, Median, 75th, IQR)를 산출합니다.
"""

leadtime_code = """from IPython.display import display, HTML

for t in thresholds:
    display(HTML(f"<hr><h3>🎯 임계값 T = {t:.4f} 리드타임 분석</h3>"))
    
    for h in horizons:
        stats = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=h)
        lead_times = stats['lead_times']
        
        if len(lead_times) > 0:
            lt_mean = np.mean(lead_times)
            lt_25 = np.percentile(lead_times, 25)
            lt_50 = np.percentile(lead_times, 50)
            lt_75 = np.percentile(lead_times, 75)
            lt_iqr = lt_75 - lt_25
        else:
            lt_mean = lt_25 = lt_50 = lt_75 = lt_iqr = 0.0
            
        print(f"▶ Horizon {h}일 기준 (TP: {len(lead_times)}대)")
        print(f"   - Mean   : {lt_mean:.1f} 일")
        print(f"   - 25th   : {lt_25:.1f} 일")
        print(f"   - Median : {lt_50:.1f} 일")
        print(f"   - 75th   : {lt_75:.1f} 일")
        print(f"   - IQR    : {lt_iqr:.1f} 일")
        print()
"""

# Append new cells
new_cells.append({"cell_type": "markdown", "metadata": {}, "source": [table_markdown]})
new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [table_code]})
new_cells.append({"cell_type": "markdown", "metadata": {}, "source": [leadtime_markdown]})
new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [leadtime_code]})

nb['cells'] = new_cells
nb['cells'][0]['source'] = ["# 08c. 다중 임계값 성능 표 및 리드타임 심층 분석\n"]

with open('notebooks/08c_final_evaluation_labeled.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Created 08c_final_evaluation_labeled.ipynb")
