import json

filename = 'notebooks/08c_final_evaluation_labeled.ipynb'

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 8 (markdown) for Section 5
new_markdown = """## 5. 최적 운영점 (T=0.9590) 기준 구간별 리드타임 및 성능 지표 (전체 리드타임 포함)
최적 운영점 `T=0.9590`에서 탐지된 전체 고장 디스크(TP)를 기준으로 삼아, 30일 이내의 각 리드타임 구간별 누적 탐지율을 산출합니다.
맨 아래에 `전체 리드타임` 행을 추가하여, 30일 이하의 구간들이 100%가 아닌 전체 대비 실제 비율로 표현되도록 구성했습니다.
또한 오탐율(FAR), 리콜(Recall), 정밀도(Precision) 지표를 함께 테이블에 포함하여 출력합니다.
"""

# Update cell 9 (code) for Section 5
new_code = """from IPython.display import display, HTML
import pandas as pd
import numpy as np
from src.eval_core import evaluate_detailed_disk_point

t_val = 0.9590
days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30]

# 1. Calculate overall TPs for T=0.9590 (overall lead times)
lead_times = []
for d in disks_data:
    if d['is_failed'] == 1:
        probs = d['probs']
        alarms = (probs >= t_val).astype(int)
        idx = np.where(alarms >= 1)[0]
        if len(idx) > 0:
            lt = len(probs) - 1 - idx[0]
            lead_times.append(lt)
            
lead_times = np.array(lead_times)
total_failed = n_failed_disks
total_tp = len(lead_times)

results = []

# 2. Loop through intervals up to 30 days
for d_th in days_thresholds:
    detected_within = np.sum(lead_times <= d_th)
    ratio_tp = (detected_within / total_tp * 100) if total_tp > 0 else 0.0
    
    # evaluate using the eval_core logic for FAR, Recall, Precision
    stats = evaluate_detailed_disk_point(disks_data, t_val, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=d_th)
    
    row = {
        '리드타임 구간': f"<= {d_th}일 전",
        '탐지 디스크 수': f"{detected_within:,} 개",
        '탐지 건(TP) 중 비율': f"{ratio_tp:.2f}%",
        '오탐율(FAR)': f"{stats['far']*100:.2f}%",
        '리콜(Recall)': f"{stats['recall']*100:.2f}%",
        '정밀도(Precision)': f"{stats['precision']*100:.2f}%"
    }
    results.append(row)

# 3. Add '전체 리드타임' row
stats_overall = evaluate_detailed_disk_point(disks_data, t_val, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=None)
row_overall = {
    '리드타임 구간': '전체 리드타임',
    '탐지 디스크 수': f"{total_tp:,} 개",
    '탐지 건(TP) 중 비율': '100.00%',
    '오탐율(FAR)': f"{stats_overall['far']*100:.2f}%",
    '리콜(Recall)': f"{stats_overall['recall']*100:.2f}%",
    '정밀도(Precision)': f"{stats_overall['precision']*100:.2f}%"
}
results.append(row_overall)

df_summary = pd.DataFrame(results)
display(HTML(f"<h3>🎯 운영점 T={t_val:.4f} 기준 구간별 리드타임 및 성능 지표 (전체 리드타임 포함)</h3>"))
display(df_summary)
"""

nb['cells'][8]['source'] = [new_markdown]
nb['cells'][9]['source'] = [new_code]
nb['cells'][9]['outputs'] = []
nb['cells'][9]['execution_count'] = None

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated section 5 with intervals and overall lead time row successfully.")
