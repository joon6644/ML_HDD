import json

filename = 'notebooks/08c_final_evaluation_labeled.ipynb'

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Rollback cell 6 (markdown) & 7 (code) to the previous versions without Miss statistics
leadtime_markdown = """## 4. 각 임계값별 전체 리드타임 심층 분석 (통합 테이블)
각 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해, Horizon에 구애받지 않고 탐지된 전체
True Positive(고장 탐지 디스크)의 **리드타임(Lead Time)** 누적 탐지율 및 분포를 산출합니다.
이를 통해 전체 탐지된 디스크 중 특정 일수 이내에 탐지된 비율을 100%가 아닌 실제 비율로 확인할 수 있습니다.
"""

leadtime_code = """from IPython.display import display, HTML
import pandas as pd
import numpy as np
from src.eval_core import evaluate_detailed_disk_point

results = []
days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30, 40, 60]

for t in thresholds:
    # 1. Calculate overall TPs for this threshold without horizon limits
    lead_times = []
    for d in disks_data:
        if d['is_failed'] == 1:
            probs = d['probs']
            alarms = (probs >= t).astype(int)
            idx = np.where(alarms >= 1)[0]
            if len(idx) > 0:
                lt = len(probs) - 1 - idx[0]
                lead_times.append(lt)
                
    lead_times = np.array(lead_times)
    total_failed = n_failed_disks
    total_tp = len(lead_times)
    
    # 2. Loop over thresholds and evaluate metrics
    for d_th in days_thresholds:
        # manual calculation for cumulative rates
        detected_within = np.sum(lead_times <= d_th)
        ratio_tp = (detected_within / total_tp * 100) if total_tp > 0 else 0.0
        
        # evaluate using the eval_core logic for consistent FAR, Recall, Precision
        stats = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=d_th)
        
        row = {
            'Threshold': f"{t:.4f}",
            '리드타임 구간': f"<= {d_th}일 전",
            '탐지 디스크 수': f"{detected_within:,}",
            '탐지 건(TP) 중 비율': f"{ratio_tp:.1f}%",
            '오탐율(FAR)': f"{stats['far']*100:.2f}%",
            '리콜(Recall)': f"{stats['recall']*100:.2f}%",
            '정밀도(Precision)': f"{stats['precision']*100:.2f}%"
        }
        results.append(row)

df_all = pd.DataFrame(results)

# display settings
pd.set_option('display.max_rows', None)
display(HTML("<h3>📊 통합 리드타임 및 성능 지표 테이블 (복붙용)</h3>"))
display(df_all)
"""

nb['cells'][6]['source'] = [leadtime_markdown]
nb['cells'][7]['source'] = [leadtime_code]
nb['cells'][7]['outputs'] = []
nb['cells'][7]['execution_count'] = None

# Create new cells for 0.9590 & 30 days analysis
new_markdown = """## 5. 최적 운영점 (T=0.9590, Horizon=30일) 탐지 및 미탐 상세 분석
연구 논문 및 보고서 작성을 위해 최적 운영점 `T=0.9590` 및 예측 기간 `30일` 기준의 
탐지 성공(True Positive)과 미탐(False Negative)의 건수 및 전체 고장 대비 비율을 명확하게 요약하여 제공합니다.
"""

new_code = """from IPython.display import display, HTML
import pandas as pd
import numpy as np

t_val = 0.9590
h_val = 30

# 1. Calculate lead times
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
detected_count = np.sum(lead_times <= h_val)
missed_count = total_failed - detected_count

# 2. Build summary table
summary_data = [
    {
        '구분': '탐지 성공 (True Positive)',
        '디스크 수': f"{detected_count:,} 개",
        '비율 (전체 고장 대비)': f"{detected_count / total_failed * 100:.2f}%",
        '정의': f"고장 발생 {h_val}일 이내에 최초 경보(Alarm) 발생 성공"
    },
    {
        '구분': '미탐 (False Negative)',
        '디스크 수': f"{missed_count:,} 개",
        '비율 (전체 고장 대비)': f"{missed_count / total_failed * 100:.2f}%",
        '정의': f"고장 발생 {h_val}일 이내에 경보가 울리지 않음 (알람이 없거나 {h_val}일 이전 발생)"
    },
    {
        '합계 (전체 고장)': 'Total Failures',
        '디스크 수': f"{total_failed:,} 개",
        '비율 (전체 고장 대비)': '100.00%',
        '정의': '검증 대상 전체 고장 디스크 개수'
    }
]

df_summary = pd.DataFrame(summary_data)
display(HTML(f"<h3>🎯 운영점 T={t_val:.4f}, Horizon={h_val}일 기준 탐지 vs 미탐 분석 요약</h3>"))
display(df_summary)
"""

# Append new cells if they don't exist yet
if len(nb['cells']) <= 8:
    nb['cells'].append({"cell_type": "markdown", "metadata": {}, "source": [new_markdown]})
    nb['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [new_code]})
else:
    nb['cells'][8] = {"cell_type": "markdown", "metadata": {}, "source": [new_markdown]}
    nb['cells'][9] = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [new_code]}

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Restored cell 7 and added section 5 successfully.")
