import json

filename = 'notebooks/08c_final_evaluation_labeled.ipynb'

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

leadtime_markdown = """## 4. 각 임계값별 전체 리드타임 심층 분석 (통합 테이블)
각 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해, 전체 True Positive(고장 탐지 디스크) 기준의 누적 탐지율을 산출합니다.
또한 복붙하여 보고서에 활용하기 쉽도록 모든 구간(0, 1, 3, 5, 7, 10, 14, 21, 30, 40, 60일)의 
미탐(Miss) 건수/비율, 오탐율(FAR), 리콜(Recall), 정밀도(Precision)를 하나의 테이블로 통합하여 출력합니다.
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
        
        # calculate missed count and missed rate relative to total failed disks
        missed_count = total_failed - detected_within
        miss_rate = (missed_count / total_failed * 100)
        
        # evaluate using the eval_core logic for consistent FAR, Recall, Precision
        stats = evaluate_detailed_disk_point(disks_data, t, 1, n_failed_disks, n_normal_disks, window_size=0, horizon=d_th)
        
        row = {
            'Threshold': f"{t:.4f}",
            '리드타임 구간': f"<= {d_th}일 전",
            '탐지 디스크 수': f"{detected_within:,}",
            '탐지 건(TP) 중 비율': f"{ratio_tp:.1f}%",
            '미탐 디스크 수': f"{missed_count:,}",
            '미탐 비율(전체 대비)': f"{miss_rate:.2f}%",
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

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Modified cell 6 and 7 to include missed count and rate.")
