import json

filename = 'notebooks/08c_final_evaluation_labeled.ipynb'

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

leadtime_markdown = """## 4. 각 임계값별 전체 리드타임 심층 분석
각 임계값(0.9820, 0.9720, 0.9590, 0.9401)에 대해, Horizon에 구애받지 않고 탐지된 전체
True Positive(고장 탐지 디스크)의 **리드타임(Lead Time)** 누적 탐지율 및 분포를 산출합니다.
이를 통해 전체 탐지된 디스크 중 특정 일수 이내에 탐지된 비율을 100%가 아닌 실제 비율로 확인할 수 있습니다.
"""

leadtime_code = """from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

for t in thresholds:
    display(HTML(f"<hr><h2 style='color:#2c3e50'>🎯 임계값 T = {t:.4f} 전체 리드타임 심층 분석</h2>"))
    
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
    
    print(f"============================================================")
    print(f"📊 리드타임(Lead Time) 심층 기술통계 (전체 탐지 기준)")
    print(f"============================================================")
    print(f"  - 전체 고장 디스크 수 : {total_failed:,} 개")
    print(f"  - 전체 탐지 성공(TP)  : {total_tp:,} 개 (최대 Recall: {total_tp/total_failed*100:.2f}%)")
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
        
        days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30, 40, 60]
        
        print(f"  리드타임 구간 | 탐지 디스크 수 | 탐지 건(TP) 중 비율 | 전체 고장 대비 비율")
        print(f"  ------------------------------------------------------------")
        for d_th in days_thresholds:
            detected_within = np.sum(lead_times <= d_th)
            ratio_tp = detected_within / total_tp * 100
            ratio_all = detected_within / total_failed * 100
            print(f"   <= {d_th:2d}일 전   | {detected_within:10,} 개 | {ratio_tp:15.1f}% | {ratio_all:16.2f}%")
        print(f"============================================================\\n")
"""

nb['cells'][6]['source'] = [leadtime_markdown]
nb['cells'][7]['source'] = [leadtime_code]
# Clear output of cell 7
nb['cells'][7]['outputs'] = []
nb['cells'][7]['execution_count'] = None

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Modified cell 6 and 7 to use overall baseline.")
