import json

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. 리드타임(Lead Time) 누적 탐지율 및 기술통계 심층 분석\n",
        "\n",
        "최적 운영점(T=0.9590)에서 탐지된 **고장 디스크(True Positive)**들의 리드타임 분포를 세밀하게 분석합니다. \n",
        "단순 평균이 아닌, 실무적으로 유의미한 **누적 탐지 비율(Cumulative Detection Rate)**과 심층 기술통계를 통해 유지보수 골든타임을 도출합니다."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "\n",
        "# 1. 고장 디스크에 대한 리드타임 추출 (T=0.9590, N=1, Horizon=30)\n",
        "lead_times = []\n",
        "horizon = 30\n",
        "ALARM_WINDOW = 1\n",
        "\n",
        "for d in disks_data:\n",
        "    if d['is_failed'] == 1:\n",
        "        probs = d['probs']\n",
        "        alarms = (probs >= BEST_T).astype(int)\n",
        "        rolling = pd.Series(alarms).rolling(window=ALARM_WINDOW, min_periods=1).sum()\n",
        "        idx = np.where(rolling >= ALARM_WINDOW)[0]\n",
        "        \n",
        "        if len(idx) > 0:\n",
        "            # 알람 발생 시점부터 실제 고장일까지의 일수\n",
        "            lt = len(probs) - 1 - idx[0]\n",
        "            if lt <= horizon:\n",
        "                lead_times.append(lt)\n",
        "\n",
        "lead_times = np.array(lead_times)\n",
        "total_failed = n_failed_disks\n",
        "total_tp = len(lead_times)\n",
        "\n",
        "print(f\"============================================================\")\n",
        "print(f\"📊 리드타임(Lead Time) 심층 기술통계 (Horizon: {horizon}일 기준)\")\n",
        "print(f\"============================================================\")\n",
        "print(f\"  - 전체 고장 디스크 수 : {total_failed:,} 개\")\n",
        "print(f\"  - {horizon}일 이내 탐지 성공(TP): {total_tp:,} 개 (Recall: {total_tp/total_failed*100:.2f}%)\")\n",
        "print(f\"\")\n",
        "print(f\"  [기술통계 요약]\")\n",
        "print(f\"    • 평균(Mean)   : {np.mean(lead_times):.2f} 일\")\n",
        "print(f\"    • 표준편차(Std): {np.std(lead_times):.2f} 일\")\n",
        "print(f\"    • 최소(Min)    : {np.min(lead_times)} 일\")\n",
        "print(f\"    • 10% 백분위수 : {np.percentile(lead_times, 10):.1f} 일\")\n",
        "print(f\"    • 25% 백분위수 : {np.percentile(lead_times, 25):.1f} 일\")\n",
        "print(f\"    • 중앙값(Med)  : {np.median(lead_times):.1f} 일\")\n",
        "print(f\"    • 75% 백분위수 : {np.percentile(lead_times, 75):.1f} 일\")\n",
        "print(f\"    • 90% 백분위수 : {np.percentile(lead_times, 90):.1f} 일\")\n",
        "print(f\"    • 최대(Max)    : {np.max(lead_times)} 일\")\n",
        "print(f\"\")\n",
        "\n",
        "print(f\"============================================================\")\n",
        "print(f\"📈 구간별 누적 탐지 비율 (Cumulative Detection Rate)\")\n",
        "print(f\"============================================================\")\n",
        "# 특정 기간 내 탐지된 디스크 비율 (전체 고장 디스크 대비 / TP 대비)\n",
        "days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30]\n",
        "\n",
        "print(f\"  리드타임 구간 | 탐지 디스크 수 | 탐지 건(TP) 중 비율 | 전체 고장 대비 비율\")\n",
        "print(f\"  ------------------------------------------------------------\")\n",
        "for d_th in days_thresholds:\n",
        "    detected_within = np.sum(lead_times <= d_th)\n",
        "    ratio_tp = detected_within / total_tp * 100\n",
        "    ratio_all = detected_within / total_failed * 100\n",
        "    print(f\"   <= {d_th:2d}일 전   | {detected_within:10,} 개 | {ratio_tp:15.1f}% | {ratio_all:16.2f}%\")\n",
        "print(f\"============================================================\\n\")\n",
        "\n",
        "# 시각화: 히스토그램 및 ECDF (누적 분포 함수)\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "fig.patch.set_facecolor('#ffffff')\n",
        "\n",
        "# 1. 히스토그램\n",
        "sns.histplot(lead_times, bins=np.arange(0, horizon+2)-0.5, kde=True, color=\"#3498db\", ax=ax1)\n",
        "ax1.set_title(\"Lead Time Histogram (Days before failure)\", fontsize=13, fontweight='bold', color='#2c3e50')\n",
        "ax1.set_xlabel(\"Lead Time (Days)\", fontsize=11, fontweight='bold')\n",
        "ax1.set_ylabel(\"Count of Disks\", fontsize=11, fontweight='bold')\n",
        "ax1.grid(axis='y', linestyle='--', alpha=0.5)\n",
        "ax1.axvline(np.median(lead_times), color='#e74c3c', linestyle='--', lw=2, label=f\"Median ({np.median(lead_times):.1f}d)\")\n",
        "ax1.axvline(np.mean(lead_times), color='#2ecc71', linestyle=':', lw=2, label=f\"Mean ({np.mean(lead_times):.1f}d)\")\n",
        "ax1.legend()\n",
        "\n",
        "# 2. 누적 분포(ECDF)\n",
        "sns.ecdfplot(lead_times, color=\"#e74c3c\", lw=3, ax=ax2)\n",
        "ax2.set_title(\"Lead Time Cumulative Distribution\", fontsize=13, fontweight='bold', color='#2c3e50')\n",
        "ax2.set_xlabel(\"Lead Time (Days)\", fontsize=11, fontweight='bold')\n",
        "ax2.set_ylabel(\"Cumulative Ratio (within TPs)\", fontsize=11, fontweight='bold')\n",
        "ax2.grid(linestyle='--', alpha=0.5)\n",
        "ax2.set_xlim(0, horizon)\n",
        "\n",
        "# ECDF에 주요 마커 표시\n",
        "for d_th in [1, 3, 7, 14]:\n",
        "    y_val = np.sum(lead_times <= d_th) / total_tp\n",
        "    ax2.plot(d_th, y_val, marker='o', markersize=8, color='#2c3e50')\n",
        "    ax2.text(d_th+0.5, y_val-0.05, f\"<= {d_th}d\\n({y_val*100:.1f}%)\", color='#2c3e50', fontweight='bold', fontsize=9)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

nb_file = 'notebooks/08c_final_evaluation_labeled.ipynb'
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if we already appended
if not any("4. 리드타임(Lead Time) 누적 탐지율" in ''.join(c.get('source', [])) for c in nb['cells']):
    nb['cells'].append(markdown_cell)
    nb['cells'].append(code_cell)
    
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("Added deep dive lead time analysis cell to 08c.")
else:
    print("Deep dive already present.")
