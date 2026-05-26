import json
import copy
import os

with open('notebooks/07b_disk_threshold_tuning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Change title and markdown
nb['cells'][0]['source'] = [
    '# 07c. 디스크 단위 임계값 튜닝 및 운영점 선정 (Labeling Horizons: 10, 20, 30 Days)\n',
    '\n',
    '이 노트북은 **07b** 노트북의 로직을 기반으로 하되, 고장 디스크에 대하여 리드타임(Lead Time) 조건(10일, 20일, 30일 이내)을 만족하는 알람만 True Positive(정탐)로 인정하도록 레이블링(Labeling)을 동적으로 변경해 가며 그리드 서치를 수행하고 성능을 평가합니다.\n'
]

# Find the cell index for "5. 초고속 그리드 서치 수행"
idx_start = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '5. 초고속 그리드 서치 수행' in ''.join(cell['source']):
        idx_start = i
        break

if idx_start == -1:
    idx_start = 9 # fallback

# Everything before idx_start is kept as is.
new_cells = nb['cells'][:idx_start]

# We will create one big code cell that loops over horizons
big_code_lines = [
    "import pandas as pd\n",
    "import copy\n",
    "from IPython.display import display\n\n",
    "horizons = [10, 20, 30]\n",
    "thresholds = np.linspace(0.001, 0.999, 1000)\n",
    "min_alarms_list = [1]\n",
    "target_far_caps = [0.001, 0.005, 0.010, 0.020, 0.030, 0.050]\n\n",
    "for horizon in horizons:\n",
    "    print('\\n' + '='*80)\n",
    "    print(f'🚀 리드타임 {horizon}일 이내 탐지 기준 (Horizon: {horizon} Days)')\n",
    "    print('='*80)\n\n",
    "    # 1. Horizon 조건에 맞게 데이터셋 마스킹 (고장 디스크에 대해서만)\n",
    "    labeled_disks_data = []\n",
    "    for disk in disks_data:\n",
    "        disk_copy = disk.copy()\n",
    "        if disk_copy['is_failed'] == 1:\n",
    "            dates = pd.to_datetime(disk_copy['dates'])\n",
    "            last_date = dates.iloc[-1]\n",
    "            days_to_fail = (last_date - dates).dt.days\n",
    "            \n",
    "            valid_mask = (days_to_fail <= horizon)\n",
    "            # horizon 기간 바깥의 확률은 0으로 마스킹하여 알람이 발생하지 않도록 처리\n",
    "            probs_masked = disk_copy['probs'].copy()\n",
    "            probs_masked[~valid_mask] = 0.0\n",
    "            disk_copy['probs'] = probs_masked\n",
    "        labeled_disks_data.append(disk_copy)\n\n",
    "    # 2. 그리드 서치 수행\n",
    "    t_grid = time.perf_counter()\n",
    "    df_grid = run_disk_level_grid_search(labeled_disks_data, thresholds, min_alarms_list, n_failed_disks, n_normal_disks, log_dir=cfg.MODEL_SAVE_DIR, window_size=0)\n",
    "    grid_csv_path = os.path.join(cfg.MODEL_SAVE_DIR, f'disk_level_grid_search_results_{horizon}d.csv')\n",
    "    df_grid.to_csv(grid_csv_path, index=False, encoding='utf-8-sig')\n",
    "    print(f'✅ 벡터화 그리드 서치 완료: 총 {len(df_grid)}개 조합 탐색 완료 (소요: {time.perf_counter() - t_grid:.4f}초)')\n\n",
    "    # 3. FAR Cap별 최적 운영점 출력\n",
    "    print('\\n[ 오탐율(Disk FAR) 상한별 최적 운영점 도출 ]')\n",
    "    print('=' * 115)\n",
    "    print(f\"{'FPR Cap':^10} | {'min_alarms':^10} | {'threshold':^10} | {'Disk Recall':^12} | {'Disk FAR':^10} | {'Lead Time':^10} | {'Precision':^10} | {'Persist.':^10}\")\n",
    "    print('-' * 115)\n",
    "    \n",
    "    best_row_for_plot = None\n",
    "    for cap in target_far_caps:\n",
    "        sub = df_grid[df_grid['far'] <= cap]\n",
    "        if len(sub) == 0:\n",
    "            print(f\"{cap*100:^9.2f}% | {'-':^10} | {'-':^10} | {'-':^12} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10}\")\n",
    "            continue\n",
    "        best_row = sub.sort_values('recall', ascending=False).iloc[0]\n",
    "        if cap == 0.01:\n",
    "            best_row_for_plot = best_row\n",
    "            \n",
    "        det = evaluate_detailed_disk_point(\n",
    "            labeled_disks_data, best_row['threshold'], int(best_row['min_alarms']),\n",
    "            n_failed_disks, n_normal_disks, far_cap=cap, window_size=0\n",
    "        )\n",
    "        print(f\"{cap*100:^9.2f}% | {int(best_row['min_alarms']):^10d} | {best_row['threshold']:^10.4f} | \"\n",
    "              f\"{best_row['recall']*100:^11.2f}% | {best_row['far']*100:^9.2f}% | \"\n",
    "              f\"{det.get('lead_time', 0):^10.1f} | {det.get('precision', 0)*100:^9.2f}% | {det.get('persistence', 0)*100:^9.2f}%\")\n",
    "    print('=' * 115 + '\\n')\n\n",
    "    # 4. 그래프 출력\n",
    "    if best_row_for_plot is not None:\n",
    "        print(f'🎯 [Horizon {horizon} Days] 적용할 최적 운영점 (FAR 1.0% 상한): 임계값 T = {best_row_for_plot[\"threshold\"]:.4f} (단일 알람 기준)')\n",
    "        plot_disk_far_recall_curves(df_grid, min_alarms_list, best_row_for_plot)\n",
    "    else:\n",
    "        print(f'[Horizon {horizon} Days] FAR 1.0% 이하를 만족하는 운영점이 없습니다.')\n\n"
]

big_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "loop_cell",
    "metadata": {},
    "outputs": [],
    "source": big_code_lines
}

new_cells.append({
    "cell_type": "markdown",
    "id": "markdown_desc",
    "metadata": {},
    "source": [
        "## 5. Horizon별(10일, 20일, 30일) 레이블링 변환 및 그리드 서치 수행\n",
        "- `disks_data`에서 고장 디스크(`is_failed=1`)의 예측 확률(`probs`) 배열 중, **고장 발생일로부터 Horizon 이내**인 기간의 데이터만 유지하고 그 이전의 확률은 0.0으로 마스킹합니다.\n",
        "- 이를 통해 지정된 리드타임 이내에 발생한 알람만 True Positive(정탐)로 집계되도록 합니다.\n",
        "- 정상 디스크(`is_failed=0`)는 마스킹 없이 그대로 평가하여 모든 오탐(False Positive)을 엄격하게 집계합니다."
    ]
})

new_cells.append(big_cell)

nb['cells'] = new_cells

with open('notebooks/07c_disk_threshold_tuning_labeled.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
