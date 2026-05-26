import json

with open('notebooks/07e_disk_threshold_tuning_labeled_calib.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the big code cell
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('horizons = [10, 20, 30]' in line for line in cell['source']):
        target_idx = i
        break

if target_idx != -1:
    new_code = [
        "import pandas as pd\n",
        "import copy\n",
        "import time\n",
        "import os\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from IPython.display import display\n\n",
        "horizons = [10, 20, 30]\n",
        "thresholds = np.linspace(0.001, 0.999, 1000)\n",
        "min_alarms_list = [1]\n",
        "target_far_caps = [0.001, 0.005, 0.010, 0.020]  # 0.1, 0.5, 1, 2 만 유지\n\n",
        "all_grids = {}\n\n",
        "for horizon in horizons:\n",
        "    print('\\n' + '='*100)\n",
        "    print(f'🚀 리드타임 {horizon}일 이내 탐지 기준 (Horizon: {horizon} Days)')\n",
        "    print('='*100)\n\n",
        "    # 1. Horizon 조건에 맞게 데이터셋 마스킹 (고장 디스크에 대해서만)\n",
        "    labeled_disks_data = []\n",
        "    for disk in disks_data:\n",
        "        disk_copy = disk.copy()\n",
        "        if disk_copy['is_failed'] == 1:\n",
        "            dates = pd.Series(pd.to_datetime(disk_copy['dates']))\n",
        "            last_date = dates.iloc[-1]\n",
        "            days_to_fail = (last_date - dates).dt.days\n",
        "            \n",
        "            valid_mask = (days_to_fail <= horizon)\n",
        "            probs_masked = disk_copy['probs'].copy()\n",
        "            probs_masked[~valid_mask] = 0.0\n",
        "            disk_copy['probs'] = probs_masked\n",
        "        labeled_disks_data.append(disk_copy)\n\n",
        "    # 2. 그리드 서치 수행\n",
        "    t_grid = time.perf_counter()\n",
        "    df_grid = run_disk_level_grid_search(labeled_disks_data, thresholds, min_alarms_list, n_failed_disks, n_normal_disks, log_dir=cfg.MODEL_SAVE_DIR, window_size=0)\n",
        "    all_grids[horizon] = df_grid\n",
        "    grid_csv_path = os.path.join(cfg.MODEL_SAVE_DIR, f'disk_level_grid_search_results_calib_enhanced_{horizon}d.csv')\n",
        "    df_grid.to_csv(grid_csv_path, index=False, encoding='utf-8-sig')\n",
        "    print(f'✅ 벡터화 그리드 서치 완료: 총 {len(df_grid)}개 조합 탐색 완료 (소요: {time.perf_counter() - t_grid:.4f}초)')\n\n",
        "    # 3. FAR Cap별 최적 운영점 출력\n",
        "    print('\\n[ 오탐율(Disk FAR) 상한별 최적 운영점 도출 ]')\n",
        "    print('-' * 115)\n",
        "    print(f\"{'FPR Cap':^10} | {'min_alarms':^10} | {'threshold':^10} | {'Disk Recall':^12} | {'Disk FAR':^10} | {'Lead Time':^10} | {'Precision':^10} | {'Persist.':^10}\")\n",
        "    print('-' * 115)\n",
        "    \n",
        "    for cap in target_far_caps:\n",
        "        sub = df_grid[df_grid['far'] <= cap]\n",
        "        if len(sub) == 0:\n",
        "            print(f\"{cap*100:^9.2f}% | {'-':^10} | {'-':^10} | {'-':^12} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10}\")\n",
        "            continue\n",
        "        best_row = sub.sort_values('recall', ascending=False).iloc[0]\n",
        "            \n",
        "        det = evaluate_detailed_disk_point(\n",
        "            labeled_disks_data, best_row['threshold'], int(best_row['min_alarms']),\n",
        "            n_failed_disks, n_normal_disks, far_cap=cap, window_size=0\n",
        "        )\n",
        "        print(f\"{cap*100:^9.2f}% | {int(best_row['min_alarms']):^10d} | {best_row['threshold']:^10.4f} | \"\n",
        "              f\"{best_row['recall']*100:^11.2f}% | {best_row['far']*100:^9.2f}% | \"\n",
        "              f\"{det.get('lead_time', 0):^10.1f} | {det.get('precision', 0)*100:^9.2f}% | {det.get('persistence', 0)*100:^9.2f}%\")\n",
        "    print('-' * 115 + '\\n')\n\n",
        "# 4. 통합 그래프 출력 (x축 3까지)\n",
        "print('\\n' + '='*100)\n",
        "print('📈 통합 FAR-Recall 곡선 (Horizons: 10, 20, 30 Days)')\n",
        "print('='*100)\n",
        "plt.figure(figsize=(10, 6))\n",
        "colors = ['#1f77b4', '#ff7f0e', '#2ca02c']\n",
        "for i, horizon in enumerate(horizons):\n",
        "    df_grid = all_grids[horizon]\n",
        "    sub_df = df_grid[df_grid['min_alarms'] == 1].sort_values('threshold', ascending=False)\n",
        "    plot_df = sub_df[sub_df['far'] > 0]\n",
        "    plt.plot(plot_df['far'] * 100, plot_df['recall'] * 100, color=colors[i], lw=2.5,\n",
        "             label=f'Horizon {horizon} Days')\n\n",
        "target_fars = [0.1, 0.5, 1.0, 2.0]\n",
        "for tf in target_fars:\n",
        "    plt.axvline(tf, color='#7f7f7f', linestyle=':', alpha=0.6, lw=1.2)\n",
        "    plt.text(tf, 72.5, f'{tf}% FAR', color='#555555', fontsize=9, ha='center', fontweight='normal')\n\n",
        "plt.xlim([0.0, 3.0])\n",
        "plt.ylim([-1.0, 75.0])\n",
        "plt.xticks(np.arange(0, 4, 1))\n\n",
        "plt.xlabel('Disk-Level False Alarm Rate (Disk FAR, %)', fontsize=11, labelpad=8)\n",
        "plt.ylabel('Disk-Level Recall (%)', fontsize=11, labelpad=8)\n",
        "plt.title('Disk-Level FAR-Recall Curve (Calib Set)', fontsize=12, fontweight='bold', pad=15)\n",
        "plt.legend(loc='lower right', fontsize=10.5)\n",
        "plt.grid(True, which='both', linestyle=':', alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
    nb['cells'][target_idx]['source'] = new_code

with open('notebooks/07e_disk_threshold_tuning_labeled_calib.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
