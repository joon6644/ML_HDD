import json

def add_plots_to_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. 오탐 분석의 다각화: 정상 개체 순수 오탐 vs 고장 개체 조기 오탐\n",
            "\n",
            "현실적인 데이터 센터 환경에서 \"정상 디스크\" 역시 관측 기간 이후 결국 고장날 수 있으므로, 두 상태를 결정론적으로 이분하는 것은 분석의 한계를 낳을 수 있습니다. \n",
            "따라서 고장 디스크(`is_failed == 1`) 중 **유지보수 목표 기간(30일)보다 훨씬 이른 시점(> 30일 전)에 경보가 울린 조기 오탐(Early Alarm)**의 분포를 시각화하여 순수 오탐(정상 개체) 분포와 나란히 비교합니다."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# 1. 0.9590 기준의 df_best_disk 구성 (노트북 독립성 보장)\n",
            "t_val = 0.9590\n",
            "best_disk_records = []\n",
            "for disk in disks_data:\n",
            "    probs = disk['probs']\n",
            "    y_pred = (probs >= t_val).astype(int)\n",
            "    total_alarms = y_pred.sum()\n",
            "    \n",
            "    lead_time = np.nan\n",
            "    is_alarmed = 0\n",
            "    \n",
            "    if disk['is_failed'] == 1:\n",
            "        dates = pd.Series(pd.to_datetime(disk['dates']))\n",
            "        last_date = dates.iloc[-1]\n",
            "        days_to_fail = (last_date - dates).dt.days\n",
            "        \n",
            "        # 고장 이전 모든 알람 중 첫 번째 알람의 Lead Time\n",
            "        if y_pred.any():\n",
            "            trigger_idx = np.where(y_pred == 1)[0][0]\n",
            "            trigger_date = dates.iloc[trigger_idx]\n",
            "            lead_time = (last_date - trigger_date).days\n",
            "            is_alarmed = 1\n",
            "    else:\n",
            "        if y_pred.any():\n",
            "            is_alarmed = 1\n",
            "            \n",
            "    best_disk_records.append({\n",
            "        'base_serial': disk['base_serial'],\n",
            "        'is_failed': disk['is_failed'],\n",
            "        'is_alarmed': is_alarmed,\n",
            "        'total_alarms': total_alarms,\n",
            "        'lead_time': lead_time,\n",
            "    })\n",
            "    \n",
            "df_best_disk_local = pd.DataFrame(best_disk_records)\n",
            "\n",
            "# 2. 정상 디스크의 오탐 일수 (Pure False Alarms)\n",
            "normal_fa_days = df_best_disk_local[(df_best_disk_local[\"is_failed\"] == 0) & (df_best_disk_local[\"total_alarms\"] > 0)][\"total_alarms\"]\n",
            "\n",
            "# 3. 고장 디스크의 조기 오탐 일수 (>30일 전 알람 일수)\n",
            "early_fa_days = []\n",
            "for d in disks_data:\n",
            "    if d['is_failed'] == 1:\n",
            "        probs = d['probs']\n",
            "        y_pred = (probs >= t_val).astype(int)\n",
            "        dates = pd.Series(pd.to_datetime(d['dates']))\n",
            "        last_date = dates.iloc[-1]\n",
            "        days_to_fail = (last_date - dates).dt.days\n",
            "        \n",
            "        # 고장 30일 초과 전에 발생한 알람 일수 합산\n",
            "        early_count = np.sum(y_pred[days_to_fail > 30])\n",
            "        if early_count > 0:\n",
            "            early_fa_days.append(early_count)\n",
            "\n",
            "early_fa_days = pd.Series(early_fa_days)\n",
            "\n",
            "# 4. 한국어 폰트 설정\n",
            "import matplotlib.font_manager as fm\n",
            "for name in [\"Malgun Gothic\", \"NanumGothic\", \"AppleGothic\", \"DejaVu Sans\"]:\n",
            "    if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):\n",
            "        plt.rcParams[\"font.family\"] = name\n",
            "        break\n",
            "plt.rcParams[\"axes.unicode_minus\"] = False\n",
            "\n",
            "# 5. 시각화: Side-by-Side 비교\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
            "fig.patch.set_facecolor('#ffffff')\n",
            "\n",
            "# Left: 정상 디스크 오탐\n",
            "sns.histplot(normal_fa_days, binwidth=1, kde=False, color=\"coral\", edgecolor=\"white\", ax=ax1)\n",
            "ax1.set_title(\"정상 디스크별 오탐 일수 분포 (순수 오탐)\\n(Pure False Alarms on Normal Disks)\", fontsize=12, fontweight=\"bold\", pad=10)\n",
            "ax1.set_xlabel(\"False Alarm Days (Per Disk)\", fontsize=11)\n",
            "ax1.set_ylabel(\"Normal Disks Count\", fontsize=11)\n",
            "ax1.grid(axis=\"y\", linestyle=\":\", alpha=0.6)\n",
            "\n",
            "# Right: 고장 디스크 조기 오탐 (> 30일 전)\n",
            "if len(early_fa_days) > 0:\n",
            "    sns.histplot(early_fa_days, binwidth=1, kde=False, color=\"crimson\", edgecolor=\"white\", ax=ax2)\n",
            "    ax2.set_title(\"고장 디스크별 조기 오탐 일수 분포 (> 30일 전)\\n(Early Alarms on Failed Disks, > 30d before failure)\", fontsize=12, fontweight=\"bold\", pad=10)\n",
            "    ax2.set_xlabel(\"Early Alarm Days (Per Disk)\", fontsize=11)\n",
            "    ax2.set_ylabel(\"Failed Disks Count\", fontsize=11)\n",
            "    ax2.grid(axis=\"y\", linestyle=\":\", alpha=0.6)\n",
            "else:\n",
            "    ax2.text(0.5, 0.5, \"조기 경보가 발생한 고장 개체가 없습니다.\", ha=\"center\", va=\"center\", transform=ax2.transAxes)\n",
            "\n",
            "plt.suptitle(\"오탐 정의의 다각화: 정상 개체 순수 오탐 vs 고장 개체 조기 오탐 (30일 리드타임 기준)\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    }

    # Clean existing section 6 if present
    nb['cells'] = [c for c in nb['cells'] if "## 6. 오탐 분석의 다각화" not in ''.join(c.get('source', [])) and "early_fa_days = []" not in ''.join(c.get('source', []))]
    
    nb['cells'].append(markdown_cell)
    nb['cells'].append(code_cell)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        
    print(f"Added early alarm plot to {filepath} successfully.")

add_plots_to_notebook('notebooks/08b_final_evaluation_labeled.ipynb')
add_plots_to_notebook('notebooks/08c_final_evaluation_labeled.ipynb')
