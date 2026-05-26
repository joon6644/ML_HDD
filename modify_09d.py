import json

new_code = """window_days = cfg.TEMPORAL_WINDOW_DAYS
top_n = cfg.TEMPORAL_TOP_N_FEATS

for serial in target_serials:
    df_entity = df_test[df_test['base_serial'] == serial].sort_values(fe_cfg.DATE_COL)
    info = df_best_disk[df_best_disk['base_serial'] == serial].iloc[0]
    
    is_failed = info['is_failed']
    is_alarmed = info['is_alarmed']
    
    alarms = (df_entity['_prob'] >= BEST_T).astype(int)
    rolling = alarms.rolling(window=ALARM_WINDOW, min_periods=1).sum()
    idx = np.where(rolling >= BEST_N)[0]
    first_alarm_date = df_entity[fe_cfg.DATE_COL].iloc[idx[0]] if len(idx) > 0 else None
    
    fail_date = df_entity[fe_cfg.DATE_COL].max().date() if is_failed == 1 else df_entity[fe_cfg.DATE_COL].max().date()
    
    if is_failed == 1:
        # 조기 알람이므로, 첫 알람 시점을 중심으로 윈도우 설정
        if first_alarm_date is not None:
            half_win = window_days // 2
            ref_date = min(first_alarm_date + pd.Timedelta(days=half_win), df_entity[fe_cfg.DATE_COL].max())
            title_suffix = "Failure Date (D-0)"
        else:
            ref_date = df_entity[fe_cfg.DATE_COL].max()
            title_suffix = "Failure Date (D-0)"
    else:
        ref_date = df_entity[fe_cfg.DATE_COL].max()
        title_suffix = "Observation End (D-0)"

    start_date = ref_date - pd.Timedelta(days=window_days - 1)
    df_window = df_entity[(df_entity[fe_cfg.DATE_COL] >= start_date) & (df_entity[fe_cfg.DATE_COL] <= ref_date)].copy().reset_index(drop=True)
    
    print(f"\\n\\n" + "="*70)
    print(f"[§9.3] 시간 기반 SHAP Trajectory — 개체: {serial}")
    print(f"       분석 기간: {start_date.date()} ~ {ref_date.date()} ({len(df_window)}일)")
    print("="*70)

    if len(df_window) == 0:
        print("⚠️  분석 기간 내 데이터가 없습니다. TEMPORAL_WINDOW_DAYS를 늘려보세요.")
        continue

    X_window = df_window[FEATURE_COLS].reset_index(drop=True)

    sv_window_list = []
    for model in models:
        exp = shap.TreeExplainer(model)
        sv_w = exp.shap_values(X_window)
        if isinstance(sv_w, list):
            sv_w = sv_w[1]
        sv_window_list.append(sv_w)
    mean_sv_window = np.mean(sv_window_list, axis=0)

    max_pos_window = np.max(mean_sv_window, axis=0) 
    top_feat_idx = np.argsort(max_pos_window)[::-1][:top_n]
    top_feat_names = [FEATURE_COLS[idx] for idx in top_feat_idx]

    dates = df_window[fe_cfg.DATE_COL].dt.date.values
    # X축: 실제 고장일(fail_date) 기준 D-n 계산
    days_from_last = [- (fail_date - d).days for d in dates]

    fig, axes = plt.subplots(top_n + 1, 1, figsize=(12, 3 * (top_n + 1)), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    
    ax_prob = axes[0]
    probs = df_window["_prob"].values
    
    ax_prob.plot(days_from_last, probs, marker='o', color='#2c3e50', lw=2.5, markersize=5, label="Prediction Probability")
    ax_prob.axhline(BEST_T, color='#e74c3c', linestyle='--', lw=2, label=f"Threshold (T={BEST_T:.3f})")
    ax_prob.fill_between(days_from_last, probs, BEST_T, where=(probs >= BEST_T), color='#e74c3c', alpha=0.15)
    
    if first_alarm_date is not None:
        alarm_day = - (fail_date - first_alarm_date.date()).days
        # Plot within window if it exists
        if min(days_from_last) <= alarm_day <= max(days_from_last):
            ax_prob.axvline(alarm_day, color='#e74c3c', linestyle=':', lw=2.5, alpha=0.8)

    ax_prob.set_ylabel("Probability", fontsize=11, fontweight="bold", color='#34495e')
    ax_prob.set_title("Model Prediction Probability", fontsize=13, fontweight="bold", loc='left', color='#2c3e50')
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#ecf0f1')
    ax_prob.grid(axis='y', linestyle='--', alpha=0.4)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['left'].set_color('#bdc3c7')
    ax_prob.spines['bottom'].set_color('#bdc3c7')

    pos_color = '#e6614f'
    neg_color = '#4a90e2'
    
    for ax, feat_name, feat_idx in zip(axes[1:], top_feat_names, top_feat_idx):
        shap_vals = mean_sv_window[:, feat_idx]
        bars = ax.bar(days_from_last, shap_vals, 
                      color=[pos_color if v > 0 else neg_color for v in shap_vals], 
                      alpha=0.85, width=0.8)
        ax.axhline(0, color="#7f8c8d", lw=1.2, ls="-")
        
        if first_alarm_date is not None:
            if min(days_from_last) <= alarm_day <= max(days_from_last):
                ax.axvline(alarm_day, color='#e74c3c', linestyle=':', lw=2.5, alpha=0.5, zorder=0)

        max_abs = max(abs(shap_vals.min()), abs(shap_vals.max())) * 1.1
        if max_abs > 0:
            ax.set_ylim(-max_abs, max_abs)
            
        ax.set_ylabel("SHAP Value", fontsize=11, fontweight="bold", color='#34495e')
        ax.set_title(f"Feature: {feat_name}", fontsize=12, fontweight="bold", loc='left', color='#2c3e50')
        ax.grid(axis="y", linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')

    # Tick generation relative to fail_date
    tick_locs = np.arange(min(days_from_last), max(days_from_last) + 1, 5)
    
    xticklabels = []
    for x in tick_locs:
        real_date = (fail_date + pd.Timedelta(days=int(x))).strftime('%m-%d')
        xticklabels.append(f"D{int(x)}\\n({real_date})")

    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels(xticklabels, fontsize=10, color='#34495e')
    axes[-1].set_xlabel(f"Timeline to Failure Date (D-0)", fontsize=12, fontweight="bold", color='#2c3e50', labelpad=15)

    fig.suptitle(
        f"Temporal SHAP Evolution & Risk Trajectory (Serial: {serial})",
        fontsize=16, fontweight="bold", color='#2c3e50', y=1.02
    )
    plt.tight_layout()
    plt.show()
    print(f"\\n개체 {serial} 분석 완료. 상위 {top_n}개 피처: {top_feat_names}")
"""

nb_file = 'notebooks/09d_early_alarm_analysis.ipynb'
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'window_days = cfg.TEMPORAL_WINDOW_DAYS' in src and 'df_window' in src:
            cell['source'] = [line + '\\n' for line in new_code.split('\n')]
            break

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated 09d notebook.")
