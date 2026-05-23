import json
import pandas as pd

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_9_3 = r"""# §9.3 — 고장 전 N일 창 SHAP Trajectory (각 개체별)
window_days = cfg.TEMPORAL_WINDOW_DAYS
top_n = cfg.TEMPORAL_TOP_N_FEATS

for serial in target_serials:
    df_entity = df_test[df_test[cfg.SERIAL_COL] == serial].sort_values(cfg.DATE_COL)
    info = df_best_disk[df_best_disk['base_serial'] == serial].iloc[0]
    
    is_failed = info['is_failed']
    is_alarmed = info['is_alarmed']
    
    # 알람 발령일 계산 (세로선 표시용)
    alarms = (df_entity['_prob'] >= BEST_T).astype(int)
    rolling = alarms.rolling(window=ALARM_WINDOW, min_periods=1).sum()
    idx = np.where(rolling >= BEST_N)[0]
    first_alarm_date = df_entity[cfg.DATE_COL].iloc[idx[0]] if len(idx) > 0 else None
    
    if is_failed == 1:
        ref_date = df_entity[cfg.DATE_COL].max()
        title_suffix = "Failure Date (D-0)"
    else:
        if is_alarmed == 1:
            if first_alarm_date is not None:
                ref_date = first_alarm_date
            else:
                ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "False Alarm Date (D-0)"
        else:
            ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "Observation End (D-0)"

    start_date = ref_date - pd.Timedelta(days=window_days - 1)
    df_window = df_entity[(df_entity[cfg.DATE_COL] >= start_date) & (df_entity[cfg.DATE_COL] <= ref_date)].copy().reset_index(drop=True)
    
    print(f"\n\n" + "="*70)
    print(f"[§9.3] 시간 기반 SHAP Trajectory — 개체: {serial}")
    print(f"       분석 기간: {start_date.date()} ~ {ref_date.date()} ({len(df_window)}일)")
    print("="*70)

    if len(df_window) == 0:
        print("⚠️  분석 기간 내 데이터가 없습니다. TEMPORAL_WINDOW_DAYS를 늘려보세요.")
        continue

    X_window = df_window[FEATURE_COLS].reset_index(drop=True)

    # 윈도우 구간 SHAP 계산
    sv_window_list = []
    for model in models:
        exp = shap.TreeExplainer(model)
        sv_w = exp.shap_values(X_window)
        if isinstance(sv_w, list):
            sv_w = sv_w[1]
        sv_window_list.append(sv_w)
    mean_sv_window = np.mean(sv_window_list, axis=0)

    # 양수 방향(위험 증가) 최대값을 기준으로 정렬
    max_pos_window = np.max(mean_sv_window, axis=0) 
    top_feat_idx = np.argsort(max_pos_window)[::-1][:top_n]
    top_feat_names = [FEATURE_COLS[idx] for idx in top_feat_idx]

    dates = df_window[cfg.DATE_COL].dt.date.values
    days_from_last = [- (ref_date.date() - d).days for d in dates]

    # 시각화 (종합 확률 + 상위 피처 SHAP)
    # 폰트 및 스타일 고급화
    fig, axes = plt.subplots(top_n + 1, 1, figsize=(12, 3 * (top_n + 1)), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    
    # 1. 최상단: 전체 모델 예측 확률 궤적 및 임계값
    ax_prob = axes[0]
    probs = df_window["_prob"].values
    
    ax_prob.plot(days_from_last, probs, marker='o', color='#2c3e50', lw=2.5, markersize=5, label="Prediction Probability")
    ax_prob.axhline(BEST_T, color='#e74c3c', linestyle='--', lw=2, label=f"Threshold (T={BEST_T:.3f})")
    ax_prob.fill_between(days_from_last, probs, BEST_T, where=(probs >= BEST_T), color='#e74c3c', alpha=0.15)
    
    # 고장 예측일 세로선 표시 (확률 그래프)
    if first_alarm_date is not None:
        alarm_day = - (ref_date.date() - first_alarm_date.date()).days
        if -window_days <= alarm_day <= 0:
            ax_prob.axvline(alarm_day, color='#e74c3c', linestyle=':', lw=2.5, alpha=0.8)

    ax_prob.set_ylabel("Probability", fontsize=11, fontweight="bold", color='#34495e')
    ax_prob.set_title("Model Prediction Probability", fontsize=13, fontweight="bold", loc='left', color='#2c3e50')
    ax_prob.set_ylim(-0.05, 1.05)
    
    # 깔끔한 범례
    ax_prob.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#ecf0f1')
    ax_prob.grid(axis='y', linestyle='--', alpha=0.4)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['left'].set_color('#bdc3c7')
    ax_prob.spines['bottom'].set_color('#bdc3c7')

    # 2. 하단: 상위 피처별 SHAP 기여도
    # 세련된 컬러 팔레트 (위험=coral, 안전=cornflowerblue)
    pos_color = '#e6614f'
    neg_color = '#4a90e2'
    
    for ax, feat_name, feat_idx in zip(axes[1:], top_feat_names, top_feat_idx):
        shap_vals = mean_sv_window[:, feat_idx]

        # SHAP 값 (막대 그래프)
        bars = ax.bar(days_from_last, shap_vals, 
                      color=[pos_color if v > 0 else neg_color for v in shap_vals], 
                      alpha=0.85, width=0.8)
        
        ax.axhline(0, color="#7f8c8d", lw=1.2, ls="-")
        
        # 고장 예측일 세로선 표시
        if first_alarm_date is not None:
            alarm_day = - (ref_date.date() - first_alarm_date.date()).days
            if -window_days <= alarm_day <= 0:
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

    # X축 눈금 설정 (날짜 포함)
    tick_locs = np.arange(-window_days + 1, 1, 5)
    if 0 not in tick_locs:
        tick_locs = np.append(tick_locs, 0)
    
    # 틱 라벨 생성 (D-Day 및 실제 날짜 포함)
    xticklabels = []
    for x in tick_locs:
        if x in days_from_last:
            idx = days_from_last.index(x)
            real_date = dates[idx].strftime('%m-%d')
        else:
            real_date = (ref_date + pd.Timedelta(days=int(x))).strftime('%m-%d')
        xticklabels.append(f"D{int(x)}\n({real_date})")

    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels(xticklabels, fontsize=10, color='#34495e')
    axes[-1].set_xlabel(f"Timeline to {title_suffix}", fontsize=12, fontweight="bold", color='#2c3e50', labelpad=15)

    fig.suptitle(
        f"Temporal SHAP Evolution & Risk Trajectory (Serial: {serial})",
        fontsize=16, fontweight="bold", color='#2c3e50', y=1.02
    )
    plt.tight_layout()
    plt.show()
    print(f"\n개체 {serial} 분석 완료. 상위 {top_n}개 피처: {top_feat_names}")
"""

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if '# §9.3 — 고장 전 N일 창 SHAP Trajectory' in s:
            c['source'] = [line + '\n' for line in cell_9_3.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
