import json

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
        title_suffix = "고장 직전 (D-0) ▶"
    else:
        if is_alarmed == 1:
            if first_alarm_date is not None:
                ref_date = first_alarm_date
            else:
                ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "최초 오탐 발령 (D-0) ▶"
        else:
            ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "관측 종료 (D-0) ▶"

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

    dates = df_window[cfg.DATE_COL].values
    days_from_last = [- (ref_date - pd.Timestamp(d)).days for d in dates]

    # 시각화 (종합 확률 + 상위 피처 SHAP)
    fig, axes = plt.subplots(top_n + 1, 1, figsize=(12, 3 * (top_n + 1)), sharex=True)
    
    # 1. 최상단: 전체 모델 예측 확률 궤적 및 임계값
    ax_prob = axes[0]
    probs = df_window["_prob"].values
    
    ax_prob.plot(days_from_last, probs, marker='o', color='purple', lw=2, markersize=5, label="모델 고장 예측 확률")
    ax_prob.axhline(BEST_T, color='red', linestyle='--', lw=2, label=f"위험 임계값 (T={BEST_T:.3f})")
    ax_prob.fill_between(days_from_last, probs, BEST_T, where=(probs >= BEST_T), color='red', alpha=0.15)
    
    # 고장 예측일 세로선 표시 (확률 그래프)
    if first_alarm_date is not None:
        alarm_day = - (ref_date - first_alarm_date).days
        if -window_days <= alarm_day <= 0:
            ax_prob.axvline(alarm_day, color='red', linestyle=':', lw=2, alpha=0.8)
            ha_val = 'right' if alarm_day == 0 else 'left'
            ax_prob.text(alarm_day, ax_prob.get_ylim()[1]*0.9, ' 최초 위험 감지 ', 
                         color='red', fontsize=10, fontweight='bold', va='top', ha=ha_val)

    ax_prob.set_ylabel("고장 예측 확률", fontsize=11, fontweight="bold")
    ax_prob.set_title("▶ 종합 고장 위험도 (Model Prediction Probability)", fontsize=13, fontweight="bold", loc='left')
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc='upper left')
    ax_prob.grid(axis='y', alpha=0.3)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)

    # 2. 하단: 상위 피처별 SHAP 기여도
    colors = plt.cm.tab10.colors
    for ax, feat_name, feat_idx, color in zip(axes[1:], top_feat_names, top_feat_idx, colors):
        shap_vals = mean_sv_window[:, feat_idx]

        # SHAP 값 (막대 그래프)
        bars = ax.bar(days_from_last, shap_vals, 
                      color=["tomato" if v > 0 else "steelblue" for v in shap_vals], 
                      alpha=0.8, width=0.8)
        
        ax.axhline(0, color="black", lw=1.2, ls="-")
        
        # 고장 예측일 세로선 표시
        if first_alarm_date is not None:
            alarm_day = - (ref_date - first_alarm_date).days
            if -window_days <= alarm_day <= 0:
                ax.axvline(alarm_day, color='red', linestyle='--', lw=2, alpha=0.4, zorder=0)

        max_abs = max(abs(shap_vals.min()), abs(shap_vals.max())) * 1.1
        if max_abs > 0:
            ax.set_ylim(-max_abs, max_abs)
            
        ax.set_ylabel("SHAP 기여도", fontsize=11, fontweight="bold")
        ax.set_title(f"▷ 원인 피처: {feat_name}", fontsize=12, fontweight="bold", loc='left')
        ax.grid(axis="y", alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # X축 눈금 간격을 5일 단위로 조절
    axes[-1].set_xticks(np.arange(-window_days, 1, 5))
    axes[-1].set_xticklabels([f"D{int(x)}" if x <= 0 else "" for x in axes[-1].get_xticks()])
    axes[-1].set_xlabel(f"◀ 과거 (D-N)                               경과 일수 (D-Day)                               {title_suffix}", 
                        fontsize=12, fontweight="bold")

    fig.suptitle(
        f"고장 예측 창({window_days}일) 내 종합 위험도 및 주요 원인 피처 SHAP 추이 (대상 개체: {serial})",
        fontsize=15, fontweight="bold", y=1.02
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
