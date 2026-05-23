import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_9_2 = r"""# §9.2 Waterfall Plot — 고장 확률이 가장 높은 시점 (각 개체별)
for serial in target_serials:
    df_entity = df_test[df_test[cfg.SERIAL_COL] == serial].sort_values(cfg.DATE_COL)
    X_entity = df_entity[FEATURE_COLS].reset_index(drop=True)

    # 앙상블 전체의 SHAP 평균 계산
    sv_entity_list = []
    for model in models:
        exp = shap.TreeExplainer(model)
        sv_e = exp.shap_values(X_entity)
        if isinstance(sv_e, list):
            sv_e = sv_e[1]
        sv_entity_list.append(sv_e)
    mean_sv_entity = np.mean(sv_entity_list, axis=0)

    # 고장 확률이 가장 높은 시점 선택
    peak_idx = df_entity["_prob"].values.argmax()
    peak_date = df_entity[cfg.DATE_COL].iloc[peak_idx].date()
    peak_prob = df_entity["_prob"].iloc[peak_idx]
    
    print(f"\n" + "="*70)
    print(f"[§9.2] Waterfall Plot — 개체: {serial}")
    print(f"       최고 위험 시점: {peak_date} (예측 확률={peak_prob:.4f})")
    print("="*70)

    # 첫 번째 모델 기준 expected_value 사용
    base_vals = np.mean([shap.TreeExplainer(m).expected_value for m in models])
    if isinstance(base_vals, np.ndarray):
        base_vals = base_vals[1]

    # data=None으로 설정하여 좌측의 연한 회색 원본 피처값을 숨김 (보다 세련된 뷰)
    explanation = shap.Explanation(
        values=mean_sv_entity[peak_idx],
        base_values=float(base_vals),
        feature_names=FEATURE_COLS,
    )

    shap.plots.waterfall(explanation, max_display=15, show=False)
    
    # SHAP 라이브러리 내부에서 사용하는 유니코드 마이너스 기호(\u2212) 폰트 깨짐 해결
    fig = plt.gcf()
    for ax in fig.axes:
        for t in ax.texts:
            t.set_text(t.get_text().replace('\u2212', '-'))
        labels = [l.get_text().replace('\u2212', '-') for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        labels = [l.get_text().replace('\u2212', '-') for l in ax.get_yticklabels()]
        ax.set_yticklabels(labels)
        
    plt.tight_layout()
    plt.show()
"""

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

    # 시각화
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 3 * top_n), sharex=True)
    if top_n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for ax, feat_name, feat_idx, color in zip(axes, top_feat_names, top_feat_idx, colors):
        shap_vals = mean_sv_window[:, feat_idx]

        # SHAP 값 (막대 그래프)
        bars = ax.bar(days_from_last, shap_vals, 
                      color=["tomato" if v > 0 else "steelblue" for v in shap_vals], 
                      alpha=0.8, width=0.8)
        
        ax.axhline(0, color="black", lw=1.2, ls="-")
        
        # 고장 예측일(최초 알람 발령일) 세로선 표시
        if first_alarm_date is not None:
            alarm_day = - (ref_date - first_alarm_date).days
            if -window_days <= alarm_day <= 0:
                ax.axvline(alarm_day, color='red', linestyle='--', lw=2, alpha=0.6, zorder=0)
                if ax == axes[0]:
                    ha_val = 'right' if alarm_day == 0 else 'left'
                    ax.text(alarm_day, ax.get_ylim()[1]*0.9, ' 모델 고장 예측일 ', 
                            color='red', fontsize=10, fontweight='bold', va='top', ha=ha_val)

        max_abs = max(abs(shap_vals.min()), abs(shap_vals.max())) * 1.1
        if max_abs > 0:
            ax.set_ylim(-max_abs, max_abs)
            
        # (양수 = 위험 증가) 텍스트 제거하고 깔끔하게 변경
        ax.set_ylabel("SHAP 기여도", fontsize=11, fontweight="bold")
        ax.set_title(f"{feat_name}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # X축 눈금 간격을 5일 단위로 조절
    axes[-1].set_xticks(np.arange(-window_days, 1, 5))
    axes[-1].set_xticklabels([f"D{int(x)}" if x <= 0 else "" for x in axes[-1].get_xticks()])
    # 맨 아래 X축 문구 깔끔하게 수정
    axes[-1].set_xlabel(f"◀ 과거 (D-N)                               경과 일수 (D-Day)                               {title_suffix}", 
                        fontsize=12, fontweight="bold")

    fig.suptitle(
        f"[§9.3] 시간 흐름에 따른 고장 위험도(SHAP) 변화 궤적\n"
        f"개체: {serial}  |  고장 전 {window_days}일 상위 {top_n}개 원인 피처",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()
    print(f"\n개체 {serial} 분석 완료. 상위 {top_n}개 피처: {top_feat_names}")
"""

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if '# §9.2 Waterfall Plot' in s:
            c['source'] = [line + '\n' for line in cell_9_2.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
        elif '# §9.3 — 고장 전 N일 창 SHAP Trajectory' in s:
            c['source'] = [line + '\n' for line in cell_9_3.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
