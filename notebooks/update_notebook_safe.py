import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_9_2 = r"""def print_lifecycle(serial_num, disk_info):
    df_entity = df_test[df_test[cfg.SERIAL_COL] == serial_num].sort_values(cfg.DATE_COL)
    first_obs = df_entity[cfg.DATE_COL].min().date()
    last_obs = df_entity[cfg.DATE_COL].max().date()
    
    print(f"\n  [개체 생애 (Lifecycle) 요약: {serial_num}]")
    print(f"  - 데이터 수집 기간 : {first_obs} ~ {last_obs} (총 {len(df_entity)}일간 관측)")
    
    raw_alarms = df_entity[df_entity['_prob'] >= BEST_T]
    
    if disk_info['is_alarmed'] == 1:
        if disk_info['is_failed'] == 1:
            first_alarm = last_obs - pd.Timedelta(days=disk_info['lead_time']) if not np.isnan(disk_info['lead_time']) else "해당 없음"
            print(f"  - 시스템 최종 확정 : {first_alarm} (고장 {disk_info['lead_time']}일 전 감지)")
        else:
            alarms = (df_entity['_prob'] >= BEST_T).astype(int)
            rolling = alarms.rolling(window=ALARM_WINDOW, min_periods=1).sum()
            idx = np.where(rolling >= BEST_N)[0]
            first_alarm_date = df_entity[cfg.DATE_COL].iloc[idx[0]].date() if len(idx) > 0 else "N/A"
            days_before = (last_obs - first_alarm_date).days if len(idx) > 0 else "N/A"
            print(f"  - 시스템 최종 확정 : {first_alarm_date} (알람 발령 후 {days_before}일 동안 고장 없이 정상 동작함)")
    else:
        print(f"  - 시스템 최종 확정 : 없음 (경보 미발령)")
        
    if len(raw_alarms) > 0:
        alarm_details = []
        for d in raw_alarms[cfg.DATE_COL].dt.date.values:
            days_before = (last_obs - d).days
            alarm_details.append(f"D-{days_before}")
        print(f"  - 모델 단일 경고 이력: 총 {len(raw_alarms)}회 ({', '.join(alarm_details)})")
    else:
        print(f"  - 모델 단일 경고 이력: 없음")
        
    if disk_info['is_failed'] == 1:
        status = '✅ 탐지 성공 (Hit)' if disk_info['is_alarmed'] else '❌ 미탐 (Miss)'
        fail_text = '고장'
    else:
        status = '❌ 오탐 (False Alarm)' if disk_info['is_alarmed'] else '✅ 정상 정탐 (True Negative)'
        fail_text = '정상'
        
    print(f"  - 최종 평가 결과   : {status} ({fail_text} 디스크)\n")

for serial in target_serials:
    info = df_best_disk[df_best_disk['base_serial'] == serial].iloc[0]
    print("="*60)
    print_lifecycle(serial, info)
"""

cell_9_3 = r"""# §9.3 — 고장 전 N일 창 SHAP Trajectory (각 개체별)
window_days = cfg.TEMPORAL_WINDOW_DAYS
top_n = cfg.TEMPORAL_TOP_N_FEATS

for serial in target_serials:
    df_entity = df_test[df_test[cfg.SERIAL_COL] == serial].sort_values(cfg.DATE_COL)
    info = df_best_disk[df_best_disk['base_serial'] == serial].iloc[0]
    
    is_failed = info['is_failed']
    is_alarmed = info['is_alarmed']
    
    if is_failed == 1:
        ref_date = df_entity[cfg.DATE_COL].max()
        title_suffix = "과거 (D-N)  ➡️  고장 직전 (D-0)"
    else:
        if is_alarmed == 1:
            alarms = (df_entity['_prob'] >= BEST_T).astype(int)
            rolling = alarms.rolling(window=ALARM_WINDOW, min_periods=1).sum()
            idx = np.where(rolling >= BEST_N)[0]
            if len(idx) > 0:
                ref_date = df_entity[cfg.DATE_COL].iloc[idx[0]]
            else:
                ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "과거 (D-N)  ➡️  최초 오탐 발령 (D-0)"
        else:
            ref_date = df_entity[cfg.DATE_COL].max()
            title_suffix = "과거 (D-N)  ➡️  관측 종료 (D-0)"

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
        feat_vals = X_window[feat_name].values

        # 원본 피처값 (오른쪽 y축) - 시선 분산을 막기 위해 투명도 약하게 조절
        ax2 = ax.twinx()
        ax2.plot(days_from_last, feat_vals, color="gray", lw=1.2,
                 linestyle="--", label=f"원본값", alpha=0.3)
        ax2.set_ylabel("원본 피처값", color="gray", fontsize=9, alpha=0.5)
        ax2.tick_params(axis="y", labelcolor="gray", colors="lightgray")

        # SHAP 값 (왼쪽 y축)
        bars = ax.bar(days_from_last, shap_vals, 
                      color=["tomato" if v > 0 else "steelblue" for v in shap_vals], 
                      alpha=0.8, width=0.8)
        
        ax.axhline(0, color="black", lw=1.2, ls="-")
        
        max_abs = max(abs(shap_vals.min()), abs(shap_vals.max())) * 1.1
        if max_abs > 0:
            ax.set_ylim(-max_abs, max_abs)
            
        ax.set_ylabel("SHAP 기여도\n(양수 = 위험 증가)", fontsize=10, fontweight="bold")
        ax.set_title(f"피처: {feat_name}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # X축 눈금 간격을 5일 단위로 조절
    axes[-1].set_xticks(np.arange(-window_days, 1, 5))
    axes[-1].set_xticklabels([f"D{int(x)}" if x <= 0 else "" for x in axes[-1].get_xticks()])
    axes[-1].set_xlabel(f"◀ 과거 (D-N)                               경과 일수 (D-Day)                               {title_suffix}", 
                        fontsize=12, fontweight="bold")

    fig.suptitle(
        f"[§9.3] 시간 흐름에 따른 고장 위험도(SHAP) 변화 궤적\n"
        f"개체: {serial}  |  고장 전 {window_days}일 상위 {top_n}개 주요 원인 피처",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()
    print(f"\n개체 {serial} 분석 완료. 고장 위험을 가장 크게 높인 상위 {top_n}개 피처: {top_feat_names}")
"""

cell_9_4 = r"""# §9.4 — 실무 활용 요약 출력
SEP = "=" * 65

print(SEP)
print("  §9.4  실무 활용 효과 요약")
print(SEP)

# 상위 피처 목록 (§9.1 결과 기반)
top5_global = shap_rank_df["feature"].head(5).tolist()
print("\n▶ [전역 해석 - §9.1] 주요 고장 기여 피처 Top 5:")
for rank, feat in enumerate(top5_global, 1):
    score = shap_rank_df.loc[shap_rank_df["feature"] == feat, "mean_abs_shap"].values[0]
    print(f"    {rank}. {feat:<35} (Mean |SHAP| = {score:.5f})")

print(f"\n▶ [국소 해석 - §9.2] 분석 개체: {', '.join(target_serials)}")
print(f"    → Waterfall Plot에서 해당 시점의 개별 피처 기여도 확인 가능")

print(f"\n▶ [시간 해석 - §9.3] 고장/오탐 전 {window_days}일 창 분석")
print(f"    → 어느 피처가 언제부터 급격히 기여도가 상승했는지 궤적 확인 가능")

print(f"\n▶ 실무 의미:")
print(f"    1. 조기 경보 원인 설명: 특정 SMART 지표가 언제부터 이상 신호를 보냈는지 설명")
print(f"    2. 유지보수 판단 근거: 엔지니어에게 '이 디스크는 A 피처 때문에 경보 발령'이라는 근거 제공")
print(f"    3. 모델 신뢰성 향상: Black-box 모델의 의사결정 과정을 투명하게 공개")
print(SEP)
"""

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if 'def print_lifecycle(serial_num, disk_info):' in s:
            c['source'] = [line + '\n' for line in cell_9_2.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
        elif '# §9.3 — 고장 전 N일 창 SHAP Trajectory' in s:
            c['source'] = [line + '\n' for line in cell_9_3.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
        elif '# §9.4 — 실무 활용 요약 출력' in s:
            c['source'] = [line + '\n' for line in cell_9_4.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
