import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_serials_code = r"""# 분석 대상 시리얼 선택 (비교 분석을 위해 다중 타겟 지원)
target_serials = []
if cfg.TARGET_SERIAL is not None:
    target_serials.append(cfg.TARGET_SERIAL)
    print(f"[수동 지정] 분석 대상: {target_serials[0]}")
else:
    hit_disks = df_best_disk[(df_best_disk['is_failed'] == 1) & (df_best_disk['is_alarmed'] == 1)]
    miss_disks = df_best_disk[(df_best_disk['is_failed'] == 1) & (df_best_disk['is_alarmed'] == 0)]
    fa_disks = df_best_disk[(df_best_disk['is_failed'] == 0) & (df_best_disk['is_alarmed'] == 1)]
    
    selected_types = []
    if len(hit_disks) > 0:
        target_serials.append(hit_disks.sample(1, random_state=cfg.SHAP_SEED).iloc[0]['base_serial'])
        selected_types.append("Hit(탐지 성공)")
    if len(miss_disks) > 0:
        target_serials.append(miss_disks.sample(1, random_state=cfg.SHAP_SEED).iloc[0]['base_serial'])
        selected_types.append("Miss(미탐)")
        
    if len(fa_disks) > 0:
        # 🌟 단순 라벨상 오탐이 아니라, '알람 발령 후 30일 이상 멀쩡히 살아남은 진짜 오탐' 개체 필터링
        real_fa_serial = None
        for s in fa_disks.sample(frac=1, random_state=cfg.SHAP_SEED)['base_serial']:
            df_s = df_test[df_test[cfg.SERIAL_COL] == s]
            alarms = (df_s['_prob'] >= BEST_T).astype(int)
            rolling = alarms.rolling(window=ALARM_WINDOW, min_periods=1).sum()
            idx = np.where(rolling >= BEST_N)[0]
            if len(idx) > 0:
                first_alarm = df_s[cfg.DATE_COL].iloc[idx[0]]
                last_obs = df_s[cfg.DATE_COL].max()
                if (last_obs - first_alarm).days >= 30:
                    real_fa_serial = s
                    break
                    
        if real_fa_serial is not None:
            target_serials.append(real_fa_serial)
            selected_types.append("FA(진짜 오탐)")
        else:
            target_serials.append(fa_disks.sample(1, random_state=cfg.SHAP_SEED).iloc[0]['base_serial'])
            selected_types.append("FA(오탐)")
        
    print(f"[자동 선택] 비교 분석을 위해 {', '.join(selected_types)} 개체를 각각 1개씩 추출했습니다.")

"""

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if 'def print_lifecycle(serial_num, disk_info):' in s:
            if 'target_serials = []' not in s:
                prefix = [line + '\n' for line in target_serials_code.split('\n')]
                c['source'] = prefix[:-1] + c['source']
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
