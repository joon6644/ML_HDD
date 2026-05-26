import json
import io

with open('notebooks/07b_disk_threshold_tuning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "BEST_N = int(opt_row['min_alarms'])" in line:
                continue
            if 'BEST_N' in line and 'evaluate_detailed_disk_point' in line:
                line = line.replace('BEST_N', '1')
            if 'print(f"  [' in line and 'Rolling W=' in line:
                line = 'print(f"  [최적 운영점] 임계값(Threshold) = {BEST_T:.4f} (단일 알람 기준)\\n")\n'
            if 'print(f"🎯 적용할' in line:
                line = 'print(f"🎯 적용할 최적 운영점: 임계값 T = {BEST_T:.4f} (단일 알람 기준)\\n")\n'
            new_source.append(line)
        
        source_str = ''.join(new_source)
        
        target_block1 = '''    if cfg.ALARM_WINDOW is not None and cfg.ALARM_WINDOW > 0:
        rolling_alarms = pd.Series(y_pred).rolling(window=cfg.ALARM_WINDOW, min_periods=1).sum().values
        is_alarmed = int((rolling_alarms >= BEST_N).any())
    else:
        is_alarmed = int(total_alarms >= BEST_N)'''
        
        replacement1 = '''    is_alarmed = int(total_alarms >= 1)'''
        source_str = source_str.replace(target_block1, replacement1)
        
        target_block2 = '''        if cfg.ALARM_WINDOW is not None and cfg.ALARM_WINDOW > 0:
            rolling_alarms = pd.Series(y_pred).rolling(window=cfg.ALARM_WINDOW, min_periods=1).sum().values
            trigger_idx = np.where(rolling_alarms >= BEST_N)[0][0]
        else:
            trigger_idx = np.where(y_pred == 1)[0][BEST_N - 1]'''
            
        replacement2 = '''        trigger_idx = np.where(y_pred == 1)[0][0]'''
        source_str = source_str.replace(target_block2, replacement2)
        
        source_str = source_str.replace('len(thresholds)개 조합 (window_size=0)', 'len(thresholds)개 조합 (단일 알람 기준)')
        
        lines = []
        for l in source_str.split('\n'):
            lines.append(l + '\n')
        # remove the last extra newline if it was empty
        if len(lines) > 0 and lines[-1] == '\n':
            lines.pop()
        elif len(lines) > 0 and lines[-1].endswith('\n\n'):
            lines[-1] = lines[-1][:-1]
            
        cell['source'] = lines

with open('notebooks/07b_disk_threshold_tuning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
