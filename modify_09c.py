import json

file_path = 'notebooks/09c_normal_false_alarm_analysis.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'ref_date = min(first_alarm_date + pd.Timedelta(days=5), df_entity[fe_cfg.DATE_COL].max())' in line:
                source[i] = line.replace('days=5', 'days=window_days // 2')
            if 'title_suffix = "Observation End (D-0)"' in line and i > 0 and 'ref_date = min(' in source[i-2]:
                source[i] = line.replace('"Observation End (D-0)"', 'f"Observation (First Alarm + {window_days // 2}d as D-0)"')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Modified 09c notebook!")
