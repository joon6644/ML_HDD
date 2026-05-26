import json
with open('notebooks/07c_disk_threshold_tuning_labeled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        if any('df_val' in line or 'df_test' in line for line in cell['source']):
            source = "".join(cell['source']).encode('utf-8').decode('unicode_escape', errors='ignore')
            print(source)
