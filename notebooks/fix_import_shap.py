import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if '# SHAP 계산용 5:5 균형 추출' in s:
            if 'import shap' not in s:
                c['source'].insert(0, 'import shap\n\n')
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
