import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        
        if '# §9.3 — 고장 전 N일 창 SHAP Trajectory' in s:
            new_source = []
            for line in c['source']:
                if 'ax_prob.set_title("▶ 종합 고장 위험도' in line:
                    line = line.replace('▶ 종합 고장 위험도', '종합 고장 예측 확률')
                if 'ax.set_title(f"▷ 원인 피처:' in line:
                    line = line.replace('▷ 원인 피처:', '피처:')
                new_source.append(line)
            c['source'] = new_source
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
