import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        
        if '# §9.2 Waterfall Plot' in s:
            new_source = []
            for line in c['source']:
                if 'plt.tight_layout()' in line:
                    new_source.append('    fig.suptitle(f"최대 고장 위험 시점의 피처별 SHAP 기여도 분석 (대상 개체: {serial})", fontsize=14, fontweight="bold", y=1.05)\n')
                new_source.append(line)
            c['source'] = new_source
            
        elif '# §9.3 — 고장 전 N일 창 SHAP Trajectory' in s:
            new_source = []
            skip = False
            for line in c['source']:
                if 'fig.suptitle(' in line:
                    new_source.append('    fig.suptitle(\n')
                    new_source.append('        f"고장 예측 창({window_days}일) 내 주요 피처의 시간적 SHAP 기여도 추이 (대상 개체: {serial})",\n')
                    new_source.append('        fontsize=14, fontweight="bold", y=1.02\n')
                    new_source.append('    )\n')
                    skip = True
                elif skip and ')' in line and 'plt.tight_layout' not in line:
                    skip = False
                elif skip:
                    pass
                else:
                    new_source.append(line)
            c['source'] = new_source
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
