import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if '# SHAP 계산용 5:5 균형 추출' in s:
            new_source = []
            for line in c['source']:
                if 'np.save(shap_cache_path, mean_shap)' in line:
                    new_source.append(line)
                    new_source.append("    try:\n")
                    new_source.append("        import subprocess\n")
                    new_source.append("        subprocess.run(['git', 'add', str(shap_cache_path)], check=True)\n")
                    new_source.append("        print(f'\\n✅  SHAP 계산 및 캐시 저장(Git Add 자동화) 완료!')\n")
                    new_source.append("    except Exception as e:\n")
                    new_source.append("        print(f'\\n✅  SHAP 계산 및 캐시 저장 완료! (Git 자동 등록 실패: {e})')\n")
                elif 'print(f"\\n✅  SHAP 계산 및 캐시 저장 완료!")' in line:
                    pass # skip the original print
                else:
                    new_source.append(line)
            c['source'] = new_source
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
