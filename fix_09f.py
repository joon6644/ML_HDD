import json

nb_file = 'notebooks/09f_failure_observability_analysis.ipynb'
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][3]['source'])
src = src.replace("d['serial']", "d['base_serial']")
nb['cells'][3]['source'] = src

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed serial to base_serial")
