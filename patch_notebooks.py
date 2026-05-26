import json
import glob

code_to_prepend = """
# 누락된 변수 초기화
df_test['base_serial'] = df_test[fe_cfg.SERIAL_COL].str.replace(r'_\\d+$', '', regex=True)
df_test['_prob'] = y_prob

"""

for nb_file in glob.glob('notebooks/09[b-e]*.ipynb'):
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if 'def print_lifecycle' in src:
                if '# 누락된 변수 초기화' not in src:
                    cell['source'] = [code_to_prepend] + cell['source']
                break
                
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebooks patched successfully!")
