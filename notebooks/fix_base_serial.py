import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        
        # 1. Add base_serial to df_test
        if 'df_test[cfg.DATE_COL] = pd.to_datetime(df_test[cfg.DATE_COL])' in s:
            new_source = []
            for line in c['source']:
                new_source.append(line)
                if 'df_test[cfg.DATE_COL] = pd.to_datetime(df_test[cfg.DATE_COL])' in line:
                    if "df_test['base_serial']" not in s:  # Prevent duplicate insertions
                        new_source.append("df_test['base_serial'] = df_test[cfg.SERIAL_COL].str.replace(r'_\\d+$', '', regex=True)\n")
            c['source'] = new_source
        
        # 2. Replace cfg.SERIAL_COL with 'base_serial' in filtering
        c['source'] = [line.replace('df_test[cfg.SERIAL_COL] == serial_num', "df_test['base_serial'] == serial_num") for line in c['source']]
        c['source'] = [line.replace('df_test[cfg.SERIAL_COL] == serial]', "df_test['base_serial'] == serial]") for line in c['source']]
        # Catch edge cases
        c['source'] = [line.replace('df_test[cfg.SERIAL_COL] == serial ', "df_test['base_serial'] == serial ") for line in c['source']]
        c['source'] = [line.replace('df_test[cfg.SERIAL_COL] == s]', "df_test['base_serial'] == s]") for line in c['source']]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
