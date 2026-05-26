import json

filename = 'notebooks/08c_final_evaluation_labeled.ipynb'

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify cell 4 (markdown)
src_4 = ''.join(nb['cells'][4]['source'])
src_4 = src_4.replace("10일, 20일, 30일", "10일, 20일, 30일, 40일, 60일")
nb['cells'][4]['source'] = [src_4]

# Modify cell 5 (code)
src_5 = ''.join(nb['cells'][5]['source'])
src_5 = src_5.replace("horizons = [10, 20, 30]", "horizons = [10, 20, 30, 40, 60]")
src_5 = src_5.replace("'Disk Recall (30일)', 'Precision (30일)']", "'Disk Recall (30일)', 'Precision (30일)', 'Disk Recall (40일)', 'Precision (40일)', 'Disk Recall (60일)', 'Precision (60일)']")
nb['cells'][5]['source'] = [src_5]

# Modify cell 6 (markdown)
src_6 = ''.join(nb['cells'][6]['source'])
src_6 = src_6.replace("10일, 20일, 30일", "10일, 20일, 30일, 40일, 60일")
nb['cells'][6]['source'] = [src_6]

# Modify cell 7 (code)
src_7 = ''.join(nb['cells'][7]['source'])
src_7 = src_7.replace("horizons = [10, 20, 30]", "horizons = [10, 20, 30, 40, 60]")
src_7 = src_7.replace("days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30]", "days_thresholds = [0, 1, 3, 5, 7, 10, 14, 21, 30, 40, 60]")
nb['cells'][7]['source'] = [src_7]

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Modified horizons in 08c successfully.")
