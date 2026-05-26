import json
with open('notebooks/07b_disk_threshold_tuning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('out_07b.txt', 'w', encoding='utf-8') as fout:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for out in cell['outputs']:
                if out.get('output_type') == 'stream':
                    text = "".join(out.get('text', []))
                    if 'Failed Disks' in text or 'Normal Disks' in text:
                        fout.write(text)
