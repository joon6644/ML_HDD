import json
import glob

with open('scratch_dump.py', 'w', encoding='utf-8') as out:
    for f in glob.glob('notebooks/09*.ipynb'):
        out.write(f"# --- {f} ---\n")
        try:
            d = json.load(open(f, encoding='utf-8'))
            for c in d.get('cells', []):
                if c.get('cell_type') == 'code':
                    src = ''.join(c.get('source', []))
                    if 'shap' in src or 'plot' in src or 'ax.' in src:
                        out.write(src)
                        out.write("\n\n")
        except Exception as e:
            out.write(f"# Error: {e}\n\n")
