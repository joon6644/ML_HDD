import json

with open('notebooks/08b_final_evaluation_labeled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"Cell {i} ({cell['cell_type']}):")
    src = ''.join(cell.get('source', []))
    print(src[:200] + '...' if len(src) > 200 else src)
    print("---")
