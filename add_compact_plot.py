import json
import glob

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Compact Temporal SHAP Trajectory (논문 본문용)\n",
        "- 정보량이 많아 세로로 길어지는 기존 플롯 대신, 논문 본문에 삽입하기 좋은 **Compact Version**입니다.\n",
        "- 핵심 피처 3개만 표시하며, Probability 패널의 비중을 높여 시각적 안정감을 줍니다."
    ]
}

notebooks = glob.glob('notebooks/09[b-e]*.ipynb')

for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Check if we already appended
    if any("Compact Temporal SHAP Trajectory" in ''.join(c.get('source', [])) for c in nb['cells']):
        print(f"Already applied to {nb_file}")
        continue
        
    # Find the temporal SHAP cell (should be the last code cell, or the one with "window_days = cfg.TEMPORAL_WINDOW_DAYS")
    target_cell = None
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if 'window_days = cfg.TEMPORAL_WINDOW_DAYS' in src and 'df_window' in src:
                target_cell = cell
                
    if not target_cell:
        print(f"Could not find target cell in {nb_file}")
        continue
        
    import copy
    new_cell = copy.deepcopy(target_cell)
    
    # Modify the source of the new cell
    source = new_cell['source']
    for i, line in enumerate(source):
        if 'top_n = cfg.TEMPORAL_TOP_N_FEATS' in line:
            source[i] = line.replace('top_n = cfg.TEMPORAL_TOP_N_FEATS', 'top_n = 3  # 논문용 압축 버전')
        elif 'fig, axes = plt.subplots(top_n + 1, 1, figsize=(12, 3 * (top_n + 1)), sharex=True)' in line:
            source[i] = line.replace(
                'figsize=(12, 3 * (top_n + 1)), sharex=True)', 
                "figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5] + [1]*top_n})"
            )
        elif '[§9.3]' in line:
            source[i] = line.replace('[§9.3]', '[§9.3 Compact]')
            
    nb['cells'].append(markdown_cell)
    nb['cells'].append(new_cell)
    
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        
    print(f"Added compact plot to {nb_file}")

