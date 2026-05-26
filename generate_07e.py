import json

with open('notebooks/07c_disk_threshold_tuning_labeled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        source = source.replace('07c. 디스크 단위 임계값 튜닝 및 운영점 선정', '07e. 디스크 단위 임계값 튜닝 및 운영점 선정 (val_calib_with_failure_date)')
        source = source.replace('**07b** 노트북', '**07c** 노트북')
        source = source.replace('val_calib 데이터', 'val_calib_with_failure_date 데이터')
        import io
        cell['source'] = [l for l in io.StringIO(source)]
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Replace data path
        source = source.replace('val_calib_path = Path(cfg.VAL_CALIB_PATH)', 'val_calib_path = Path("../data/split_group_stratified/val_calib_with_failure_date.parquet")')
        source = source.replace('val_calib 데이터 로드', 'val_calib_with_failure_date 데이터 로드')
        
        # Replace cache path
        source = source.replace('"val_calib_probs.npy"', '"val_calib_with_failure_date_probs.npy"')
        
        # Replace grid csv output path so we know it's from the enhanced calib set
        source = source.replace('f\'disk_level_grid_search_results_{horizon}d.csv\'', 'f\'disk_level_grid_search_results_calib_enhanced_{horizon}d.csv\'')
        
        import io
        cell['source'] = [l for l in io.StringIO(source)]

with open('notebooks/07e_disk_threshold_tuning_labeled_calib.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
