import pandas as pd

try:
    df = pd.read_csv('models/underbagging_ensemble_4/disk_level_grid_search_results_calib_enhanced_10d.csv')
    row = df[df['threshold'] >= 0.982].iloc[0]
    print(f"Threshold: {row['threshold']:.4f}")
    print(f"Recall: {row['recall']*100:.2f}%")
    print(f"FAR: {row['far']*100:.2f}%")
except Exception as e:
    print(e)
