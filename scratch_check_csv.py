import pandas as pd
import numpy as np

for h in [10, 20, 30]:
    try:
        df = pd.read_csv(f'models/underbagging_ensemble_4/disk_level_grid_search_results_{h}d.csv')
        sub = df[df['far'] <= 0.01]
        best_row = sub.sort_values('recall', ascending=False).iloc[0]
        print(f"Horizon {h}: Best T = {best_row['threshold']:.4f}, Recall = {best_row['recall']*100:.2f}%, FAR = {best_row['far']*100:.2f}%")
    except Exception as e:
        print(f'Error reading {h}d: {e}')
