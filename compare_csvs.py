import pandas as pd

# Load old results (07b)
try:
    df_old = pd.read_csv('models/underbagging_ensemble_4/disk_level_grid_search_results.csv')
    print("Old Results loaded")
except:
    print("Could not load old results")
    df_old = None

# Load new results (07c)
try:
    df_new = pd.read_csv('models/underbagging_ensemble_4/disk_level_grid_search_results_10d.csv')
    print("New Results loaded")
except:
    print("Could not load new results")
    df_new = None

if df_old is not None and df_new is not None:
    # Filter for threshold 0.9820
    old_row = df_old[df_old['threshold'] >= 0.982].iloc[0]
    new_row = df_new[df_new['threshold'] >= 0.982].iloc[0]
    
    print("\n--- OLD (07b) ---")
    print(f"Threshold: {old_row['threshold']:.4f}")
    print(f"Recall:    {old_row['recall']*100:.2f}%")
    print(f"FAR:       {old_row['far']*100:.2f}%")
    
    print("\n--- NEW (07c - 10d) ---")
    print(f"Threshold: {new_row['threshold']:.4f}")
    print(f"Recall:    {new_row['recall']*100:.2f}%")
    print(f"FAR:       {new_row['far']*100:.2f}%")
