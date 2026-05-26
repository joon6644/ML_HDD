import pandas as pd
import glob
import os

files = glob.glob('models/underbagging_ensemble_4/*.csv')
for f in files:
    try:
        df = pd.read_csv(f)
        if 'threshold' in df.columns and 'recall' in df.columns and 'far' in df.columns:
            sub = df[df['threshold'] >= 0.982]
            if len(sub) > 0:
                row = sub.iloc[0]
                if abs(row['recall'] - 0.1437) < 0.005 or abs(row['far'] - 0.0016) < 0.005:
                    print(f"Match found in {os.path.basename(f)}:")
                    print(f"Threshold: {row['threshold']:.4f}, Recall: {row['recall']*100:.2f}%, FAR: {row['far']*100:.2f}%")
    except Exception as e:
        pass
