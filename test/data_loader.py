import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def prepare_dataset(parquet_path):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Sort by date
    df = df.sort_values(by=['serial_number', 'date'])
    
    # Group by serial_number
    grouped = df.groupby('serial_number')
    
    # Max date and failure mapping per drive
    df['max_date'] = grouped['date'].transform('max')
    df['has_failed'] = grouped['failure'].transform('max')
    
    # RUL calculation (Target variable)
    df['RUL'] = (df['max_date'] - df['date']).dt.days
    
    # Censoring indicator: 1 if right-censored (no failure observed), 0 if exact RUL (failure observed)
    df['censored'] = 1 - df['has_failed']
    
    # Drop temp columns
    df = df.drop(columns=['max_date', 'has_failed'])
    
    # Select features
    # Exclude non-feature columns, plus 'smart_10_raw' due to near-zero variance and extreme 16-bit encoding outliers.
    exclude_cols = ['serial_number', 'date', 'failure', 'RUL', 'censored', 'smart_10_raw']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Fill missing values if any
    df[features] = df[features].fillna(0)
    
    return df, features

def get_data(data_dir=r'C:\Workspace\projects\26_1_COIN\data2\03_splitting'):
    # Loading val_tune_raw as train for faster execution during test.
    # For actual training, you can replace 'val_tune_raw.parquet' with 'train_raw.parquet'
    train_df, features = prepare_dataset(os.path.join(data_dir, 'val_tune_raw.parquet'))
    test_df, _ = prepare_dataset(os.path.join(data_dir, 'test_raw.parquet'))
    
    # Standardize features using parameters from the training set
    print("Standardizing features...")
    scaler = StandardScaler()
    
    # Fit on train and transform train
    train_df[features] = scaler.fit_transform(train_df[features])
    
    # Transform test using train parameters
    test_df[features] = scaler.transform(test_df[features])
    
    # Strictly clip all standardized features to [-10.0, 10.0] to neutralize any remaining extreme anomalies
    train_df[features] = train_df[features].clip(-10.0, 10.0)
    test_df[features] = test_df[features].clip(-10.0, 10.0)
    
    return train_df, test_df, features
