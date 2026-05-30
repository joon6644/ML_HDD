"""
data_splitter.py  ─  비대칭 언더배깅 서브셋 분할기
──────────────────────────────────────────────────────────
역할:
  학습 데이터를 AsymmetricSampler 로 N개의 서브셋으로 분할하고
  parquet 파일로 저장하는 전처리 단계에서만 사용.

  → 학습 파이프라인(train_core.py) 은 미리 저장된 서브셋을
    로드하기만 하므로 이 모듈을 import 하지 않음.

사용 위치:
  notebooks/19_data_preprocessing_subsets.ipynb
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AsymmetricSampler:
    """
    고장 데이터(Positive)는 모두 사용하고, 정상 데이터(Negative)는
    여러 개의 서브셋으로 나누어 언더샘플링을 수행함.
    고장 임박 데이터(near-failure)에 더 높은 샘플링 가중치를 부여.
    """

    def __init__(
        self,
        n_subsets: int = 10,
        neg_ratio: float = 10.0,
        near_window: int = 30,
        near_weight: float = 3.0,
        seed: int = 42,
    ):
        self.n_subsets = n_subsets
        self.neg_ratio = neg_ratio
        self.near_window = near_window
        self.near_weight = near_weight
        self.seed = seed

    def _build_near_failure_weights(self, df_neg: pd.DataFrame, target_col: str) -> np.ndarray:
        """정상 데이터 내에서 고장 임박 데이터에 높은 가중치 부여."""
        weights = np.ones(len(df_neg))
        if "days_to_failure" in df_neg.columns:
            dtf = df_neg["days_to_failure"].values
            near_mask = (dtf >= 1) & (dtf <= self.near_window)
            weights[near_mask] = self.near_weight
        return weights / weights.sum()

    def split(self, df: pd.DataFrame, target_col: str = "failure") -> list[pd.DataFrame]:
        """전체 데이터를 n_subsets 개의 언더샘플링된 서브셋으로 분할."""
        # Calculate days_to_failure on the fly if missing but required metadata columns exist
        if "days_to_failure" not in df.columns and "serial_number" in df.columns and "date" in df.columns:
            df = df.copy()
            base_serials = df['serial_number'].str.replace(r'_\d+$', '', regex=True)
            fail_dates = df[df[target_col] == 1].groupby(base_serials)['date'].max().reset_index()
            fail_dates.columns = ['base_serial', 'fail_date']
            df['base_serial'] = base_serials
            df = df.merge(fail_dates, on='base_serial', how='left')
            df['days_to_failure'] = (pd.to_datetime(df['fail_date']) - pd.to_datetime(df['date'])).dt.days
            df.drop(columns=['base_serial', 'fail_date'], inplace=True)

        df_pos = df[df[target_col] == 1].copy()
        df_neg = df[df[target_col] == 0].copy()

        n_pos = len(df_pos)
        n_neg_per_subset = int(n_pos * self.neg_ratio)

        neg_weights = self._build_near_failure_weights(df_neg, target_col)
        rng = np.random.default_rng(self.seed)

        subsets = []
        for i in range(self.n_subsets):
            selected_neg_idx = rng.choice(
                df_neg.index,
                size=min(n_neg_per_subset, len(df_neg)),
                replace=False,
                p=neg_weights,
            )
            df_sub = pd.concat([df_pos, df_neg.loc[selected_neg_idx]]).sample(frac=1, random_state=self.seed + i)
            subsets.append(df_sub)

        return subsets
