"""
train_core.py  ─  모델 학습 파이프라인 엔진  (README §6)
──────────────────────────────────────────────────────────
구조:
  1. AsymmetricSampler  : 10-subset 비대칭 언더배깅 샘플링
                          + near-failure importance sampling
  2. SubsetTrainer      : 서브셋 단일 LightGBM 학습 (검증 생략 모드)
  3. UnderbaggingEnsemble : 10개 모델 soft-voting 앙상블
  4. OptunaObjective    : Optuna 목적 함수 (VAL PR-AUC 최대화)
  5. run_training       : 노트북에서 한 셀로 호출하는 메인 함수
  6. Visualization      : 혼동행렬, PR-AUC 등 시각화 함수
"""

from __future__ import annotations

import warnings
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


# ── 한글 폰트 설정 ──────────────────────────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()


# ════════════════════════════════════════════════════════════
#  0. 타입 힌트용 결과 컨테이너
# ════════════════════════════════════════════════════════════

@dataclass
class SubsetResult:
    """서브셋 하나의 학습 결과."""
    subset_id: int
    model: LGBMClassifier
    val_prauc: float                        # val_tune 기준 PR-AUC
    n_train_pos: int                        # 학습에 사용된 고장 행 수
    n_train_neg: int                        # 학습에 사용된 정상 행 수


@dataclass
class EnsembleResult:
    """앙상블 최종 결과."""
    subset_results: list[SubsetResult]
    val_tune_prauc: float                   # val_tune soft-voting PR-AUC
    val_tune_probs: np.ndarray              # val_tune 예측 확률
    val_tune_y_true: np.ndarray             # 실제 정답 (샘플링 대응)
    models: list[LGBMClassifier] = field(default_factory=list)

    def __post_init__(self):
        self.models = [r.model for r in self.subset_results]


# ════════════════════════════════════════════════════════════
#  1. ASYMMETRIC SAMPLER (README §6.1)
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
#  2. SUBSET TRAINER
# ════════════════════════════════════════════════════════════

class SubsetTrainer:
    """단일 서브셋 모델 학습기."""

    def __init__(self, lgbm_params: dict, target_col: str = "failure"):
        self.lgbm_params = lgbm_params
        self.target_col = target_col
        self._meta = {"serial_number", "date", "days_to_failure", target_col}

    def _get_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in self._meta]

    def train(
        self,
        subset_id: int,
        df_subset: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> SubsetResult:
        """단일 서브셋 학습 (검증 완전 생략)."""
        feats = feature_cols or self._get_features(df_subset)

        X_tr = df_subset[feats]
        y_tr = df_subset[self.target_col]

        model = LGBMClassifier(**self.lgbm_params)
        model.fit(X_tr, y_tr)

        return SubsetResult(
            subset_id=subset_id,
            model=model,
            val_prauc=0.0,
            n_train_pos=int(y_tr.sum()),
            n_train_neg=int((y_tr == 0).sum()),
        )


# ════════════════════════════════════════════════════════════
#  3. UNDERBAGGING ENSEMBLE
# ════════════════════════════════════════════════════════════

class UnderbaggingEnsemble:
    """비대칭 언더배깅 앙상블 (학습 후 일괄 검증 방식)."""

    def __init__(self, sampler: AsymmetricSampler, trainer: SubsetTrainer):
        self.sampler = sampler
        self.trainer = trainer
        self._result: Optional[EnsembleResult] = None

    def fit(
        self,
        df_train: pd.DataFrame | list[pd.DataFrame],
        df_val_tune: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "failure",
    ) -> EnsembleResult:
        """전체 앙상블 학습."""
        # 1) 서브셋 분할 (파일이 있으면 로드된 리스트 사용)
        if isinstance(df_train, list):
            subsets = df_train
            print(f"✅  [Ensemble] 사전 분할된 {len(subsets)}개 서브셋 사용.")
        else:
            subsets = self.sampler.split(df_train, target_col=target_col)

        # 2) 각 서브셋 학습 (순수 학습만 수행)
        subset_results: list[SubsetResult] = []
        for i, sub in enumerate(subsets):
            print(f"  🏋️  Subset {i+1}/{len(subsets)} 학습 중...", end="\r")
            res = self.trainer.train(
                subset_id=i,
                df_subset=sub,
                feature_cols=feature_cols,
            )
            subset_results.append(res)

        print(f"\n✅  {len(subsets)}개 모델 학습 완료. 이제 전체 검증을 시작합니다...")

        # 3) Soft-voting (학습이 끝난 후 단 1회 예측)
        feats = feature_cols or [
            c for c in df_val_tune.columns
            if c not in {"serial_number", "date", "days_to_failure", target_col}
        ]
        X_val = df_val_tune[feats]
        y_val = df_val_tune[target_col]

        probs_list = []
        for i, r in enumerate(subset_results):
            print(f"  🔍  모델 {i+1} 예측 및 평가 중...", end="\r")
            p = r.model.predict_proba(X_val)[:, 1]
            probs_list.append(p)
            # 개별 모델 점수 업데이트
            r.val_prauc = average_precision_score(y_val, p)
        
        probs = np.mean(probs_list, axis=0)
        ensemble_prauc = average_precision_score(y_val, probs)

        self._result = EnsembleResult(
            subset_results=subset_results,
            val_tune_prauc=ensemble_prauc,
            val_tune_probs=probs,
            val_tune_y_true=y_val.values if hasattr(y_val, "values") else y_val,
        )

        print(f"\n✨  앙상블 최종 VAL_TUNE PR-AUC = {ensemble_prauc:.5f}")
        return self._result

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """학습 완료 후 새 데이터에 대한 soft-voting 확률 반환."""
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        X = df[feature_cols]
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self._result.models], axis=0)
        return probs


# ════════════════════════════════════════════════════════════
#  4. OPTUNA OBJECTIVE
# ════════════════════════════════════════════════════════════

def make_optuna_objective(
    df_train: pd.DataFrame | list[pd.DataFrame],
    df_val_tune: pd.DataFrame,
    feature_cols: list[str],
    sampler_kwargs: dict,
    target_col: str = "failure",
    device: str = "cpu",
):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "n_estimators": 500,
            "verbosity": -1,
            "device": device,
            "random_state": 42,
            "max_bin": 63,
        }
        
        sampler = AsymmetricSampler(**sampler_kwargs)
        trainer = SubsetTrainer(lgbm_params=params, target_col=target_col)
        ens     = UnderbaggingEnsemble(sampler=sampler, trainer=trainer)

        result  = ens.fit(df_train, df_val_tune, feature_cols=feature_cols, target_col=target_col)
        return result.val_tune_prauc

    return objective


# ════════════════════════════════════════════════════════════
#  5. MAIN RUNNER (run_training)
# ════════════════════════════════════════════════════════════

def run_training(
    cfg,
    feature_cols: Optional[list[str]] = None,
    *,
    run_optuna: bool = False,
    optuna_n_trials: int = 30,
    optuna_timeout: Optional[int] = None,
    show_plots: bool = True,
) -> dict:
    # [데이터 로드]
    subset_dir = Path(getattr(cfg, "SUBSET_DIR", ""))
    subset_files = sorted(list(subset_dir.glob("subset_*.parquet"))) if subset_dir.exists() else []
    if not subset_files:
        raise FileNotFoundError(f"❌ [Error] 사전 분할 데이터가 {subset_dir} 에 없습니다.")

    df_train = [pd.read_parquet(f) for f in subset_files]
    df_val_tune = pd.read_parquet(cfg.VAL_TUNE_PATH)

    print(f"  [Debug] Train Subsets: {len(df_train)} files")
    print(f"  [Debug] Val Tune Rows: {len(df_val_tune):,}")

    # [검증 샘플링]
    _sample_size = getattr(cfg, "VAL_TUNE_SAMPLE_SIZE", None)
    if _sample_size and len(df_val_tune) > _sample_size:
        df_val_tune = df_val_tune.sample(_sample_size, random_state=cfg.SEED)

    feats = feature_cols or getattr(cfg, "FEATURE_COLS", None)
    _device = cfg.LGBM_PARAMS.get("device", "cpu")

    # [Optuna 튜닝]
    best_params = cfg.LGBM_PARAMS.copy()
    if run_optuna:
        import optuna
        obj = make_optuna_objective(df_train, df_val_tune, feats, cfg.SAMPLER_KWARGS, cfg.TARGET_COL, _device)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.SEED))
        study.optimize(obj, n_trials=optuna_n_trials, timeout=optuna_timeout)
        best_params.update(study.best_params)

    # [최종 학습]
    sampler = AsymmetricSampler(**cfg.SAMPLER_KWARGS)
    trainer = SubsetTrainer(lgbm_params=best_params, target_col=cfg.TARGET_COL)
    ens     = UnderbaggingEnsemble(sampler=sampler, trainer=trainer)
    result = ens.fit(df_train, df_val_tune, feature_cols=feats, target_col=cfg.TARGET_COL)

    return {"ensemble_result": result, "best_params": best_params, "feature_cols": feats}


# ════════════════════════════════════════════════════════════
#  6. VISUALIZATION FUNCTIONS
# ════════════════════════════════════════════════════════════

def plot_confusion_matrix(result: EnsembleResult, cfg):
    """임계값별 혼동행렬 시각화 (개수 및 비율 포함)."""
    y_true = result.val_tune_y_true
    probs  = result.val_tune_probs
    thresholds = getattr(cfg, "EVAL_THRESHOLDS", [0.1, 0.2, 0.3, 0.4, 0.5])
    
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.5 * len(thresholds), 4.5))
    if len(thresholds) == 1: axes = [axes]

    for ax, thr in zip(axes, thresholds):
        preds = (probs >= thr).astype(int)
        cm    = confusion_matrix(y_true, preds)
        ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            count = cm[i, j]
            pct = count / cm.sum() * 100
            ax.text(j, i, f"{count:,}\n({pct:.2f}%)",
                    ha="center", color="white" if count > thresh else "black",
                    fontsize=11, fontweight='bold')

        ax.set_title(f'Threshold = {thr}', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Failure'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Failure'])
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')

    plt.suptitle('Confusion Matrix (Count & Percentage)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

def print_ensemble_summary(result: EnsembleResult):
    print("\n" + "="*60)
    print("              UNDERBAGGING ENSEMBLE SUMMARY")
    print("="*60)
    print(f"  Final Ensemble PR-AUC: {result.val_tune_prauc:.5f}")
    print("\n  서브셋별 VAL_TUNE PR-AUC:")
    scores = [r.val_prauc for r in result.subset_results]
    for i, s in enumerate(scores):
        print(f"    Subset {i+1:02d}: {s:.5f}")
    print(f"\n  평균 (단순): {np.mean(scores):.5f}  ±  {np.std(scores):.5f}")
    print("="*60)

def plot_subset_prauc(result: EnsembleResult):
    scores = [r.val_prauc for r in result.subset_results]
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(scores)+1), scores, color='skyblue', edgecolor='navy')
    plt.axhline(result.val_tune_prauc, color='red', linestyle='--', label=f'Ensemble ({result.val_tune_prauc:.4f})')
    plt.title('PR-AUC by Subset Model', fontweight='bold')
    plt.xlabel('Subset ID'); plt.ylabel('PR-AUC')
    plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.show()
