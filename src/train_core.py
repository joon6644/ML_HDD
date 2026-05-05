"""
train_core.py  ─  모델 학습 파이프라인 엔진  (README §6)
──────────────────────────────────────────────────────────
구조:
  1. AsymmetricSampler  : 10-subset 비대칭 언더배깅 샘플링
                          + near-failure importance sampling
  2. SubsetTrainer      : 서브셋 단일 LightGBM 학습
  3. UnderbaggingEnsemble : 10개 모델 soft-voting 앙상블
  4. OptunaObjective    : Optuna 목적 함수 (VAL PR-AUC 최대화)
  5. run_training       : 노트북에서 한 셀로 호출하는 메인 함수

설정은 config/train_config.py 에서만.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


# ── 한글 폰트 (fs_core.py 와 동일 패턴) ──────────────────────
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
    models: list[LGBMClassifier] = field(default_factory=list)

    def __post_init__(self):
        self.models = [r.model for r in self.subset_results]


# ════════════════════════════════════════════════════════════
#  1. ASYMMETRIC SAMPLER
# ════════════════════════════════════════════════════════════

class AsymmetricSampler:
    """
    README §6.1 비대칭 언더배깅 샘플러.

    - failure 행: 100% 포함 (전 서브셋 공유)
    - normal  행: 비복원 추출, neg_ratio 배수 (기본 10:1)
    - near-failure importance sampling:
        D-1 ~ D-near_window 구간의 정상 행에 weight 부여
        → 고장 직전 패턴을 더 자주 학습
    """

    def __init__(
        self,
        n_subsets: int = 10,
        neg_ratio: int = 10,
        near_window: int = 30,       # 고장 전 n일을 중요 구간으로 정의
        near_weight: float = 3.0,    # 중요 구간 정상 행의 상대 가중치
        seed: int = 42,
    ):
        self.n_subsets   = n_subsets
        self.neg_ratio   = neg_ratio
        self.near_window = near_window
        self.near_weight = near_weight
        self.seed        = seed

    # ----------------------------------------------------------
    def _build_near_failure_weights(
        self,
        df_neg: pd.DataFrame,
    ) -> np.ndarray:
        """
        정상 행(y=0)에 대해 near-failure 가중치 벡터를 생성.

        near_window 계산 기준:
          'days_to_failure' 컬럼이 있으면 사용,
          없으면 모든 정상 행에 균등 가중치 1.0 적용.
        """
        weights = np.ones(len(df_neg), dtype=float)

        if "days_to_failure" in df_neg.columns:
            dtf = df_neg["days_to_failure"].values
            # 1 ~ near_window 사이인 정상 행 → 높은 가중치
            near_mask = (dtf >= 1) & (dtf <= self.near_window)
            weights[near_mask] = self.near_weight

        # 확률로 정규화
        weights /= weights.sum()
        return weights

    # ----------------------------------------------------------
    def split(
        self,
        df: pd.DataFrame,
        target_col: str = "failure",
    ) -> list[pd.DataFrame]:
        """
        Parameters
        ----------
        df         : 학습 데이터 전체 (serial_number, date 포함 가능)
        target_col : 타겟 컬럼명

        Returns
        -------
        list of n_subsets DataFrames
            각 df 는 failure 전체 + 정상 샘플링 결과
        """
        rng = np.random.default_rng(self.seed)

        df_pos = df[df[target_col] == 1].reset_index(drop=True)
        df_neg = df[df[target_col] == 0].reset_index(drop=True)

        n_pos        = len(df_pos)
        n_neg_needed = n_pos * self.neg_ratio

        if n_neg_needed > len(df_neg):
            print(
                f"⚠️  [Sampler] 정상 행 부족: 필요={n_neg_needed:,} "
                f"/ 보유={len(df_neg):,} → 보유 전체 사용"
            )
            n_neg_needed = len(df_neg)

        weights = self._build_near_failure_weights(df_neg)

        subsets: list[pd.DataFrame] = []
        # 비복원 추출을 위해 전체 인덱스 풀에서 순서대로 뽑음
        # (n_subsets * n_neg_needed 가 전체보다 크면 반복 허용)
        total_needed = self.n_subsets * n_neg_needed

        if total_needed <= len(df_neg):
            # 완전 비복원: 한 번에 섞어서 분배
            all_neg_idx = rng.choice(
                len(df_neg),
                size=total_needed,
                replace=False,
                p=weights,
            )
            splits = np.array_split(all_neg_idx, self.n_subsets)
        else:
            # 정상 행이 부족 → 서브셋마다 독립 샘플링 (복원 허용 최소화)
            splits = [
                rng.choice(len(df_neg), size=n_neg_needed, replace=True, p=weights)
                for _ in range(self.n_subsets)
            ]

        for i, neg_idx in enumerate(splits):
            df_neg_sub = df_neg.iloc[neg_idx].copy()
            sub = pd.concat([df_pos, df_neg_sub], axis=0).sample(
                frac=1, random_state=self.seed + i
            ).reset_index(drop=True)
            subsets.append(sub)

        # 요약 출력
        print(f"✅  [Sampler] {self.n_subsets}개 서브셋 생성 완료")
        print(f"   pos={n_pos:,}  neg/subset={n_neg_needed:,}  "
              f"total/subset={n_pos + n_neg_needed:,}")
        if "days_to_failure" in df.columns:
            near = (df_neg["days_to_failure"].between(1, self.near_window)).sum()
            print(f"   near-failure rows (D-1~D-{self.near_window}): {near:,}  "
                  f"(weight={self.near_weight}x)")

        return subsets


# ════════════════════════════════════════════════════════════
#  2. SUBSET TRAINER
# ════════════════════════════════════════════════════════════

class SubsetTrainer:
    """
    서브셋 하나에 LightGBM 을 학습시키고 SubsetResult 반환.

    val_tune 은 학습에 사용되지 않고 평가에만 쓰임.
    """

    def __init__(self, lgbm_params: dict, target_col: str = "failure"):
        self.lgbm_params = lgbm_params
        self.target_col  = target_col
        # 메타 컬럼은 feature에서 제외
        self._meta = {"serial_number", "date", "days_to_failure", target_col}

    def _get_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in self._meta]

    def train(
        self,
        subset_id: int,
        df_subset: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> SubsetResult:
        """
        Parameters
        ----------
        subset_id    : 서브셋 번호 (0-indexed)
        df_subset    : AsymmetricSampler 가 만든 서브셋
        df_val       : val_tune 검증 데이터 (실제 분포)
        feature_cols : None 이면 메타 제외 전체 컬럼 사용
        """
        feats = feature_cols or self._get_features(df_subset)

        X_tr = df_subset[feats]
        y_tr = df_subset[self.target_col]
        X_val = df_val[feats]
        y_val = df_val[self.target_col]

        model = LGBMClassifier(**self.lgbm_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(stopping_rounds=30, verbose=False),
                log_evaluation(period=-1),   # silent
            ],
        )

        val_prob   = model.predict_proba(X_val)[:, 1]
        val_prauc  = average_precision_score(y_val, val_prob)

        return SubsetResult(
            subset_id=subset_id,
            model=model,
            val_prauc=val_prauc,
            n_train_pos=int(y_tr.sum()),
            n_train_neg=int((y_tr == 0).sum()),
        )


# ════════════════════════════════════════════════════════════
#  3. UNDERBAGGING ENSEMBLE
# ════════════════════════════════════════════════════════════

class UnderbaggingEnsemble:
    """
    README §6.2~6.3 비대칭 언더배깅 앙상블.

    - 10개 LightGBM 서브셋 모델 학습
    - soft-voting (probability averaging) 으로 최종 예측
    """

    def __init__(
        self,
        sampler: AsymmetricSampler,
        trainer: SubsetTrainer,
    ):
        self.sampler = sampler
        self.trainer = trainer
        self._result: Optional[EnsembleResult] = None

    # ----------------------------------------------------------
    def fit(
        self,
        df_train: pd.DataFrame,
        df_val_tune: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "failure",
    ) -> EnsembleResult:
        """
        전체 앙상블 학습.

        Returns
        -------
        EnsembleResult
        """
        # 1) 서브셋 분할
        subsets = self.sampler.split(df_train, target_col=target_col)

        # 2) 각 서브셋 학습
        subset_results: list[SubsetResult] = []
        for i, sub in enumerate(subsets):
            print(f"  🔧  Subset {i+1}/{len(subsets)} 학습 중...", end=" ")
            res = self.trainer.train(
                subset_id=i,
                df_subset=sub,
                df_val=df_val_tune,
                feature_cols=feature_cols,
            )
            print(f"VAL PR-AUC = {res.val_prauc:.5f}")
            subset_results.append(res)

        # 3) soft-voting
        feats = feature_cols or [
            c for c in df_val_tune.columns
            if c not in {"serial_number", "date", "days_to_failure", target_col}
        ]
        X_val = df_val_tune[feats]
        y_val = df_val_tune[target_col]

        probs = np.mean(
            [r.model.predict_proba(X_val)[:, 1] for r in subset_results],
            axis=0,
        )
        ensemble_prauc = average_precision_score(y_val, probs)

        self._result = EnsembleResult(
            subset_results=subset_results,
            val_tune_prauc=ensemble_prauc,
            val_tune_probs=probs,
        )

        print(f"\n✅  앙상블 VAL_TUNE PR-AUC = {ensemble_prauc:.5f}")
        return self._result

    # ----------------------------------------------------------
    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """학습 완료 후 새 데이터에 대한 soft-voting 확률 반환."""
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        X = df[feature_cols]
        probs = np.mean(
            [m.predict_proba(X)[:, 1] for m in self._result.models],
            axis=0,
        )
        return probs


# ════════════════════════════════════════════════════════════
#  4. OPTUNA OBJECTIVE
# ════════════════════════════════════════════════════════════

def make_optuna_objective(
    df_train: pd.DataFrame,
    df_val_tune: pd.DataFrame,
    feature_cols: list[str],
    sampler_kwargs: Optional[dict] = None,
    target_col: str = "failure",
):
    """
    Optuna 목적 함수 팩토리.

    README §6.4 : maximize VAL_TUNE PR-AUC

    사용 예시:
        import optuna
        objective = make_optuna_objective(df_train, df_val_tune, feats)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

    Parameters
    ----------
    sampler_kwargs : AsymmetricSampler 생성 인자 override dict.
                     None 이면 기본값 사용.
    """
    _sampler_kwargs = sampler_kwargs or {}

    def objective(trial):
        # ── 탐색 공간 ───────────────────────────────────────
        params = {
            "objective":        "binary",
            "metric":           "average_precision",
            "verbosity":        -1,
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples":trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            # 클래스 불균형: scale_pos_weight 는 샘플링 비율에 맞춰 자동 계산
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        }

        sampler = AsymmetricSampler(**_sampler_kwargs)
        trainer = SubsetTrainer(lgbm_params=params, target_col=target_col)
        ens     = UnderbaggingEnsemble(sampler=sampler, trainer=trainer)

        result  = ens.fit(
            df_train=df_train,
            df_val_tune=df_val_tune,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        return result.val_tune_prauc

    return objective


# ════════════════════════════════════════════════════════════
#  5. 유틸: 결과 시각화 / 요약
# ════════════════════════════════════════════════════════════

def print_ensemble_summary(result: EnsembleResult):
    """앙상블 결과 요약 출력."""
    SEP = "=" * 65
    print(SEP)
    print("  ENSEMBLE RESULT SUMMARY")
    print(SEP)
    scores = [r.val_prauc for r in result.subset_results]
    print(f"\n  서브셋별 VAL_TUNE PR-AUC:")
    for r in result.subset_results:
        bar = "█" * int(r.val_prauc * 50)
        print(f"    Subset {r.subset_id+1:02d}: {r.val_prauc:.5f}  {bar}")
    print(f"\n  평균 (단순): {np.mean(scores):.5f}  ±  {np.std(scores):.5f}")
    print(f"  앙상블 soft-voting PR-AUC: {result.val_tune_prauc:.5f}")
    print(SEP)


def plot_subset_prauc(result: EnsembleResult):
    """서브셋별 PR-AUC 막대 그래프."""
    scores = [r.val_prauc for r in result.subset_results]
    ids    = [f"S{r.subset_id+1}" for r in result.subset_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
    bars    = ax.bar(ids, scores, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(np.mean(scores), color="tomato", linestyle="--",
               linewidth=1.5, label=f"평균: {np.mean(scores):.4f}")
    ax.axhline(result.val_tune_prauc, color="royalblue", linestyle="-.",
               linewidth=1.5, label=f"앙상블: {result.val_tune_prauc:.4f}")

    ax.set_ylabel("VAL_TUNE PR-AUC")
    ax.set_title("서브셋별 PR-AUC (Asymmetric Underbagging)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(max(0, min(scores) * 0.95), min(1.0, max(scores) * 1.02))
    ax.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{score:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
#  6. MAIN PIPELINE  (노트북 한 셀 호출용)
# ════════════════════════════════════════════════════════════

def run_training(
    cfg,
    feature_cols: Optional[list[str]] = None,
    *,
    run_optuna: bool = False,
    optuna_n_trials: int = 30,
    optuna_timeout: Optional[int] = None,    # seconds
    show_plots: bool = True,
) -> dict:
    """
    전체 학습 파이프라인.

    Parameters
    ----------
    cfg          : config/train_config.py 모듈
    feature_cols : 사용할 피처 리스트.
                   None 이면 cfg.FEATURE_COLS 사용.
    run_optuna   : True 면 Optuna 튜닝 후 best params 로 최종 학습.
                   False 면 cfg.LGBM_PARAMS 그대로 학습.
    optuna_n_trials : Optuna 트라이얼 수
    optuna_timeout  : Optuna 타임아웃 (초). None 이면 n_trials 로만 제한.
    show_plots   : 결과 시각화 여부

    Returns
    -------
    dict with keys:
        ensemble_result, best_params (Optuna 사용 시), feature_cols
    """
    import os

    # ── 0. 경로 검증 ──────────────────────────────────────────
    for attr in ("TRAIN_PATH", "VAL_TUNE_PATH"):
        path = getattr(cfg, attr, None)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"[CONFIG] {attr}='{path}' 파일을 찾을 수 없습니다.")

    # ── 1. 데이터 로드 ────────────────────────────────────────
    print("📂  데이터 로드 중...")
    df_train    = pd.read_parquet(cfg.TRAIN_PATH)
    df_val_tune = pd.read_parquet(cfg.VAL_TUNE_PATH)

    feats = feature_cols or getattr(cfg, "FEATURE_COLS", None)
    if feats is None:
        _meta = {"serial_number", "date", "days_to_failure", cfg.TARGET_COL}
        feats = [c for c in df_train.columns if c not in _meta]

    print(f"   train={len(df_train):,}  val_tune={len(df_val_tune):,}  "
          f"features={len(feats)}")
    print(f"   pos_rate: train={df_train[cfg.TARGET_COL].mean():.5f}  "
          f"val_tune={df_val_tune[cfg.TARGET_COL].mean():.5f}")

    # ── 2. Optuna 튜닝 (선택) ─────────────────────────────────
    best_params = cfg.LGBM_PARAMS.copy()

    if run_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n🔍  Optuna 하이퍼파라미터 탐색 시작 "
              f"(n_trials={optuna_n_trials}, timeout={optuna_timeout}s)...")

        objective = make_optuna_objective(
            df_train=df_train,
            df_val_tune=df_val_tune,
            feature_cols=feats,
            sampler_kwargs=getattr(cfg, "SAMPLER_KWARGS", {}),
            target_col=cfg.TARGET_COL,
        )
        study = optuna.create_study(
            direction="maximize",
            study_name="underbagging_prauc",
            sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
        )
        study.optimize(
            objective,
            n_trials=optuna_n_trials,
            timeout=optuna_timeout,
            show_progress_bar=True,
        )

        best_params.update(study.best_params)
        print(f"\n✅  Best VAL_TUNE PR-AUC: {study.best_value:.5f}")
        print(f"   Best params: {study.best_params}")
    else:
        study = None

    # ── 3. 최종 앙상블 학습 ───────────────────────────────────
    print("\n🏋️  최종 앙상블 학습 시작...")
    sampler = AsymmetricSampler(**getattr(cfg, "SAMPLER_KWARGS", {}))
    trainer = SubsetTrainer(lgbm_params=best_params, target_col=cfg.TARGET_COL)
    ens     = UnderbaggingEnsemble(sampler=sampler, trainer=trainer)

    result = ens.fit(
        df_train=df_train,
        df_val_tune=df_val_tune,
        feature_cols=feats,
        target_col=cfg.TARGET_COL,
    )

    # ── 4. 결과 출력 / 시각화 ─────────────────────────────────
    print_ensemble_summary(result)

    if show_plots:
        plot_subset_prauc(result)

    return {
        "ensemble_result": result,
        "ensemble":        ens,
        "best_params":     best_params,
        "feature_cols":    feats,
        "optuna_study":    study,
    }
