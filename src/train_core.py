"""
train_core.py  ─  모델 학습 파이프라인 엔진  (README §6)
──────────────────────────────────────────────────────────
구조:
1. SubsetTrainer      : 서브셋 단일 LightGBM 학습 (검증 생략 모드)
2. UnderbaggingEnsemble : 사전 분할된 서브셋 리스트로 soft-voting 앙상블
3. OptunaObjective    : Optuna 목적 함수 (VAL PR-AUC 최대화)
4. run_training       : 노트북에서 한 셀로 호출하는 메인 함수
5. Visualization      : 혼동행렬, PR-AUC 등 시각화 함수

참고:
데이터셋 분할(AsymmetricSampler)은 src/data_splitter.py 에서 담당.
notebooks/19_data_preprocessing_subsets.ipynb 에서 서브셋을 미리 생성한 뒤
이 모듈은 저장된 parquet 파일만 로드하여 학습에 사용함.
"""

from __future__ import annotations

import warnings
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import optuna
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
#  1. SUBSET TRAINER
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
    """비대칭 언더배깅 앙상블 (학습 후 일괄 검증 방식).

    학습 데이터는 notebooks/19_data_preprocessing_subsets.ipynb 에서
    사전 분할된 list[pd.DataFrame] 형태로 전달받음.
    """

    def __init__(self, trainer: SubsetTrainer):
        self.trainer = trainer
        self._result: Optional[EnsembleResult] = None

    def fit(
        self,
        df_train_list: list[pd.DataFrame],
        df_val_tune: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "failure",
        trial: Optional[optuna.Trial] = None,
    ) -> EnsembleResult:
        subsets = df_train_list
        print(f"✅  [Ensemble] 사전 분할된 {len(subsets)}개 서브셋 사용.")

        feats = feature_cols or [
            c for c in df_val_tune.columns
            if c not in {"serial_number", "date", "days_to_failure", target_col}
        ]
        X_val = df_val_tune[feats]
        y_val = df_val_tune[target_col]

        subset_results: list[SubsetResult] = []
        probs_list = []
        
        # 각 서브셋 학습 및 (필요 시) Pruning 중간 평가
        for i, sub in enumerate(subsets):
            print(f"  🏋️  Subset {i+1}/{len(subsets)} 학습 중...", end="\r")
            res = self.trainer.train(
                subset_id=i,
                df_subset=sub,
                feature_cols=feature_cols,
            )
            subset_results.append(res)
            
            # 모델 예측 (Pruning 또는 최종 Soft-voting용)
            p = res.model.predict_proba(X_val)[:, 1]
            probs_list.append(p)
            res.val_prauc = average_precision_score(y_val, p)
            
            if trial is not None:
                cur_probs = np.mean(probs_list, axis=0)
                cur_score = average_precision_score(y_val, cur_probs)
                trial.report(cur_score, step=i)
                if trial.should_prune():
                    print(f"\n🚫  [Pruned] Trial {trial.number} pruned at step {i} (score: {cur_score:.5f})")
                    raise optuna.TrialPruned()

        print(f"\n✅  {len(subsets)}개 모델 학습 및 평가 완료.")

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
    df_train: list[pd.DataFrame],
    df_val_tune: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "failure",
    device: str = "cpu",
    bounds: dict = None,
    tune_estimators: bool = False,
):
    if bounds is None:
        # 기본 Stage 1 (Coarse) 탐색 범위
        bounds = {
            "learning_rate": (0.01, 0.1),
            "max_depth": (4, 10),
            "num_leaves": (16, 128),
            "min_child_samples": (20, 100),
            # feature_fraction: Noisy SMART + Correlated feature 환경에서 
            # 앙상블 다양성을 주되, 부스팅 과정이 망가지지 않는 선(최소 절반)으로 하한 방어
            "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.6, 1.0),
            "lambda_l1": (1e-4, 1.0),
            "lambda_l2": (1e-8, 10.0),
            "n_estimators": (400, 800), # Stage 2에서만 사용됨
        }

    def objective(trial):
        max_depth   = trial.suggest_int("max_depth", *bounds["max_depth"])
        max_leaves  = min(2 ** max_depth, bounds["num_leaves"][1])
        num_leaves  = trial.suggest_int("num_leaves", bounds["num_leaves"][0], max_leaves)
        
        n_estimators = trial.suggest_int("n_estimators", *bounds["n_estimators"]) if tune_estimators else 400

        params = {
            "learning_rate":     trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            "max_depth":         max_depth,
            "num_leaves":        num_leaves,
            "min_child_samples": trial.suggest_int("min_child_samples", *bounds["min_child_samples"]),
            "feature_fraction":  trial.suggest_float("feature_fraction", *bounds["feature_fraction"]),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", *bounds["bagging_fraction"]),
            "bagging_freq":      1,
            "lambda_l1":         trial.suggest_float("lambda_l1", *bounds["lambda_l1"], log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", *bounds["lambda_l2"], log=True),
            "n_estimators":      n_estimators,
            "verbosity":         -1,
            "device":            device,
            "random_state":      42,
            "max_bin":           63,
        }

        trainer = SubsetTrainer(lgbm_params=params, target_col=target_col)
        ens     = UnderbaggingEnsemble(trainer=trainer)

        result  = ens.fit(df_train, df_val_tune, feature_cols=feature_cols, target_col=target_col, trial=trial)
        return result.val_tune_prauc

    return objective


def _narrow_bounds(best_params: dict, orig_bounds: dict) -> dict:
    """Stage 1의 best_params를 기반으로 Stage 2 탐색 범위를 파라미터 특성에 맞게 축소합니다."""
    new_bounds = {}
    
    # 1. 좁게 축소 (학습률) - 민감함
    lr = best_params["learning_rate"]
    # 로그 스케일이므로 대략 ±30% 범위로 축소
    new_bounds["learning_rate"] = (max(orig_bounds["learning_rate"][0], lr * 0.7), 
                                   min(orig_bounds["learning_rate"][1], lr * 1.3))
    
    # 2. 중간 축소 (구조적 파라미터)
    depth = best_params["max_depth"]
    new_bounds["max_depth"] = (max(orig_bounds["max_depth"][0], depth - 1), 
                               min(orig_bounds["max_depth"][1], depth + 2))
    
    leaves = best_params["num_leaves"]
    new_bounds["num_leaves"] = (max(orig_bounds["num_leaves"][0], int(leaves * 0.7)), 
                                min(orig_bounds["num_leaves"][1], int(leaves * 1.3)))
    
    child = best_params["min_child_samples"]
    new_bounds["min_child_samples"] = (max(orig_bounds["min_child_samples"][0], int(child * 0.8)), 
                                       min(orig_bounds["min_child_samples"][1], int(child * 1.2)))
    
    # 3. 넓게 유지 (Regularization & Sampling) - Interaction을 위해 폭 유지
    new_bounds["feature_fraction"] = orig_bounds["feature_fraction"]
    new_bounds["bagging_fraction"] = orig_bounds["bagging_fraction"]
    new_bounds["lambda_l1"] = orig_bounds["lambda_l1"]
    new_bounds["lambda_l2"] = orig_bounds["lambda_l2"]
    
    # n_estimators는 Stage 2에서 해방되므로 원본 바운드 사용
    new_bounds["n_estimators"] = orig_bounds["n_estimators"]
    
    return new_bounds


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
    optuna_rerank_delta: float = 0.005,
    optuna_rerank_cap: int = 5,
    show_plots: bool = True,
) -> dict:
    # [데이터 로드]
    subset_dir = Path(getattr(cfg, "SUBSET_DIR", ""))
    subset_files = sorted(list(subset_dir.glob("subset_*.parquet"))) if subset_dir.exists() else []
    if not subset_files:
        raise FileNotFoundError(f"❌ [Error] 사전 분할 데이터가 {subset_dir} 에 없습니다.")

    df_train = [pd.read_parquet(f) for f in subset_files]
    df_val_tune_full = pd.read_parquet(cfg.VAL_TUNE_PATH)

    print(f"  [Debug] Train Subsets: {len(df_train)} files")
    print(f"  [Debug] Val Tune (Full) Rows: {len(df_val_tune_full):,}")

    # [검증 샘플링]
    _sample_size = getattr(cfg, "VAL_TUNE_SAMPLE_SIZE", None)
    if _sample_size and len(df_val_tune_full) > _sample_size:
        df_val_optuna = df_val_tune_full.sample(_sample_size, random_state=cfg.SEED)
        print(f"  [Debug] Val Tune (Sampled for Optuna) Rows: {len(df_val_optuna):,}")
    else:
        df_val_optuna = df_val_tune_full

    feats = feature_cols or getattr(cfg, "FEATURE_COLS", None)
    _device = cfg.LGBM_PARAMS.get("device", "cpu")

    # [Optuna 튜닝]
    best_params = cfg.LGBM_PARAMS.copy()
    if run_optuna:
        db_path = getattr(cfg, "OPTUNA_DB_PATH", "optuna_study.db")
        base_study_name = getattr(cfg, "OPTUNA_STUDY_NAME", "hdd_failure_prediction")
        storage_url = f"sqlite:///{db_path}"

        # ─── Stage 1 (Coarse Exploration) ───
        print("\n  [Stage 1] 구조적 탐색 시작 (Coarse Exploration, 고정 n_estimators=400)")
        s1_trials = int(optuna_n_trials * 0.6)
        study_s1_name = f"{base_study_name}_s1"
        
        study_s1 = optuna.create_study(
            study_name=study_s1_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3), # Stage 1 적극적 Pruning
            load_if_exists=True,
        )
        
        completed_s1 = len(study_s1.trials)
        if completed_s1 < s1_trials:
            obj_s1 = make_optuna_objective(df_train, df_val_optuna, feats, cfg.TARGET_COL, _device, tune_estimators=False)
            study_s1.optimize(obj_s1, n_trials=s1_trials - completed_s1, timeout=optuna_timeout)
        
        s1_best_params = study_s1.best_params
        print(f"  [Stage 1] 완료. Best PR-AUC: {study_s1.best_value:.5f}")

        # ─── Stage 2 (Refinement) ───
        print("\n  [Stage 2] 정밀 탐색 시작 (Narrowed Bounds & n_estimators Tuning)")
        s2_trials = optuna_n_trials - s1_trials
        study_s2_name = f"{base_study_name}_s2"
        
        default_bounds = {
            "learning_rate": (0.01, 0.1),
            "max_depth": (4, 10),
            "num_leaves": (16, 128),
            "min_child_samples": (20, 100),
            "feature_fraction": (0.6, 1.0),
            "bagging_fraction": (0.6, 1.0),
            "lambda_l1": (1e-4, 1.0),
            "lambda_l2": (1e-8, 10.0),
            "n_estimators": (400, 800),
        }
        narrowed_bounds = _narrow_bounds(s1_best_params, default_bounds)
        
        study_s2 = optuna.create_study(
            study_name=study_s2_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.SEED + 1), # 다른 시드로 다양한 탐색
            pruner=optuna.pruners.NopPruner(), # Stage 2는 미세 조정 단계이므로 Pruning 안함
            load_if_exists=True,
        )
        
        # 웜스타트 주입 (Enqueuing)
        if len(study_s2.trials) == 0:
            warm_params = s1_best_params.copy()
            warm_params["n_estimators"] = 400 # 기본값 명시
            study_s2.enqueue_trial(warm_params)
            print("  [Stage 2] Stage 1 Best 파라미터 웜스타트 주입 완료.")
            
        completed_s2 = len(study_s2.trials)
        if completed_s2 < s2_trials:
            obj_s2 = make_optuna_objective(df_train, df_val_optuna, feats, cfg.TARGET_COL, _device, bounds=narrowed_bounds, tune_estimators=True)
            study_s2.optimize(obj_s2, n_trials=s2_trials - completed_s2, timeout=optuna_timeout)

        print(f"  [Stage 2] 완료. Best PR-AUC: {study_s2.best_value:.5f}")

        # ─── Margin-based Reranking ───
        if _sample_size and optuna_rerank_delta > 0:
            print(f"\n  [Rerank] Margin-based Reranking 시작 (Full Validation, 대상: Stage 2)")
            df_trials = study_s2.trials_dataframe(states=(optuna.trial.TrialState.COMPLETE,))
            
            if not df_trials.empty:
                best_val = df_trials['value'].max()
                
                candidates = df_trials[df_trials['value'] >= best_val - optuna_rerank_delta]
                candidates = candidates.sort_values('value', ascending=False).head(optuna_rerank_cap)
                
                print(f"  [Rerank] 후보 수: {len(candidates)} (best: {best_val:.5f}, delta: {optuna_rerank_delta})")
                
                best_rerank_score = -1.0
                best_rerank_params = None
                
                for idx, row in candidates.iterrows():
                    cand_params = {k.replace('params_', ''): v for k, v in row.items() if k.startswith('params_')}
                    
                    merged_params = cfg.LGBM_PARAMS.copy()
                    merged_params.update(cand_params)
                    
                    trainer = SubsetTrainer(lgbm_params=merged_params, target_col=cfg.TARGET_COL)
                    ens = UnderbaggingEnsemble(trainer=trainer)
                    res = ens.fit(df_train, df_val_tune_full, feature_cols=feats, target_col=cfg.TARGET_COL)
                    
                    score = res.val_tune_prauc
                    print(f"    - Trial {row['number']}: Sampled PR-AUC = {row['value']:.5f} -> Full PR-AUC = {score:.5f}")
                    
                    if score > best_rerank_score:
                        best_rerank_score = score
                        best_rerank_params = merged_params
                
                best_params = best_rerank_params
                print(f"  [Rerank] 최종 선택된 Full PR-AUC: {best_rerank_score:.5f}")
            else:
                best_params.update(study_s2.best_params)
        else:
            best_params.update(study_s2.best_params)

    # [최종 학습] (전체 검증셋 사용)
    print("\n  [Final] 최적 파라미터로 최종 앙상블 학습 (Full Validation)")
    trainer = SubsetTrainer(lgbm_params=best_params, target_col=cfg.TARGET_COL)
    ens     = UnderbaggingEnsemble(trainer=trainer)
    result = ens.fit(df_train, df_val_tune_full, feature_cols=feats, target_col=cfg.TARGET_COL)

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
