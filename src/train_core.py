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
from src.data_splitter import AsymmetricSampler

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

        feats = feature_cols or [
            c for c in df_val_tune.columns
            if c not in {"serial_number", "date", "days_to_failure", target_col}
        ]

        # 🔴 Feature Schema Validation 추가 (결측 컬럼 검증)
        missing_val = set(feats) - set(df_val_tune.columns)
        if missing_val:
            raise ValueError(f"❌ [Error] 검증 데이터(val_tune)에 다음 특성이 누락되어 있습니다: {missing_val}")
            
        for i, sub in enumerate(subsets):
            missing_sub = set(feats) - set(sub.columns)
            if missing_sub:
                raise ValueError(f"❌ [Error] 훈련 서브셋 {i} 에 다음 특성이 누락되어 있습니다: {missing_sub}")

        X_val = df_val_tune[feats].astype(np.float32)
        y_val = df_val_tune[target_col]

        subset_results: list[SubsetResult] = []
        probs_list = []
        
        is_optuna = trial is not None
        
        # 각 서브셋 학습 및 (필요 시) Pruning 중간 평가
        for i, sub in enumerate(subsets):
            if is_optuna:
                msg = f"  🏋️  Trial {trial.number} - Subset {i+1}/{len(subsets)} 학습 중..."
            else:
                msg = f"  🏋️  Subset {i+1}/{len(subsets)} 학습 중..."
            print(f"\r{msg.ljust(60)}", end="", flush=True)
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
                    print(f"\r  🚫  [Pruned] Trial {trial.number} pruned at step {i} (score: {cur_score:.5f})".ljust(60))
                    raise optuna.TrialPruned()

        if is_optuna:
            # Optuna의 기본 로그가 깔끔하게 출력되도록 임시 진행 상태줄을 비움
            print("\r" + " " * 60 + "\r", end="", flush=True)
        else:
            print(f"\r✅  {len(subsets)}개 모델 학습 및 평가 완료.".ljust(60))

        probs = np.mean(probs_list, axis=0)
        ensemble_prauc = average_precision_score(y_val, probs)

        self._result = EnsembleResult(
            subset_results=subset_results,
            val_tune_prauc=ensemble_prauc,
            val_tune_probs=probs,
            val_tune_y_true=y_val.values if hasattr(y_val, "values") else y_val,
        )

        if not is_optuna:
            print(f"\n✨  앙상블 최종 VAL_TUNE PR-AUC = {ensemble_prauc:.5f} (참고용 - 최종 Test Metric 아님)")
        return self._result

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """학습 완료 후 새 데이터에 대한 soft-voting 확률 반환."""
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        X = df[feature_cols].astype(np.float32)
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
    save_model_dir: str = None,
):
    if bounds is None:
        raise ValueError("❌ 'bounds' (탐색 범위)가 지정되지 않았습니다. cfg.OPTUNA_BOUNDS를 확인하세요.")

    def objective(trial):
        max_depth   = trial.suggest_int("max_depth", *bounds["max_depth"])
        max_leaves  = min(2 ** max_depth, bounds["num_leaves"][1])
        min_leaves  = min(bounds["num_leaves"][0], max_leaves)
        num_leaves  = trial.suggest_int("num_leaves", min_leaves, max_leaves)
        
        n_estimators = trial.suggest_int("n_estimators", *bounds["n_estimators"])
        scale_pos_weight = trial.suggest_float("scale_pos_weight", *bounds["scale_pos_weight"]) if "scale_pos_weight" in bounds else 1.0

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
            "scale_pos_weight":  scale_pos_weight,
            "verbosity":         -1,
            "device":            device,
            "random_state":      42,
            "max_bin":           63,
        }

        trainer = SubsetTrainer(lgbm_params=params, target_col=target_col)
        ens     = UnderbaggingEnsemble(trainer=trainer)

        result  = ens.fit(df_train, df_val_tune, feature_cols=feature_cols, target_col=target_col, trial=trial)
        
        if save_model_dir is not None:
            import joblib
            import uuid
            trial_id_str = f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"
            trial_dir = Path(save_model_dir) / trial_id_str
            trial_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(result.models):
                joblib.dump(model, trial_dir / f"model_{i}.pkl")
            trial.set_user_attr("model_dir", trial_id_str)
                
        return result.val_tune_prauc

    return objective




# ════════════════════════════════════════════════════════════
#  5. EVALUATION UTILS & MAIN RUNNER
# ════════════════════════════════════════════════════════════

def evaluate_saved_models(
    models: list[LGBMClassifier],
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "failure"
) -> EnsembleResult:
    """로드된 모델들을 이용해 데이터셋을 평가하고 EnsembleResult 객체를 반환합니다."""
    X = df[feature_cols].astype(np.float32)
    y = df[target_col]
    
    subset_results = []
    probs_list = []
    
    for i, model in enumerate(models):
        p = model.predict_proba(X)[:, 1]
        probs_list.append(p)
        val_prauc = average_precision_score(y, p)
        
        res = SubsetResult(
            subset_id=i,
            model=model,
            val_prauc=val_prauc,
            n_train_pos=-1,  # 로드된 모델이므로 정보 없음
            n_train_neg=-1
        )
        subset_results.append(res)
        
    ensemble_probs = np.mean(probs_list, axis=0)
    ensemble_prauc = average_precision_score(y, ensemble_probs)
    
    ens_result = EnsembleResult(
        subset_results=subset_results,
        val_tune_prauc=ensemble_prauc,
        val_tune_probs=ensemble_probs,
        val_tune_y_true=y.values if hasattr(y, 'values') else y,
    )
    
    return ens_result

def run_training(
    cfg,
    feature_cols: Optional[list[str]] = None,
    *,
    run_optuna: bool = False,
    optuna_trials: Optional[int] = None,
    optuna_timeout: Optional[int] = None,
    optuna_rerank_delta: float = 0.005,
    optuna_rerank_cap: int = 5,
    optuna_rerank_ids: Optional[list[int]] = None,
    interactive_rerank: bool = False,
    show_plots: bool = True,
    cleanup_optuna_temp: bool = True,
) -> dict:
    from config.path_utils import validate_path_contract

    required_paths = list(getattr(cfg, "REQUIRED_DATA_PATHS", []))
    if run_optuna:
        required_paths.append(("sampled val_tune data", "file", cfg.VAL_TUNE_SAMPLED_PATH))
    validate_path_contract(required_paths)

    # [데이터 로드]
    subset_dir = Path(getattr(cfg, "SUBSET_DIR", ""))
    subset_files = sorted(list(subset_dir.glob("subset_*.parquet"))) if subset_dir.exists() else []
    if not subset_files:
        raise FileNotFoundError(f"❌ [Error] 사전 분할 데이터가 {subset_dir} 에 없습니다.")

    df_train = [pd.read_parquet(f) for f in subset_files]
    val_tune_path = getattr(cfg, "VAL_TUNE_PATH", None)
    if not val_tune_path or not Path(val_tune_path).exists():
        raise FileNotFoundError(f"❌ [Error] 원본 검증셋(VAL_TUNE_PATH)이 없습니다. 리랭크 및 최종 검증에 필수적입니다.")
    df_val_tune_full = pd.read_parquet(val_tune_path)

    print(f"  [Debug] Train Subsets: {len(df_train)} files")
    print(f"  [Debug] Val Tune (Full) Rows: {len(df_val_tune_full):,}")
    print(f"\n✅  [Ensemble] 사전 분할된 {len(df_train)}개 서브셋 학습 준비 완료.")
    # [검증 데이터 로드 (Optuna vs Single Run)]
    sampled_path = getattr(cfg, "VAL_TUNE_SAMPLED_PATH", None)
    
    if run_optuna:
        # Optuna 튜닝 모드: 분할(샘플링)된 검증셋 필수
        if not sampled_path or not Path(sampled_path).exists():
            raise FileNotFoundError("❌ [Error] Optuna 튜닝 모드(run_optuna=True)에서는 샘플링된 검증셋(VAL_TUNE_SAMPLED_PATH)이 필수입니다.")
        df_val_optuna = pd.read_parquet(sampled_path)
        print(f"  [Debug] Val Tune (Sampled for Optuna) loaded: {len(df_val_optuna):,} rows")
        is_val_sampled = True
    else:
        # 단일 평가 모드 (Single Run): 노트북 설정에 따라 선택적 로드
        if sampled_path and Path(sampled_path).exists():
            df_val_optuna = pd.read_parquet(sampled_path)
            print(f"  [Debug] Val Tune (Sampled for Single Run) loaded: {len(df_val_optuna):,} rows")
            is_val_sampled = True
        else:
            df_val_optuna = df_val_tune_full
            print(f"  [Debug] Val Tune (Full for Single Run) Rows: {len(df_val_optuna):,}")
            is_val_sampled = False

    feats = feature_cols or getattr(cfg, "FEATURE_COLS", None)
    _device = cfg.LGBM_PARAMS.get("device", "cpu")

    # [Optuna 튜닝]
    best_params = cfg.LGBM_PARAMS.copy()
    if run_optuna:
        db_path = getattr(cfg, "OPTUNA_DB_PATH", "optuna_study.db")
        base_study_name = getattr(cfg, "OPTUNA_STUDY_NAME", "hdd_failure_prediction")
        storage_url = f"sqlite:///{db_path}"

        # ─── Single Stage Optuna Tuning ───
        trials = optuna_trials if optuna_trials is not None else getattr(cfg, "OPTUNA_TRIALS", 300)
        print(f"\n  [Optuna Tuning] 하이퍼파라미터 탐색 시작 (설정된 목표 횟수: {trials}회, 가지치기 활성화)")
        study_name = base_study_name
        
        optuna_temp_dir = Path(cfg.MODEL_SAVE_DIR).parent / "optuna_temp"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3), # 가지치기 활성화
            load_if_exists=True,
        )
        
        total_trials = len(study.trials)
        if total_trials < trials:
            obj = make_optuna_objective(
                df_train, df_val_optuna, feats, cfg.TARGET_COL, _device,
                bounds=cfg.OPTUNA_BOUNDS, save_model_dir=str(optuna_temp_dir)
            )
            study.optimize(obj, n_trials=trials - total_trials, timeout=optuna_timeout)
        
        valid_trials = [t for t in study.trials if t.number < trials and t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials:
            raise ValueError(f"완료(COMPLETE)된 트라이얼이 없습니다. (목표 횟수: {trials})")
            
        best_trial = max(valid_trials, key=lambda t: t.value)
        print(f"  [Optuna Tuning] 완료. Best PR-AUC (처음 {trials}회 기준): {best_trial.value:.5f}")

        # ─── Reranking ───
        if is_val_sampled and (optuna_rerank_delta > 0 or optuna_rerank_ids is not None or interactive_rerank):
            print(f"\n  [Rerank] Reranking 시작 (Full Validation, 대상: Tuning 완료 모델)")
            
            # 설정된 trials 횟수를 넘기지 않은 유효 트라이얼들만 리랭킹 대상으로 삼음
            completed_trials = valid_trials
            
            if completed_trials:
                # 1. Trial 점수 표 출력
                df_trials = pd.DataFrame([
                    {"Trial": t.number, "Sampled PR-AUC": t.value}
                    for t in completed_trials
                ]).sort_values("Sampled PR-AUC", ascending=False)
                
                print("\n📊 [Optuna Trial 결과 요약]")
                print(df_trials.to_markdown(index=False))
                
                # 2. Interactive Input
                if interactive_rerank:
                    valid_ids = {t.number for t in completed_trials}
                    while True:
                        user_input = input("\n📝 리랭크할 Trial 번호를 쉼표(,)로 구분하여 입력하세요 (예: 0,2 / Enter 입력 시 자동 선택): ")
                        if not user_input.strip():
                            break  # 비워두면 기존 Margin 기반 자동 탐색으로 넘어감
                        
                        try:
                            parsed_ids = [int(x.strip()) for x in user_input.split(",")]
                            
                            # 목록에 없는 번호를 썼는지 검증
                            invalid_ids = [x for x in parsed_ids if x not in valid_ids]
                            if invalid_ids:
                                print(f"  ⚠️ 에러: 트라이얼 번호 {invalid_ids}는 위 목록에 존재하지 않습니다. 다시 입력해주세요.")
                                continue
                                
                            optuna_rerank_ids = parsed_ids
                            break
                        except ValueError:
                            print("  ⚠️ 에러: 숫자와 쉼표(,)만 입력하셔야 합니다. 다시 입력해주세요.")
                        
                if optuna_rerank_ids is not None:
                    # 사용자 지정 트라이얼 번호로 필터링
                    candidates = [t for t in completed_trials if t.number in optuna_rerank_ids]
                    print(f"  [Rerank] 지정된 트라이얼 리랭킹: {len(candidates)}개 (IDs: {optuna_rerank_ids})")
                else:
                    # Margin 기반 자동 필터링
                    best_val = max(t.value for t in completed_trials)
                    candidates = [t for t in completed_trials if t.value >= best_val - optuna_rerank_delta]
                    candidates = sorted(candidates, key=lambda t: t.value, reverse=True)[:optuna_rerank_cap]
                    print(f"  [Rerank] 자동 후보 선택: {len(candidates)}개 (best: {best_val:.5f}, delta: {optuna_rerank_delta})")
                
                best_rerank_score = -1.0
                best_rerank_params = None
                
                import joblib
                
                for trial_obj in candidates:
                    trial_number = trial_obj.number
                    cand_params = trial_obj.params
                    
                    merged_params = cfg.LGBM_PARAMS.copy()
                    merged_params.update(cand_params)
                    
                    model_dir_name = trial_obj.user_attrs.get("model_dir")
                    if model_dir_name:
                        trial_dir = optuna_temp_dir / model_dir_name
                    else:
                        trial_dir = optuna_temp_dir / f"trial_{trial_number}"
                    
                    if trial_dir.exists():
                        # 파일 존재 여부 및 무결성 검증 방어 로직
                        all_exist = all((trial_dir / f"model_{i}.pkl").exists() for i in range(len(df_train)))
                        
                        if not all_exist:
                            print(f"    - Trial {trial_number} (Skipped): 일부 모델 파일 누락 (Corrupted)")
                            continue
                            
                        try:
                            # 안내 메시지 추가 (침묵 방지)
                            print(f"    - Trial {trial_number} 검증 중 (전체 데이터 추론)...", end="\r")
                            # 저장된 모델 Reuse (Inference Only)
                            models = []
                            for i in range(len(df_train)):
                                model_path = trial_dir / f"model_{i}.pkl"
                                models.append(joblib.load(model_path))
                            
                            X_val = df_val_tune_full[feats]
                            y_val = df_val_tune_full[cfg.TARGET_COL]
                            probs = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
                            score = average_precision_score(y_val, probs)
                            print(f"    - Trial {trial_number} (Reuse): Sampled PR-AUC = {trial_obj.value:.5f} -> Full PR-AUC = {score:.5f}")
                            
                            if score > best_rerank_score:
                                best_rerank_score = score
                                best_rerank_params = merged_params
                                
                        except Exception as e:
                            print(f"    - Trial {trial_number} (Skipped): 모델 로드 중 에러 발생 ({type(e).__name__})")
                            continue
                    else:
                        # Stochastic inconsistency 방지를 위해 모델 파일이 없는 trial은 안전하게 건너뜀 (Retrain 배제)
                        print(f"    - Trial {trial_number} (Skipped): 저장된 모델 없음 (재학습 시 Stochastic 편향 방지)")
                
                if best_rerank_params is not None:
                    best_params = best_rerank_params
                    print(f"  [Rerank] 최종 선택된 Full PR-AUC: {best_rerank_score:.5f}")
            elif best_trial:
                best_params.update(best_trial.params)
        elif best_trial:
            best_params.update(best_trial.params)

    # [최종 학습] (전체 검증셋 사용)
    print("\n  [Final] 최적 파라미터로 최종 앙상블 학습 (Val-Tune 기반, 최종 성능 평가용 아님)")
    print("          ⚠️ 현재 출력되는 점수는 Parameter Selection 과정에 반영된 결과(Biased)이므로,")
    print("          ⚠️ unbiased metric 확보를 위해서는 반드시 분리된 Test 셋으로 최종 리포트를 작성해야 합니다.")
    trainer = SubsetTrainer(lgbm_params=best_params, target_col=cfg.TARGET_COL)
    ens     = UnderbaggingEnsemble(trainer=trainer)
    result = ens.fit(df_train, df_val_tune_full, feature_cols=feats, target_col=cfg.TARGET_COL)

    # 임시 모델 폴더 정리
    if run_optuna and cleanup_optuna_temp:
        optuna_temp_dir = Path(cfg.MODEL_SAVE_DIR).parent / "optuna_temp"
        if optuna_temp_dir.exists():
            import shutil
            shutil.rmtree(optuna_temp_dir, ignore_errors=True)
            print(f"\n  [Cleanup] {optuna_temp_dir} 삭제 완료.")

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

def plot_shap_importance(
    models: list[LGBMClassifier],
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = 1000,
    seed: int = 42,
    max_display: int = 30
):
    """저장된 앙상블 모델들에 대해 SHAP 중요도를 계산하고 Summary Plot을 시각화합니다."""
    import shap
    import warnings
    warnings.filterwarnings('ignore')
    
    X_val_shap = df[feature_cols].copy()
    sample_size = min(sample_size, len(X_val_shap))
    X_sample = X_val_shap.sample(sample_size, random_state=seed).reset_index(drop=True)
    
    print(f'SHAP 계산 중... (샘플={sample_size}, 모델={len(models)}개)')
    
    sv_list = []
    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list): sv = sv[1]
        sv_list.append(sv)
        print(f'  모델 {i+1}/{len(models)} 완료', end='\r')
        
    mean_shap = np.mean(sv_list, axis=0)
    print('\nSHAP 계산 완료!')
    
    shap.summary_plot(mean_shap, X_sample, feature_names=feature_cols, max_display=max_display, show=True)
