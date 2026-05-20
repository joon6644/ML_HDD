"""
train_config.py  ─  모델 학습 파이프라인 설정  (README §6)
──────────────────────────────────────────────────────────
설정값을 여기서만 관리하고, train_core.py 에 cfg 로 넘김.
"""

import numpy as np
import lightgbm as lgb
import os
from pathlib import Path
from config.path_utils import PROJECT_ROOT, data_path

def check_gpu() -> bool:
    """LightGBM GPU 사용 가능 여부 확인."""
    try:
        lgb.train(
            {"device": "gpu", "verbose": -1},
            lgb.Dataset(np.zeros((1, 1)), label=[0]),
            num_boost_round=1,
        )
        return True
    except Exception:
        return False

def get_lgbm_device() -> str:
    """Select a portable LightGBM device.

    Set ML_HDD_LGBM_DEVICE=gpu on machines where the GPU build is known to work.
    Set ML_HDD_LGBM_DEVICE=auto to run the import-time GPU probe.
    """
    device = os.environ.get("ML_HDD_LGBM_DEVICE", "cpu").lower()
    if device == "auto":
        return "gpu" if check_gpu() else "cpu"
    if device not in {"cpu", "gpu"}:
        raise ValueError("ML_HDD_LGBM_DEVICE must be one of: cpu, gpu, auto")
    return device

_LGBM_DEVICE = get_lgbm_device()

# ════════════════════════════════════════════════════════════
#  기본 설정 (SEED)
# ════════════════════════════════════════════════════════════

SEED       = 42
TARGET_COL = "failure"

# ════════════════════════════════════════════════════════════
#  경로 설정
# ════════════════════════════════════════════════════════════

_BASE = PROJECT_ROOT

# ── 입력 데이터 ──────────────────────────────────────────────
# 1. 학습 데이터 (피처 선택용 샘플을 쓸 경우 fs_sample_data 하위 참조)
TRAIN_PATH = data_path("06a_feature_engineering/train.parquet")
# 2. 검증 및 테스트 데이터 (06a_feature_engineering 하위 참조)
VAL_TUNE_PATH = data_path("06a_feature_engineering/val_tune.parquet")
VAL_CALIB_PATH = data_path("06a_feature_engineering/val_calib.parquet")
TEST_PATH = data_path("06a_feature_engineering/test.parquet")

# 3. 사전 분할된 서브셋 데이터 (SEED에 따라 동적 결정)
SUBSET_DIR = data_path(f"06b_subset_generation/seed_{SEED}")
VAL_TUNE_SAMPLED_PATH = str(Path(SUBSET_DIR) / "val_sampled.parquet")

REQUIRED_DATA_PATHS = [
    ("train feature data", "file", TRAIN_PATH),
    ("val_tune feature data", "file", VAL_TUNE_PATH),
    ("val_calib feature data", "file", VAL_CALIB_PATH),
    ("test feature data", "file", TEST_PATH),
    ("underbagging subset directory", "dir", SUBSET_DIR),
]

# 4. 학습 효율을 위한 검증 데이터 샘플링 (None이면 전체 사용)
# 800만 건 PR-AUC 계산 병목 해결을 위해 100만~200만 권장
VAL_TUNE_SAMPLE_SIZE = 1_000_000

# ── 출력 디렉토리 ────────────────────────────────────────────
MODEL_SAVE_DIR = str(_BASE / "models" / "underbagging_ensemble")


# ════════════════════════════════════════════════════════════
#  사용할 피처 리스트
# ════════════════════════════════════════════════════════════
# None 이면 train_core.run_training() 에서 메타 컬럼 제외 전체 사용
FEATURE_COLS: list[str] | None = [
"s187_days_since_first",
"smart_184_raw",
"error_density_14d",
"smart_187_raw",
"smart_198_raw",
"total_seeks_28d_asfd",
"s187_28d_sum",
"s5_days_since_first",
"s199_days_since_last",
"multi_error_count",
"age_weighted_workload",
"smart_242_raw",
"smart_241_raw",
"smart_9_raw",
"total_seeks_diff",
"s241_diff",
"s194_28d_dai",
"s194_28d_std",
"s242_28d_dai",
"total_reads_7d_asfd",
"total_reads_7d_max",
"s194_14d_max",
"s194_28d_ewma",
"s194_14d_std",
"s190_28d_zscore",
"s190_28d_ewma",
"s190_28d_mean",
]


# ════════════════════════════════════════════════════════════
#  AsymmetricSampler 설정  (README §6.1)
# ════════════════════════════════════════════════════════════

SAMPLER_KWARGS = dict(
    n_subsets   = 10,       # 서브셋 개수 (앙상블 모델 수)
    neg_ratio   = 10,       # 정상:고장 = 10:1
    near_window = 30,       # D-1 ~ D-30 구간을 near-failure 로 정의
    near_weight = 3.0,      # near-failure 정상 행 가중치 배율
    seed        = SEED,
)


# ════════════════════════════════════════════════════════════
#  LightGBM 파라미터  (README §6.2)
# ════════════════════════════════════════════════════════════
# Optuna 튜닝 전 기본값 / 튜닝 후 best_params 로 자동 교체됨 (파일 수정 아닌 메모리 상에서)

LGBM_PARAMS = dict(
    objective         = "binary",
    metric            = "average_precision",
    verbosity         = -1,
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 50,
    min_child_samples = 50,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,     # underbagging 구조에서 추가 탐색 불필요
    lambda_l1         = 0.1,
    lambda_l2         = 0.1,
    scale_pos_weight  = 1.0,
    random_state      = SEED,
    n_jobs            = -1,
    device            = _LGBM_DEVICE,
    # ── [GPU 최적화] ──────────────────────────
    max_bin           = 63,    # GPU 연산 및 메모리 급상승
    gpu_use_dp        = False, # 배정밀도 미사용 (속도 향상)
)


# ════════════════════════════════════════════════════════════
#  Optuna 설정  (README §6.4)
# ════════════════════════════════════════════════════════════

OPTUNA_TRIALS = 300         # 탐색 횟수
OPTUNA_TIMEOUT  = None       # 초 단위 타임아웃. None = n_trials 로만 제한
OPTUNA_DB_PATH  = str(_BASE / "models" / "optuna_study.db")  # SQLite 저장 경로 (절대 경로)
OPTUNA_STUDY_NAME = "hdd_failure_prediction" # Study 이름

# Optuna 탐색 범위
OPTUNA_BOUNDS = {
    "learning_rate": (0.01, 0.1),
    "max_depth": (4, 10),
    "num_leaves": (16, 128),
    "min_child_samples": (20, 100),
    # feature_fraction: 앙상블 다양성을 주되 하한(0.5) 방어
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "lambda_l1": (1e-4, 1.0),
    "lambda_l2": (1e-8, 5.0),
    "n_estimators": (400, 800)
}


# ════════════════════════════════════════════════════════════
#  평가 임계값 설정
# ════════════════════════════════════════════════════════════

# 혼동행렬에 표시할 임계값 목록
EVAL_THRESHOLDS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]

# Classification Report 출력에 사용할 단일 임계값
REPORT_THRESHOLD: float = 0.3
