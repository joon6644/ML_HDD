"""
train_config.py  ─  모델 학습 파이프라인 설정  (README §6)
──────────────────────────────────────────────────────────
설정값을 여기서만 관리하고, train_core.py 에 cfg 로 넘김.
"""

from pathlib import Path

# ════════════════════════════════════════════════════════════
#  경로 설정
# ════════════════════════════════════════════════════════════

_BASE = Path(r"C:\Workspace\06_ML_projdect\26_1_COIN")

# ── 입력 데이터 ──────────────────────────────────────────────
# 1. 학습 데이터 (피처 선택용 샘플을 쓸 경우 fs_sample_data 하위 참조)
TRAIN_PATH    = str(_BASE / "data" / "split_group_stratified" / "train_raw.parquet")
# 2. 검증 및 테스트 데이터 (split_group_stratified 하위 참조)
VAL_TUNE_PATH  = str(_BASE / "data" / "split_group_stratified" / "val_tune_raw.parquet")
VAL_CALIB_PATH = str(_BASE / "data" / "split_group_stratified" / "val_calib_raw.parquet")
TEST_PATH      = str(_BASE / "data" / "split_group_stratified" / "test_raw.parquet")

# ── 출력 디렉토리 ────────────────────────────────────────────
MODEL_SAVE_DIR = str(_BASE / "models" / "underbagging_ensemble")

TARGET_COL = "failure"
SEED       = 42

# ── 사용할 피처 리스트 ────────────────────────────────────────
# None 이면 train_core.run_training() 에서 메타 컬럼 제외 전체 사용
# 특성 선택 완료 후 아래에 직접 채울 것
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
# Optuna 튜닝 전 기본값 / 튜닝 후 best_params 로 자동 교체됨

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
    bagging_freq      = 5,
    lambda_l1         = 0.1,
    lambda_l2         = 0.1,
    # 언더샘플링(10:1) 상태이므로 scale_pos_weight 는 낮게 시작
    scale_pos_weight  = 5.0,
    random_state      = SEED,
    n_jobs            = -1,
)


# ════════════════════════════════════════════════════════════
#  Optuna 설정  (README §6.4)
# ════════════════════════════════════════════════════════════

OPTUNA_N_TRIALS = 50         # 탐색 트라이얼 수
OPTUNA_TIMEOUT  = None       # 초 단위 타임아웃. None = n_trials 로만 제한


# ════════════════════════════════════════════════════════════
#  평가 임계값 설정
# ════════════════════════════════════════════════════════════

# 혼동행렬에 표시할 임계값 목록
EVAL_THRESHOLDS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]

# Classification Report 출력에 사용할 단일 임계값
REPORT_THRESHOLD: float = 0.3
