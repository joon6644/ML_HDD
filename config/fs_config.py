# ============================================================
#  fs_config.py  ─  RFE 실험 설정 (여기만 수정)
# ============================================================
#
#  MODE 설명
#  ---------
#  "add"   : 빈 feature set에서 출발 → INCLUDE_GROUPS / INCLUDE_COLS 에 명시한 것만 사용
#  "drop"  : feature_group.json 전체에서 출발 → EXCLUDE_GROUPS / EXCLUDE_COLS 에 명시한 것을 제거
#
# ============================================================

import os
import numpy as np
import lightgbm as lgb
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

# ─── 경로 ────────────────────────────────────────────────────
BASE_DIR        = PROJECT_ROOT
TRAIN_PATH      = data_path("05_feature_selection/fs_train.parquet")
TEST_PATH       = data_path("05_feature_selection/fs_validation.parquet")
FEATURE_GROUP_PATH = str(BASE_DIR / "config" / "json" / "feature_groups.json")

REQUIRED_DATA_PATHS = [
    ("feature-selection train data", "file", TRAIN_PATH),
    ("feature-selection validation data", "file", TEST_PATH),
    ("feature group definition", "file", FEATURE_GROUP_PATH),
]


# ─── 실험 모드 ───────────────────────────────────────────────
#  "add"  or  "drop"
MODE = "drop"

# ─── [add 모드] 포함할 그룹 / 개별 컬럼 ──────────────────────
#  MODE == "add" 일 때만 유효
#  비워두면 아무 feature도 없음 → 반드시 하나 이상 지정해야 경고 없이 실행됨
INCLUDE_GROUPS: list[str] = [

]

INCLUDE_COLS: list[str] = [
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




# ----- 원본 피처 -----
#     "smart_5_raw", 
# "smart_184_raw", 
# "smart_187_raw", 
# "smart_197_raw", 
# "smart_198_raw", 
# "timeout_5s", 
# "timeout_total", 
# "seek_error_count", 
# "smart_9_raw", 
# "smart_183_raw", 
# "smart_189_raw", 
# "smart_190_raw", 
# "smart_191_raw", 
# "smart_194_raw", 
# "smart_199_raw", 
# "smart_241_raw", 
# "smart_242_raw", 
# "total_reads", 
# "total_seeks",

]

# ─── 제외할 메타 컬럼 (항상 무시됨) ──────────────────────
META_COLS: list[str] = ["serial_number", "date"]

# ─── [drop 모드] 제외할 그룹 / 개별 컬럼 ─────────────────────
#  MODE == "drop" 일 때만 유효
EXCLUDE_GROUPS: list[str] = [
    # ── 부하 / 사용량 ───────────────────────────────────────
    # "최근 부하 압력",   
    # "총 탐색량",      
    # "총 읽기량",       
    # "총 기록량",         
    # "누적 사용량",        
    # ── 에러 / 이상 ─────────────────────────────────────────
    # "Burst 이상",          
    # "Reallocated / Pending", 
    # "급성 Spike",         
    # "Sector 열화",         
    # "읽기/쓰기 안정성",  
    # "Seek 경로 이상",      
    # "기본 I/O 이상",     
    # ── 물리 / 열 ───────────────────────────────────────────
    # "물리 스트레스 상호작용",  
    # "기계적 충격",       
    # "열 스트레스",        
    # "온도 수준",          
    # ── 시스템 / 열화 ────────────────────────────────────────
    # "시스템성 실패",     
    # "펌웨어 실패",       
    # "지속 악화",       
    # ── 발생 시점 / 이력 ─────────────────────────────────────
    # "최근 발생 시점",     
    # "최초 발생 시점",   
    # "과거 누적 발생",     
    # "직접 손상 발생",  
]

EXCLUDE_COLS: list[str] = [
    # 그룹 제거 이후에도 개별로 추가 제거하고 싶은 컬럼


# "s194_28d_mean",


]

# ─── 타깃 컬럼 ───────────────────────────────────────────────
TARGET_COL = "failure"

# ─── SEED ───────────────────────────────────────────────────
SEED = 42

# ─── LightGBM 하이퍼파라미터 ──────────────
_LGBM_DEVICE = get_lgbm_device()

LGBM_PARAMS = {
    "objective":              "binary",
    "importance_type":        "gain",
    "class_weight":           "balanced",
    "max_depth":              6,
    "num_leaves":             40,
    "n_estimators":           200,
    "learning_rate":          0.07,
    "device":                 _LGBM_DEVICE,
    "random_state":           SEED,
    "bagging_seed":           SEED,
    "feature_fraction_seed":  SEED,
    "data_random_seed":       SEED,
    "n_jobs":                 -1,
    "verbose":                -1,
}

# ─── Cross-Validation ────────────────────────────────────────
CV_N_SPLITS  = 5
CV_SHUFFLE   = True
CV_SEED      = SEED

# ─── SHAP 샘플 수 (속도 vs 정밀도) ──────────────────────────
SHAP_SAMPLE_N = 5000           # None 이면 전체 사용 (느림)
SHAP_TOP_N    = 30            # 중요도 시각화 상위 N개

# ─── 중요도 시각화 상위 N개 (Gain) ──────────────────────────
GAIN_TOP_N = 30
