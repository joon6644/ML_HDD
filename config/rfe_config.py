# ============================================================
#  rfe_config.py  ─  RFE 실험 설정 (여기만 수정)
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

# ─── 경로 ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH      = os.path.join(BASE_DIR, "data", "rfe_sample_data", "rfe_train.parquet")
TEST_PATH       = os.path.join(BASE_DIR, "data", "rfe_sample_data", "rfe_test.parquet")
FEATURE_GROUP_PATH = r"..\config\feature_groups.json"


# ─── 실험 모드 ───────────────────────────────────────────────
#  "add"  or  "drop"
MODE = "add"

# ─── [add 모드] 포함할 그룹 / 개별 컬럼 ──────────────────────
#  MODE == "add" 일 때만 유효
#  비워두면 아무 feature도 없음 → 반드시 하나 이상 지정해야 경고 없이 실행됨
INCLUDE_GROUPS: list[str] = [

]

INCLUDE_COLS: list[str] = [
"smart_189_raw", 
"s190_28d_mean", 
"total_reads_7d_asfd", 
"smart_184_raw", 
"total_reads_28d_std", 
"s194_28d_max", 
"s242_28d_dai", 
"s190_28d_ewma", 
"s190_7d_ewma", 
"write_spike_ratio", 
"s241_14d_mean", 
"s194_7d_std", 
"s194_28d_zscore", 
"total_seeks_diff", 
"cumulative_error_score", 
"s194_28d_ewma", 
"total_reads_7d_max", 
"s5_days_since_first", 
"s190_28d_max", 
"total_reads_28d_asfd", 
"total_reads_28d_max", 
"s241_28d_asfd", 
"s241_diff", 
"smart_183_raw", 
"s187_28d_sum", 
"s190_28d_std", 
"s242_28d_asfd", 
"age_weighted_workload", 
"total_reads_14d_max", 
"s241_7d_max", 
"total_seeks_28d_max", 
"s194_14d_std", 
"s194_14d_mean", 
"total_seeks", 
"s187_days_since_first", 
"total_seeks_14d_mean", 
"s191_days_since_last", 
"s194_28d_mean", 
"s198_28d_max", 
"workload_intensity", 
"s242_14d_mean", 
"smart_241_raw", 
"error_density_14d", 
"smart_242_raw", 
"smart_9_raw", 
"total_seeks_28d_asfd", 
"smart_187_raw", 
"write_stability_ratio", 
"smart_197_raw",
]

# ─── [drop 모드] 제거할 그룹 / 개별 컬럼 ─────────────────────
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
    "timeout_5s_diff",  




]

# ─── 타깃 컬럼 ───────────────────────────────────────────────
TARGET_COL = "failure"

# ─── LightGBM 하이퍼파라미터 ──────────────
_HAS_GPU = check_gpu()

LGBM_PARAMS = {
    "objective":              "binary",
    "importance_type":        "gain",
    "class_weight":           "balanced",
    "max_depth":              6,
    "num_leaves":             40,
    "n_estimators":           200,
    "learning_rate":          0.07,
    "device":                 "gpu" if _HAS_GPU else "cpu",
    "random_state":           42,
    "bagging_seed":           42,
    "feature_fraction_seed":  42,
    "data_random_seed":       42,
    "n_jobs":                 -1,
    "verbose":                -1,
}

# ─── Cross-Validation ────────────────────────────────────────
CV_N_SPLITS  = 3
CV_SHUFFLE   = True
CV_SEED      = 42

# ─── SHAP 샘플 수 (속도 vs 정밀도) ──────────────────────────
SHAP_SAMPLE_N = 200           # None 이면 전체 사용 (느림)
SHAP_TOP_N    = 30            # 중요도 시각화 상위 N개

# ─── 중요도 시각화 상위 N개 (Gain) ──────────────────────────
GAIN_TOP_N = 30
