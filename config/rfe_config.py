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
FEATURE_GROUP_PATH = os.path.join(BASE_DIR, "data", "feature_group.json")


# ─── 실험 모드 ───────────────────────────────────────────────
#  "add"  or  "drop"
MODE = "drop"

# ─── [add 모드] 포함할 그룹 / 개별 컬럼 ──────────────────────
#  MODE == "add" 일 때만 유효
#  비워두면 아무 feature도 없음 → 반드시 하나 이상 지정해야 경고 없이 실행됨
INCLUDE_GROUPS: list[str] = [
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

INCLUDE_COLS: list[str] = [

# "workload_intensity",
# "age_weighted_workload",
# "is_warmup_14d",
# "workload_7d_accel",
# "total_seeks_28d_asfd",
# "s242_28d_asfd",
# "s242_14d_mean",
# "s242_7d_dai",
# "total_reads_28d_asfd",
# "smart_241_raw",
# "s241_28d_asfd",
# # "s187_damaged",
# # "s197_damaged",
# "write_stability_ratio",
# "smart_183_raw",
# "s194_28d_mean",
# "s194_28d_dai",
# "thermal_stress_index",
# "zero_to_hero_count",
# "s184_14d_max",
# "shock_to_highfly_ratio",
# "smart_9_raw",
# "io_asymmetry_index",
# "timeout_total_28d_max",
# "timeout_read_density",
# "s199_error_density",
# "s191_days_since_last",
# "write_spike_ratio",
# "uncorrectable_spike_ratio",
# "timeout_seek_density",
# "s197_recovery_flag",
# # "smart_197_raw",
# # "s198_28d_max",
# # "error_density_14d",
# # "reallocated_pending_ratio",
# "s5_28d_max",
# "s5_relative_score_14d",
# "s183_28d_max"
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
    "s197_damaged",
    "smart_197_raw",
    "s198_damaged",
    "smart_198_raw",
    "s187_damaged",
    "smart_187_raw",
    "s187_days_since_first",
    "s187_ever_flag",
    "error_density_14d",
    "s198_28d_max",
    "reallocated_pending_ratio",
    "total_seeks_28d_asfd",
    "s187_28d_sum",
    "s187_28d_max",
    "smart_183_raw",
    "total_seeks_28d_max",
    "s187_14d_sum",
    "s187_14d_burst_index",
    "s187_14d_max",
    "cumulative_error_score",
    "s5_damaged",
    "s5_relative_score_14d",
    "s5_days_since_first",
    "s5_ever_flag",
    "s5_28d_max",
    "multi_error_count",
    "error_growth_ratio",
    "smart_9_raw",
    "write_stability_ratio",
    "smart_241_raw",
    "age_weighted_workload",
    "workload_intensity",
    "smart_242_raw",
    "s187_diff",
    "s183_28d_max",
    "s198_diff",
    "uncorrectable_spike_ratio",
    "s183_14d_max",
    "late_stage_degradation",
    "",
    "",



]

# ─── 타깃 컬럼 ───────────────────────────────────────────────
TARGET_COL = "failure"

# ─── LightGBM 하이퍼파라미터 (기존 그대로 유지) ──────────────
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
CV_N_SPLITS  = 5
CV_SHUFFLE   = True
CV_SEED      = 42

# ─── SHAP 샘플 수 (속도 vs 정밀도) ──────────────────────────
SHAP_SAMPLE_N = 200           # None 이면 전체 사용 (느림)
SHAP_TOP_N    = 30            # 중요도 시각화 상위 N개

# ─── 중요도 시각화 상위 N개 (Gain) ──────────────────────────
GAIN_TOP_N = 30
