"""
final_eval_config.py  ─  최종 모델 평가 설정  (README §8)
──────────────────────────────────────────────────────────
08_final_evaluation.ipynb 전용. 설정값은 여기서만 관리합니다.
"""

from config.path_utils import PROJECT_ROOT, fe_data_path

_BASE = PROJECT_ROOT

# ── 학습 산출물 (06c + 07에서 저장된 앙상블·임계값) ─────────
MODEL_SAVE_DIR = str(_BASE / "models" / "underbagging_ensemble_4")
N_SUBSETS = 10  # subset_*.pkl 개수 (06c SAMPLER_KWARGS.n_subsets 와 일치)

# ── 입력 경로 ────────────────────────────────────────────────
TEST_PATH = fe_data_path("test_with_failure_date.parquet")
# TEST_PATH = fe_data_path("test.parquet")

REQUIRED_DATA_PATHS = [
    ("model save directory", "dir", MODEL_SAVE_DIR),
    ("best threshold metadata", "file", f"{MODEL_SAVE_DIR}/best_threshold.json"),
    ("test feature data", "file", TEST_PATH),
]

# ── 최적 임계값 수동 오버라이드 (Optional) ─────────────────────
# 값을 지정하지 않으면 (None), 모델 폴더 내 best_threshold.json 에서 자동으로 불러옵니다.
MANUAL_BEST_T = 0.9261    
MANUAL_BEST_N = 2

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

ALARM_WINDOW = 14  # 알람 탐지 슬라이딩 윈도우 크기 (일 단위)

