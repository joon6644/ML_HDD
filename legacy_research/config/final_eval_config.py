"""
final_eval_config.py  ─  최종 모델 평가 설정  (README §8)
──────────────────────────────────────────────────────────
08_final_evaluation.ipynb 전용. 설정값은 여기서만 관리합니다.
"""

from config.path_utils import PROJECT_ROOT, fe_data_path, DATA_ROOT

_BASE = PROJECT_ROOT

# ── 학습 산출물 (06c + 07에서 저장된 앙상블·임계값) ─────────
MODEL_SAVE_DIR = str(_BASE / "notebooks2" / "models2" / "06d_optuna_tuning" / "seed_42")

# 서브셋 개수 동적 탐색 (기본값 5)
_SUBSET_DIR = DATA_ROOT / "06_subset_generation" / "seed_42"
if _SUBSET_DIR.is_dir():
    N_SUBSETS = len(list(_SUBSET_DIR.glob("subset_*.parquet")))
    if N_SUBSETS == 0:
        N_SUBSETS = 5
else:
    N_SUBSETS = 5

# ── 입력 경로 ────────────────────────────────────────────────
# D-day 행이 data2/03_splitting/test.parquet 안에 이미 포함되어 있음
# test_with_failure_date.parquet 는 중복 추가본으로 사용하지 않음
TEST_PATH = str(_BASE / "data2" / "03_splitting" / "test.parquet")

REQUIRED_DATA_PATHS = [
    ("model save directory", "dir", MODEL_SAVE_DIR),
    ("best threshold metadata", "file", f"{MODEL_SAVE_DIR}/best_threshold.json"),
    ("test feature data", "file", TEST_PATH),
]

# ── 최적 임계값 수동 오버라이드 (Optional) ─────────────────────
# 값을 지정하지 않으면 (None), 모델 폴더 내 best_threshold.json 에서 자동으로 불러옵니다.
MANUAL_BEST_T = None
MANUAL_BEST_N = None

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

ALARM_WINDOW = 14  # 알람 탐지 슬라이딩 윈도우 크기 (일 단위)

