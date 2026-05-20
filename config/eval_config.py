"""
eval_config.py  ─  임계값 튜닝 + 모델 평가 설정  (README §7, §8)
"""
from config.path_utils import PROJECT_ROOT, data_path

_BASE = PROJECT_ROOT

# ── 입력 경로 ────────────────────────────────────────────────
VAL_CALIB_PATH = data_path("06a_feature_engineering/val_calib.parquet")
TEST_PATH = data_path("06a_feature_engineering/test.parquet")

REQUIRED_DATA_PATHS = [
    ("val_calib feature data", "file", VAL_CALIB_PATH),
    ("test feature data", "file", TEST_PATH),
]

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

# ── §7 임계값 튜닝 설정 ──────────────────────────────────────
MAX_FPR          = 0.01    # FPR 상한 (1%)
THRESHOLD_N_GRID = 1000    # 그리드서치 격자 수
