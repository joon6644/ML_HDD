"""
eval_config.py  ─  임계값 튜닝 + 모델 평가 설정  (README §7, §8)
"""
from pathlib import Path

_BASE = Path(__file__).parent.parent

# ── 입력 경로 ────────────────────────────────────────────────
VAL_CALIB_PATH = str(_BASE / "data" / "03_data_splitting" / "val_calib_raw.parquet")
TEST_PATH      = str(_BASE / "data" / "03_data_splitting" / "test_raw.parquet")

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

# ── §7 임계값 튜닝 설정 ──────────────────────────────────────
MAX_FPR          = 0.01    # FPR 상한 (1%)
THRESHOLD_N_GRID = 1000    # 그리드서치 격자 수
