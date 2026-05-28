"""
threshold_config.py  ─  임계값 튜닝 설정  (README §7)
──────────────────────────────────────────────────────────
07_threshold_tuning.ipynb 전용. 설정값은 여기서만 관리합니다.
"""

from config.path_utils import PROJECT_ROOT, fe_data_path

_BASE = PROJECT_ROOT

# ── 학습 산출물 (06c에서 저장된 앙상블) ─────────────────────
MODEL_SAVE_DIR = str(_BASE / "models2" / "06d_optuna_tuning" / "underbagging_ensemble_4")
N_SUBSETS = 10  # subset_*.pkl 개수 (06c SAMPLER_KWARGS.n_subsets 와 일치)

# ── 입력 경로 ────────────────────────────────────────────────
VAL_CALIB_PATH = fe_data_path("val_calib.parquet")

REQUIRED_DATA_PATHS = [
    ("model save directory", "dir", MODEL_SAVE_DIR),
    ("val_calib feature data", "file", VAL_CALIB_PATH),
]

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

# ── §7 임계값 튜닝 설정 ──────────────────────────────────────
# README §7: FPR 상한별 Recall 표를 먼저 보고, 그다음 FPR–Recall 곡선을 그림
FPR_LEVELS = [0.001, 0.005, 0.01, 0.05]  # 간판 표 (각 FPR% 이하에서 Recall 최대)

# 08_final_evaluation 저장용: 표에서 이 FPR 상한 행의 threshold 사용. None이면 JSON에 threshold 미저장
SAVE_OPERATING_FPR_CAP = 0.005

THRESHOLD_N_GRID = 1000    # 그리드서치 격자 수
ALARM_WINDOW = 14          # 알람 탐지 슬라이딩 윈도우 크기 (일 단위)

# ── CSV 저장 (None이면 MODEL_SAVE_DIR 에 저장) ───────────────
THRESHOLD_RESULT_DIR = None
FPR_RECALL_TABLE_CSV = "fpr_recall_table.csv"
THRESHOLD_GRID_CSV     = "threshold_grid.csv"

