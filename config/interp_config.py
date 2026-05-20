"""
interp_config.py  ─  모델 해석 설정  (README §9)
──────────────────────────────────────────────────────────
§9.1 전역적 해석 (Global SHAP Summary Plot)
§9.2 국소적 해석 (Local SHAP Waterfall Plot)
§9.3 시간 기반 해석 (Temporal SHAP Trajectory)

설정값을 여기서만 관리하고, 09_model_interpretation.ipynb 에 cfg 로 넘김.
"""

from config.path_utils import PROJECT_ROOT, data_path

_BASE = PROJECT_ROOT

# ── 입력 경로 ────────────────────────────────────────────────
TEST_PATH = data_path("06a_feature_engineering/test.parquet")

REQUIRED_DATA_PATHS = [
    ("test feature data", "file", TEST_PATH),
]

# ── 컬럼명 ───────────────────────────────────────────────────
TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL   = "date"

# ── §9.1 전역적 해석 (Global SHAP) ──────────────────────────
SHAP_SAMPLE_SIZE  = 2000   # SHAP 계산에 사용할 샘플 수 (클수록 정밀하나 느림)
SHAP_MAX_DISPLAY  = 27     # Summary Plot에 표시할 최대 피처 수
SHAP_SEED         = 42     # 재현성 보장 시드

# ── §9.2 국소적 해석 (Local SHAP Waterfall) ──────────────────
# None 이면 노트북 실행 시 자동으로 미탐 개체 목록 중 첫 번째를 선택
TARGET_SERIAL     = None   # 분석할 특정 시리얼 번호 (예: "S300XXXX")

# ── §9.3 시간 기반 해석 (Temporal Trajectory) ────────────────
TEMPORAL_WINDOW_DAYS = 30  # 고장 전 분석할 기간 (일수)
TEMPORAL_TOP_N_FEATS =  5  # Trajectory 추적할 상위 N개 피처
