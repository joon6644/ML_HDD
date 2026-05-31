from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks2"
DATA_ROOT = PROJECT_ROOT / "data2"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Path contract fragments used by the feature-engineering notebooks.
FE_SPLIT_REL = "03_splitting"

TRAIN_PATH = DATA_ROOT / "06_hyperparameter_tuning" / "train.parquet"
VAL_TUNE_PATH = DATA_ROOT / "06_hyperparameter_tuning" / "val_tune.parquet"
VAL_CALIB_PATH = DATA_ROOT / "06_hyperparameter_tuning" / "val_calib.parquet"
TEST_PATH = DATA_ROOT / "06_hyperparameter_tuning" / "test.parquet"

MODEL_SAVE_DIR = RESULTS_ROOT / "models" / "seed_42"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions"
OPTUNA_DB_PATH = RESULTS_ROOT / "databases" / "optuna_study.db"
METRICS_DIR = RESULTS_ROOT / "metrics"
LOGS_DIR = RESULTS_ROOT / "logs"

TARGET_COL = "failure"
SERIAL_COL = "serial_number"
DATE_COL = "date"

ALARM_WINDOW = 14
EVAL_HORIZON_DAYS = 30
ROLLING_WINDOW_SIZE = 0

SHAP_SAMPLE_SIZE = 6000
SHAP_MAX_DISPLAY = 27
SHAP_SEED = 42
TEMPORAL_WINDOW_DAYS = 30
TEMPORAL_TOP_N_FEATS = 5
TARGET_SERIAL = None

THRESHOLD_GRID = [round(x / 100, 2) for x in range(1, 100)]


def validate_path_contract(*_args, **_kwargs) -> bool:
    """Compatibility hook for old notebooks; local paths are explicit above."""
    return True


def fe_data_path(filename: str) -> Path:
    return DATA_ROOT / FE_SPLIT_REL / filename
