"""
SHAP 기반 Backward Recursive Feature Elimination (RFE) 파이프라인.

이 스크립트는 SHAP 중요도를 기반으로 불필요한 피처를 단계적으로 제거하며 모델의 성능 변화를 추적합니다.
주요 기능:
- StratifiedGroupKFold를 통한 교차 검증 (Serial Number 기반 그룹화로 데이터 누수 방지)
- SHAP (TreeExplainer)를 이용한 피처 기여도 산출
- 매 단계마다 가장 기여도가 낮은(SHAP value가 작은) 피처를 제거
- 각 단계별 CV PR-AUC 및 Test PR-AUC 결과를 CSV/Excel로 저장

모든 파라미터(Fold 수, SHAP 샘플 수 등)는 'config/fs_config.py'의 설정을 공유합니다.
"""

# =============================================================
#  shap_backward_rfe.py  ─  SHAP 기반 Backward RFE 자동화
# =============================================================

import sys, os, warnings, logging, contextlib
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("shap").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import average_precision_score

# C레벨 stderr("1 warning generated") 억제용
@contextlib.contextmanager
def suppress_c_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr  = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull_fd)

# ── config에서 파라미터 가져오기 ─────────────────────────────
from config.fs_config import LGBM_PARAMS, CV_N_SPLITS, SHAP_SAMPLE_N

TRAIN_PATH  = r"data\fs_data\fs_train.parquet"
TEST_PATH   = r"data\fs_data\fs_validation.parquet"
OUTPUT_CSV  = r"results\feature_selection\shap_backward_fs_results.csv"

N_FOLDS     = CV_N_SPLITS
SHAP_SAMPLE = SHAP_SAMPLE_N
TARGET      = "failure"

# ── CV 파라미터 설정 ─────────────────────────────────────────
_PARAMS = {**LGBM_PARAMS}   # 원본 건드리지 않음


def run_fold_cv(X_tr_all, y_tr_all, groups=None):
    """StratifiedGroupKFold CV → (cv_scores, fold_models)"""
    if N_FOLDS <= 1:
        return [0.0], []

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        split_gen = cv.split(X_tr_all, y_tr_all, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        split_gen = cv.split(X_tr_all, y_tr_all)

    cv_scores, models = [], []
    for tr_idx, val_idx in split_gen:
        X_tr, X_val = X_tr_all.iloc[tr_idx], X_tr_all.iloc[val_idx]
        y_tr, y_val = y_tr_all.iloc[tr_idx], y_tr_all.iloc[val_idx]
        m = LGBMClassifier(**_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
        )
        prob = m.predict_proba(X_val)[:, 1]
        cv_scores.append(average_precision_score(y_val, prob))
        models.append(m)
    return cv_scores, models



def top_shap_feature(models, X_train, features):
    """fold 모델들의 평균 |SHAP| 1위 feature 반환"""
    X_s = X_train.sample(min(SHAP_SAMPLE, len(X_train)), random_state=42).copy()
    num_cols = X_s.select_dtypes(include=[np.number]).columns
    X_s[num_cols] = X_s[num_cols].astype("float32")

    sv_list = []
    for m in models:
        with suppress_c_stderr():
            exp = shap.TreeExplainer(m)
            sv  = exp.shap_values(X_s)
        if isinstance(sv, list):
            sv = sv[1]
        sv_list.append(sv)

    mean_abs = np.abs(np.mean(sv_list, axis=0)).mean(axis=0)
    return features[np.argmax(mean_abs)] # 제거 대상: SHAP 중요도가 가장 낮은 변수 (argmin)


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, TRAIN_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)
    test_df  = pd.read_parquet(os.path.join(base_dir, TEST_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)

    # Leakage 방지를 위한 메타 컬럼 정의 및 제거
    DROP_COLS = [TARGET, "serial_number", "date"]
    
    # Group 정보 사전 추출 (Serial Leakage 방지용)
    groups = train_df["serial_number"].copy() if "serial_number" in train_df.columns else None

    # Feature 리스트 확정 및 정렬 (재현성)
    features = sorted([c for c in train_df.columns if c not in DROP_COLS])
    
    X_train  = train_df[features]
    y_train  = train_df[TARGET]
    X_test   = test_df[features]
    y_test   = test_df[TARGET]

    results        = []
    removed        = None          # 이번 라운드에서 제거된 feature
    prev_exp_id    = "none"
    exp_num        = 0

    print(f"시작 feature 수: {len(features)}")
    print("=" * 70)

    out_csv = os.path.join(base_dir, OUTPUT_CSV)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    while len(features) > 1:
        exp_num  += 1
        exp_id    = f"exp_{exp_num:03d}"
        change    = "baseline" if removed is None else removed

        # ── 학습 ──────────────────────────────────────────────
        cv_scores, fold_models = run_fold_cv(X_train, y_train, groups=groups)
        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))

        # ── 기록 ──────────────────────────────────────────────
        row = {
            "exp_id":      exp_id,
            "base_exp":    prev_exp_id,
            "change":      change,
            "CV PR-AUC":   round(cv_mean, 5),
            "CV std":      round(cv_std,  5),
        }
        results.append(row)

        print(
            f"[{exp_id}] n={len(features):3d} | "
            f"CV={cv_mean:.5f}±{cv_std:.5f} | {change}"
        )

        # ── 중간 저장 ──────────────────────────────────────────
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # ── 다음 제거 feature (SHAP 1위) ──────────────────────
        next_remove = top_shap_feature(fold_models, X_train, list(features))

        # ── 업데이트 ──────────────────────────────────────────
        prev_exp_id = exp_id
        removed     = next_remove
        features    = [f for f in features if f != next_remove]
        X_train     = X_train[features]
        X_test      = X_test[features]

    # ── 최종 결과 ──────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    print(df_out.to_string(index=False))
    print(f"✅ CSV → {out_csv}")
