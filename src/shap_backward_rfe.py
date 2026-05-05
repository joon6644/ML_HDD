# =============================================================
#  shap_backward_rfe.py  ─  SHAP 기반 Backward RFE 자동화
#  - SHAP 중요도 1위 feature를 재귀적으로 제거
#  - 3-fold CV + 전체 train 재학습 test 평가
#  - 변수 1개 남을 때까지 반복
#  - 결과를 CSV / Excel로 저장
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
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
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
from config.fs_config import LGBM_PARAMS

TRAIN_PATH  = r"data\fs_data\fs_train.parquet"
TEST_PATH   = r"data\fs_data\fs_validation.parquet"
OUTPUT_CSV  = r"scripts\shap_backward_fs_results.csv"
OUTPUT_XLSX = r"scripts\shap_backward_fs_results.xlsx"

N_FOLDS     = 3
SHAP_SAMPLE = 200
TARGET      = "failure"

# ── 3-fold 파라미터로 덮어쓰기 ───────────────────────────────
_PARAMS = {**LGBM_PARAMS}   # 원본 건드리지 않음


def run_fold_cv(X_tr_all, y_tr_all):
    """3-fold CV → (cv_scores, fold_models)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cv_scores, models = [], []
    for tr_idx, val_idx in skf.split(X_tr_all, y_tr_all):
        X_tr, X_val = X_tr_all.iloc[tr_idx], X_tr_all.iloc[val_idx]
        y_tr, y_val = y_tr_all.iloc[tr_idx], y_tr_all.iloc[val_idx]
        m = LGBMClassifier(**_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(30, verbose=False)],
        )
        prob = m.predict_proba(X_val)[:, 1]
        cv_scores.append(average_precision_score(y_val, prob))
        models.append(m)
    return cv_scores, models


def get_test_prauc(X_tr_all, y_tr_all, X_te, y_te):
    """전체 train 재학습 단일 모델 → test PR-AUC"""
    m = LGBMClassifier(**_PARAMS)
    m.fit(X_tr_all, y_tr_all)
    prob = m.predict_proba(X_te)[:, 1]
    return average_precision_score(y_te, prob)


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
    return features[np.argmin(mean_abs)] ########################### argmax, argmin


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, TRAIN_PATH))
    test_df  = pd.read_parquet(os.path.join(base_dir, TEST_PATH))

    features = [c for c in train_df.columns if c != TARGET]
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

    while len(features) > 1:
        exp_num  += 1
        exp_id    = f"exp_{exp_num:03d}"
        change    = "baseline" if removed is None else f"remove: {removed}"

        # ── 학습 ──────────────────────────────────────────────
        cv_scores, fold_models = run_fold_cv(X_train, y_train)
        test_prauc             = get_test_prauc(X_train, y_train, X_test, y_test)
        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))

        # ── 기록 ──────────────────────────────────────────────
        row = {
            "exp_id":      exp_id,
            "base_exp":    prev_exp_id,
            "change":      change,
            "n_features":  len(features),
            "CV PR-AUC":   round(cv_mean, 4),
            "CV std":      round(cv_std,  4),
            "test PR-AUC": round(test_prauc, 4),
        }
        results.append(row)

        print(
            f"[{exp_id}] n={len(features):3d} | "
            f"CV={cv_mean:.4f}±{cv_std:.4f} | "
            f"test={test_prauc:.4f} | {change}"
        )

        # ── 다음 제거 feature (SHAP 1위) ──────────────────────
        next_remove = top_shap_feature(fold_models, X_train, list(features))

        # ── 업데이트 ──────────────────────────────────────────
        prev_exp_id = exp_id
        removed     = next_remove
        features    = [f for f in features if f != next_remove]
        X_train     = X_train[features]
        X_test      = X_test[features]

    # ── 저장 ──────────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    out_csv  = os.path.join(base_dir, OUTPUT_CSV)
    out_xlsx = os.path.join(base_dir, OUTPUT_XLSX)

    df_out.to_csv(out_csv,  index=False, encoding="utf-8-sig")
    df_out.to_excel(out_xlsx, index=False)

    print("\n" + "=" * 70)
    print(df_out.to_string(index=False))
    print(f"\n✅ CSV  → {out_csv}")
    print(f"✅ XLSX → {out_xlsx}")
