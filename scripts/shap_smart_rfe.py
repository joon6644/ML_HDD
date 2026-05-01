# =============================================================
#  shap_smart_rfe.py  ─  SHAP Smart Backward RFE
#  - 매 라운드: SHAP 1위(top) vs 꼴찌(bottom) 두 후보 비교
#  - 제거 후 CV 점수 하락이 더 적은 쪽을 제거 (greedy optimal)
#  - 연산량: 라운드당 CV 2회 + final model 1회
#  - 변수 1개 남을 때까지 반복
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

from config.rfe_config import LGBM_PARAMS

TRAIN_PATH  = r"data\rfe_sample_data\rfe_train.parquet"
TEST_PATH   = r"data\rfe_sample_data\rfe_test.parquet"
OUTPUT_CSV  = r"scripts\shap_smart_rfe_results.csv"
OUTPUT_XLSX = r"scripts\shap_smart_rfe_results.xlsx"

N_FOLDS     = 3
SHAP_SAMPLE = 200
TARGET      = "failure"

_PARAMS = {**LGBM_PARAMS}


# ── C레벨 stderr 억제 ─────────────────────────────────────────
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


# ── 함수 ──────────────────────────────────────────────────────
def run_cv_score(X, y):
    """3-fold CV → (cv_mean, cv_std, models)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores, models = [], []
    for tr_idx, val_idx in skf.split(X, y):
        m = LGBMClassifier(**_PARAMS)
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="average_precision",
            callbacks=[early_stopping(30, verbose=False)],
        )
        prob = m.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], prob))
        models.append(m)
    return float(np.mean(scores)), float(np.std(scores)), models


def get_test_prauc(X_tr, y_tr, X_te, y_te):
    """전체 train 재학습 단일 모델 → test PR-AUC"""
    m = LGBMClassifier(**_PARAMS)
    m.fit(X_tr, y_tr)
    return average_precision_score(y_te, m.predict_proba(X_te)[:, 1])


def get_shap_ranking(models, X_train, features):
    """fold 모델 평균 |SHAP| → (top_feature, bottom_feature)"""
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
    feat_arr = np.array(features)
    return feat_arr[np.argmax(mean_abs)], feat_arr[np.argmin(mean_abs)]


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, TRAIN_PATH))
    test_df  = pd.read_parquet(os.path.join(base_dir, TEST_PATH))

    features = [c for c in train_df.columns if c != TARGET]
    X_train  = train_df[features]
    y_train  = train_df[TARGET]
    X_test   = test_df[features]
    y_test   = test_df[TARGET]

    results     = []
    removed     = None
    prev_exp_id = "none"
    exp_num     = 0

    print(f"시작 feature 수: {len(features)}")
    print("=" * 80)

    # ── baseline (라운드 0) ────────────────────────────────────
    cv_mean, cv_std, fold_models = run_cv_score(X_train, y_train)
    test_prauc = get_test_prauc(X_train, y_train, X_test, y_test)
    exp_num   += 1
    exp_id     = f"exp_{exp_num:03d}"
    results.append({
        "exp_id":      exp_id,
        "base_exp":    prev_exp_id,
        "change":      "baseline",
        "removed":     "none",
        "decision":    "-",
        "top_cand":    "-",
        "top_cv":      "-",
        "bot_cand":    "-",
        "bot_cv":      "-",
        "n_features":  len(features),
        "CV PR-AUC":   round(cv_mean, 4),
        "CV std":      round(cv_std,  4),
        "test PR-AUC": round(test_prauc, 4),
    })
    print(f"[{exp_id}] n={len(features):3d} | CV={cv_mean:.4f}±{cv_std:.4f} | test={test_prauc:.4f} | baseline")

    while len(features) > 1:
        # ── 1. SHAP으로 top/bottom 후보 결정 ──────────────────
        top_feat, bot_feat = get_shap_ranking(fold_models, X_train, list(features))

        # ── 2. top 제거 시 CV ─────────────────────────────────
        feat_a = [f for f in features if f != top_feat]
        cv_a, std_a, _ = run_cv_score(X_train[feat_a], y_train)

        # ── 3. bottom 제거 시 CV ──────────────────────────────
        feat_b = [f for f in features if f != bot_feat]
        cv_b, std_b, _ = run_cv_score(X_train[feat_b], y_train)

        # ── 4. 제거 후 점수 하락이 덜한 쪽(덜 중요한 쪽) 제거 ──
        if cv_a >= cv_b:
            next_remove = top_feat
            new_features = feat_a
            decision = f"top({top_feat}) cv={cv_a:.4f} >= bot({bot_feat}) cv={cv_b:.4f}"
        else:
            next_remove = bot_feat
            new_features = feat_b
            decision = f"bot({bot_feat}) cv={cv_b:.4f} > top({top_feat}) cv={cv_a:.4f}"

        # ── 5. 실제 학습 (승자 feature set) ───────────────────
        X_train = X_train[new_features]
        X_test  = X_test[new_features]
        features = new_features

        cv_mean, cv_std, fold_models = run_cv_score(X_train, y_train)
        test_prauc = get_test_prauc(X_train, y_train, X_test, y_test)

        prev_exp_id = exp_id
        exp_num    += 1
        exp_id      = f"exp_{exp_num:03d}"

        results.append({
            "exp_id":      exp_id,
            "base_exp":    prev_exp_id,
            "change":      f"remove: {next_remove}",
            "removed":     next_remove,
            "decision":    "top" if next_remove == top_feat else "bot",
            "top_cand":    top_feat,
            "top_cv":      round(cv_a, 4),
            "bot_cand":    bot_feat,
            "bot_cv":      round(cv_b, 4),
            "n_features":  len(features),
            "CV PR-AUC":   round(cv_mean, 4),
            "CV std":      round(cv_std,  4),
            "test PR-AUC": round(test_prauc, 4),
        })

        print(
            f"[{exp_id}] n={len(features):3d} | CV={cv_mean:.4f}±{cv_std:.4f} | "
            f"test={test_prauc:.4f} | remove: {next_remove} "
            f"[top_cv={cv_a:.4f} / bot_cv={cv_b:.4f}]"
        )

    # ── 저장 ──────────────────────────────────────────────────
    df_out   = pd.DataFrame(results)
    out_csv  = os.path.join(base_dir, OUTPUT_CSV)
    out_xlsx = os.path.join(base_dir, OUTPUT_XLSX)

    df_out.to_csv(out_csv,  index=False, encoding="utf-8-sig")
    df_out.to_excel(out_xlsx, index=False)

    print("\n" + "=" * 80)
    print(df_out[["exp_id","change","n_features","CV PR-AUC","CV std","test PR-AUC","decision"]].to_string(index=False))
    print(f"\n✅ CSV  → {out_csv}")
    print(f"✅ XLSX → {out_xlsx}")
