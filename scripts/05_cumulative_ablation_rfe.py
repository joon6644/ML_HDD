import sys, os, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import average_precision_score

from config import fs_config as cfg
from src import fs_core

OUTPUT_CSV  = r"results\feature_selection\cumulative_ablation_results.csv"

# =========================================================
# 사용자가 제거할 변수를 순서대로 나열합니다.
# config의 기본 베이스 변수에서 시작하여, 아래 리스트의 변수를 1개씩 순차적으로 누적하면서 제거(ablation)합니다.
# =========================================================
FEATURES_TO_REMOVE = [
"io_asymmetry_index",
"timeout_seek_density",
"s187_ever_flag",
"smart_197_raw",
"s197_damaged",
"s187_damaged",
"timeout_total_14d_max",
"total_reads_diff",
"s194_28d_mean",
"error_growth_ratio",
"s190_28d_std",
"s190_28d_max",
"s242_14d_mean",
"total_reads_14d_max",
"s194_28d_max",
"s241_7d_max",
"smart_241_raw",
"s190_28d_zscore",
"s190_14d_zscore",
"s194_28d_dai",
"s194_28d_ewma",
"age_weighted_workload",
"s242_28d_dai",
"s190_28d_ewma",
"s194_14d_std",
"read_spike_ratio",
"s184_3d_max",
"total_reads_7d_asfd",
"s190_28d_mean",
"s194_28d_std",
"s194_14d_max",
"s199_days_since_last",
"s187_28d_sum",
"s187_days_since_first",
"total_seeks_28d_asfd",
"total_reads_7d_max",
"s5_days_since_first",
"multi_error_count",
"smart_242_raw",
"s241_diff",
"smart_9_raw",
"smart_184_raw",
"error_density_14d",
"total_seeks_diff",
"smart_187_raw",
]

# config 파라미터 가져오기
_PARAMS = {**cfg.LGBM_PARAMS}
N_FOLDS = cfg.CV_N_SPLITS

def run_fold_cv(X_tr_all, y_tr_all, groups=None):
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

def get_val_prauc(X_tr_all, y_tr_all, X_val, y_val):
    m = LGBMClassifier(**_PARAMS)
    m.fit(X_tr_all, y_tr_all)
    prob = m.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, prob)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, cfg.TRAIN_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)
    test_df  = pd.read_parquet(os.path.join(base_dir, cfg.TEST_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)

    # Group 정보 사전 추출 (Serial Leakage 방지용)
    groups = train_df["serial_number"].copy() if "serial_number" in train_df.columns else None

    # fs_core.resolve_features를 이용해 config에 정의된 변수만 추출 (시작 기본 베이스 변수들)
    features_resolved, audit = fs_core.resolve_features(cfg, train_df.columns.tolist())
    
    # 시작점: 베이스 변수 전체
    selected_features = sorted(features_resolved)
    
    # Validation에 사용할 전체 컬럼 검증
    valid_features_to_remove = []
    for f in FEATURES_TO_REMOVE:
        if f not in selected_features:
            print(f"[Warning] 변수 '{f}'는 베이스 변수(config)에 포함되어 있지 않아 제거할 수 없습니다. (건너뜁니다)")
        else:
            valid_features_to_remove.append(f)

    # 전체 사용될 변수집합(미리 추출용)
    features_all = sorted(list(set(selected_features)))
    
    X_train_full = train_df[features_all]
    y_train = train_df[cfg.TARGET_COL]
    X_test_full  = test_df[features_all]
    y_test  = test_df[cfg.TARGET_COL]

    results = []
    prev_exp_id = "none"
    exp_num = 0

    print(f"시작 기본(베이스) feature 수: {len(selected_features)}")
    print(f"누적 제거할 feature 수: {len(valid_features_to_remove)}")
    print("=" * 70)

    # ── [Step 1] Baseline 평가 (시작 베이스 변수들 전체) ────────────────
    exp_num += 1
    exp_id = f"exp_{exp_num:03d}"
    print(f"[{exp_id}] Baseline 평가 중... (n={len(selected_features)})")
    
    if len(selected_features) > 0:
        cv_scores, _ = run_fold_cv(X_train_full[selected_features], y_train, groups=groups)
        val_prauc = get_val_prauc(X_train_full[selected_features], y_train, X_test_full[selected_features], y_test)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    else:
        cv_mean, cv_std, val_prauc = 0.0, 0.0, 0.0

    results.append({
        "exp_id": exp_id,
        "base_exp": "none",
        "change": "baseline",
        "CV PR-AUC": round(cv_mean, 5),
        "CV std": round(cv_std, 5),
        "val PR-AUC": round(val_prauc, 5),
    })
    prev_exp_id = exp_id

    print(f"[{exp_id}] Baseline   CV={cv_mean:.5f}±{cv_std:.5f}   val={val_prauc:.5f}\n")

    # ── [Step 2] Cumulative Ablation Loop (순차적 누적 제거) ────────
    current_features = list(selected_features)
    
    for i, feature_to_remove in enumerate(valid_features_to_remove):
        exp_num += 1
        exp_id = f"exp_{exp_num:03d}"
        
        # 변수 누적 제거
        current_features.remove(feature_to_remove)
        
        print(f"[{exp_id}] 변수 누적 제거 중 ({i+1}/{len(valid_features_to_remove)}): - {feature_to_remove}")
        
        if len(current_features) == 0:
            print("  -> 남은 변수가 없어 평가를 건너뜁니다.")
            curr_cv_mean, curr_cv_std, curr_val_prauc = 0.0, 0.0, 0.0
        else:
            # CV 수행
            cv_scores, _ = run_fold_cv(X_train_full[current_features], y_train, groups=groups)
            curr_cv_mean = float(np.mean(cv_scores))
            curr_cv_std  = float(np.std(cv_scores))
            
            # Validation 수행
            curr_val_prauc = get_val_prauc(X_train_full[current_features], y_train, X_test_full[current_features], y_test)
        
        # 결과 기록
        results.append({
            "exp_id": exp_id,
            "base_exp": prev_exp_id,
            "change": f"- {feature_to_remove}",
            "CV PR-AUC": round(curr_cv_mean, 5),
            "CV std": round(curr_cv_std, 5),
            "val PR-AUC": round(curr_val_prauc, 5),
        })

        print(
            f"  -> n={len(current_features):3d}   "
            f"CV={curr_cv_mean:.5f}±{curr_cv_std:.5f}   "
            f"val={curr_val_prauc:.5f}\n"
        )
        
        prev_exp_id = exp_id

    # ── 저장 ──────────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    out_csv  = os.path.join(base_dir, OUTPUT_CSV)
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df_out.to_csv(out_csv,  index=False, encoding="utf-8-sig")

    print("=" * 70)
    print(df_out.to_string(index=False))
    print(f"\n✅ CSV  → {out_csv}")
