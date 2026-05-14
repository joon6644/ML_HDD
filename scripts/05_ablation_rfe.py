# greedy backward elimination

import sys, os, warnings, logging, contextlib
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

# C레벨 stderr 억제용
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

OUTPUT_CSV  = r"results\feature_selection\ablation_rfe_results.csv"

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


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, cfg.TRAIN_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)
    test_df  = pd.read_parquet(os.path.join(base_dir, cfg.TEST_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)

    # Group 정보 사전 추출 (Serial Leakage 방지용)
    groups = train_df["serial_number"].copy() if "serial_number" in train_df.columns else None

    # fs_core.resolve_features를 이용해 config에 정의된 변수만 추출
    features, audit = fs_core.resolve_features(cfg, train_df.columns.tolist())
    features = sorted(features)  # 재현성을 위한 정렬
    
    X_train  = train_df[features]
    y_train  = train_df[cfg.TARGET_COL]
    X_test   = test_df[features]
    y_test   = test_df[cfg.TARGET_COL]

    results = []
    removed = None
    prev_exp_id = "none"
    exp_num = 0

    out_csv  = os.path.join(base_dir, OUTPUT_CSV)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"시작 feature 수: {len(features)}")
    print("=" * 70)

    # ── [Step 1] Baseline 평가 (모든 변수 포함) ────────────────
    exp_num += 1
    exp_id = f"exp_{exp_num:03d}"
    print(f"[{exp_id}] Baseline 평가 중... (n={len(features)})")
    
    cv_scores, _ = run_fold_cv(X_train, y_train, groups=groups)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    results.append({
        "exp_id": exp_id,
        "base_exp": "none",
        "change": "baseline",
        "CV PR-AUC": round(cv_mean, 5),
        "CV std": round(cv_std, 5),
    })
    prev_exp_id = exp_id

    print(f"[{exp_id}] Baseline   CV={cv_mean:.5f}±{cv_std:.5f}\n")
    
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ── [Step 2+] Ablation Loop (하나씩 제거하며 성능 최적화) ──────
    while len(features) > 1:
        exp_num += 1
        exp_id = f"exp_{exp_num:03d}"
        
        best_cv_mean = -1.0
        best_cv_std = 0.0
        feature_to_remove = None

        print(f"[{exp_id}] 최적 제거 대상 탐색 중 (후보: {len(features)}개)...")
        print(f"  {'No.':8s} {'Feature Name':30s}   {'CV PR-AUC':10s}   {'CV std':8s}")
        print(f"  {'-'*66}")
        
        # 현재 남은 변수들을 하나씩 제거해보기
        for i, f in enumerate(features):
            temp_features = [col for col in features if col != f]
            
            # CV 수행 (판단 기준)
            cv_scores, _ = run_fold_cv(X_train[temp_features], y_train, groups=groups)
            curr_cv_mean = float(np.mean(cv_scores))
            curr_cv_std  = float(np.std(cv_scores))
            
            # 표 형식으로 실시간 출력
            print(f"  ({i+1:2d}/{len(features):2d}) {f:30s}   {curr_cv_mean:10.5f}   {curr_cv_std:8.5f}", flush=True)

            # 제거했을 때 CV 성능이 가장 좋은(가장 덜 중요한) 변수 업데이트
            if curr_cv_mean > best_cv_mean:
                best_cv_mean = curr_cv_mean
                best_cv_std  = curr_cv_std
                feature_to_remove = f

        # 결과 기록
        results.append({
            "exp_id": exp_id,
            "base_exp": prev_exp_id,
            "change": feature_to_remove,
            "CV PR-AUC": round(best_cv_mean, 5),
            "CV std": round(best_cv_std, 5),
        })

        print(
            f"[{exp_id}] n={len(features)-1:3d}   "
            f"CV={best_cv_mean:.5f}±{best_cv_std:.5f}   "
            f"제거: {feature_to_remove}\n"
        )

        # 상태 업데이트
        features.remove(feature_to_remove)
        X_train = X_train[features]
        X_test  = X_test[features]
        prev_exp_id = exp_id
        
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ── 최종 결과 ──────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(df_out.to_string(index=False))
    print(f"\n✅ CSV  → {out_csv}")
