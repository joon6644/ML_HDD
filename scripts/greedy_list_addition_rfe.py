# greedy list forward selection

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

OUTPUT_CSV  = r"results\feature_selection\greedy_list_addition_results.csv"

# =========================================================
# 사용자가 지정한 후보 변수 풀입니다.
# config의 기본 베이스 변수에, 아래 리스트의 변수 중 성능을 가장 많이 올리는
# 변수부터 하나씩 그리디(Greedy)하게 찾아가며 누적 추가합니다.
# =========================================================
ADDITIONAL_FEATURES = [
"total_reads_7d_dai",
"read_spike_ratio",
"s187_ever_flag",
"timeout_total_14d_max",
"s190_14d_zscore",
"error_growth_ratio",
"total_reads_diff",
"s184_3d_max",
"multi_error_count",
"smart_198_raw",
"total_seeks",
"s242_28d_asfd",
"s197_damaged",
"s190_28d_mean",
"total_reads_28d_max",
"workload_intensity",
"smart_183_raw",
"s187_damaged",
"total_seeks_28d_max",
"total_reads_28d_std",
"write_stability_ratio",
"total_seeks_14d_mean",
"s198_28d_max",
"age_weighted_workload",
"s241_14d_mean",
"total_reads_7d_max",
"s190_7d_ewma",
"s5_damaged",
"s194_28d_dai",
"s190_28d_ewma",
"s242_diff",
"s194_14d_max",
"io_asymmetry_index",
"s194_28d_std",
"s199_days_since_last",
"s190_28d_max",
"s191_days_since_last",
"s5_days_since_first",
"total_reads_7d_asfd",
"s190_28d_zscore",
"s5_relative_score_14d",
"s190_14d_mean",
"s241_28d_asfd",
"timeout_seek_density",
"cumulative_error_score",
"s242_28d_dai",
"s241_diff",
"total_reads_28d_asfd",
"s194_28d_mean",
"s194_28d_max",
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


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    train_df = pd.read_parquet(os.path.join(base_dir, cfg.TRAIN_PATH)).sort_values(["serial_number", "date"]).reset_index(drop=True)

    # Group 정보 사전 추출 (Serial Leakage 방지용)
    groups = train_df["serial_number"].copy() if "serial_number" in train_df.columns else None

    # fs_core.resolve_features를 이용해 config에 정의된 변수만 추출 (시작 기본 베이스 변수들)
    features_resolved, audit = fs_core.resolve_features(cfg, train_df.columns.tolist())
    
    # 전체 데이터셋 컬럼 중 메타 컬럼 제외
    meta_cols = ["serial_number", "date", cfg.TARGET_COL]
    if hasattr(cfg, "META_COLS"):
        meta_cols.extend(cfg.META_COLS)
    all_available = [c for c in train_df.columns if c not in meta_cols]

    # 시작점: 베이스 변수
    selected_features = sorted(features_resolved)
    
    # 설정할 후보군 필터링
    valid_additional_features = []
    for f in ADDITIONAL_FEATURES:
        if f not in all_available:
            print(f"[Warning] 변수 '{f}'가 데이터셋에 없어 건너뜁니다.")
        elif f in selected_features:
            print(f"[Warning] 변수 '{f}'는 이미 기본 변수(config)에 포함되어 있어 건너뜁니다.")
        else:
            valid_additional_features.append(f)

    # 전체 사용될 변수집합(미리 추출용)
    features_all = sorted(list(set(selected_features + valid_additional_features)))
    
    X_train_full = train_df[features_all]
    y_train = train_df[cfg.TARGET_COL]

    results = []
    prev_exp_id = "none"
    exp_num = 0

    print(f"시작 기본(베이스) feature 수: {len(selected_features)}")
    print(f"탐색 후보(ADDITIONAL_FEATURES) 수: {len(valid_additional_features)}")
    print("=" * 70)

    out_csv  = os.path.join(base_dir, OUTPUT_CSV)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── [Step 1] Baseline 평가 (시작 베이스 변수들만) ────────────────
    exp_num += 1
    exp_id = f"exp_{exp_num:03d}"
    print(f"[{exp_id}] Baseline 평가 중... (n={len(selected_features)})")
    
    if len(selected_features) > 0:
        cv_scores, _ = run_fold_cv(X_train_full[selected_features], y_train, groups=groups)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    else:
        cv_mean, cv_std = 0.0, 0.0

    results.append({
        "exp_id": exp_id,
        "base_exp": "none",
        "change": "baseline",
        "CV PR-AUC": round(cv_mean, 5),
        "CV std": round(cv_std, 5),
    })
    prev_exp_id = exp_id

    print(f"[{exp_id}] Baseline   CV={cv_mean:.5f}±{cv_std:.5f}\n")
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ── [Step 2] Greedy Addition Loop ────────────────────────────────
    candidate_features = list(valid_additional_features)
    current_features = list(selected_features)
    
    while len(candidate_features) > 0:
        exp_num += 1
        exp_id = f"exp_{exp_num:03d}"
        
        best_cv_mean = -1.0
        best_cv_std = 0.0
        feature_to_add = None

        print(f"[{exp_id}] 최적 추가 대상 탐색 중 (남은 후보: {len(candidate_features)}개)...")
        print(f"  {'No.':8s} {'Feature Name':30s}   {'CV PR-AUC':10s}   {'CV std':8s}")
        print(f"  {'-'*66}")
        
        # 남은 후보 변수들을 현재 조합에 하나씩 추가해보기
        for i, f in enumerate(candidate_features):
            temp_features = current_features + [f]
            
            # CV 수행 (판단 기준)
            cv_scores, _ = run_fold_cv(X_train_full[temp_features], y_train, groups=groups)
            curr_cv_mean = float(np.mean(cv_scores))
            curr_cv_std  = float(np.std(cv_scores))
            
            # 표 형식으로 실시간 출력
            print(f"  ({i+1:2d}/{len(candidate_features):2d}) {f:30s}   {curr_cv_mean:10.5f}   {curr_cv_std:8.5f}", flush=True)

            # 추가했을 때 CV 성능이 가장 좋아지는 변수 업데이트
            if curr_cv_mean > best_cv_mean:
                best_cv_mean = curr_cv_mean
                best_cv_std = curr_cv_std
                feature_to_add = f
                
        # 상태 업데이트 (선택된 변수를 확정 짓고 후보에서 제거)
        current_features.append(feature_to_add)
        candidate_features.remove(feature_to_add)
        
        print(f"  => [선택됨] {feature_to_add} (CV PR-AUC: {best_cv_mean:.5f})\n")
        
        # 결과 기록
        results.append({
            "exp_id": exp_id,
            "base_exp": prev_exp_id,
            "change": f"+ {feature_to_add}",
            "CV PR-AUC": round(best_cv_mean, 5),
            "CV std": round(best_cv_std, 5),
        })

        prev_exp_id = exp_id
        
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ── 최종 결과 ──────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    print("=" * 70)
    print(df_out.to_string(index=False))
    print(f"\n✅ CSV  → {out_csv}")
