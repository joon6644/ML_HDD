# =============================================================
#  addition_test.py  ─  INCLUDE_COLS 외부 변수 Forward Addition 테스트
#
#  동작 방식
#  ---------
#  1. fs_config.py 의 INCLUDE_COLS 전체로 baseline 학습 → PR-AUC 기록
#  2. 데이터에 존재하지만 INCLUDE_COLS에 없는 변수를 하나씩 추가하며 재학습 → PR-AUC 기록
#  3. 총 M+1 번 실험 (baseline 1회 + 변수 추가 M회)
#
#  출력
#  ----
#  - 콘솔: 실험 진행 현황 실시간 출력 (ablation_test.py 동일 포맷)
#
#  실행
#  ----
#  python scripts\addition_test.py
# =============================================================

import sys, os, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# ── config 로드 ──────────────────────────────────────────────
from config import fs_config as cfg
from config.fs_config import (
    TRAIN_PATH,
    TEST_PATH,
    TARGET_COL,
    LGBM_PARAMS,
    CV_N_SPLITS,
    CV_SHUFFLE,
    CV_SEED,
)
from src import fs_core

# ── 출력 경로 ────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_CSV  = os.path.join(BASE_DIR, "results", "feature_selection", "addition_test_results.csv")

# ── 파라미터 복사 (원본 건드리지 않음) ──────────────────────
_PARAMS = {**LGBM_PARAMS}


# ── CV + test 평가 ───────────────────────────────────────────
def evaluate(X_tr_all: pd.DataFrame, y_tr_all: pd.Series,
             X_te: pd.DataFrame,    y_te: pd.Series) -> dict:
    """
    Stratified K-Fold CV → CV PR-AUC 통계 + 전체 train 재학습 test PR-AUC.
    Returns
    -------
    dict: cv_mean, cv_std, cv_scores
    """
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=CV_SHUFFLE, random_state=CV_SEED)
    cv_scores = []

    for tr_idx, val_idx in skf.split(X_tr_all, y_tr_all):
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

    return {
        "cv_mean":    float(np.mean(cv_scores)),
        "cv_std":     float(np.std(cv_scores)),
        "cv_scores":  cv_scores,
    }


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 데이터 로드 ───────────────────────────────────────────
    print("[DATA] 데이터 로드 중...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    # ── Feature 설정 ─────────────────────────────────────
    valid_base, audit = fs_core.resolve_features(cfg, train_df.columns.tolist())

    available_cols = set(train_df.columns) - {TARGET_COL}
    meta_cols = set(getattr(cfg, "META_COLS", []))
    available_cols -= meta_cols

    # 추출된 컬럼에 없지만 데이터에 존재하는 변수 → 추가 대상
    candidate_cols = [c for c in sorted(available_cols) if c not in set(valid_base)]

    if len(valid_base) == 0:
        raise RuntimeError("추출된 유효한 컬럼이 없습니다. config를 확인하세요.")

    if len(candidate_cols) == 0:
        raise RuntimeError("추가할 수 있는 변수가 없습니다. 모든 변수가 이미 추출되었습니다.")

    y_train = train_df[TARGET_COL]
    y_test  = test_df[TARGET_COL]

    total_exp = len(candidate_cols) + 1   # baseline + 변수 1개씩 M회 추가
    results   = []
    prev_exp  = "none"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"\n[INFO] 현재 포함된 변수(baseline) 수: {len(valid_base)}개")
    print(f"[INFO] 추가 테스트 대상 변수 수     : {len(candidate_cols)}개")
    print(f"[INFO] 총 실험 수                  : {total_exp}회  (baseline + 변수 추가 {len(candidate_cols)}회)")
    print(f"       pos rate: train={y_train.mean():.4f}, test={y_test.mean():.4f}")
    print("=" * 80)
    print(f"{'exp_id':<10} {'base_exp':<10} {'change':<30} {'CV PR-AUC':>10} {'CV std':>8}")
    print("-" * 80)

    # ── 실험 루프 ─────────────────────────────────────────────
    # exp 0 : baseline (INCLUDE_COLS 그대로)
    # exp i : valid_base + candidate_cols[i-1] 추가한 feature set
    for exp_idx in range(total_exp):
        if exp_idx == 0:
            current_cols = valid_base[:]
            added_col    = None
            change_str   = "baseline"
        else:
            added_col    = candidate_cols[exp_idx - 1]
            current_cols = valid_base + [added_col]
            change_str   = added_col

        exp_id = f"exp_{exp_idx + 1:03d}"

        X_tr = train_df[current_cols]
        X_te = test_df[current_cols]

        res = evaluate(X_tr, y_train, X_te, y_test)

        row = {
            "exp_id":       exp_id,
            "base_exp":     prev_exp,
            "change":       change_str,
            "n_features":   len(current_cols),
            "CV PR-AUC":    round(res["cv_mean"],    5),
            "CV std":       round(res["cv_std"],      5),
            "CV scores":    str([round(s, 5) for s in res["cv_scores"]]),
        }
        results.append(row)

        print(
            f"{exp_id:<10} {prev_exp:<10} {change_str:<30} "
            f"{res['cv_mean']:>10.5f} {res['cv_std']:>8.5f}",
            flush=True,
        )

        prev_exp = exp_id

        df_out = pd.DataFrame(results)
        df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
