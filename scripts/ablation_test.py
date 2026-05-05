# =============================================================
#  ablation_test.py  ─  INCLUDE_COLS 변수 Ablation 테스트
#
#  동작 방식
#  ---------
#  1. fs_config.py 의 INCLUDE_COLS 전체로 baseline 학습 → PR-AUC 기록
#  2. INCLUDE_COLS 순서대로 변수 하나씩 제거하며 재학습 → PR-AUC 기록
#  3. 총 N+1 번 실험 (baseline 1회 + 변수 제거 N회)
#
#  출력
#  ----
#  - 콘솔: 실험 진행 현황 실시간 출력
#  - CSV / Excel: scripts/ablation_test_results.csv & .xlsx
#
#  실행
#  ----
#  cd c:\Workspace\COIN\ML_HDD
#  python scripts\ablation_test.py
# =============================================================

import sys, os, warnings, logging, contextlib
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# ── config 로드 ──────────────────────────────────────────────
from config.fs_config import (
    TRAIN_PATH,
    TEST_PATH,
    INCLUDE_COLS,
    TARGET_COL,
    LGBM_PARAMS,
    CV_N_SPLITS,
    CV_SHUFFLE,
    CV_SEED,
)

# ── 출력 경로 ────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_CSV  = os.path.join(BASE_DIR, "scripts", "ablation_test_results.csv")
OUTPUT_XLSX = os.path.join(BASE_DIR, "scripts", "ablation_test_results.xlsx")

# ── 파라미터 복사 (원본 건드리지 않음) ──────────────────────
_PARAMS = {**LGBM_PARAMS}


# ── CV + test 평가 ───────────────────────────────────────────
def evaluate(X_tr_all: pd.DataFrame, y_tr_all: pd.Series,
             X_te: pd.DataFrame,    y_te: pd.Series) -> dict:
    """
    Stratified K-Fold CV → CV PR-AUC 통계 + 전체 train 재학습 test PR-AUC.
    Returns
    -------
    dict: cv_mean, cv_std, cv_scores, test_prauc
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
            callbacks=[early_stopping(30, verbose=False)],
        )
        prob = m.predict_proba(X_val)[:, 1]
        cv_scores.append(average_precision_score(y_val, prob))

    # 전체 train 재학습 → test 평가
    final_m = LGBMClassifier(**_PARAMS)
    final_m.fit(X_tr_all, y_tr_all)
    test_prob   = final_m.predict_proba(X_te)[:, 1]
    test_prauc  = average_precision_score(y_te, test_prob)

    return {
        "cv_mean":    float(np.mean(cv_scores)),
        "cv_std":     float(np.std(cv_scores)),
        "cv_scores":  cv_scores,
        "test_prauc": test_prauc,
    }


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 데이터 로드 ───────────────────────────────────────────
    print("[DATA] 데이터 로드 중...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    # ── INCLUDE_COLS 검증 ─────────────────────────────────────
    available_cols = set(train_df.columns) - {TARGET_COL}
    valid_cols = [c for c in INCLUDE_COLS if c in available_cols]
    missing    = [c for c in INCLUDE_COLS if c not in available_cols]

    if missing:
        print(f"\n[WARN] 데이터에 없는 컬럼 {len(missing)}개 제외:")
        for c in missing:
            print(f"   - {c}")

    if len(valid_cols) == 0:
        raise RuntimeError("INCLUDE_COLS 에서 유효한 컬럼이 없습니다. config를 확인하세요.")

    y_train = train_df[TARGET_COL]
    y_test  = test_df[TARGET_COL]

    total_exp  = len(valid_cols) + 1   # baseline + 변수 1개씩 N회 제거
    results    = []
    prev_exp   = "none"  # base_exp 추적

    print(f"\n[INFO] 유효 변수 수: {len(valid_cols)}개")
    print(f"[INFO] 총 실험 수  : {total_exp}회  (baseline + 변수 제거 {len(valid_cols)}회)")
    print(f"       pos rate: train={y_train.mean():.4f}, test={y_test.mean():.4f}")
    print("=" * 80)
    print(f"{'exp_id':<10} {'base_exp':<10} {'change':<30} {'CV PR-AUC':>10} {'CV std':>8} {'test PR-AUC':>12}")
    print("-" * 80)

    # ── 실험 루프 ─────────────────────────────────────────────
    # exp 0 : baseline (전체 INCLUDE_COLS)
    # exp i : valid_cols[i-1] 를 제거한 feature set
    for exp_idx in range(total_exp):
        if exp_idx == 0:
            # baseline
            current_cols = valid_cols[:]
            removed_col  = None
            label        = "baseline"
        else:
            # i번째 변수를 제거한 feature set
            removed_col  = valid_cols[exp_idx - 1]
            current_cols = [c for c in valid_cols if c != removed_col]
            label        = f"remove: {removed_col}"

        exp_id = f"exp_{exp_idx + 1:03d}"

        X_tr = train_df[current_cols]
        X_te = test_df[current_cols]

        res = evaluate(X_tr, y_train, X_te, y_test)

        change_str = "baseline" if removed_col is None else removed_col

        row = {
            "exp_id":       exp_id,
            "base_exp":     prev_exp,
            "change":       change_str,
            "n_features":   len(current_cols),
            "CV PR-AUC":    round(res["cv_mean"],    5),
            "CV std":       round(res["cv_std"],      5),
            "test PR-AUC":  round(res["test_prauc"],  5),
            "CV scores":    str([round(s, 5) for s in res["cv_scores"]]),
        }
        results.append(row)

        print(
            f"{exp_id:<10} {prev_exp:<10} {change_str:<30} "
            f"{res['cv_mean']:>10.5f} {res['cv_std']:>8.5f} {res['test_prauc']:>12.5f}",
            flush=True,
        )

        prev_exp = exp_id
