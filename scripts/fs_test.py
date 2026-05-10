import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

def check_gpu() -> bool:
    """LightGBM GPU 사용 가능 여부 확인."""
    try:
        lgb.train(
            {"device": "gpu", "verbose": -1},
            lgb.Dataset(np.zeros((1, 1)), label=[0]),
            num_boost_round=1,
        )
        return True
    except Exception:
        return False

# =====================================================================
# 1. 전역 설정 (Hyperparameters)
# =====================================================================
SEED = 42
TARGET_COL = "failure"

LGBM_PARAMS = {
    "objective": "binary",
    "importance_type": "gain",
    "class_weight": "balanced",
    "max_depth": 6,
    "num_leaves": 40,
    "n_estimators": 200,
    "learning_rate": 0.07,
    "device": "gpu" if check_gpu() else "cpu",
    "random_state": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "data_random_seed": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

CV_N_SPLITS = 5
SHAP_SAMPLE_N = 5000
SHAP_TOP_N = 30

# =====================================================================
# 2. 핵심 파이프라인 함수
# =====================================================================
def run_pipeline(train_path: str, test_path: str, features: list):
    print("데이터를 로드합니다...")
    df_train = pd.read_parquet(train_path).sort_values(["serial_number", "date"]).reset_index(drop=True)
    df_test = pd.read_parquet(test_path).sort_values(["serial_number", "date"]).reset_index(drop=True)

    # 개체 누수 방지를 위한 그룹 정보(serial_number) 사전 구출
    groups_train = df_train["serial_number"].copy() if "serial_number" in df_train.columns else None

    # 모델에 들어가면 안 되는 메타 데이터 삭제
    drop_meta = ["serial_number", "date"]
    for df in [df_train, df_test]:
        found = [c for c in drop_meta if c in df.columns]
        if found:
            df.drop(columns=found, inplace=True)

    X_train, y_train = df_train[features], df_train[TARGET_COL]
    X_test, y_test = df_test[features], df_test[TARGET_COL]

    print(f"\n학습 시작 (Feature: {len(features)}개)")
    
    # ── 교차 검증 (GroupKFold 적용) ──
    if groups_train is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
        split_gen = cv_splitter.split(X_train, y_train, groups=groups_train)
        print("StratifiedGroupKFold 적용 완료 (개체 단위 분할)")
    else:
        cv_splitter = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
        split_gen = cv_splitter.split(X_train, y_train)
        print("StratifiedKFold 적용 완료 (행 단위 분할)")

    cv_scores = []
    models = []

    for fold, (tr_idx, val_idx) in enumerate(split_gen, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="average_precision")
        
        val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_prob)
        cv_scores.append(score)
        models.append(model)
        print(f"  - Fold {fold} PR-AUC: {score:.5f}")

    print(f"\nCV PR-AUC: {np.mean(cv_scores):.5f} (± {np.std(cv_scores):.5f})")

    # ── 전체 데이터 단일 모델 재학습 및 VAL 평가 ──
    final_model = LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X_train, y_train)
    val_prob = final_model.predict_proba(X_test)[:, 1]
    final_prauc = average_precision_score(y_test, val_prob)
    print(f"VAL PR-AUC: {final_prauc:.5f}")

    # ── SHAP 중요도 계산 (층화 샘플링 적용) ──
    print(f"\nSHAP 중요도를 계산합니다... (Sample: {SHAP_SAMPLE_N})")
    
    # 고장 비율을 유지하며 샘플링
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(SHAP_SAMPLE_N, len(X_train)), random_state=SEED)
    idx, _ = next(sss.split(X_train, y_train))
    X_sample = X_train.iloc[idx].astype("float32")

    shap_vals_list = []
    for m in models:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list): sv = sv[1]
        shap_vals_list.append(sv)

    shap_mean = np.mean(shap_vals_list, axis=0)
    mean_abs_shap = np.abs(shap_mean).mean(axis=0)
    
    shap.summary_plot(shap_mean, X_sample, show=False, max_display=SHAP_TOP_N)
    plt.title(f"SHAP Beeswarm Plot (Top {SHAP_TOP_N})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    df_shap = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap})
    df_shap = df_shap.sort_values("mean_abs_shap", ascending=False).head(SHAP_TOP_N).reset_index(drop=True)

    # SHAP 시각화
    plt.figure(figsize=(10, max(6, SHAP_TOP_N * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_shap)))
    plt.barh(df_shap["feature"][::-1], df_shap["mean_abs_shap"][::-1], color=colors)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP Feature Importance (Top {SHAP_TOP_N})")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"cv_mean": np.mean(cv_scores), "val_prauc": final_prauc, "shap_df": df_shap}

# =====================================================================
# 사용 예시
# =====================================================================
if __name__ == "__main__":
    # 1. 파일 경로 설정
    TRAIN_PATH = "./data/fs_sample_data/fs_train.parquet"
    VAL_PATH   = "./data/fs_sample_data/fs_validation.parquet"

    # 2. 특성 넣기
    my_features = [
        "s187_days_since_first",
        "error_density_14d",
        "smart_187_raw",
        "smart_197_raw",
        "s187_28d_sum",
        "total_seeks_28d_max",
        "total_seeks_28d_asfd",
        "cumulative_error_score",
        "s5_days_since_first",
        "age_weighted_workload",
        "s242_14d_mean",
        "smart_241_raw",
        "smart_242_raw",
        "smart_9_raw",
        "s241_14d_mean",
        "s241_diff",
        "s241_7d_max",
        "total_seeks_14d_mean",
        "s190_28d_mean",
        "s190_28d_ewma",
        "s190_7d_ewma",
        "total_reads_7d_max",
        "total_reads_14d_max",
        "s194_14d_std",
        "s194_28d_mean",
        "smart_184_raw",
        "total_reads_28d_std",
        "total_reads_7d_dai",
    ]
    
    # 3. 파이프라인 실행!
    results = run_pipeline(
        train_path=TRAIN_PATH, 
        test_path=VAL_PATH, 
        features=my_features
    )