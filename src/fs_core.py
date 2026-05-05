"""
fs_core.py  ─  RFE 파이프라인 엔진
────────────────────────────────────
역할:  데이터 로드 → feature 결정 → CV 학습 → 평가 → 중요도
설정은 config/fs_config.py 에서만.
"""

from __future__ import annotations

import json
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 (Windows) ────────────────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()


# ════════════════════════════════════════════════════════════
#  1. CONFIG VALIDATION
# ════════════════════════════════════════════════════════════

def validate_config(cfg) -> list[str]:
    """
    config 오류를 사전에 감지해서 경고 메시지 리스트로 반환.
    빈 리스트 = 문제 없음.
    """
    issues: list[str] = []

    # 모드 체크
    if cfg.MODE not in ("add", "drop"):
        issues.append(f"[CRITICAL] MODE='{cfg.MODE}' 는 'add' 또는 'drop' 이어야 합니다.")

    # add 모드인데 아무것도 지정 안 한 경우
    if cfg.MODE == "add":
        if not cfg.INCLUDE_GROUPS and not cfg.INCLUDE_COLS:
            issues.append(
                "[CRITICAL] MODE='add' 인데 INCLUDE_GROUPS 와 INCLUDE_COLS 가 모두 비어 있습니다. "
                "feature가 0개가 됩니다."
            )

    # 경로 존재 체크
    import os
    for attr, path in [
        ("TRAIN_PATH", cfg.TRAIN_PATH),
        ("TEST_PATH",  cfg.TEST_PATH),
        ("FEATURE_GROUP_PATH", cfg.FEATURE_GROUP_PATH),
    ]:
        if not os.path.exists(path):
            issues.append(f"[CRITICAL] {attr}='{path}' 파일이 존재하지 않습니다.")

    return issues


# ════════════════════════════════════════════════════════════
#  2. FEATURE RESOLVER
# ════════════════════════════════════════════════════════════

def resolve_features(cfg, df_columns: list[str]) -> tuple[list[str], dict]:
    """
    config 설정에 따라 최종 feature 리스트를 결정.
    Returns:
        final_features: 사용할 feature 리스트
        audit: 추적 정보 dict (어떤 그룹/컬럼이 포함/제외됐는지)
    """
    with open(cfg.FEATURE_GROUP_PATH, encoding="utf-8") as f:
        group_map: dict[str, list[str]] = json.load(f)

    # 전체 그룹 feature 집합
    all_grouped: set[str] = set(c for cols in group_map.values() for c in cols)
    # 실제 데이터에 있는 컬럼만 (타깃 + 메타 컬럼 제외)
    meta_cols = set(getattr(cfg, "META_COLS", []))
    available: set[str] = set(df_columns) - {cfg.TARGET_COL} - meta_cols

    audit = {
        "mode": cfg.MODE,
        "total_in_group_json": len(all_grouped),
        "total_in_data": len(available),
        "included_groups": {},
        "excluded_groups": {},
        "included_cols": [],
        "excluded_cols": [],
        "missing_in_data": [],   # group json에 있지만 데이터에 없는 컬럼
        "warnings": [],
    }

    # ── add 모드 ─────────────────────────────────────────────
    if cfg.MODE == "add":
        feature_set: set[str] = set()

        for g in cfg.INCLUDE_GROUPS:
            if g not in group_map:
                audit["warnings"].append(
                    f"[WARNING] INCLUDE_GROUPS: '{g}' 는 feature_group.json에 없는 그룹명입니다."
                )
                continue
            cols = [c for c in group_map[g] if c in available]
            feature_set.update(cols)
            audit["included_groups"][g] = cols

        for c in cfg.INCLUDE_COLS:
            if c not in available:
                audit["warnings"].append(
                    f"[WARNING] INCLUDE_COLS: '{c}' 는 데이터에 없는 컬럼입니다."
                )
            else:
                feature_set.add(c)
                audit["included_cols"].append(c)

    # ── drop 모드 ────────────────────────────────────────────
    else:
        feature_set = available.copy()

        for g in cfg.EXCLUDE_GROUPS:
            if g not in group_map:
                audit["warnings"].append(
                    f"[WARNING] EXCLUDE_GROUPS: '{g}' 는 feature_group.json에 없는 그룹명입니다."
                )
                continue
            cols_in_json = set(group_map[g])
            cols_removed = [c for c in cols_in_json if c in feature_set]
            cols_not_in_data = [c for c in cols_in_json if c not in available]
            feature_set -= cols_in_json
            audit["excluded_groups"][g] = {
                "removed": cols_removed,
                "not_in_data": cols_not_in_data,
            }

        for c in cfg.EXCLUDE_COLS:
            if c not in available:
                audit["warnings"].append(
                    f"[WARNING] EXCLUDE_COLS: '{c}' 는 데이터에 없는 컬럼입니다."
                )
            else:
                feature_set.discard(c)
                audit["excluded_cols"].append(c)

    # group json에 있지만 데이터에 없는 컬럼 추적
    audit["missing_in_data"] = sorted(all_grouped - available)

    final = sorted(feature_set)
    audit["final_feature_count"] = len(final)

    return final, audit


# ════════════════════════════════════════════════════════════
#  3. CV TRAINING
# ════════════════════════════════════════════════════════════

def run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg,
) -> dict:
    """
    Stratified K-Fold CV → PR-AUC 통계 + test 평가 반환.
    Returns dict with keys: cv_scores, cv_mean, cv_std, test_prauc,
                            oof_pred, val_pred, models
    """
    skf = StratifiedKFold(
        n_splits=cfg.CV_N_SPLITS,
        shuffle=cfg.CV_SHUFFLE,
        random_state=cfg.CV_SEED,
    )

    cv_scores: list[float] = []
    oof_pred  = np.zeros(len(X_train))
    models: list[LGBMClassifier] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = LGBMClassifier(**cfg.LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(30, verbose=False),
            ],
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        fold_prauc = average_precision_score(y_val, val_prob)
        cv_scores.append(fold_prauc)
        oof_pred[val_idx] = val_prob
        models.append(model)

        print(f"  Fold {fold}: PR-AUC = {fold_prauc:.5f}")

    # ── 전체 train으로 단일 모델 재학습 → validation 평가 ──────────────
    print("\n  전체 train 재학습 중 (validation 평가용)...")
    final_model = LGBMClassifier(**cfg.LGBM_PARAMS)
    final_model.fit(X_train, y_train)
    final_val_pred = final_model.predict_proba(X_test)[:, 1]
    val_prauc = average_precision_score(y_test, final_val_pred)

    return {
        "cv_scores":   cv_scores,
        "cv_mean":     float(np.mean(cv_scores)),
        "cv_std":      float(np.std(cv_scores)),
        "val_prauc":  val_prauc,
        "oof_pred":    oof_pred,
        "val_pred":   final_val_pred,
        "models":      models,
        "final_model": final_model,
    }


# ════════════════════════════════════════════════════════════
#  4. IMPORTANCE: GAIN
# ════════════════════════════════════════════════════════════

def plot_gain_importance(models: list[LGBMClassifier], features: list[str], top_n: int = 30):
    """앙상블 모델들의 Gain 기반 평균 중요도 시각화."""
    gain_sum = np.zeros(len(features))
    for m in models:
        imp = m.booster_.feature_importance(importance_type="gain")
        gain_sum += imp
    gain_mean = gain_sum / len(models)

    df_imp = (
        pd.DataFrame({"feature": features, "gain": gain_mean})
        .sort_values("gain", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(df_imp)))
    ax.barh(df_imp["feature"][::-1], df_imp["gain"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean Gain (CV 앙상블)")
    ax.set_title(f"Gain 기반 Feature 중요도 (Top {top_n})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_imp


# ════════════════════════════════════════════════════════════
#  5. IMPORTANCE: SHAP
# ════════════════════════════════════════════════════════════

def plot_shap_importance(
    models: list[LGBMClassifier],
    X_train: pd.DataFrame,
    top_n: int = 30,
    sample_n: Optional[int] = 5000,
):
    """앙상블 모델들의 평균 SHAP 절댓값 → beeswarm 플롯."""
    X_sample = X_train.sample(min(sample_n, len(X_train)), random_state=42) \
        if sample_n else X_train.copy()

    # float32 캐스팅: LightGBM+SHAP beeswarm 회색 버그 fix
    num_cols = X_sample.select_dtypes(include=[np.number]).columns
    X_sample = X_sample.copy()
    X_sample[num_cols] = X_sample[num_cols].astype("float32")

    shap_vals_list = []
    for m in models:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_sample)
        # LightGBM binary → sv shape: (n, p) or list[(n,p),(n,p)]
        if isinstance(sv, list):
            sv = sv[1]
        shap_vals_list.append(sv)

    shap_mean = np.mean(shap_vals_list, axis=0)   # (n_sample, n_features)

    # Top N 선별
    mean_abs = np.abs(shap_mean).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    X_plot   = X_sample.iloc[:, top_idx]
    sv_plot  = shap_mean[:, top_idx]

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    shap.summary_plot(sv_plot, X_plot, show=False, max_display=top_n)
    plt.title(f"SHAP Beeswarm (Top {top_n})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ── SHAP 절댓값 bar chart ────────────────────────────────
    df_shap = pd.DataFrame({
        "feature":       X_sample.columns[top_idx],
        "mean_abs_shap": mean_abs[top_idx],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_shap)))
    ax.barh(df_shap["feature"][::-1], df_shap["mean_abs_shap"][::-1], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP 절댓값 Feature 중요도 (Top {top_n})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_shap


# ════════════════════════════════════════════════════════════
#  5b. SHAP INTERACTION MATRIX
# ════════════════════════════════════════════════════════════

def plot_shap_interaction(
    models: list[LGBMClassifier],
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    interaction_cols: Optional[list[str]] = None,
    top_n: int = 15,
    sample_n: int = 200,
):
    """
    SHAP Interaction Values 절댓값 평균 매트릭스 시각화.

    Parameters
    ----------
    interaction_cols : 직접 지정할 feature 리스트.
                       None 이면 mean|SHAP| 주대각 상위 top_n 자동 선별.
    y_train         : 레이블 시리즈. 제공 시 실패/정상 비율 유지 층화 샘플링 수행.
    top_n           : interaction_cols 미지정 시 상위 feature 수.
    sample_n        : 상호작용 계산에 사용할 샘플 수.

    주대각 = 직접 효과, 비주대각 = 상호작용 강도
    """
    # ── 층화 샘플링 ───────────────────────────────────────────
    n = min(sample_n, len(X_train))
    if y_train is not None and len(y_train.unique()) > 1:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=42)
        idx, _ = next(sss.split(X_train, y_train))
        X_s = X_train.iloc[idx].copy()
        print(f"  [Stratified] sample={n}  "
              f"pos={int(y_train.iloc[idx].sum())} ({y_train.iloc[idx].mean()*100:.1f}%)")
    else:
        X_s = X_train.sample(n, random_state=42).copy()
        print(f"  [Random] sample={n}")

    num_cols = X_s.select_dtypes(include=[np.number]).columns
    X_s[num_cols] = X_s[num_cols].astype("float32")

    # ── 상호작용 값 수집 (fold 모델 평균) ──────────────────────
    sv_inter_list = []
    for m in models:
        explainer = shap.TreeExplainer(m)
        sv_inter  = explainer.shap_interaction_values(X_s)   # (n, p, p)
        if isinstance(sv_inter, list):
            sv_inter = sv_inter[1]
        sv_inter_list.append(sv_inter)

    mean_inter = np.mean(sv_inter_list, axis=0)   # (n, p, p)
    abs_mean   = np.abs(mean_inter).mean(axis=0)  # (p, p)

    # ── feature 선별 ───────────────────────────────────────────
    all_cols = list(X_s.columns)

    if interaction_cols is not None:
        # 직접 지정: 데이터에 없는 컬럼 경고 후 제외
        missing_ic = [c for c in interaction_cols if c not in all_cols]
        if missing_ic:
            print(f"  [WARN] interaction_cols 중 데이터 없는 컬럼 무시: {missing_ic}")
        sel_cols = [c for c in interaction_cols if c in all_cols]
        if not sel_cols:
            raise ValueError("interaction_cols 에 유효한 컬럼이 없습니다.")
        sel_idx      = [all_cols.index(c) for c in sel_cols]
        title_suffix = f"user-defined ({len(sel_cols)} cols)"
    else:
        # 자동: 주대각(mean |SHAP|) 상위 top_n
        diag_vals = np.diag(abs_mean)
        sel_idx   = np.argsort(diag_vals)[::-1][:top_n].tolist()
        sel_cols  = [all_cols[i] for i in sel_idx]
        title_suffix = f"Top {top_n} by mean|SHAP|"

    feat_names = np.array(sel_cols)
    mat        = abs_mean[np.ix_(sel_idx, sel_idx)]
    n_feat     = len(feat_names)

    # ── 플롯 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_feat * 0.7), max(6, n_feat * 0.7)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="mean |SHAP interaction|")

    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_title(
        f"SHAP Interaction Matrix ({title_suffix}, sample={n})",
        fontsize=12, fontweight="bold"
    )

    # 셀 값 텍스트
    vmax = mat.max()
    for i in range(n_feat):
        for j in range(n_feat):
            ax.text(j, i, f"{mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=max(5, 8 - n_feat // 4),
                    color="white" if vmax > 0 and mat[i, j] > vmax * 0.6 else "black")

    plt.tight_layout()
    plt.show()

    df_inter = pd.DataFrame(mat, index=feat_names, columns=feat_names)
    return df_inter


# ════════════════════════════════════════════════════════════
#  5c. CONFUSION MATRIX
# ════════════════════════════════════════════════════════════

def plot_confusion(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    title: str = "Confusion Matrix (Test)",
):
    """
    이진 분류 혼동행렬 시각화.
    수치(count) + 비율(%) 동시 표시.
    """
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    labels = ["Normal (0)", "Failure (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{title}  (thr={threshold})", fontsize=12, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    plt.tight_layout()
    plt.show()

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))

    tn, fp, fn, tp = cm.ravel()
    print(f"  Precision (Failure) : {tp/(tp+fp):.4f}")
    print(f"  Recall    (Failure) : {tp/(tp+fn):.4f}")
    print(f"  Specificity         : {tn/(tn+fp):.4f}")

    return cm


# ════════════════════════════════════════════════════════════
#  6. AUDIT REPORT  (설정 투명성)
# ════════════════════════════════════════════════════════════

def print_audit(audit: dict):
    """feature 결정 과정 전체를 콘솔에 출력."""
    SEP = "=" * 65

    print(SEP)
    print(f"  FEATURE AUDIT  │  MODE = {audit['mode'].upper()}")
    print(SEP)

    # ── 경고 먼저 ─────────────────────────────────────────────
    if audit["warnings"]:
        print("\n⚠️  CONFIG 경고 (아래 항목을 확인하세요)")
        for w in audit["warnings"]:
            print(f"  {w}")

    # ── 포함 그룹 ─────────────────────────────────────────────
    if audit["included_groups"]:
        print("\n✅  INCLUDE_GROUPS (포함된 그룹)")
        for g, cols in audit["included_groups"].items():
            print(f"  [{g}]  {len(cols)} cols")

    if audit["included_cols"]:
        print(f"\n✅  INCLUDE_COLS (개별 포함): {audit['included_cols']}")

    # ── 제외 그룹 ─────────────────────────────────────────────
    if audit["excluded_groups"]:
        print("\n🗑️   EXCLUDE_GROUPS (제외된 그룹)")
        for g, info in audit["excluded_groups"].items():
            not_in = info["not_in_data"]
            removed = info["removed"]
            print(f"  [{g}]  {len(removed)} cols 제거됨", end="")
            if not_in:
                print(f"  ⚠ {len(not_in)} cols 데이터 없음: {not_in[:5]}{'...' if len(not_in)>5 else ''}", end="")
            print()

    if audit["excluded_cols"]:
        print(f"\n🗑️   EXCLUDE_COLS (개별 제거): {audit['excluded_cols']}")

    # ── 요약 ─────────────────────────────────────────────────
    print(f"\n📊  데이터 내 전체 컬럼: {audit['total_in_data']}")
    print(f"📊  그룹 JSON 내 정의된 컬럼: {audit['total_in_group_json']}")

    if audit["missing_in_data"]:
        print(f"⚠️   그룹 JSON에 있지만 데이터 없음: {len(audit['missing_in_data'])} cols")
        if len(audit["missing_in_data"]) <= 10:
            print(f"    {audit['missing_in_data']}")

    print(f"\n✅  최종 사용 feature 수: {audit['final_feature_count']}")
    print(SEP)


# ════════════════════════════════════════════════════════════
#  7. RESULT SUMMARY
# ════════════════════════════════════════════════════════════

def print_results(result: dict, cfg):
    """CV 및 test 결과 요약 출력."""
    SEP = "=" * 65

    print(SEP)
    print("  RESULT SUMMARY")
    print(SEP)
    print(f"\n  CV PR-AUC  : {result['cv_mean']:.5f}")
    print(f"  CV std  : {result['cv_std']:.5f}")
    print(f"  CV scores  : {[round(s, 5) for s in result['cv_scores']]}")
    print(f"\n  VAL PR-AUC: {result['val_prauc']:.5f}")
    print(SEP)


# ════════════════════════════════════════════════════════════
#  8. MAIN PIPELINE  (노트북에서 한 셀로 호출)
# ════════════════════════════════════════════════════════════

def run_pipeline(
    cfg,
    *,
    show_gain: bool = True,
    show_shap: bool = True,
    show_interaction: bool = False,
    show_confusion: bool = True,
    confusion_threshold: float = 0.5,
    interaction_cols: Optional[list[str]] = None,
    interaction_top_n: int = 15,
    interaction_sample_n: int = 200,
) -> dict:
    """
    전체 파이프라인 실행.

    Parameters
    ----------
    cfg                  : config 모듈
    show_gain            : Gain 중요도 플롯
    show_shap            : SHAP beeswarm + bar 플롯
    show_interaction     : SHAP 상호작용 매트릭스 (기본 Off)
    show_confusion       : test 혼동행렬
    confusion_threshold  : 혼동행렬 분류 임계값
    interaction_cols     : 상호작용 매트릭스에 사용할 feature 리스트.
                           None 이면 mean|SHAP| 상위 interaction_top_n 자동 선별.
    interaction_top_n    : interaction_cols=None 일 때 상위 feature 수
    interaction_sample_n : 상호작용 계산용 층화 샘플 수

    Returns
    -------
    dict with keys:
        features, audit, cv_result, df_gain, df_shap, df_interaction, cm
    """
    # ── 0. config validation ──────────────────────────────────
    issues = validate_config(cfg)
    if issues:
        print("\n" + "!" * 65)
        print("  CONFIG 오류 발견 — 실행을 중단합니다")
        print("!" * 65)
        for msg in issues:
            print(f"  {msg}")
        print("!" * 65 + "\n")
        raise RuntimeError("CONFIG 오류로 파이프라인 실행 중단. 위 메시지를 확인하세요.")

    # ── 1. 데이터 로드 ────────────────────────────────────────
    print("📂  데이터 로드 중...")
    df_train = pd.read_parquet(cfg.TRAIN_PATH)
    df_test  = pd.read_parquet(cfg.TEST_PATH)

    # ── 메타 컬럼 자동 제거 (serial_number, date) ────────────
    _DROP_META = ["serial_number", "date"]
    for _df_name, _df in [("train", df_train), ("test", df_test)]:
        _found = [c for c in _DROP_META if c in _df.columns]
        if _found:
            _df.drop(columns=_found, inplace=True)

    # ── 2. feature 결정 ───────────────────────────────────────
    features, audit = resolve_features(cfg, df_train.columns.tolist())
    print_audit(audit)

    if audit["warnings"]:
        print("\n⚠️  경고가 있습니다. audit 결과를 확인하고 config를 수정하세요.")

    if len(features) == 0:
        raise RuntimeError(
            "최종 feature 수가 0입니다. "
            "INCLUDE_GROUPS / INCLUDE_COLS (add 모드) 또는 "
            "EXCLUDE_GROUPS / EXCLUDE_COLS (drop 모드) 설정을 확인하세요."
        )

    X_train = df_train[features]
    y_train = df_train[cfg.TARGET_COL]
    X_test  = df_test[features]
    y_test  = df_test[cfg.TARGET_COL]

    print(f"\n🏋️  학습 시작  │  features={len(features)}, "
          f"train={len(X_train):,}, test={len(X_test):,}")
    print(f"   pos rate: train={y_train.mean():.4f}, test={y_test.mean():.4f}\n")

    # ── 3. CV 학습 ────────────────────────────────────────────
    cv_result = run_cv(X_train, y_train, X_test, y_test, cfg)
    print_results(cv_result, cfg)

    # ── 4. 중요도 ─────────────────────────────────────────────
    df_gain = df_shap = df_interaction = cm = None

    if show_gain:
        print("\n\U0001f4ca  Gain 중요도 계산 중...")
        df_gain = plot_gain_importance(cv_result["models"], features, top_n=cfg.GAIN_TOP_N)

    if show_shap:
        print(f"\n\U0001f50d  SHAP 계산 중 (sample={cfg.SHAP_SAMPLE_N})...")
        df_shap = plot_shap_importance(
            cv_result["models"], X_train,
            top_n=cfg.SHAP_TOP_N,
            sample_n=cfg.SHAP_SAMPLE_N,
        )

    if show_interaction:
        label_str = f"cols={interaction_cols}" if interaction_cols else f"top_n={interaction_top_n}"
        print(f"\n\U0001f9e9  SHAP 상호작용 매트릭스 계산 중 ({label_str}, sample={interaction_sample_n})...")
        df_interaction = plot_shap_interaction(
            cv_result["models"], X_train,
            y_train=y_train,
            interaction_cols=interaction_cols,
            top_n=interaction_top_n,
            sample_n=interaction_sample_n,
        )

    if show_confusion:
        print("\n\U0001f4ca  혼동행렬 (테스트 세트) 출력 중...")
        cm = plot_confusion(
            y_test,
            cv_result["val_pred"],
            threshold=confusion_threshold,
            title="Confusion Matrix",
        )

    return {
        "features":      features,
        "audit":         audit,
        "cv_result":     cv_result,
        "df_gain":       df_gain,
        "df_shap":       df_shap,
        "df_interaction": df_interaction,
        "cm":            cm,
    }
