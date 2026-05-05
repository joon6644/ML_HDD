"""
eval_core.py  ─  임계값 튜닝 + 모델 평가 엔진  (README §7, §8)
──────────────────────────────────────────────────────────────
§7  ThresholdTuner      : val_calib 기반 FPR 제약 최적 임계값 탐색
§8.1 evaluate_row_level : 행 단위 PR-AUC / MCC / F1 평가
§8.2 EntityLevelEvaluator : 개체 단위 롤링 평가 (Hit/Miss/FA + Lead Time)

설정은 config/eval_config.py 에서만.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ── 한글 폰트 ────────────────────────────────────────────────
def _set_korean_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

SEP = "=" * 65


# ════════════════════════════════════════════════════════════
#  §7.  THRESHOLD TUNER
#  val_calib 에서 그리드서치 → FPR 상한 내 Recall 최대화
# ════════════════════════════════════════════════════════════

class ThresholdTuner:
    """
    FPR ≤ max_fpr 제약 하에서 Recall 을 최대화하는 임계값 탐색.

    Parameters
    ----------
    max_fpr   : FPR 상한 (예: 0.01 = 1%)
    n_grid    : 임계값 탐색 격자 수
    """

    def __init__(self, max_fpr: float = 0.01, n_grid: int = 1000):
        self.max_fpr = max_fpr
        self.n_grid  = n_grid
        self._result:   Optional[dict]         = None
        self._grid_df:  Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """
        val_calib 데이터로 최적 임계값 탐색.

        Returns
        -------
        dict: threshold, recall, fpr, precision, f1, mcc, pr_auc
        """
        y_true = np.asarray(y_true)
        n_pos  = int(y_true.sum())
        n_neg  = int((y_true == 0).sum())

        thresholds = np.linspace(0.0, 1.0, self.n_grid + 1)

        records = []
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            recall    = tp / n_pos if n_pos > 0 else 0.0
            fpr       = fp / n_neg if n_neg > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            records.append(dict(threshold=thr, recall=recall, fpr=fpr,
                                precision=precision, f1=f1,
                                tp=tp, fp=fp, fn=fn, tn=tn))

        self._grid_df = pd.DataFrame(records)

        valid = self._grid_df[self._grid_df["fpr"] <= self.max_fpr]
        if len(valid) == 0:
            print(f"⚠️  FPR ≤ {self.max_fpr:.1%} 를 만족하는 임계값 없음 → 최소 FPR 임계값 선택")
            best_row = self._grid_df.loc[self._grid_df["fpr"].idxmin()]
        else:
            best_row = valid.loc[valid["recall"].idxmax()]

        best_thr  = float(best_row["threshold"])
        y_pred_b  = (y_prob >= best_thr).astype(int)
        mcc       = matthews_corrcoef(y_true, y_pred_b)
        pr_auc    = average_precision_score(y_true, y_prob)

        self._result = dict(
            threshold         = best_thr,
            recall            = float(best_row["recall"]),
            fpr               = float(best_row["fpr"]),
            precision         = float(best_row["precision"]),
            f1                = float(best_row["f1"]),
            mcc               = float(mcc),
            pr_auc            = float(pr_auc),
            max_fpr_constraint= self.max_fpr,
            n_pos=n_pos, n_neg=n_neg,
        )
        self._print_result()
        return self._result

    # ----------------------------------------------------------
    def _print_result(self):
        r = self._result
        print(SEP)
        print(f"  THRESHOLD TUNING  (FPR 상한 = {r['max_fpr_constraint']:.1%})")
        print(SEP)
        print(f"  최적 임계값   : {r['threshold']:.4f}")
        print(f"  Recall (탐지율): {r['recall']:.4f}  ({r['recall']*100:.2f}%)")
        print(f"  FPR   (오탐율) : {r['fpr']:.4f}  ({r['fpr']*100:.2f}%)")
        print(f"  Precision      : {r['precision']:.4f}")
        print(f"  F1-score       : {r['f1']:.4f}")
        print(f"  MCC            : {r['mcc']:.4f}")
        print(f"  PR-AUC         : {r['pr_auc']:.5f}")
        print(f"\n  → FPR {r['fpr']*100:.2f}%에서 Recall {r['recall']*100:.2f}%")
        print(SEP)

    # ----------------------------------------------------------
    def plot(self):
        """Recall / FPR vs Threshold 커브 + FPR-Recall Trade-off 플롯."""
        if self._grid_df is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")

        gdf     = self._grid_df
        best    = self._result

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── 왼쪽: Recall & FPR vs Threshold
        ax = axes[0]
        ax.plot(gdf["threshold"], gdf["recall"], color="steelblue", lw=2, label="Recall (탐지율)")
        ax.plot(gdf["threshold"], gdf["fpr"],    color="tomato",    lw=2, label="FPR (오탐율)")
        ax.axhline(self.max_fpr, color="tomato",  ls="--", alpha=0.6, label=f"FPR 상한 {self.max_fpr:.1%}")
        ax.axvline(best["threshold"], color="green", ls=":", lw=2,
                   label=f"최적 임계값 {best['threshold']:.4f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
        ax.set_title("Recall & FPR vs Threshold", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        # ── 오른쪽: FPR-Recall Trade-off
        ax = axes[1]
        ax.plot(gdf["fpr"], gdf["recall"], color="purple", lw=2)
        ax.axvline(self.max_fpr, color="tomato", ls="--", alpha=0.6,
                   label=f"FPR 상한 {self.max_fpr:.1%}")
        ax.scatter([best["fpr"]], [best["recall"]], color="green", s=120, zorder=5,
                   label=f"최적점 Recall={best['recall']:.3f}")
        ax.set_xlabel("FPR (오탐율)"); ax.set_ylabel("Recall (탐지율)")
        ax.set_title("Recall vs FPR Trade-off", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        plt.suptitle("임계값 튜닝 결과 (val_calib 기반)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    @property
    def best_threshold(self) -> float:
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        return self._result["threshold"]


# ════════════════════════════════════════════════════════════
#  §8.1  ROW-LEVEL EVALUATION
#  PR-AUC / MCC / F1 / Precision / Recall / FPR
# ════════════════════════════════════════════════════════════

def evaluate_row_level(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str = "Test Set",
) -> dict:
    """
    README §8.1 행 단위 점수 평가.

    평가 지표: PR-AUC, MCC, F1, Precision, Recall, FPR
    시각화  : PR Curve + Confusion Matrix
    """
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    mcc    = matthews_corrcoef(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)

    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    result = dict(pr_auc=pr_auc, mcc=mcc, f1=f1, precision=float(prec),
                  recall=float(rec), fpr=fpr, threshold=threshold,
                  n_pos=int(y_true.sum()), n_neg=int((y_true==0).sum()),
                  tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    print(SEP)
    print(f"  ROW-LEVEL EVALUATION  ({title})")
    print(SEP)
    print(f"  임계값    : {threshold:.4f}")
    print(f"  PR-AUC    : {pr_auc:.5f}")
    print(f"  MCC       : {mcc:.5f}")
    print(f"  F1-score  : {f1:.5f}")
    print(f"  Precision : {prec:.5f}")
    print(f"  Recall    : {rec:.5f}  ({rec*100:.2f}%)")
    print(f"  FPR       : {fpr:.5f}  ({fpr*100:.2f}%)")
    print(f"\n  → FPR {fpr*100:.2f}%에서 Recall {rec*100:.2f}%")
    print(SEP)

    # ── 시각화 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PR Curve
    ax = axes[0]
    pc, rc, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(rc, pc, color="steelblue", lw=2, label=f"PR-AUC={pr_auc:.4f}")
    ax.scatter([rec], [prec], color="tomato", s=120, zorder=5,
               label=f"운영점 thr={threshold:.3f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", ls="--", alpha=0.5, label=f"Baseline={baseline:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # Confusion Matrix
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["Normal (0)", "Failure (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(labels)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (thr={threshold:.4f})", fontweight="bold")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n({v/total*100:.2f}%)", ha="center", va="center",
                    fontsize=11, color="white" if v > cm.max()*0.5 else "black")

    plt.suptitle(f"행 단위 평가 ({title})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return result


# ════════════════════════════════════════════════════════════
#  §8.2  ENTITY-LEVEL ROLLING EVALUATION
#  개체 단위: Hit Rate / Miss Rate / FA Rate / Lead Time
# ════════════════════════════════════════════════════════════

class EntityLevelEvaluator:
    """
    README §8.2 실무형 롤링 평가.

    평가 단위: serial_number (개체)
    - 고장 개체: 적어도 하나의 failure=1 행이 있는 시리얼
    - 정상 개체: 모든 failure=0

    핵심 지표
    ─────────
    Hit Rate  : 고장 개체 중 알람이 발생한 비율
    Miss Rate : 고장 개체 중 알람이 발생하지 않은 비율  (= 1 - Hit Rate)
    FA Rate   : 정상 개체 중 알람이 발생한 비율 (오탐율)
    Lead Time : 탐지된 고장 개체에서 첫 알람~D-1 까지의 일수 평균
                (탐지 못한 개체는 제외)

    Lead Time 계산 기준
    ───────────────────
    마지막 관측일(≈ D-1) - 첫 알람 발생일
    값이 클수록 더 일찍 경보를 발령한 것.

    Parameters
    ----------
    serial_col : serial_number 컬럼명
    date_col   : date 컬럼명
    target_col : failure 컬럼명
    """

    def __init__(
        self,
        serial_col: str = "serial_number",
        date_col:   str = "date",
        target_col: str = "failure",
    ):
        self.serial_col = serial_col
        self.date_col   = date_col
        self.target_col = target_col

    # ----------------------------------------------------------
    def evaluate(
        self,
        df: pd.DataFrame,
        y_prob: np.ndarray,
        threshold: float,
    ) -> dict:
        """
        Parameters
        ----------
        df        : test 데이터 (serial_number, date, failure 컬럼 포함)
                    인덱스가 y_prob 와 정렬되어 있어야 함.
        y_prob    : 앙상블 예측 확률 (df 와 동일 순서)
        threshold : 임계값 (ThresholdTuner.best_threshold 사용 권장)

        Returns
        -------
        dict: hit_rate, miss_rate, fa_rate, mean_lead_time, entity_df
        """
        df = df[[self.serial_col, self.date_col, self.target_col]].copy()
        df["_prob"]  = np.asarray(y_prob)
        df["_alarm"] = (df["_prob"] >= threshold).astype(int)
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        records = []
        for sn, grp in df.groupby(self.serial_col):
            grp = grp.sort_values(self.date_col)

            is_failure = int(grp[self.target_col].max())  # 1 if any row == 1
            alarm_rows = grp[grp["_alarm"] == 1]
            has_alarm  = len(alarm_rows) > 0

            last_date  = grp[self.date_col].max()
            first_alarm_date = alarm_rows[self.date_col].min() if has_alarm else pd.NaT

            # Lead Time: last_date - first_alarm_date (일수)
            lead_time = (
                (last_date - first_alarm_date).days
                if (has_alarm and is_failure)
                else np.nan
            )

            records.append(dict(
                serial_number = sn,
                is_failure    = is_failure,
                has_alarm     = int(has_alarm),
                first_alarm_date = first_alarm_date,
                last_date     = last_date,
                lead_time_days= lead_time,
            ))

        entity_df = pd.DataFrame(records)

        # ── 지표 계산 ─────────────────────────────────────────
        fail_ent   = entity_df[entity_df["is_failure"] == 1]
        normal_ent = entity_df[entity_df["is_failure"] == 0]

        n_fail   = len(fail_ent)
        n_normal = len(normal_ent)

        hit_count  = int(fail_ent["has_alarm"].sum())
        miss_count = n_fail - hit_count
        fa_count   = int(normal_ent["has_alarm"].sum())

        hit_rate  = hit_count  / n_fail   if n_fail   > 0 else 0.0
        miss_rate = miss_count / n_fail   if n_fail   > 0 else 0.0
        fa_rate   = fa_count   / n_normal if n_normal > 0 else 0.0

        hit_lead_times = entity_df[
            (entity_df["is_failure"] == 1) & (entity_df["has_alarm"] == 1)
        ]["lead_time_days"]
        mean_lead_time = float(hit_lead_times.mean()) if len(hit_lead_times) > 0 else np.nan

        result = dict(
            n_failure_entities = n_fail,
            n_normal_entities  = n_normal,
            hit_count    = hit_count,
            miss_count   = miss_count,
            fa_count     = fa_count,
            hit_rate     = hit_rate,
            miss_rate    = miss_rate,
            fa_rate      = fa_rate,
            mean_lead_time_days = mean_lead_time,
            entity_df    = entity_df,
        )

        self._print_result(result)
        return result

    # ----------------------------------------------------------
    @staticmethod
    def _print_result(r: dict):
        print(SEP)
        print("  ENTITY-LEVEL ROLLING EVALUATION")
        print(SEP)
        print(f"  고장 개체 수 : {r['n_failure_entities']:,}")
        print(f"  정상 개체 수 : {r['n_normal_entities']:,}")
        print()
        print(f"  고장 탐지율 (Hit Rate)  : {r['hit_rate']:.4f}  "
              f"({r['hit_count']}/{r['n_failure_entities']})")
        print(f"  미탐율      (Miss Rate) : {r['miss_rate']:.4f}  "
              f"({r['miss_count']}/{r['n_failure_entities']})")
        print(f"  오탐율      (FA Rate)   : {r['fa_rate']:.4f}  "
              f"({r['fa_count']}/{r['n_normal_entities']})")
        print()
        if not np.isnan(r["mean_lead_time_days"]):
            print(f"  평균 사전 경보 시간 (Lead Time) : {r['mean_lead_time_days']:.1f}일")
            print("  (탐지 못한 개체 제외)")
        print(SEP)

    # ----------------------------------------------------------
    @staticmethod
    def plot(result: dict):
        """Hit / Miss / FA / Lead Time 시각화."""
        r  = result
        ed = r["entity_df"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── 왼쪽: Hit / Miss / FA 막대
        ax = axes[0]
        categories = ["Hit\n(고장 탐지)", "Miss\n(고장 미탐)", "FA\n(정상 오탐)"]
        counts     = [r["hit_count"], r["miss_count"], r["fa_count"]]
        colors     = ["steelblue", "tomato", "orange"]
        bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("개체 수")
        ax.set_title("개체 단위 탐지 결과", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # 비율 표시 (이중 y축 대신 텍스트)
        ax.text(0, counts[0]/2, f"{r['hit_rate']*100:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
        ax.text(1, counts[1]/2 if counts[1] > 0 else 0.5,
                f"{r['miss_rate']*100:.1f}%",
                ha="center", va="center", color="white" if counts[1]>0 else "black",
                fontweight="bold")
        ax.text(2, counts[2]/2 if counts[2] > 0 else 0.5,
                f"{r['fa_rate']*100:.1f}%",
                ha="center", va="center", color="white" if counts[2]>0 else "black",
                fontweight="bold")

        # ── 오른쪽: Lead Time 분포
        ax = axes[1]
        hit_lt = ed[(ed["is_failure"]==1) & (ed["has_alarm"]==1)]["lead_time_days"].dropna()
        if len(hit_lt) > 0:
            ax.hist(hit_lt, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(hit_lt.mean(), color="tomato", ls="--", lw=2,
                       label=f"평균 {hit_lt.mean():.1f}일")
            ax.axvline(hit_lt.median(), color="orange", ls=":", lw=2,
                       label=f"중앙값 {hit_lt.median():.1f}일")
            ax.set_xlabel("Lead Time (일)")
            ax.set_ylabel("개체 수")
            ax.set_title("사전 경보 시간 분포 (탐지 성공 개체)", fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "탐지된 고장 개체 없음", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title("사전 경보 시간 분포", fontweight="bold")

        plt.suptitle("실무형 롤링 평가 결과", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════
#  MAIN PIPELINE  (노트북 한 셀 호출용)
# ════════════════════════════════════════════════════════════

def run_evaluation(
    cfg,
    ensemble,               # UnderbaggingEnsemble (train_core.py)
    feature_cols: list[str],
    *,
    show_plots: bool = True,
) -> dict:
    """
    §7 + §8 전체 평가 파이프라인.

    Parameters
    ----------
    cfg           : config/eval_config.py
    ensemble      : 학습 완료된 UnderbaggingEnsemble 객체
    feature_cols  : 모델에 사용된 피처 리스트
    show_plots    : 시각화 여부

    Returns
    -------
    dict: tuner, threshold, row_result, entity_result
    """
    import pandas as pd

    # ── 1. 데이터 로드 ────────────────────────────────────────
    print("📂  평가 데이터 로드 중...")
    df_calib = pd.read_parquet(cfg.VAL_CALIB_PATH)
    df_test  = pd.read_parquet(cfg.TEST_PATH)

    # ── 2. §7 임계값 튜닝 (val_calib 기반) ───────────────────
    print("\n🎯  §7 임계값 튜닝 (val_calib)...")
    y_calib_true = df_calib[cfg.TARGET_COL].values
    y_calib_prob = ensemble.predict_proba(df_calib, feature_cols)

    tuner = ThresholdTuner(max_fpr=cfg.MAX_FPR, n_grid=cfg.THRESHOLD_N_GRID)
    tuner.fit(y_calib_true, y_calib_prob)

    if show_plots:
        tuner.plot()

    threshold = tuner.best_threshold

    # ── 3. §8.1 행 단위 평가 (test) ──────────────────────────
    print("\n📊  §8.1 행 단위 평가 (test)...")
    y_test_true = df_test[cfg.TARGET_COL].values
    y_test_prob = ensemble.predict_proba(df_test, feature_cols)

    row_result = evaluate_row_level(
        y_test_true, y_test_prob, threshold, title="Test Set"
    )

    # ── 4. §8.2 개체 단위 롤링 평가 (test) ──────────────────
    print("\n🔄  §8.2 개체 단위 롤링 평가 (test)...")
    ev = EntityLevelEvaluator(
        serial_col=cfg.SERIAL_COL,
        date_col=cfg.DATE_COL,
        target_col=cfg.TARGET_COL,
    )
    entity_result = ev.evaluate(df_test, y_test_prob, threshold)

    if show_plots:
        EntityLevelEvaluator.plot(entity_result)

    return dict(
        tuner         = tuner,
        threshold     = threshold,
        row_result    = row_result,
        entity_result = entity_result,
    )
