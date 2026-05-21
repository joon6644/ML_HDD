"""
eval_core.py  ─  임계값 튜닝 + 모델 평가 엔진  (README §7, §8)
──────────────────────────────────────────────────────────────
§7  ThresholdTuner      : val_calib 그리드서치 → FPR별 Recall 표 → FPR–Recall 곡선
§8.1 evaluate_row_level : 행 단위 PR-AUC / MCC / F1 평가
§8.2 EntityLevelEvaluator : 개체 단위 롤링 평가 (Hit/Miss/FA + Lead Time)

설정은 config/threshold_config.py, config/final_eval_config.py 에서만.
"""

from __future__ import annotations

import time
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


def _log(msg: str, *, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def _log_step(step: str, detail: str = "") -> None:
    line = f"[§7] {step}"
    if detail:
        line += f" — {detail}"
    _log(line)


def save_threshold_tuning_csvs(
    tuner: "ThresholdTuner",
    cfg,
    out_dir,
) -> dict[str, str]:
    """§7 튜닝 결과를 CSV로 저장. 반환: {fpr_recall_table, threshold_grid} 경로."""
    from pathlib import Path

    if tuner._result is None or tuner._grid_df is None:
        raise RuntimeError("ThresholdTuner.fit() 이후에만 CSV 저장 가능합니다.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_name = getattr(cfg, "FPR_RECALL_TABLE_CSV", "fpr_recall_table.csv")
    grid_name = getattr(cfg, "THRESHOLD_GRID_CSV", "threshold_grid.csv")

    table_rows = tuner._result["fpr_recall_table"]
    df_table = pd.DataFrame(table_rows)
    df_table.insert(0, "fpr_cap_pct", (df_table["fpr_cap"] * 100).round(3))
    df_table["fpr_actual_pct"] = (df_table["fpr_actual"] * 100).round(4)
    df_table["recall_pct"] = (df_table["recall"] * 100).round(4)

    path_table = out_dir / table_name
    path_grid = out_dir / grid_name
    df_table.to_csv(path_table, index=False, encoding="utf-8-sig")
    tuner._grid_df.to_csv(path_grid, index=False, encoding="utf-8-sig")

    return {
        "fpr_recall_table": str(path_table),
        "threshold_grid": str(path_grid),
    }


# ════════════════════════════════════════════════════════════
#  §7.  THRESHOLD TUNER
#  val_calib 에서 그리드서치 → FPR 상한 내 Recall 최대화
# ════════════════════════════════════════════════════════════

def summarize_recall_at_fpr_levels(
    grid_df: pd.DataFrame,
    fpr_levels: list[float],
    *,
    operating_fpr: float | None = None,
) -> list[dict]:
    """각 FPR 상한에서 Recall을 최대화한 운영점 요약 (README §7 간판 표)."""
    rows = []
    for cap in fpr_levels:
        valid = grid_df[grid_df["fpr"] <= cap]
        if len(valid) == 0:
            best_row = grid_df.loc[grid_df["fpr"].idxmin()]
        else:
            best_row = valid.loc[valid["recall"].idxmax()]

        rows.append({
            "fpr_cap": cap,
            "recall": float(best_row["recall"]),
            "fpr_actual": float(best_row["fpr"]),
            "precision": float(best_row["precision"]),
            "f1": float(best_row["f1"]),
            "threshold": float(best_row["threshold"]),
            "is_save_point": operating_fpr is not None and cap == operating_fpr,
        })
    return rows


class ThresholdTuner:
    """
    val_calib 그리드서치 후 FPR 상한별 Recall 표를 출력하고 FPR–Recall 곡선을 그림.

    Parameters
    ----------
    fpr_levels            : README §7 간판 표에 쓸 FPR 상한 목록 (필수)
    n_grid                : 임계값 탐색 격자 수
    save_operating_fpr_cap: 08 저장용으로 표에서 선택할 FPR 상한 (예: 0.01). None이면 미선택
    """

    def __init__(
        self,
        fpr_levels: list[float],
        n_grid: int = 1000,
        save_operating_fpr_cap: float | None = None,
    ):
        if not fpr_levels:
            raise ValueError("fpr_levels must be a non-empty list (e.g. [0.001, 0.005, 0.01, 0.05])")
        self.fpr_levels = list(fpr_levels)
        self.n_grid = n_grid
        self.save_operating_fpr_cap = save_operating_fpr_cap
        self._result: Optional[dict] = None
        self._grid_df: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """
        val_calib 데이터로 최적 임계값 탐색.

        Returns
        -------
        dict: threshold, recall, fpr, precision, f1, pr_auc, fpr_recall_table
        """
        y_true = np.asarray(y_true)
        n_pos  = int(y_true.sum())
        n_neg  = int((y_true == 0).sum())
        n_rows = len(y_true)

        thresholds = np.linspace(0.0, 1.0, self.n_grid + 1)
        n_thr = len(thresholds)
        _log(
            f"임계값 그리드 서치: {n_thr:,}개 후보 × {n_rows:,}행 "
            f"(FPR 표: {[f'{c*100:.1f}%' for c in self.fpr_levels]})",
            indent=1,
        )
        t_grid = time.perf_counter()

        records = []
        for i, thr in enumerate(thresholds):
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

            if i == 0 or i == n_thr - 1 or (i + 1) % 250 == 0:
                elapsed = time.perf_counter() - t_grid
                pct = 100.0 * (i + 1) / n_thr
                _log(f"... {i + 1:,}/{n_thr:,} ({pct:.0f}%) — {elapsed:.1f}s 경과", indent=2)

        _log(f"그리드 서치 완료 ({time.perf_counter() - t_grid:.1f}s)", indent=1)

        self._grid_df = pd.DataFrame(records)
        _log("PR-AUC 계산 중...", indent=1)
        pr_auc = average_precision_score(y_true, y_prob)
        _log("FPR별 Recall 표 집계 중...", indent=1)

        fpr_recall_table = summarize_recall_at_fpr_levels(
            self._grid_df,
            self.fpr_levels,
            operating_fpr=self.save_operating_fpr_cap,
        )

        save_row = None
        if self.save_operating_fpr_cap is not None:
            save_row = next(
                (r for r in fpr_recall_table if r["fpr_cap"] == self.save_operating_fpr_cap),
                None,
            )
            if save_row is None:
                raise ValueError(
                    f"save_operating_fpr_cap={self.save_operating_fpr_cap} not in fpr_levels"
                )

        self._result = dict(
            threshold=float(save_row["threshold"]) if save_row else None,
            recall=float(save_row["recall"]) if save_row else None,
            fpr=float(save_row["fpr_actual"]) if save_row else None,
            precision=float(save_row["precision"]) if save_row else None,
            f1=float(save_row["f1"]) if save_row else None,
            pr_auc=float(pr_auc),
            save_operating_fpr_cap=self.save_operating_fpr_cap,
            fpr_recall_table=fpr_recall_table,
            n_pos=n_pos,
            n_neg=n_neg,
        )

        self.print_table()
        return self._result

    # ----------------------------------------------------------
    def print_table(self) -> None:
        """README §7 간판 표 (FPR 상한별 Recall) — 그래프보다 먼저 출력."""
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        r = self._result
        table = r["fpr_recall_table"]

        print(SEP)
        print("  THRESHOLD TUNING — val_calib (README §7)")
        print(SEP)
        print(f"  PR-AUC (확률 품질, 임계값 무관): {r['pr_auc']:.5f}")
        print(f"  표본: 고장 {r['n_pos']:,}행 / 정상 {r['n_neg']:,}행\n")

        print("  [FPR 상한별 최대 Recall — 간판 표]")
        print(f"  {'FPR 상한':>10}  {'Recall':>10}  {'실제 FPR':>10}  {'threshold':>10}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for row in table:
            cap_pct = row["fpr_cap"] * 100
            rec_pct = row["recall"] * 100
            fpr_pct = row["fpr_actual"] * 100
            tag = " ← 08 저장" if row.get("is_save_point") else ""
            print(
                f"  {cap_pct:>9.1f}%  {rec_pct:>9.2f}%  {fpr_pct:>9.2f}%  {row['threshold']:>10.4f}{tag}"
            )
            print(f"      → FPR {fpr_pct:.2f}%에서 Recall {rec_pct:.2f}%")

        if r.get("threshold") is not None and r.get("save_operating_fpr_cap") is not None:
            cap = r["save_operating_fpr_cap"] * 100
            print(
                f"\n  [08 저장용] FPR 상한 {cap:.1f}% 행 → threshold = {r['threshold']:.4f}"
            )
        print(SEP)

    # ----------------------------------------------------------
    def plot(self):
        """README §7: FPR(x)–Recall(y) 곡선 (표의 각 운영점 표시). fit()·print_table() 이후 호출."""
        if self._grid_df is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")

        gdf = self._grid_df
        table = self._result["fpr_recall_table"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(gdf["fpr"], gdf["recall"], color="purple", lw=2, label="FPR–Recall curve")
        for row in table:
            ax.scatter(
                row["fpr_actual"],
                row["recall"],
                s=100,
                zorder=5,
                label=f"FPR≤{row['fpr_cap']*100:.1f}%",
            )
            ax.annotate(
                f"{row['fpr_cap']*100:.1f}%\nR={row['recall']:.2f}",
                (row["fpr_actual"], row["recall"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax.set_xlabel("FPR (오탐율)")
        ax.set_ylabel("Recall (탐지율, TPR)")
        ax.set_title("Recall vs FPR (val_calib)", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @property
    def best_threshold(self) -> float:
        if self._result is None:
            raise RuntimeError("fit() 을 먼저 호출하세요.")
        thr = self._result.get("threshold")
        if thr is None:
            raise RuntimeError(
                "저장용 threshold 없음. threshold_config.SAVE_OPERATING_FPR_CAP 을 설정하세요."
            )
        return float(thr)


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


class SavedEnsemble:
    """Inference wrapper for a saved underbagging ensemble."""

    def __init__(self, models):
        self.models = models

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        n_rows = len(df)
        n_models = len(self.models)
        probs = []
        for i, m in enumerate(self.models):
            if verbose:
                _log(f"모델 {i + 1}/{n_models} 예측 중 ({n_rows:,}행)...", indent=2)
                t0 = time.perf_counter()
            probs.append(m.predict_proba(df[feature_cols])[:, 1])
            if verbose:
                _log(f"모델 {i + 1}/{n_models} 완료 ({time.perf_counter() - t0:.1f}s)", indent=2)
        if verbose:
            _log("앙상블 soft-voting 평균 계산...", indent=2)
        return np.mean(probs, axis=0)


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _summarize_dataset(df: pd.DataFrame, cfg, *, label: str, path: str) -> dict:
    n_rows = len(df)
    n_pos = int(df[cfg.TARGET_COL].sum())
    n_neg = int((df[cfg.TARGET_COL] == 0).sum())
    summary = {"path": path, "rows": n_rows, "positives": n_pos, "negatives": n_neg}

    print(f"[{label}]")
    print(f"  path: {path}")
    print(f"  rows: {n_rows:,}")
    print(f"  positives: {n_pos:,}")
    print(f"  negatives: {n_neg:,}")

    if getattr(cfg, "DATE_COL", None) in df.columns:
        dates = pd.to_datetime(df[cfg.DATE_COL])
        summary["date_min"] = dates.min().isoformat()
        summary["date_max"] = dates.max().isoformat()
        print(f"  date_range: {dates.min()} ~ {dates.max()}")

    if getattr(cfg, "SERIAL_COL", None) in df.columns:
        n_entities = int(df[cfg.SERIAL_COL].nunique())
        summary["entities"] = n_entities
        print(f"  entities: {n_entities:,}")

    return summary


def _validate_columns(df: pd.DataFrame, required_cols: list[str], *, dataset_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing {len(missing)} required columns: {missing[:20]}"
        )


def load_saved_ensemble(cfg, *, require_threshold: bool = False) -> dict:
    """Load saved models, feature columns, and optionally threshold metadata."""
    import json
    import joblib
    from pathlib import Path
    from config.path_utils import validate_path_contract

    save_dir = Path(cfg.MODEL_SAVE_DIR)
    feature_cols_path = save_dir / "feature_cols.json"
    threshold_path = save_dir / "best_threshold.json"
    model_paths = sorted(save_dir.glob("subset_*.pkl"))

    required_paths = [
        ("model save directory", "dir", str(save_dir)),
        ("feature columns", "file", str(feature_cols_path)),
    ]
    if require_threshold:
        required_paths.append(("best threshold metadata", "file", str(threshold_path)))
    validate_path_contract(required_paths)

    if not model_paths:
        raise FileNotFoundError(f"No subset model files found: {save_dir / 'subset_*.pkl'}")

    expected_models = getattr(cfg, "N_SUBSETS", None)
    if expected_models is not None and len(model_paths) != expected_models:
        raise ValueError(
            f"Expected {expected_models} subset models, found {len(model_paths)} in {save_dir}"
        )

    with open(feature_cols_path, encoding="utf-8") as f:
        feature_cols = json.load(f)

    threshold_meta = None
    threshold = None
    if require_threshold:
        with open(threshold_path, encoding="utf-8") as f:
            threshold_meta = json.load(f)
        threshold = float(threshold_meta["threshold"])

        if threshold_meta.get("n_features") is not None and threshold_meta["n_features"] != len(feature_cols):
            raise ValueError(
                f"Threshold metadata feature count mismatch: "
                f"{threshold_meta['n_features']} != {len(feature_cols)}"
            )
        if threshold_meta.get("model_files") is not None:
            current_model_files = [p.name for p in model_paths]
            if threshold_meta["model_files"] != current_model_files:
                raise ValueError("Threshold metadata model_files do not match current model files.")

    print("[Model Artifact Contract]")
    print(f"  model_dir: {save_dir}")
    print(f"  feature_cols: {feature_cols_path} ({len(feature_cols)} features)")
    if require_threshold:
        print(f"  threshold_path: {threshold_path}")
        print(f"  threshold: {threshold:.6f}")
        print(f"  threshold_source_val_calib: {threshold_meta.get('val_calib_path')}")
    for p in model_paths:
        print(f"  model: {p.name} ({p.stat().st_size:,} bytes)")

    models = []
    for i, p in enumerate(model_paths):
        _log(f"joblib 로드 {i + 1}/{len(model_paths)}: {p.name}", indent=1)
        t0 = time.perf_counter()
        models.append(joblib.load(p))
        _log(f"  → {time.perf_counter() - t0:.1f}s", indent=2)
    ensemble = SavedEnsemble(models)

    return {
        "save_dir": save_dir,
        "feature_cols_path": feature_cols_path,
        "threshold_path": threshold_path,
        "model_paths": model_paths,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "threshold_meta": threshold_meta,
        "ensemble": ensemble,
    }


def run_threshold_tuning_from_saved_model(
    cfg,
    *,
    show_plot: bool = True,
    save: bool = True,
) -> dict:
    """README section 7: tune threshold on val_calib using a saved ensemble."""
    import json
    from datetime import datetime
    from config.path_utils import validate_path_contract

    from pathlib import Path
    from config.path_utils import val_calib_missing_hint

    pipeline_t0 = time.perf_counter()
    _log_step("시작", "임계값 튜닝 (val_calib)")

    calib_path = Path(cfg.VAL_CALIB_PATH)
    if not calib_path.is_file():
        raise FileNotFoundError(val_calib_missing_hint(calib_path))

    _log_step("1/5", "경로·산출물 확인")
    validate_path_contract(list(getattr(cfg, "REQUIRED_DATA_PATHS", [])))

    _log_step("2/5", "앙상블 모델 로드")
    artifacts = load_saved_ensemble(cfg, require_threshold=False)
    feature_cols = artifacts["feature_cols"]
    ensemble = artifacts["ensemble"]

    _log_step("3/5", f"val_calib parquet 읽기 — {calib_path.name}")
    t_read = time.perf_counter()
    df_calib = pd.read_parquet(cfg.VAL_CALIB_PATH)
    _log(f"로드 완료: {len(df_calib):,}행 ({time.perf_counter() - t_read:.1f}s)", indent=1)
    _validate_columns(df_calib, feature_cols + [cfg.TARGET_COL], dataset_name="val_calib")
    calib_summary = _summarize_dataset(
        df_calib, cfg, label="Calibration Data", path=cfg.VAL_CALIB_PATH
    )

    _log_step(
        "4/5",
        f"앙상블 추론 — {len(df_calib):,}행 × {len(ensemble.models)}모델 "
        "(가장 오래 걸릴 수 있음)",
    )
    t_inf = time.perf_counter()
    y_calib_prob = ensemble.predict_proba(df_calib, feature_cols, verbose=True)
    _log(f"추론 전체 완료 ({time.perf_counter() - t_inf:.1f}s)", indent=1)

    fpr_levels = list(cfg.FPR_LEVELS)
    save_cap = getattr(cfg, "SAVE_OPERATING_FPR_CAP", getattr(cfg, "MAX_FPR", None))
    _log_step("5/5", f"임계값 그리드 + FPR 표 (n_grid={cfg.THRESHOLD_N_GRID})")
    tuner = ThresholdTuner(
        fpr_levels=fpr_levels,
        n_grid=cfg.THRESHOLD_N_GRID,
        save_operating_fpr_cap=save_cap,
    )
    tuner_result = tuner.fit(df_calib[cfg.TARGET_COL].values, y_calib_prob)

    if show_plot:
        _log("\n[FPR–Recall 곡선 그리기]", indent=0)
        tuner.plot()

    threshold = float(tuner.best_threshold) if save_cap is not None else None

    threshold_metadata = {
        "threshold": threshold,
        "tuning_result": {k: _jsonable(v) for k, v in tuner_result.items()},
        "save_operating_fpr_cap": save_cap,
        "fpr_levels": fpr_levels,
        "fpr_recall_table": tuner_result.get("fpr_recall_table"),
        "threshold_n_grid": cfg.THRESHOLD_N_GRID,
        "target_col": cfg.TARGET_COL,
        "val_calib_path": cfg.VAL_CALIB_PATH,
        "val_calib_rows": calib_summary["rows"],
        "val_calib_positives": calib_summary["positives"],
        "val_calib_negatives": calib_summary["negatives"],
        "model_dir": str(artifacts["save_dir"]),
        "model_files": [p.name for p in artifacts["model_paths"]],
        "feature_cols_path": str(artifacts["feature_cols_path"]),
        "n_features": len(feature_cols),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    csv_paths: dict[str, str] = {}
    if save:
        result_dir = getattr(cfg, "THRESHOLD_RESULT_DIR", None) or artifacts["save_dir"]
        _log("CSV 저장 중...", indent=1)
        csv_paths = save_threshold_tuning_csvs(tuner, cfg, result_dir)
        for label, path in csv_paths.items():
            _log(f"  {label}: {path}", indent=2)

        if threshold is None:
            print("⚠️  SAVE_OPERATING_FPR_CAP=None — best_threshold.json 미저장")
        else:
            threshold_metadata["csv_paths"] = csv_paths
            with open(artifacts["threshold_path"], "w", encoding="utf-8") as f:
                json.dump(threshold_metadata, f, ensure_ascii=False, indent=2)
            print(f"Saved threshold metadata: {artifacts['threshold_path']}")

    _log_step("완료", f"총 소요 {time.perf_counter() - pipeline_t0:.1f}s")

    return {
        "tuner": tuner,
        "threshold": threshold,
        "tuner_result": tuner_result,
        "threshold_metadata": threshold_metadata,
        "csv_paths": csv_paths,
        "artifacts": artifacts,
        "calib_summary": calib_summary,
    }


def run_final_evaluation_from_saved_model(
    cfg,
    *,
    show_plots: bool = True,
    save: bool = True,
) -> dict:
    """README section 8: evaluate a saved ensemble on the held-out test set."""
    import json
    from datetime import datetime
    from config.path_utils import validate_path_contract

    validate_path_contract(list(getattr(cfg, "REQUIRED_DATA_PATHS", [])))

    artifacts = load_saved_ensemble(cfg, require_threshold=True)
    feature_cols = artifacts["feature_cols"]
    ensemble = artifacts["ensemble"]
    threshold = artifacts["threshold"]

    df_test = pd.read_parquet(cfg.TEST_PATH)
    _validate_columns(
        df_test,
        feature_cols + [cfg.TARGET_COL, cfg.SERIAL_COL, cfg.DATE_COL],
        dataset_name="test data",
    )
    test_summary = _summarize_dataset(
        df_test, cfg, label="Test Data", path=cfg.TEST_PATH
    )

    y_test_prob = ensemble.predict_proba(df_test, feature_cols)

    row_result = evaluate_row_level(
        df_test[cfg.TARGET_COL].values,
        y_test_prob,
        threshold,
        title="Test Set",
    )

    ev = EntityLevelEvaluator(
        serial_col=cfg.SERIAL_COL,
        date_col=cfg.DATE_COL,
        target_col=cfg.TARGET_COL,
    )
    entity_result = ev.evaluate(df_test, y_test_prob, threshold)
    if show_plots:
        EntityLevelEvaluator.plot(entity_result)

    entity_df = entity_result["entity_df"]
    miss_list = entity_df[(entity_df["is_failure"] == 1) & (entity_df["has_alarm"] == 0)]
    print(f"Missed failure entities: {len(miss_list)}")

    entity_summary = {k: _jsonable(v) for k, v in entity_result.items() if k != "entity_df"}
    evaluation_metadata = {
        "threshold": threshold,
        "threshold_path": str(artifacts["threshold_path"]),
        "threshold_metadata": artifacts["threshold_meta"],
        "test_path": cfg.TEST_PATH,
        "test_rows": test_summary["rows"],
        "test_positives": test_summary["positives"],
        "test_negatives": test_summary["negatives"],
        "test_entities": test_summary.get("entities"),
        "model_dir": str(artifacts["save_dir"]),
        "model_files": [p.name for p in artifacts["model_paths"]],
        "feature_cols_path": str(artifacts["feature_cols_path"]),
        "n_features": len(feature_cols),
        "row_result": {k: _jsonable(v) for k, v in row_result.items()},
        "entity_result": entity_summary,
        "missed_failure_entities": int(len(miss_list)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    result_path = artifacts["save_dir"] / "final_evaluation.json"
    if save:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved final evaluation metadata: {result_path}")

    return {
        "row_result": row_result,
        "entity_result": entity_result,
        "miss_list": miss_list,
        "evaluation_metadata": evaluation_metadata,
        "result_path": result_path,
        "artifacts": artifacts,
        "test_summary": test_summary,
    }


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
    cfg           : threshold + final_eval 설정을 합친 객체
                    (VAL_CALIB_PATH, TEST_PATH, MAX_FPR, THRESHOLD_N_GRID 등)
    ensemble      : 학습 완료된 UnderbaggingEnsemble 객체
    feature_cols  : 모델에 사용된 피처 리스트
    show_plots    : 시각화 여부

    Returns
    -------
    dict: tuner, threshold, row_result, entity_result
    """
    import pandas as pd
    from config.path_utils import validate_path_contract

    required = []
    if getattr(cfg, "VAL_CALIB_PATH", None):
        required.append(("val_calib feature data", "file", cfg.VAL_CALIB_PATH))
    if getattr(cfg, "TEST_PATH", None):
        required.append(("test feature data", "file", cfg.TEST_PATH))
    validate_path_contract(required)

    # ── 1. 데이터 로드 ────────────────────────────────────────
    print("📂  평가 데이터 로드 중...")
    df_calib = pd.read_parquet(cfg.VAL_CALIB_PATH)
    df_test  = pd.read_parquet(cfg.TEST_PATH)

    # ── 2. §7 임계값 튜닝 (val_calib 기반) ───────────────────
    print("\n🎯  §7 임계값 튜닝 (val_calib)...")
    y_calib_true = df_calib[cfg.TARGET_COL].values
    y_calib_prob = ensemble.predict_proba(df_calib, feature_cols)

    fpr_levels = list(getattr(cfg, "FPR_LEVELS", [0.001, 0.005, 0.01, 0.05]))
    save_cap = getattr(cfg, "SAVE_OPERATING_FPR_CAP", getattr(cfg, "MAX_FPR", 0.01))
    tuner = ThresholdTuner(
        fpr_levels=fpr_levels,
        n_grid=getattr(cfg, "THRESHOLD_N_GRID", 1000),
        save_operating_fpr_cap=save_cap,
    )
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
