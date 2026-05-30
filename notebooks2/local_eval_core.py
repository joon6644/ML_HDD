from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from local_config import DATE_COL, MODEL_SAVE_DIR, SERIAL_COL, TARGET_COL
except ModuleNotFoundError:
    from notebooks2.local_config import DATE_COL, MODEL_SAVE_DIR, SERIAL_COL, TARGET_COL


PROB_CANDIDATES = (
    "prob",
    "proba",
    "pred_prob",
    "prediction",
    "score",
    "failure_probability",
    "y_prob",
)


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _resolve_prob_col(df: pd.DataFrame, prob_col: str | None = None) -> str:
    if prob_col and prob_col in df.columns:
        return prob_col
    for col in PROB_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Probability column was not found. Pass `prob_col=` or provide one of "
        f"{PROB_CANDIDATES}."
    )


def _ensure_eval_frame(
    df: pd.DataFrame,
    y_prob: Sequence[float] | None = None,
    *,
    prob_col: str | None = None,
    target_col: str = TARGET_COL,
    serial_col: str = SERIAL_COL,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    missing = [c for c in (target_col, serial_col, date_col) if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required evaluation columns: {missing}")

    if y_prob is not None:
        if len(y_prob) != len(out):
            raise ValueError(f"len(y_prob)={len(y_prob)} does not match len(df)={len(out)}")
        out["_rolling_prob"] = np.asarray(y_prob, dtype=float)
        resolved_prob_col = "_rolling_prob"
    else:
        resolved_prob_col = _resolve_prob_col(out, prob_col)

    out[serial_col] = out[serial_col].astype(str)
    out[date_col] = pd.to_datetime(out[date_col])
    out[target_col] = out[target_col].fillna(0).astype(int)
    return out.sort_values([serial_col, date_col]).reset_index(drop=True), resolved_prob_col


def _drop_failure_rows_and_label_disks(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    serial_col: str = SERIAL_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    first_failure = (
        df.loc[df[target_col].eq(1)]
        .groupby(serial_col, sort=False)[date_col]
        .min()
        .rename("_first_failure_date")
    )
    out = df.join(first_failure, on=serial_col)
    out["_disk_label"] = out["_first_failure_date"].notna().astype(int)
    out = out.loc[~out[target_col].eq(1)].copy()
    return out.reset_index(drop=True)


def get_rolling_n_largest(
    values: Iterable[float] | pd.Series,
    n: int = 1,
    *,
    window_size: int = 0,
) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return np.nan
    if window_size and window_size > 0:
        series = series.tail(window_size)
    n = max(1, min(int(n), len(series)))
    return float(series.nlargest(n).mean())


def prepare_disk_level_data(
    df: pd.DataFrame,
    y_prob: Sequence[float] | None = None,
    *,
    prob_col: str | None = None,
    target_col: str = TARGET_COL,
    serial_col: str = SERIAL_COL,
    date_col: str = DATE_COL,
    window_size: int = 0,
    remove_failure_rows: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    """Build disk-level rolling records and remove D-day/zombie rows.

    `window_size=0` intentionally means row-by-row rolling inference over all
    already-computed feature rows. If a disk only has failure rows, it naturally
    disappears after filtering because there is no pre-failure evidence to score.
    """
    frame, resolved_prob_col = _ensure_eval_frame(
        df,
        y_prob,
        prob_col=prob_col,
        target_col=target_col,
        serial_col=serial_col,
        date_col=date_col,
    )
    if remove_failure_rows:
        frame = _drop_failure_rows_and_label_disks(
            frame, target_col=target_col, serial_col=serial_col, date_col=date_col
        )
    else:
        frame["_disk_label"] = frame.groupby(serial_col)[target_col].transform("max").astype(int)
        frame["_first_failure_date"] = pd.NaT

    if frame.empty:
        return pd.DataFrame(
            columns=[
                serial_col,
                "disk_score",
                "disk_label",
                "n_rows",
                "first_date",
                "last_date",
                "first_failure_date",
            ]
        )

    records = []
    for serial, g in frame.groupby(serial_col, sort=False):
        g = g.sort_values(date_col)
        score = get_rolling_n_largest(g[resolved_prob_col], n=1, window_size=window_size)
        records.append(
            {
                serial_col: serial,
                "disk_score": score,
                "disk_label": int(g["_disk_label"].max()),
                "n_rows": int(len(g)),
                "first_date": g[date_col].min(),
                "last_date": g[date_col].max(),
                "first_failure_date": g["_first_failure_date"].dropna().min()
                if "_first_failure_date" in g
                else pd.NaT,
            }
        )
    return pd.DataFrame.from_records(records)


def run_disk_level_grid_search(
    df: pd.DataFrame,
    y_prob: Sequence[float] | None = None,
    *,
    thresholds: Sequence[float] | None = None,
    prob_col: str | None = None,
    target_col: str = TARGET_COL,
    serial_col: str = SERIAL_COL,
    date_col: str = DATE_COL,
    window_size: int = 0,
    horizon: int | None = 30,
    log_path: str | Path | None = None,
    verbose: bool = True,
    **kwargs,
) -> dict:
    from sklearn.metrics import average_precision_score

    start = time.time()
    disk_df = prepare_disk_level_data(
        df,
        y_prob,
        prob_col=prob_col,
        target_col=target_col,
        serial_col=serial_col,
        date_col=date_col,
        window_size=window_size,
        **kwargs,
    )

    if disk_df.empty or disk_df["disk_label"].nunique() < 2:
        prauc = np.nan
    else:
        prauc = float(average_precision_score(disk_df["disk_label"], disk_df["disk_score"]))

    thresholds = list(thresholds) if thresholds is not None else [round(x / 100, 2) for x in range(1, 100)]
    rows = []
    labels = disk_df["disk_label"].to_numpy(dtype=int) if not disk_df.empty else np.array([], dtype=int)
    scores = disk_df["disk_score"].to_numpy(dtype=float) if not disk_df.empty else np.array([], dtype=float)
    for thr in thresholds:
        pred = scores >= float(thr)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    grid_df = pd.DataFrame(rows)
    best = grid_df.sort_values(["f1", "recall", "precision"], ascending=False).head(1)
    best_threshold = float(best["threshold"].iloc[0]) if not best.empty else np.nan
    result = {
        "disk_prauc": prauc,
        "prauc": prauc,
        "best_threshold": best_threshold,
        "grid": grid_df,
        "disk_df": disk_df,
        "n_disks": int(len(disk_df)),
        "n_failure_disks": int(disk_df["disk_label"].sum()) if not disk_df.empty else 0,
        "horizon": horizon,
        "window_size": window_size,
        "elapsed_sec": time.time() - start,
    }

    if log_path:
        serializable = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
        Path(log_path).write_text(json.dumps(serializable, default=str, indent=2), encoding="utf-8")
    _log(
        f"Disk rolling PR-AUC={prauc:.6f} disks={result['n_disks']} "
        f"failure_disks={result['n_failure_disks']} window_size={window_size}",
        verbose=verbose,
    )
    return result


def evaluate_detailed_disk_point(
    df: pd.DataFrame,
    y_prob: Sequence[float] | None = None,
    *,
    threshold: float = 0.5,
    prob_col: str | None = None,
    target_col: str = TARGET_COL,
    serial_col: str = SERIAL_COL,
    date_col: str = DATE_COL,
    horizon: int | None = 30,
    window_size: int = 0,
    **kwargs,
) -> dict:
    frame, resolved_prob_col = _ensure_eval_frame(
        df,
        y_prob,
        prob_col=prob_col,
        target_col=target_col,
        serial_col=serial_col,
        date_col=date_col,
    )
    original = frame.copy()
    frame = _drop_failure_rows_and_label_disks(
        frame, target_col=target_col, serial_col=serial_col, date_col=date_col
    )
    rows = []
    for serial, g in frame.groupby(serial_col, sort=False):
        g = g.sort_values(date_col)
        first_failure = g["_first_failure_date"].dropna().min()
        alarms = g.loc[g[resolved_prob_col] >= threshold]
        first_alarm = alarms[date_col].min() if not alarms.empty else pd.NaT
        label = int(g["_disk_label"].max())
        pred = int(pd.notna(first_alarm))
        lead_days = (
            int((first_failure - first_alarm).days)
            if label and pd.notna(first_failure) and pd.notna(first_alarm)
            else np.nan
        )
        in_horizon = bool(label and pred and (horizon is None or 0 <= lead_days <= horizon))
        rows.append(
            {
                serial_col: serial,
                "disk_label": label,
                "disk_pred": pred,
                "disk_score": float(g[resolved_prob_col].max()),
                "first_alarm_date": first_alarm,
                "first_failure_date": first_failure,
                "lead_days": lead_days,
                "in_horizon": in_horizon,
                "n_rows": int(len(g)),
            }
        )

    records = pd.DataFrame.from_records(rows)
    if records.empty:
        tp = fp = fn = tn = 0
    else:
        tp = int(((records.disk_label == 1) & (records.disk_pred == 1)).sum())
        fp = int(((records.disk_label == 0) & (records.disk_pred == 1)).sum())
        fn = int(((records.disk_label == 1) & (records.disk_pred == 0)).sum())
        tn = int(((records.disk_label == 0) & (records.disk_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "records": records,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "threshold": threshold,
        "horizon": horizon,
        "window_size": window_size,
        "dropped_failure_rows": int(original[target_col].eq(1).sum()),
    }


def get_detailed_disk_records(*args, **kwargs) -> pd.DataFrame:
    result = evaluate_detailed_disk_point(*args, **kwargs)
    return result["records"]


@dataclass
class SavedEnsemble:
    models: list

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        X = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32)
        probs = []
        for i, model in enumerate(self.models, start=1):
            _log(f"[SavedEnsemble] predict_proba model {i}/{len(self.models)}", verbose=verbose)
            probs.append(model.predict_proba(X)[:, 1])
        return np.mean(probs, axis=0)


def load_saved_ensemble(
    model_dir: str | Path = MODEL_SAVE_DIR,
    *,
    model_pattern: str = "*.pkl",
    verbose: bool = True,
    **_kwargs,
) -> SavedEnsemble:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    model_paths = sorted(model_dir.glob(model_pattern))
    if not model_paths:
        model_paths = sorted(model_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No saved model files found in: {model_dir}")
    models = []
    for path in model_paths:
        _log(f"[load_saved_ensemble] loading {path.name}", verbose=verbose)
        models.append(joblib.load(path))
    return SavedEnsemble(models=models)
