"""
Path helpers for running the project across multiple local machines.

Compatibility here means "same dataset contract, different local root", not
"silently fall back to another pipeline stage". If a required file is missing,
the pipeline should fail loudly so we do not train or evaluate on the wrong
data by accident.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("ML_HDD_DATA_DIR", PROJECT_ROOT / "data2")).expanduser()

# 06a_feature_engineering.ipynb 출력 디렉터리 (train/val_*/test.parquet)
FE_SPLIT_REL = "split_group_stratified"


def data_path(relative_path: str) -> str:
    """Return the canonical path under the configured data root."""
    return str(DATA_ROOT / relative_path)


def fe_data_path(filename: str) -> str:
    """Feature-engineered split parquet (06a 노트북 산출물)."""
    return data_path(f"{FE_SPLIT_REL}/{filename}")


def val_calib_missing_hint(val_calib_path: str | Path) -> str:
    """val_calib.parquet 가 없을 때 선행 노트북 안내 문구."""
    calib = Path(val_calib_path)
    raw = DATA_ROOT / FE_SPLIT_REL / "val_calib_raw.parquet"
    lines = [f"Missing: {calib}"]
    if raw.is_file():
        lines.append(
            "Found val_calib_raw.parquet — run notebooks/06a_feature_engineering.ipynb "
            "to create val_calib.parquet."
        )
    else:
        lines.append(
            "Also missing val_calib_raw.parquet — place split files under "
            f"{DATA_ROOT / FE_SPLIT_REL} (see README 데이터 분할 후), "
            "then run notebooks/03_data_splitting.ipynb and "
            "notebooks/06a_feature_engineering.ipynb."
        )
    return "\n".join(lines)


def require_data_file(relative_path: str, *, label: str | None = None) -> str:
    """Return a required data file path or raise a precise error."""
    path = DATA_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(_missing_message(path, label or relative_path, "file"))
    return str(path)


def require_data_dir(relative_path: str, *, label: str | None = None) -> str:
    """Return a required data directory path or raise a precise error."""
    path = DATA_ROOT / relative_path
    if not path.is_dir():
        raise FileNotFoundError(_missing_message(path, label or relative_path, "directory"))
    return str(path)


def print_path_report(paths: list[tuple[str, str, str]]) -> None:
    """Print every configured data path and whether it exists."""
    print("\n[Data Path Contract]")
    for label, kind, path_str in paths:
        path = Path(path_str)
        if kind == "dir":
            exists = path.is_dir()
        elif kind == "file":
            exists = path.is_file()
        else:
            raise ValueError(f"Unknown path kind: {kind}")
        status = "OK" if exists else "MISSING"
        size = f" ({path.stat().st_size:,} bytes)" if exists and path.is_file() else ""
        print(f"  [{status}] {label}: {path}{size}")


def validate_path_contract(paths: list[tuple[str, str, str]]) -> None:
    """Print all configured paths, then fail if any required path is missing."""
    print_path_report(paths)
    missing = []
    for label, kind, path_str in paths:
        path = Path(path_str)
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            missing.append(f"- {label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Required data paths are missing. Refusing to continue with a partial "
            "or mismatched dataset.\n" + "\n".join(missing)
        )


def _missing_message(path: Path, label: str, kind: str) -> str:
    return (
        f"Required data {kind} is missing for {label}: {path}\n"
        "Keep the same relative data layout on each machine, or set "
        "ML_HDD_DATA_DIR to the machine-specific data root."
    )
