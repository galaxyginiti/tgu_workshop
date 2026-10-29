"""Общие функции: пути, конфигурация, чтение/запись CSV и московское время.

Внутри пайплайна все моменты времени хранятся как наивные datetime по Москве
(UTC+03:00, перехода на летнее время нет). В выходных CSV они записываются
в ISO 8601 с явным смещением +03:00.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOLUTION_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SOLUTION_DIR.parent
CONFIG_PATH = SOLUTION_DIR / "config" / "project.json"

STAGE2_OUT = SOLUTION_DIR / "stage2_preparation" / "output"
STAGE3_OUT = SOLUTION_DIR / "stage3_eda" / "output"
STAGE4_OUT = SOLUTION_DIR / "stage4_features" / "output"

MSK_OFFSET = pd.Timedelta(hours=3)
CSV_OPTIONS = dict(sep=";", encoding="utf-8-sig")


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["abs_paths"] = {name: REPO_ROOT / rel for name, rel in cfg["paths"].items()}
    return cfg


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Читает исходник как текст: пустая ячейка остаётся пустой строкой, не NaN и не нулём."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, **CSV_OPTIONS)
    return df.astype(object)


def read_prepared(path: Path) -> pd.DataFrame:
    """Читает подготовленную таблицу: пустая ячейка = неизвестное значение (NaN)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, **CSV_OPTIONS).replace("", np.nan).astype(object)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)
    out.to_csv(path, index=False, na_rep="", lineterminator="\n", **CSV_OPTIONS)
    return path


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_msk(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """ISO-строки -> наивное московское время.

    Значение с зоной (Z или ±HH:MM) переводится в Москву с сохранением момента.
    Значение без зоны по подтверждению заказчика считается московским.
    Возвращает (время, маска значений без зоны).
    """
    s = values.fillna("").astype(str).str.strip()
    has_zone = s.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True)
    no_zone = (s != "") & ~has_zone
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if has_zone.any():
        utc = pd.to_datetime(s[has_zone], utc=True, format="ISO8601")
        out.loc[has_zone] = (utc + MSK_OFFSET).dt.tz_localize(None).astype("datetime64[ns]")
    if no_zone.any():
        out.loc[no_zone] = pd.to_datetime(s[no_zone], format="ISO8601").astype("datetime64[ns]")
    return out, no_zone


def parse_local(values: pd.Series, fmt: str) -> pd.Series:
    s = values.fillna("").astype(str).str.strip().replace("", np.nan)
    return pd.to_datetime(s, format=fmt).astype("datetime64[ns]")


def fmt_ts(values: pd.Series) -> pd.Series:
    """Наивное московское время -> 'YYYY-MM-DDTHH:MM:SS+03:00' (пусто для NaT)."""
    v = pd.to_datetime(values)
    return v.dt.strftime("%Y-%m-%dT%H:%M:%S").add("+03:00").where(v.notna(), "")


def fmt_date(values: pd.Series) -> pd.Series:
    v = pd.to_datetime(values)
    return v.dt.strftime("%Y-%m-%d").where(v.notna(), "")


def month_start(values: pd.Series) -> pd.Series:
    v = pd.to_datetime(values)
    return v.dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.Timestamp(ts) + pd.DateOffset(months=n)).normalize()


def num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").astype(float)


def snapshot_dates(cfg: dict) -> dict[str, list[pd.Timestamp]]:
    train = pd.date_range(cfg["train_start"], cfg["train_end_inclusive"], freq="MS")
    return {
        "train": list(train),
        "validation": [pd.Timestamp(d) for d in cfg["validation_dates"]],
        "test": [pd.Timestamp(d) for d in cfg["test_dates"]],
    }


def check_row(check: str, expected, actual, passed: bool, details: str = "") -> dict:
    return {"check": check, "expected": str(expected), "actual": str(actual),
            "passed": "PASS" if passed else "FAIL", "details": details}
