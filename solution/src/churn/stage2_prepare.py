"""Этап 2. Сбор и подготовка данных.

Вход: исходники занятий 1–2 (не изменяются).
Выход (solution/stage2_preparation/output):
  subscribers_clean.csv, billing_clean.csv, contract_events_clean.csv,
  support_clean.csv, customer_month.csv, quarantine.csv,
  inventory.csv, sqlite_diagnostics.csv, quality_before.csv, quality_after.csv,
  quality_summary.json, decision_log.csv, data_dictionary_prepared.csv,
  checks_stage2.csv.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (SOLUTION_DIR, STAGE2_OUT, add_months, check_row, fmt_date, fmt_ts,
                     month_start, num, parse_local, read_raw_csv, sha256, to_msk, write_csv,
                     write_json)

SQL_DIR = SOLUTION_DIR / "sql"
TARIFF_FEE = {"Basic": 450.0, "Comfort": 750.0, "Premium": 1200.0}
TARIFF_MAP = {"basic": "Basic", "comfort": "Comfort", "premium": "Premium", "премиум": "Premium"}
TOPICS = ["network", "billing", "tariff", "services", "unknown"]
CHANNELS = ["phone", "chat", "office"]
MONEY = ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub"]
INTS = ["voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]
DECIMAL_RE = r"-?\d+(?:\.\d+)?"


# --------------------------------------------------------------------------- загрузка

def extract_sqlite(db_path: Path) -> pd.DataFrame:
    query = (SQL_DIR / "extract_support.sql").read_text(encoding="utf-8")
    with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
        cur = conn.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols).astype(object)
    return df.where(df.notna(), "").astype(str).astype(object)


def run_sqlite_diagnostics(db_path: Path) -> pd.DataFrame:
    text = (SQL_DIR / "diagnostics_support.sql").read_text(encoding="utf-8")
    blocks = re.findall(r"-- name: (\w+)\n(.*?;)", text, flags=re.S)
    out = []
    with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
        for name, sql in blocks:
            cur = conn.execute(sql)
            cols = [c[0] for c in cur.description]
            for row in cur.fetchall():
                out.append({"query": name, "result": "; ".join(f"{c}={v}" for c, v in zip(cols, row))})
    return pd.DataFrame(out)


def parse_schema_maps(schema: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Соответствие полей и словари категорий берём из schema_support.csv, а не вручную."""
    rename = dict(zip(schema.legacy_field, schema.canonical_field))
    rules = dict(zip(schema.canonical_field, schema.rules))
    topic_map = {k.strip(): v for k, v in re.findall(r"([^;=.]+)=([a-z_]+)", rules["topic"])}
    channel_map = {k.strip(): v for k, v in re.findall(r"([^;=.]+)=([a-z_]+)", rules["channel"])}
    return rename, topic_map, channel_map


# --------------------------------------------------------------------------- проверки качества

def _observed_types(df: pd.DataFrame) -> dict:
    """Фактические форматы текстового столбца (сколько значений какого вида)."""
    patterns = [
        ("integer", r"-?\d+"),
        ("decimal", r"-?\d+\.\d+"),
        ("decimal с запятой", r"-?\d+,\d+"),
        ("date YYYY-MM-DD", r"\d{4}-\d{2}-\d{2}"),
        ("date DD.MM.YYYY", r"\d{2}\.\d{2}\.\d{4}"),
        ("datetime UTC Z", r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"),
        ("datetime без зоны", r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
        ("datetime DD.MM.YYYY HH:MM:SS", r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}"),
    ]
    kinds = {}
    for c in df.columns:
        v = df[c].astype(str).str.strip()
        v = v[v != ""]
        label = pd.Series("text", index=v.index, dtype=object)
        for name, rx in patterns:
            label[(label == "text") & v.str.fullmatch(rx)] = name
        counts = label.value_counts()
        kinds[c] = counts.index[0] if len(counts) == 1 else " / ".join(f"{k} {n}" for k, n in counts.items())
        if (df[c].astype(str).str.strip() == "").any():
            kinds[c] += f" (+пусто {(df[c].astype(str).str.strip() == '').sum()})"
    return kinds


class Issues:
    """Накопитель проблем: (таблица, проверка, поле) -> булева маска по строкам."""

    def __init__(self, stage: str):
        self.stage = stage
        self.rows: list[dict] = []
        self.masks: dict[str, list[pd.Series]] = {}

    def add(self, table: str, check: str, column: str, mask: pd.Series, unit: str = "cells", note: str = ""):
        mask = mask.fillna(False).astype(bool)
        self.rows.append({"stage": self.stage, "table": table, "check": check, "column": column,
                          "unit": unit, "count": int(mask.sum()), "note": note})
        if unit == "cells":
            self.masks.setdefault(table, []).append(mask)

    def add_count(self, table: str, check: str, column: str, count: int, unit: str, note: str = ""):
        self.rows.append({"stage": self.stage, "table": table, "check": check, "column": column,
                          "unit": unit, "count": int(count), "note": note})

    def summary(self, table: str, n_rows: int) -> dict:
        masks = self.masks.get(table, [])
        if not masks:
            return {"rows_with_problems": 0, "problem_cells": 0}
        frame = pd.concat(masks, axis=1)
        return {"rows_with_problems": int(frame.any(axis=1).sum()), "problem_cells": int(frame.sum().sum())}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _not_decimal(s: pd.Series) -> pd.Series:
    t = s.astype(str)
    return ~_blank(s) & ~t.str.fullmatch(DECIMAL_RE)


def audit_subscribers(df: pd.DataFrame, iss: Issues, events: pd.DataFrame, raw: bool):
    t = "subscribers"
    iss.add_count(t, "full_duplicate_copies", "*", df.duplicated().sum(), "copies")
    iss.add(t, "key_duplicate", "customer_id", df.customer_id.duplicated(keep=False), "cells")
    iss.add(t, "id_format", "customer_id", ~df.customer_id.astype(str).str.fullmatch(r"C\d{6}"))
    iss.add(t, "missing", "activation_date", _blank(df.activation_date))
    date_txt = df.activation_date.astype(str)
    iss.add(t, "date_format", "activation_date", ~_blank(df.activation_date) & ~date_txt.str.fullmatch(r"\d{4}-\d{2}-\d{2}"))
    act = events[events.event_type == "activation"].set_index("customer_id").event_date
    linked = df.customer_id.map(act)
    iss.add(t, "activation_differs_from_event", "activation_date",
            ~_blank(df.activation_date) & linked.notna() & (date_txt != linked.astype(str)))
    for col in ["region", "acquisition_channel", "customer_segment"]:
        iss.add(t, "missing", col, _blank(df[col]))
    iss.add(t, "category", "acquisition_channel", ~df.acquisition_channel.isin(["online", "store", "partner"]))
    iss.add(t, "category", "customer_segment", ~df.customer_segment.isin(["individual", "small_business"]))


def audit_events(df: pd.DataFrame, iss: Issues, subscriber_ids: set):
    t = "contract_events"
    iss.add_count(t, "full_duplicate_copies", "*", df.duplicated().sum(), "copies")
    iss.add(t, "key_duplicate", "event_id", df.event_id.duplicated(keep=False))
    iss.add(t, "date_format", "event_date", ~df.event_date.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}"))
    iss.add(t, "category", "event_type", ~df.event_type.isin(["activation", "termination"]))
    iss.add(t, "reference", "customer_id", ~df.customer_id.astype(str).str.strip().isin(subscriber_ids))
    per = df.groupby(["customer_id", "event_type"]).size().unstack(fill_value=0)
    bad_act = set(per.index[per.get("activation", 0) != 1])
    bad_term = set(per.index[per.get("termination", 0) > 1])
    iss.add(t, "one_activation_per_customer", "customer_id", df.customer_id.isin(bad_act), "cells")
    iss.add(t, "at_most_one_termination", "customer_id", df.customer_id.isin(bad_term), "cells")
    a = df[df.event_type == "activation"].set_index("customer_id").event_date
    term = df[df.event_type == "termination"]
    iss.add(t, "termination_not_after_activation", "event_date",
            pd.Series(term.event_date.values <= term.customer_id.map(a).values, index=term.index).reindex(df.index, fill_value=False))


def audit_billing(df: pd.DataFrame, iss: Issues, subscriber_ids: set, events: pd.DataFrame, clean: bool):
    t = "billing"
    iss.add_count(t, "full_duplicate_copies", "*", df.duplicated().sum(), "copies")
    month_txt = df.billing_month.astype(str)
    uniq = df.drop_duplicates()
    iss.add_count(t, "key_duplicate_copies", "customer_id+billing_month",
                  df.duplicated(["customer_id", "billing_month"]).sum(), "copies", "лишние строки ключа, включая полные копии")
    iss.add(t, "key_conflict", "customer_id+billing_month",
            uniq.duplicated(["customer_id", "billing_month"], keep=False).reindex(df.index, fill_value=False),
            note="один ключ с разными значениями (после удаления полных копий)")
    iss.add(t, "date_format", "billing_month", ~month_txt.str.fullmatch(r"\d{4}-\d{2}-01"), note="ожидается YYYY-MM-01")
    avail_re = r"\d{4}-\d{2}-01T00:00:00\+03:00" if clean else r"\d{4}-\d{2}-01 00:00:00"
    iss.add(t, "datetime_format", "available_at", ~df.available_at.astype(str).str.fullmatch(avail_re))
    iss.add(t, "category", "tariff", ~df.tariff.isin(list(TARIFF_FEE)), note="Basic/Comfort/Premium")
    for col in MONEY + ["traffic_gb"] + INTS:
        iss.add(t, "missing", col, _blank(df[col]))
        iss.add(t, "number_format", col, _not_decimal(df[col]), note="десятичная точка")
    v = {c: pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce") for c in MONEY + ["traffic_gb"] + INTS}
    month = pd.to_datetime(month_txt, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(month_txt, format="%d.%m.%Y", errors="coerce"))
    dim = month.dt.days_in_month
    iss.add(t, "range", "voice_minutes", v["voice_minutes"] < 0, note="минуты не могут быть отрицательными")
    iss.add(t, "range", "traffic_gb", v["traffic_gb"] < 0)
    iss.add(t, "range", "active_days", (v["active_days"] < 1) | (v["active_days"] > dim))
    iss.add(t, "range", "payment_delay_days", (v["payment_delay_days"] < 0) | (v["payment_delay_days"] > dim))
    iss.add(t, "range", "tariff_changes_count", ~v["tariff_changes_count"].isin([0, 1]) & v["tariff_changes_count"].notna())
    for c in ["opening_debt_rub", "closing_debt_rub", "charged_rub", "paid_rub"]:
        iss.add(t, "range", c, v[c] < 0)
    tariff_norm = df.tariff.astype(str).str.strip().str.lower().map(TARIFF_MAP)
    iss.add(t, "fee_matches_tariff", "monthly_fee_rub", tariff_norm.map(TARIFF_FEE).ne(v["monthly_fee_rub"]) & tariff_norm.notna())
    eq = (v["opening_debt_rub"] + v["charged_rub"] - v["paid_rub"] - v["closing_debt_rub"]).abs() > 0.005
    iss.add(t, "debt_equation", "closing_debt_rub", eq & v["closing_debt_rub"].notna(),
            note="opening + charged - paid = closing")
    iss.add(t, "reference", "customer_id", ~df.customer_id.astype(str).str.strip().isin(subscriber_ids))
    if clean:
        avail, _ = to_msk(df.available_at)
    else:
        avail = pd.to_datetime(df.available_at, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    iss.add(t, "available_at_is_next_month_start", "available_at", avail.ne(month + pd.offsets.MonthBegin(1)) & month.notna())
    act = pd.to_datetime(events[events.event_type == "activation"].set_index("customer_id").event_date)
    term = pd.to_datetime(events[events.event_type == "termination"].set_index("customer_id").event_date)
    a = df.customer_id.map(act)
    e = df.customer_id.map(term)
    iss.add(t, "month_inside_contract", "billing_month", (month < month_start(a)) | (month > month_start(e)))
    start = pd.concat([month, a], axis=1).max(axis=1)
    end = pd.concat([month + pd.offsets.MonthBegin(1), e], axis=1).min(axis=1)
    iss.add(t, "active_days_match_contract", "active_days", ((end - start).dt.days != v["active_days"]) & v["active_days"].notna())


def audit_support(df: pd.DataFrame, iss: Issues, subscriber_ids: set, table: str, clean: bool,
                  topic_values: list, channel_values: list):
    iss.add_count(table, "full_duplicate_copies", "*", df.duplicated().sum(), "copies")
    uniq = df.drop_duplicates()
    iss.add_count(table, "key_duplicate_copies", "ticket_id", df.ticket_id.duplicated().sum(), "copies",
                  "лишние строки ключа, включая полные копии")
    iss.add(table, "key_conflict", "ticket_id", uniq.ticket_id.duplicated(keep=False).reindex(df.index, fill_value=False),
            note="один ticket_id с разными значениями")
    iss.add(table, "id_format", "ticket_id", ~df.ticket_id.astype(str).str.fullmatch(r"T\d{7}"))
    cid = df.customer_id.astype(str)
    iss.add(table, "missing", "customer_id", _blank(df.customer_id))
    iss.add(table, "whitespace", "customer_id", cid != cid.str.strip())
    iss.add(table, "reference", "customer_id", ~_blank(df.customer_id) & ~cid.str.strip().isin(subscriber_ids))
    iss.add(table, "missing", "topic", _blank(df.topic))
    iss.add(table, "category", "topic", ~_blank(df.topic) & ~df.topic.isin(topic_values))
    iss.add(table, "category", "channel", ~df.channel.isin(channel_values))
    score = pd.to_numeric(df.satisfaction_score, errors="coerce")
    iss.add(table, "range", "satisfaction_score", ~_blank(df.satisfaction_score) & ~score.between(1, 5))
    if table == "support_legacy" and not clean:
        parse = {c: parse_local(df[c], "%d.%m.%Y %H:%M:%S") for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]}
    else:
        parse = {}
        for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
            parse[c], no_zone = to_msk(df[c])
            if not clean:
                iss.add(table, "timezone_missing", c, no_zone, note="по заказчику: Москва")
    p = parse
    iss.add(table, "time_order", "available_at", p["available_at"] < p["created_at"], note="available_at < created_at")
    iss.add(table, "time_order", "resolved_at", p["resolved_at"] < p["created_at"], note="resolved_at < created_at")
    iss.add(table, "time_order", "resolution_available_at",
            (p["resolution_available_at"] < p["resolved_at"]) | (p["resolution_available_at"] < p["available_at"]))
    iss.add(table, "closure_pair_incomplete", "resolution_available_at", p["resolved_at"].isna() != p["resolution_available_at"].isna())
    iss.add(table, "score_without_closure", "satisfaction_score", ~_blank(df.satisfaction_score) & p["resolved_at"].isna())
    lo, hi = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-01")
    iss.add(table, "created_outside_export_window", "created_at", (p["created_at"] < lo) | (p["created_at"] >= hi))


# --------------------------------------------------------------------------- очистка

def clean_subscribers(raw: pd.DataFrame, events: pd.DataFrame, log: list) -> pd.DataFrame:
    s = raw.apply(lambda c: c.astype(str).str.strip())
    act = events[events.event_type == "activation"].set_index("customer_id").event_date
    missing = s.activation_date == ""
    restorable = missing & s.customer_id.isin(act.index)
    s["activation_date_source"] = np.where(missing, np.where(restorable, "contract_events", "unknown"), "subscribers")
    s.loc[restorable, "activation_date"] = s.loc[restorable, "customer_id"].map(act)
    s["activation_date"] = pd.to_datetime(s.activation_date.replace("", np.nan), format="%Y-%m-%d")
    log.append(dict(source="subscribers.csv", field="activation_date", problem="Пустая дата подключения",
                    action="Восстановлена по единственному событию activation в contract_events.csv; источник отмечен в activation_date_source",
                    rationale="Журнал событий полон, на абонента ровно одно подключение; при совпадающих непустых датах расхождений 0",
                    affected=int(restorable.sum()), unit="ячейки"))
    log.append(dict(source="subscribers.csv", field="activation_date", problem="Дата не восстанавливается",
                    action="Абонент сохраняется, дата остаётся пустой", rationale="Удалять абонентов запрещено",
                    affected=int((missing & ~restorable).sum()), unit="строки"))
    s = s.sort_values("customer_id").reset_index(drop=True)
    return s[["customer_id", "activation_date", "activation_date_source", "region", "acquisition_channel", "customer_segment"]]


def clean_events(raw: pd.DataFrame) -> pd.DataFrame:
    e = raw.apply(lambda c: c.astype(str).str.strip())
    e["event_date"] = pd.to_datetime(e.event_date, format="%Y-%m-%d")
    return e.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def clean_billing(raw: pd.DataFrame, log: list, quarantine: list) -> pd.DataFrame:
    copies = raw.duplicated(keep="first")
    b = raw[~copies].copy()
    log.append(dict(source="billing_monthly.csv", field="*", problem="Полные повторные строки",
                    action="Удалены копии, оставлена первая", rationale="Совпадают все поля, включая ключ; это повтор выгрузки, а не новое начисление",
                    affected=int(copies.sum()), unit="повторные копии"))
    b["customer_id"] = b.customer_id.astype(str).str.strip()
    m = b.billing_month.astype(str).str.strip()
    ru = m.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}")
    month = pd.to_datetime(m.where(~ru), format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(m.where(ru), format="%d.%m.%Y", errors="coerce"))
    assert month.notna().all() and (month.dt.day == 1).all(), "Непредусмотренный формат billing_month"
    b["billing_month"] = month.astype("datetime64[ns]")
    log.append(dict(source="billing_monthly.csv", field="billing_month", problem="Формат DD.MM.YYYY вместо YYYY-MM-DD",
                    action="Приведено к первому дню месяца YYYY-MM-DD", rationale="Значение однозначно (всегда 01-е число), совпадает с available_at",
                    affected=int(ru.sum()), unit="ячейки"))
    key_dup = b.duplicated(["customer_id", "billing_month"], keep=False)
    assert not key_dup.any(), "После нормализации месяца появились конфликты ключа"

    t = b.tariff.astype(str).str.strip().str.lower().map(TARIFF_MAP)
    assert t.notna().all(), "Неизвестное написание тарифа"
    changed = t != b.tariff
    b["tariff"] = t
    log.append(dict(source="billing_monthly.csv", field="tariff", problem="Варианты написания (COMFORT, ' basic ', Премиум)",
                    action="Нормализовано к Basic/Comfort/Premium", rationale="monthly_fee_rub во всех таких строках совпадает с ценой нормализованного тарифа",
                    affected=int(changed.sum()), unit="ячейки"))

    comma = b.paid_rub.astype(str).str.contains(",", regex=False)
    for c in MONEY + ["traffic_gb"]:
        b[c] = num(b[c].astype(str).str.replace(",", ".", regex=False).replace("", np.nan))
    log.append(dict(source="billing_monthly.csv", field="paid_rub", problem="Десятичная запятая",
                    action="Запятая заменена точкой", rationale="Формат по описанию — десятичная точка; значение не меняется",
                    affected=int(comma.sum()), unit="ячейки"))
    for c in INTS:
        b[c] = num(b[c].replace("", np.nan))

    miss_close = b.closing_debt_rub.isna()
    restored = (b.opening_debt_rub + b.charged_rub - b.paid_rub).round(2) + 0.0  # + 0.0 убирает -0.0
    b["closing_debt_restored"] = miss_close.astype(int)
    b.loc[miss_close, "closing_debt_rub"] = restored[miss_close]
    b = b.sort_values(["customer_id", "billing_month"]).reset_index(drop=True)
    nxt_open = b.groupby("customer_id").opening_debt_rub.shift(-1)
    rest_rows = b.closing_debt_restored == 1
    confirmed = int(((b.closing_debt_rub - nxt_open).abs() < 0.005)[rest_rows].sum())
    has_next = int(nxt_open[rest_rows].notna().sum())
    log.append(dict(source="billing_monthly.csv", field="closing_debt_rub", problem="Пустой конечный долг",
                    action="Восстановлен по равенству opening + charged − paid; флаг closing_debt_restored=1",
                    rationale=f"Равенство выполняется во всех непустых строках; у {has_next} из восстановленных есть следующий месяц, совпадение с его opening_debt: {confirmed}",
                    affected=int(miss_close.sum()), unit="ячейки"))

    neg_voice = b.voice_minutes < 0
    for _, r in b[neg_voice].iterrows():
        quarantine.append(dict(source_table="billing", record_id=f"{r.customer_id}|{r.billing_month:%Y-%m-%d}", scope="cell",
                               field="voice_minutes", raw_value=str(int(r.voice_minutes)), reason_code="negative_value",
                               reason="Отрицательные минуты связи; значение исключено из расчётов (пусто)",
                               customer_id=r.customer_id, month=f"{r.billing_month:%Y-%m-%d}"))
    b["voice_minutes_invalid"] = neg_voice.astype(int)
    b.loc[neg_voice, "voice_minutes"] = np.nan
    log.append(dict(source="billing_monthly.csv", field="voice_minutes", problem="Отрицательные минуты",
                    action="Значение заменено пропуском, флаг voice_minutes_invalid=1, ячейка в карантине",
                    rationale="Правдоподобное значение восстановить нельзя; строка биллинга и абонент сохраняются",
                    affected=int(neg_voice.sum()), unit="ячейки"))
    log.append(dict(source="billing_monthly.csv", field="traffic_gb", problem="Пустой трафик",
                    action="Сохранён пропуск (неизвестно), не заменяется нулём", rationale="По словарю ноль и пропуск имеют разный смысл; импутация — позже и только на train",
                    affected=int(b.traffic_gb.isna().sum()), unit="ячейки"))
    avail = pd.to_datetime(b.available_at, format="%Y-%m-%d %H:%M:%S")
    assert (avail == b.billing_month + pd.offsets.MonthBegin(1)).all()
    b["available_at"] = avail.astype("datetime64[ns]")
    for c in INTS:
        b[c] = b[c].astype("Int64")
    cols = ["customer_id", "billing_month", "available_at", "tariff"] + MONEY + ["traffic_gb"] + INTS + ["closing_debt_restored", "voice_minutes_invalid"]
    return b[cols]


def clean_support(legacy_raw: pd.DataFrame, current_raw: pd.DataFrame, schema: pd.DataFrame, subs: pd.DataFrame,
                  events: pd.DataFrame, log: list, quarantine: list) -> pd.DataFrame:
    rename, topic_map, channel_map = parse_schema_maps(schema)
    parts = []
    for name, raw in [("legacy", legacy_raw), ("current", current_raw)]:
        copies = raw.duplicated(keep="first")
        d = raw[~copies].copy()
        log.append(dict(source=f"support_{name}", field="*", problem="Полные повторные строки",
                        action="Удалены копии, оставлена первая", rationale="Одинаковый ticket_id и все поля; разные обращения одного абонента не затрагиваются",
                        affected=int(copies.sum()), unit="повторные копии"))
        if name == "legacy":
            d = d.rename(columns=rename)
            times = {c: parse_local(d[c], "%d.%m.%Y %H:%M:%S") for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]}
            no_zone = pd.Series(False, index=d.index)
            topic_raw = d.topic.replace("", np.nan).map(topic_map)
            assert topic_raw.notna().sum() == (d.topic != "").sum(), "Неизвестная тема в legacy"
            channel = d.channel.map(channel_map)
            log.append(dict(source="support_legacy", field="все поля", problem="Старые названия полей и русские категории",
                            action="Переименовано по schema_support.csv; темы и каналы переведены в коды новой системы",
                            rationale="Словарь соответствий задан заказчиком в schema_support.csv", affected=len(d), unit="строки"))
        else:
            times, no_zone = {}, None
            for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
                times[c], nz = to_msk(d[c])
                if c == "created_at":
                    no_zone = nz
            topic_raw = d.topic.replace("", np.nan)
            channel = d.channel
            log.append(dict(source="support_current", field="created_at", problem="Время без часового пояса",
                            action="Интерпретировано как московское время; флаг created_at_zone_assumed=1",
                            rationale="Подтверждение заказчика; значения с Z переведены из UTC в Москву с сохранением момента",
                            affected=int(no_zone.sum()), unit="ячейки"))
        cid_raw = d.customer_id.astype(str)
        if name == "current":
            log.append(dict(source="support_current", field="customer_id", problem="Пробелы вокруг идентификатора",
                            action="Пробелы удалены", rationale="По схеме пробелы незначимы", affected=int((cid_raw != cid_raw.str.strip()).sum()), unit="ячейки"))
        assert channel.isin(CHANNELS).all(), "Неизвестный канал обращения"
        part = pd.DataFrame({
            "ticket_id": d.ticket_id.astype(str).str.strip(),
            "customer_id": cid_raw.str.strip(),
            "source_system": name,
            "created_at": times["created_at"], "available_at": times["available_at"],
            "topic_raw": topic_raw, "channel": channel,
            "resolved_at": times["resolved_at"], "resolution_available_at": times["resolution_available_at"],
            "score_raw": d.satisfaction_score.astype(str).str.strip(),
            "created_at_zone_assumed": no_zone.astype(int).values,
            "_raw": d.astype(str).apply(lambda r: " | ".join(r.values), axis=1),
        })
        parts.append(part)
    s = pd.concat(parts, ignore_index=True)
    assert not s.ticket_id.duplicated().any(), "Конфликт ticket_id после удаления копий"

    unknown_topic = s.topic_raw.isna()
    s["topic"] = s.topic_raw.fillna("unknown")
    assert s.topic.isin(TOPICS).all()
    log.append(dict(source="support (обе системы)", field="topic", problem="Пустая тема",
                    action="Отдельная категория unknown", rationale="Схема допускает unknown; обращение остаётся в счётчиках",
                    affected=int(unknown_topic.sum()), unit="ячейки"))

    # Карантин целого обращения: нельзя установить абонента или безопасную доступность.
    act = pd.to_datetime(events[events.event_type == "activation"].set_index("customer_id").event_date)
    term = pd.to_datetime(events[events.event_type == "termination"].set_index("customer_id").event_date)
    known_ids = set(subs.customer_id)
    reasons = pd.Series("", index=s.index, dtype=object)
    reasons[s.customer_id == ""] = "customer_missing"
    reasons[(s.customer_id != "") & ~s.customer_id.isin(known_ids)] = "customer_unknown"
    reasons[(reasons == "") & (s.available_at < s.created_at)] = "available_before_created"
    lo, hi = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-01")
    reasons[(reasons == "") & ((s.created_at < lo) | (s.created_at >= hi))] = "created_outside_window"
    a, e = s.customer_id.map(act), s.customer_id.map(term)
    reasons[(reasons == "") & ((s.created_at < a) | (s.created_at >= e))] = "created_outside_contract"
    texts = {
        "customer_missing": "Пустой customer_id: обращение нельзя связать с абонентом",
        "customer_unknown": "customer_id отсутствует в реестре абонентов",
        "available_before_created": "available_at раньше created_at: безопасный момент доступности не установлен",
        "created_outside_window": "Создание вне интервала выгрузки [2025-09-01, 2026-09-01)",
        "created_outside_contract": "Создание вне периода действия договора",
    }
    q = s[reasons != ""]
    for i, r in q.iterrows():
        linked = r.customer_id in known_ids
        quarantine.append(dict(source_table=f"support_{r.source_system}", record_id=r.ticket_id, scope="row", field="*",
                               raw_value=r._raw, reason_code=reasons[i], reason=texts[reasons[i]],
                               customer_id=r.customer_id if linked else "",
                               month=f"{r.created_at:%Y-%m-01}" if linked and pd.notna(r.created_at) else ""))
    for code, cnt in reasons[reasons != ""].value_counts().sort_index().items():
        log.append(dict(source="support (обе системы)", field="*", problem=texts[code],
                        action="Обращение перенесено в карантин и исключено из агрегатов",
                        rationale="Нет абонента или безопасной даты доступности; связанные с абонентом случаи учтены в support_quarantine_count",
                        affected=int(cnt), unit="строки"))
    s = s[reasons == ""].copy()

    # Закрытие и оценка: ошибочная часть исключается, само обращение остаётся.
    bad_close = (s.resolved_at < s.created_at) | (s.resolution_available_at < s.resolved_at) | \
                (s.resolution_available_at < s.available_at) | (s.resolved_at.isna() != s.resolution_available_at.isna())
    for _, r in s[bad_close].iterrows():
        quarantine.append(dict(source_table=f"support_{r.source_system}", record_id=r.ticket_id, scope="resolution",
                               field="resolved_at,resolution_available_at,satisfaction_score",
                               raw_value=f"{r.resolved_at} | {r.resolution_available_at} | {r.score_raw}",
                               reason_code="resolution_time_order",
                               reason="Закрытие раньше создания: факт закрытия и оценка исключены, обращение сохранено",
                               customer_id=r.customer_id, month=f"{r.created_at:%Y-%m-01}"))
    s["resolution_removed"] = bad_close.astype(int)
    s.loc[bad_close, ["resolved_at", "resolution_available_at"]] = pd.NaT
    log.append(dict(source="support_current", field="resolved_at", problem="Закрытие раньше создания",
                    action="Поля закрытия и оценка очищены (resolution_removed=1), обращение остаётся в счётчиках создания",
                    rationale="Время создания и доступности корректны; неверен только факт закрытия",
                    affected=int(bad_close.sum()), unit="строки"))
    score = pd.to_numeric(s.score_raw.replace("", np.nan), errors="coerce")
    bad_score = s.score_raw.ne("") & (~score.between(1, 5) | bad_close)
    for _, r in s[bad_score & ~bad_close].iterrows():
        quarantine.append(dict(source_table=f"support_{r.source_system}", record_id=r.ticket_id, scope="cell",
                               field="satisfaction_score", raw_value=r.score_raw, reason_code="score_out_of_range",
                               reason="Оценка вне шкалы 1–5; значение исключено", customer_id=r.customer_id, month=f"{r.created_at:%Y-%m-01}"))
    s["satisfaction_score"] = score.where(~bad_score).astype("Int64")
    s["score_removed"] = (bad_score & ~bad_close).astype(int)
    log.append(dict(source="support_current", field="satisfaction_score", problem="Оценка 9 вне шкалы 1–5",
                    action="Заменена пропуском, score_removed=1, ячейка в карантине",
                    rationale="Пустая оценка допустима; угадывать балл нельзя", affected=int(s.score_removed.sum()), unit="ячейки"))
    s["created_month"] = month_start(s.created_at)
    s = s.sort_values(["created_at", "ticket_id"]).reset_index(drop=True)
    return s[["ticket_id", "customer_id", "source_system", "created_at", "available_at", "created_month", "topic", "channel",
              "resolved_at", "resolution_available_at", "satisfaction_score", "created_at_zone_assumed",
              "resolution_removed", "score_removed"]]


# --------------------------------------------------------------------------- объединение

def aggregate_support(rows: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты поддержки для строк (customer_id, billing_month, available_at).

    Обращение входит в строку, если создано в billing_month (по Москве) и его
    available_at не позже available_at строки. Закрытие и оценка учитываются,
    только если resolution_available_at не позже того же момента.
    Таблица tickets должна быть без повторных копий (dedupe до вызова).
    """
    keys = rows[["customer_id", "billing_month", "available_at"]].rename(columns={"available_at": "row_available_at"})
    t = tickets.merge(keys, left_on=["customer_id", "created_month"], right_on=["customer_id", "billing_month"], how="inner")
    t = t[t.available_at <= t.row_available_at].copy()
    closed = t.resolved_at.notna() & (t.resolution_available_at <= t.row_available_at)
    t["is_resolved"] = closed.astype(int)
    t["score_known"] = (closed & t.satisfaction_score.notna()).astype(int)
    t["score_value"] = pd.to_numeric(t.satisfaction_score, errors="coerce").astype(float).where(t.score_known == 1)
    for topic in TOPICS:
        t[f"is_{topic}"] = (t.topic == topic).astype(int)
    g = t.groupby(["customer_id", "billing_month"]).agg(
        support_count=("ticket_id", "size"),
        support_network_count=("is_network", "sum"),
        support_billing_count=("is_billing", "sum"),
        support_tariff_count=("is_tariff", "sum"),
        support_services_count=("is_services", "sum"),
        support_unknown_topic_count=("is_unknown", "sum"),
        support_resolved_count=("is_resolved", "sum"),
        support_score_count=("score_known", "sum"),
        support_score_sum=("score_value", "sum"),
    ).reset_index()
    out = rows.merge(g, on=["customer_id", "billing_month"], how="left", validate="one_to_one")
    count_cols = [c for c in g.columns if c.endswith("_count")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["support_score_mean"] = (out.support_score_sum / out.support_score_count.where(out.support_score_count > 0)).round(4)
    return out.drop(columns="support_score_sum")


def build_customer_month(billing: pd.DataFrame, subs: pd.DataFrame, support: pd.DataFrame, quarantine: pd.DataFrame) -> pd.DataFrame:
    cm = billing.merge(subs[["customer_id", "region", "acquisition_channel", "customer_segment", "activation_date"]],
                       on="customer_id", how="left", validate="many_to_one")
    cm = aggregate_support(cm, support)
    q = quarantine[(quarantine.scope == "row") & (quarantine.customer_id != "")]
    qc = q.groupby(["customer_id", "month"]).size().rename("support_quarantine_count").reset_index()
    qc["billing_month"] = pd.to_datetime(qc.month).astype("datetime64[ns]")
    cm = cm.merge(qc.drop(columns="month"), on=["customer_id", "billing_month"], how="left", validate="one_to_one")
    cm["support_quarantine_count"] = cm.support_quarantine_count.fillna(0).astype(int)
    cm["support_system"] = np.where(cm.billing_month < pd.Timestamp("2025-11-01"), "legacy", "current")
    return cm.sort_values(["customer_id", "billing_month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- проверки

def synthetic_checks() -> list[dict]:
    """Маленькие проверки на искусственных данных (шаг 5 задания)."""
    res = []
    rows = pd.DataFrame({"customer_id": ["A", "B"], "billing_month": pd.to_datetime(["2026-08-01"] * 2)})
    rows["available_at"] = pd.Timestamp("2026-09-01")

    def ticket(tid, cid, created, available, topic="network"):
        return dict(ticket_id=tid, customer_id=cid, created_at=pd.Timestamp(created), available_at=pd.Timestamp(available),
                    created_month=month_start(pd.Series([pd.Timestamp(created)]))[0], topic=topic,
                    resolved_at=pd.NaT, resolution_available_at=pd.NaT, satisfaction_score=np.nan)

    base = ticket("T1", "A", "2026-08-10 10:00", "2026-08-10 12:00")
    dup = pd.DataFrame([base, dict(base)])
    agg = aggregate_support(rows, dup.drop_duplicates(subset=["ticket_id", "customer_id", "created_at", "available_at"]))
    res.append(check_row("synthetic: повтор строки не удваивает счётчик", 1, int(agg.loc[0, "support_count"]),
                         agg.loc[0, "support_count"] == 1, "две одинаковые строки T1 -> удаление копий -> 1"))
    late = pd.DataFrame([ticket("T2", "A", "2026-08-31 23:00", "2026-09-01 09:00")])
    agg = aggregate_support(rows, late)
    res.append(check_row("synthetic: обращение после даты строки не входит", 0, int(agg.loc[0, "support_count"]),
                         agg.loc[0, "support_count"] == 0, "создано 31.08 23:00, загружено 01.09 09:00"))
    edge = pd.DataFrame([ticket("T3", "A", "2026-08-31 23:00", "2026-09-01 00:00"),
                         ticket("T4", "A", "2026-08-31 23:00", "2026-09-01 00:00:01")])
    agg = aggregate_support(rows, edge)
    res.append(check_row("synthetic: граница available_at = дата строки включается, +1 с — нет", 1,
                         int(agg.loc[0, "support_count"]), agg.loc[0, "support_count"] == 1))
    agg = aggregate_support(rows, pd.DataFrame([base]))
    ok = len(agg) == 2 and agg.set_index("customer_id").loc["B", "support_count"] == 0
    res.append(check_row("synthetic: абонент без обращений сохраняется после left join", "2 строки, B=0",
                         f"{len(agg)} строки, B={agg.set_index('customer_id').loc['B', 'support_count']}", ok))
    prev = pd.DataFrame([ticket("T5", "A", "2026-07-31 10:00", "2026-07-31 11:00")])
    agg = aggregate_support(rows, prev)
    res.append(check_row("synthetic: обращение прошлого месяца не попадает в текущий", 0, int(agg.loc[0, "support_count"]),
                         agg.loc[0, "support_count"] == 0))
    return res


# --------------------------------------------------------------------------- запуск

DICTIONARY = [
    # table, field, type, description, unit, available_when
    ("subscribers_clean", "customer_id", "String", "Идентификатор абонента", "—", "постоянное поле"),
    ("subscribers_clean", "activation_date", "Date", "Дата подключения (восстановлена по activation при пропуске)", "дата", "известна с даты подключения"),
    ("subscribers_clean", "activation_date_source", "String", "subscribers | contract_events | unknown — откуда взята дата", "—", "служебное"),
    ("subscribers_clean", "region", "String", "Регион обслуживания", "—", "постоянное поле"),
    ("subscribers_clean", "acquisition_channel", "String", "Канал подключения: online, store, partner", "—", "постоянное поле"),
    ("subscribers_clean", "customer_segment", "String", "individual | small_business", "—", "постоянное поле"),
    ("contract_events_clean", "event_id", "String", "Идентификатор события", "—", ""),
    ("contract_events_clean", "customer_id", "String", "Абонент", "—", ""),
    ("contract_events_clean", "event_date", "Date", "Дата события, действует с 00:00 МСК", "дата", "с 00:00 event_date; termination после T — только для метки"),
    ("contract_events_clean", "event_type", "String", "activation | termination", "—", ""),
    ("billing_clean", "customer_id, billing_month", "String, Date", "Составной ключ: абонент и первый день месяца услуг", "—", ""),
    ("billing_clean", "available_at", "DateTime", "Момент доступности строки биллинга", "ISO +03:00", "00:00 первого дня следующего месяца"),
    ("billing_clean", "tariff", "String", "Basic | Comfort | Premium (нормализовано)", "—", "available_at"),
    ("billing_clean", "monthly_fee_rub … closing_debt_rub", "Decimal", "Цена тарифа, начислено, оплачено, долг на начало и конец", "руб.", "available_at"),
    ("billing_clean", "traffic_gb", "Decimal nullable", "Трафик; пусто = неизвестно (не ноль)", "ГБ", "available_at"),
    ("billing_clean", "voice_minutes", "Integer nullable", "Минуты; пусто = неизвестно или отрицательное исходное значение", "мин.", "available_at"),
    ("billing_clean", "active_days, payment_delay_days, tariff_changes_count", "Integer", "Как в словаре занятия 1", "дни / шт.", "available_at"),
    ("billing_clean", "closing_debt_restored", "0/1", "Конечный долг восстановлен по равенству", "флаг", "диагностика качества"),
    ("billing_clean", "voice_minutes_invalid", "0/1", "Исходные минуты отрицательны и исключены", "флаг", "диагностика качества"),
    ("support_clean", "ticket_id", "String", "Ключ обращения (уникален после удаления копий)", "—", ""),
    ("support_clean", "customer_id", "String", "Абонент (без пробелов, есть в реестре)", "—", ""),
    ("support_clean", "source_system", "String", "legacy (сен–окт 2025) | current (с ноября 2025)", "—", ""),
    ("support_clean", "created_at", "DateTime", "Создание обращения", "ISO +03:00", "только вместе с available_at"),
    ("support_clean", "available_at", "DateTime", "Доступность записи о создании", "ISO +03:00", "определяет включение в снимок"),
    ("support_clean", "created_month", "Date", "Месяц создания по Москве", "месяц", ""),
    ("support_clean", "topic", "String", "network | billing | tariff | services | unknown", "—", "available_at"),
    ("support_clean", "channel", "String", "phone | chat | office", "—", "available_at"),
    ("support_clean", "resolved_at", "DateTime nullable", "Закрытие (пусто: не подтверждено или исключено)", "ISO +03:00", "только при resolution_available_at ≤ T"),
    ("support_clean", "resolution_available_at", "DateTime nullable", "Доступность закрытия и оценки", "ISO +03:00", ""),
    ("support_clean", "satisfaction_score", "Integer nullable", "Оценка 1–5", "балл", "только при resolution_available_at ≤ T"),
    ("support_clean", "created_at_zone_assumed, resolution_removed, score_removed", "0/1", "Флаги обработки", "флаг", "диагностика качества"),
    ("customer_month", "customer_id, billing_month", "String, Date", "Ключ: одна строка на строку billing_clean", "—", ""),
    ("customer_month", "available_at", "DateTime", "Доступность строки = available_at биллинга", "ISO +03:00", "первый день следующего месяца 00:00"),
    ("customer_month", "region, acquisition_channel, customer_segment, activation_date", "String/Date", "Постоянные поля абонента", "—", "постоянные"),
    ("customer_month", "support_count", "Integer", "Обращения, созданные в billing_month и доступные не позже available_at строки", "шт.", "available_at строки"),
    ("customer_month", "support_network_count / billing / tariff / services / unknown_topic", "Integer", "То же по темам", "шт.", "available_at строки"),
    ("customer_month", "support_resolved_count", "Integer", "Из них закрытия с resolution_available_at ≤ available_at строки", "шт.", "available_at строки"),
    ("customer_month", "support_score_count", "Integer", "Число доступных оценок", "шт.", "available_at строки"),
    ("customer_month", "support_score_mean", "Decimal nullable", "Средняя доступная оценка; пусто, если оценок 0", "балл", "available_at строки"),
    ("customer_month", "support_quarantine_count", "Integer", "Обращения абонента за месяц в карантине (диагностика, НЕ признак)", "шт.", "доступность не подтверждена"),
    ("customer_month", "support_system", "String", "Система учёта поддержки в этом месяце (legacy/current)", "—", "контекст сопоставимости"),
]


def run(cfg: dict) -> dict:
    P = cfg["abs_paths"]
    out = STAGE2_OUT
    out.mkdir(parents=True, exist_ok=True)
    raw_names = ["subscribers", "billing", "contract_events", "support_sqlite", "support_current_export", "support_legacy", "schema_support"]
    hashes_before = {n: sha256(P[n]) for n in raw_names}

    subs_raw = read_raw_csv(P["subscribers"])
    bill_raw = read_raw_csv(P["billing"])
    ev_raw = read_raw_csv(P["contract_events"])
    legacy_raw = read_raw_csv(P["support_legacy"])
    schema = read_raw_csv(P["schema_support"])
    export_raw = read_raw_csv(P["support_current_export"])
    if cfg.get("support_read_mode", "sqlite") == "sqlite":
        current_raw, read_mode = extract_sqlite(P["support_sqlite"]), "SQLite (sql/extract_support.sql, mode=ro)"
    else:
        current_raw, read_mode = export_raw.copy(), "CSV-копия (резервный способ)"
    write_csv(run_sqlite_diagnostics(P["support_sqlite"]), out / "sqlite_diagnostics.csv")
    same_as_export = current_raw.sort_values("ticket_id", kind="stable").reset_index(drop=True).equals(
        export_raw.sort_values("ticket_id", kind="stable").reset_index(drop=True))

    # Шаг 1. Инвентаризация
    inv = []
    def inventory(name, df, key, times, method, path):
        row = dict(source=name, file=Path(path).relative_to(SOLUTION_DIR.parent).as_posix(), sha256=sha256(path), read_method=method,
                   rows=len(df), columns=len(df.columns), fields=", ".join(df.columns), key=key,
                   key_unique_values=int(df[key.split("+")].drop_duplicates().shape[0]) if key else "")
        row["time_range_msk"] = "; ".join(f"{c}: {v.min():%Y-%m-%d %H:%M} … {v.max():%Y-%m-%d %H:%M}" for c, v in times.items())
        row["types_observed"] = "; ".join(f"{c}: {kind}" for c, kind in _observed_types(df).items())
        inv.append(row)
    leg_t = {c: parse_local(legacy_raw[c], "%d.%m.%Y %H:%M:%S") for c in ["opened_local", "loaded_local"]}
    cur_t = {c: to_msk(current_raw[c])[0] for c in ["created_at", "available_at"]}
    inventory("subscribers", subs_raw, "customer_id",
              {"activation_date": pd.to_datetime(subs_raw.activation_date.replace("", np.nan))}, "CSV", P["subscribers"])
    inventory("billing", bill_raw, "customer_id+billing_month", {"available_at": pd.to_datetime(bill_raw.available_at)}, "CSV", P["billing"])
    inventory("contract_events", ev_raw, "event_id", {"event_date": pd.to_datetime(ev_raw.event_date)}, "CSV", P["contract_events"])
    inventory("support_legacy", legacy_raw, "ticket_no", leg_t, "CSV", P["support_legacy"])
    inventory("support_current", current_raw, "ticket_id", cur_t, read_mode, P["support_sqlite"])
    inventory("support_current_export (только сверка)", export_raw, "ticket_id", {}, "CSV, не объединяется с SQLite", P["support_current_export"])
    inv_df = pd.DataFrame(inv)
    write_csv(inv_df, out / "inventory.csv")

    # Шаг 2. Проверка качества до преобразования
    ev_tmp = ev_raw.apply(lambda c: c.astype(str).str.strip())
    sub_ids = set(subs_raw.customer_id.astype(str).str.strip())
    rename, topic_map, channel_map = parse_schema_maps(schema)
    before = Issues("before")
    audit_subscribers(subs_raw, before, ev_tmp, raw=True)
    audit_events(ev_raw, before, sub_ids)
    audit_billing(bill_raw, before, sub_ids, ev_tmp, clean=False)
    audit_support(legacy_raw.rename(columns=rename), before, sub_ids, "support_legacy", False, list(topic_map), list(channel_map))
    audit_support(current_raw, before, sub_ids, "support_current", False, TOPICS[:-1], CHANNELS)

    # Шаг 3. Согласование и очистка
    log, quarantine = [], []
    events = clean_events(ev_raw)
    ev_str = events.assign(event_date=fmt_date(events.event_date))
    subs = clean_subscribers(subs_raw, ev_str, log)
    billing = clean_billing(bill_raw, log, quarantine)
    support = clean_support(legacy_raw, current_raw, schema, subs, ev_str, log, quarantine)
    q = pd.DataFrame(quarantine, columns=["source_table", "record_id", "scope", "field", "raw_value", "reason_code", "reason", "customer_id", "month"])
    q = q.sort_values(["source_table", "scope", "record_id"]).reset_index(drop=True)

    # Шаг 4. Объединение
    cm = build_customer_month(billing, subs, support, q)

    # Запись
    write_csv(subs.assign(activation_date=fmt_date(subs.activation_date)), out / "subscribers_clean.csv")
    write_csv(ev_str, out / "contract_events_clean.csv")
    bill_out = billing.assign(billing_month=fmt_date(billing.billing_month), available_at=fmt_ts(billing.available_at))
    write_csv(bill_out, out / "billing_clean.csv")
    sup_out = support.assign(**{c: fmt_ts(support[c]) for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]},
                             created_month=fmt_date(support.created_month))
    write_csv(sup_out, out / "support_clean.csv")
    cm_out = cm.assign(billing_month=fmt_date(cm.billing_month), available_at=fmt_ts(cm.available_at),
                       activation_date=fmt_date(cm.activation_date))
    write_csv(cm_out, out / "customer_month.csv")
    write_csv(q, out / "quarantine.csv")

    # Повтор проверок после очистки (на записанных таблицах)
    after = Issues("after")
    s2, e2, b2, sp2 = (read_raw_csv(out / f) for f in ["subscribers_clean.csv", "contract_events_clean.csv", "billing_clean.csv", "support_clean.csv"])
    audit_subscribers(s2, after, e2, raw=False)
    audit_events(e2, after, set(s2.customer_id))
    audit_billing(b2.drop(columns=["closing_debt_restored", "voice_minutes_invalid"]), after, set(s2.customer_id), e2, clean=True)
    for name in ["legacy", "current"]:
        part = sp2[sp2.source_system == name]
        audit_support(part, after, set(s2.customer_id), f"support_{name}", True, TOPICS, CHANNELS)

    qb, qa = before.frame(), after.frame()
    write_csv(qb, out / "quality_before.csv")
    write_csv(qa, out / "quality_after.csv")
    log_df = pd.DataFrame(log)
    log_df.insert(0, "decision_id", [f"D{i + 1:02d}" for i in range(len(log_df))])
    write_csv(log_df, out / "decision_log.csv")
    write_csv(pd.DataFrame(DICTIONARY, columns=["table", "field", "data_type", "description", "unit", "available_when"]),
              out / "data_dictionary_prepared.csv")

    # Шаг 5–6. Проверки
    checks = []
    hashes_after = {n: sha256(P[n]) for n in raw_names}
    checks.append(check_row("исходные файлы не изменены (sha256 до/после)", "равны", "равны" if hashes_before == hashes_after else "различаются",
                            hashes_before == hashes_after))
    checks.append(check_row("SQLite-выгрузка совпадает с CSV-экспортом", True, same_as_export, same_as_export, "используется один путь чтения"))
    checks.append(check_row("абоненты не удалены", len(subs_raw), len(subs), len(subs) == len(subs_raw)))
    checks.append(check_row("у всех абонентов есть дата подключения", 0, int(subs.activation_date.isna().sum()), subs.activation_date.notna().all()))
    checks.append(check_row("ключ billing_clean уникален", 0, int(billing.duplicated(["customer_id", "billing_month"]).sum()),
                            not billing.duplicated(["customer_id", "billing_month"]).any()))
    checks.append(check_row("billing_clean = исходные строки − копии", len(bill_raw) - int(bill_raw.duplicated().sum()), len(billing),
                            len(billing) == len(bill_raw) - int(bill_raw.duplicated().sum())))
    eq = (billing.opening_debt_rub + billing.charged_rub - billing.paid_rub - billing.closing_debt_rub).abs() > 0.005
    checks.append(check_row("равенство долга выполняется во всех строках", 0, int(eq.sum()), not eq.any()))
    checks.append(check_row("ключ support_clean уникален", 0, int(support.ticket_id.duplicated().sum()), not support.ticket_id.duplicated().any()))
    checks.append(check_row("все обращения support_clean связаны с реестром", 0, int((~support.customer_id.isin(subs.customer_id)).sum()),
                            support.customer_id.isin(subs.customer_id).all()))
    checks.append(check_row("available_at ≥ created_at в support_clean", 0, int((support.available_at < support.created_at).sum()),
                            not (support.available_at < support.created_at).any()))
    checks.append(check_row("customer_month: строк = billing_clean", len(billing), len(cm), len(cm) == len(billing)))
    checks.append(check_row("customer_month: ключ уникален", 0, int(cm.duplicated(["customer_id", "billing_month"]).sum()),
                            not cm.duplicated(["customer_id", "billing_month"]).any()))
    sums_ok = all(abs(cm[c].sum() - billing[c].sum()) < 0.005 for c in MONEY + ["traffic_gb"])
    checks.append(check_row("customer_month: суммы денег и трафика = billing_clean", "равны", "равны" if sums_ok else "различаются", sums_ok))
    checks.append(check_row("customer_month: пропуски трафика сохранены", int(billing.traffic_gb.isna().sum()), int(cm.traffic_gb.isna().sum()),
                            cm.traffic_gb.isna().sum() == billing.traffic_gb.isna().sum()))
    t = support.merge(cm[["customer_id", "billing_month", "available_at"]].rename(columns={"available_at": "row_av"}),
                      left_on=["customer_id", "created_month"], right_on=["customer_id", "billing_month"], how="left")
    no_row = int(t.row_av.isna().sum())
    checks.append(check_row("каждое обращение попадает в существующую строку биллинга", 0, no_row, no_row == 0))
    included = int((t.available_at <= t.row_av).sum())
    checks.append(check_row("сумма support_count = обращения, доступные к дате строки", included, int(cm.support_count.sum()),
                            included == int(cm.support_count.sum()), f"не вошли из-за позднего поступления: {len(t) - included}"))
    checks += synthetic_checks()
    ch = pd.DataFrame(checks)
    write_csv(ch, out / "checks_stage2.csv")

    summary = {
        "read_mode_support_current": read_mode,
        "rows_raw": {r["source"]: r["rows"] for r in inv},
        "rows_clean": {"subscribers_clean": len(subs), "contract_events_clean": len(events), "billing_clean": len(billing),
                       "support_clean": len(support), "customer_month": len(cm), "quarantine": len(q)},
        "quarantine_by_reason": q.groupby(["scope", "reason_code"]).size().rename("n").reset_index().to_dict("records"),
        "quality_before": {t_: before.summary(t_, 0) | {"removed_copies": int(qb[(qb.table == t_) & (qb.check == "full_duplicate_copies")]["count"].sum())}
                           for t_ in qb.table.unique()},
        "quality_after": {t_: after.summary(t_, 0) | {"removed_copies": int(qa[(qa.table == t_) & (qa.check == "full_duplicate_copies")]["count"].sum())}
                          for t_ in qa.table.unique()},
        "support_tickets_not_in_month_snapshot_late": int(len(t) - included),
        "customer_month_rows_with_support": int((cm.support_count > 0).sum()),
        "customer_month_share_without_support": round(float((cm.support_count == 0).mean()), 4),
        "checks_failed": int((ch.passed == "FAIL").sum()),
    }
    write_json(summary, out / "quality_summary.json")
    return summary
