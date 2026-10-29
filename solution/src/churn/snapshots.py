"""Активная база на дату расчёта T и метка оттока (общие для этапов 3 и 4).

Правило устава (active_mode = before_events): база — состояние непосредственно
перед выполнением событий на T:
    activation_date < T  и  (расторжения нет  или  termination_date >= T).
Подключение в день T в базу не входит; расторжение в день T входит в базу и в метку.

Горизонт calendar_month: churn_target = 1, если termination в [T, T + 1 месяц).
Если конец горизонта позже observation_end_exclusive, исход неизвестен (NaN), а не 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import add_months


def contract_dates(subs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на абонента: activation_date и termination_date (NaT, если не было)."""
    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    term = ev[ev.event_type == "termination"].set_index("customer_id").event_date
    out = subs[["customer_id"]].copy()
    out["activation_date"] = pd.to_datetime(subs.activation_date).values
    out["termination_date"] = out.customer_id.map(term)
    return out


def active_base(contracts: pd.DataFrame, T: pd.Timestamp, mode: str = "before_events") -> pd.DataFrame:
    if mode != "before_events":
        raise ValueError("В уставе согласовано правило before_events; другое правило требует пересогласования")
    a, e = contracts.activation_date, contracts.termination_date
    mask = (a < T) & (e.isna() | (e >= T))
    base = contracts.loc[mask, ["customer_id"]].copy()
    base.insert(1, "snapshot_at", T)
    return base.reset_index(drop=True)


def independent_active_count(events: pd.DataFrame, T: pd.Timestamp) -> int:
    """Независимый расчёт размера базы: подключения до T минус расторжения до T (по журналу)."""
    d = pd.to_datetime(events.event_date)
    return int(((events.event_type == "activation") & (d < T)).sum() - ((events.event_type == "termination") & (d < T)).sum())


def horizon_end(T: pd.Timestamp, horizon: str) -> pd.Timestamp:
    if horizon == "calendar_month":
        return add_months(T, 1)
    if horizon == "30_days":
        return T + pd.Timedelta(days=30)
    raise ValueError(horizon)


def make_labels(base: pd.DataFrame, contracts: pd.DataFrame, horizon: str, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Метка формируется отдельно от признаков; дата расторжения в результат не попадает."""
    term = base.customer_id.map(contracts.set_index("customer_id").termination_date)
    T = pd.to_datetime(base.snapshot_at)
    end = T.map(lambda t: horizon_end(t, horizon))
    observed = end <= observation_end
    churn = ((term >= T) & (term < end)).astype(float)
    out = base[["customer_id", "snapshot_at"]].copy()
    out["churn_target"] = churn.where(observed, np.nan)
    out["target_end_exclusive"] = end.values
    out["label_observed"] = observed.astype(int).values
    return out
