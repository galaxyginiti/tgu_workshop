"""Этап 4. Признаки на дату расчёта, метки, разбиение, преобразование и baseline.

Вход: таблицы этапа 2, 4_features/data/digital_usage_monthly.csv, 3_eda/data/network_quality_monthly.csv.
Выход (solution/stage4_features/output):
  features.csv, labels.csv, split.csv, feature_audit.csv, feature_catalog.csv,
  x_columns.csv, preprocessing_report.json, baseline_train_metrics.csv,
  digital_checks.csv, checks_stage4.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (STAGE2_OUT, STAGE4_OUT, add_months, check_row, fmt_date, fmt_ts, read_prepared,
                     read_raw_csv, snapshot_dates, to_msk, write_csv, write_json)
from .preprocessing import make_preprocessor
from .snapshots import active_base, contract_dates, independent_active_count, make_labels

KEYS = ["customer_id", "snapshot_at"]

# Паспорт признаков: порядок = порядок столбцов features.csv.
FEATURES = [
    # name, group, x_type, meaning, formula, source, window, unit, missing_policy, zero_policy, eda_reason, leakage_risk, include_in_X
    ("tenure_days", "договор/сегмент", "numeric", "Стаж абонента на T", "(T − activation_date) в днях", "subscribers_clean", "на T", "дни",
     "не бывает (дата восстановлена на этапе 2)", "0 невозможен: activation < T", "H4: новички уходят чаще (3,6% vs 2,6%), слабее долга", "низкий: постоянная дата", "да"),
    ("region", "договор/сегмент", "categorical", "Регион обслуживания", "значение справочника", "subscribers_clean", "постоянно", "—",
     "категория __MISSING__", "—", "различия по регионам 2,4–3,2% (churn_by_region.csv)", "низкий", "да"),
    ("acquisition_channel", "договор/сегмент", "categorical", "Канал подключения", "значение справочника", "subscribers_clean", "постоянно", "—",
     "категория __MISSING__", "—", "контроль состава; различия малы", "низкий", "да"),
    ("customer_segment", "договор/сегмент", "categorical", "Сегмент: individual / small_business", "значение справочника", "subscribers_clean", "постоянно", "—",
     "категория __MISSING__", "—", "контроль состава; различий нет", "низкий", "да"),
    ("tariff_last", "договор/сегмент", "categorical", "Тариф в предыдущем месяце", "tariff строки M−1", "billing_clean", "M−1", "—",
     "пусто, если строки M−1 нет или она недоступна на T", "—", "Premium 3,1% vs Basic 2,5%; долг сильнее внутри тарифа", "низкий: available_at = T", "да"),
    ("debt_last_rub", "биллинг", "numeric", "Долг на конец предыдущего месяца", "closing_debt_rub строки M−1", "billing_clean", "M−1", "руб.",
     "пусто, если строки M−1 нет; не 0", "0 — измеренное отсутствие долга", "H1: долг > 0 → 7,5% оттока против 1,4%", "низкий: строка M−1 доступна ровно в T", "да"),
    ("debt_months_3m", "биллинг", "numeric", "Число месяцев с долгом > 0 за 3 месяца", "Σ[closing_debt_rub > 0] по M−3…M−1", "billing_clean", "3 мес.", "мес.",
     "пусто, если доступных строк нет; полнота — billing_months_3m", "0 — долгов не было в известных месяцах", "H1: устойчивость долга", "средний: каждая строка с available_at ≤ T", "да"),
    ("debt_change_1m_rub", "биллинг", "numeric", "Изменение долга M−1 к M−2", "debt(M−1) − debt(M−2)", "billing_clean", "M−2, M−1", "руб.",
     "пусто, если нет любого из двух месяцев", "0 — долг не изменился", "H1: рост долга как сигнал", "низкий", "да"),
    ("payment_delay_last_days", "биллинг", "numeric", "Задержка оплаты в предыдущем месяце", "payment_delay_days строки M−1", "billing_clean", "M−1", "дни",
     "пусто, если строки M−1 нет", "0 — оплата вовремя", "> 10 дней: 8,7% оттока против 1,4% при 0 (churn_by_payment_delay.csv)", "низкий", "да"),
    ("paid_to_charged_last", "биллинг", "numeric", "Отношение оплаты к начислению в M−1", "paid_rub / charged_rub; при charged ≤ 0 — пусто", "billing_clean", "M−1", "доля",
     "пусто при нулевом/неизвестном начислении", "0 — ничего не оплачено; > 1 — погашение старого долга, не ошибка", "оплата = 0: 10,6% оттока (churn_by_paid_zero.csv)", "низкий", "да"),
    ("tariff_changes_3m", "биллинг", "numeric", "Число смен тарифа за 3 месяца", "Σ tariff_changes_count по M−3…M−1", "billing_clean", "3 мес.", "шт.",
     "пусто, если строк нет", "0 — смен не было", "гипотеза: смена тарифа перед уходом (не проверена в EDA)", "низкий", "да"),
    ("traffic_last_gb", "использование", "numeric", "Трафик предыдущего месяца", "traffic_gb строки M−1", "billing_clean", "M−1", "ГБ",
     "пусто = неизвестно (пропуск в биллинге или нет строки); не 0", "в данных нулей нет; 0 был бы измеренным нулём", "нижний квинтиль трафика: 3,6% vs ~2,5%", "низкий", "да"),
    ("traffic_avg_3m_gb", "использование", "numeric", "Средний трафик за 3 месяца по известным месяцам", "mean(traffic_gb) по известным из M−3…M−1", "billing_clean", "3 мес.", "ГБ",
     "пусто, если известных месяцев нет; число — traffic_known_months_3m", "—", "направление 2: изменение использования", "средний: неполный месяц занижает сумму", "да"),
    ("traffic_per_active_day_3m", "использование", "numeric", "Трафик на активный день за 3 месяца", "Σ traffic_gb / Σ active_days по месяцам с известным трафиком", "billing_clean", "3 мес.", "ГБ/день",
     "пусто при нулевом знаменателе (нет месяцев с известным трафиком)", "—", "нормировка неполного месяца подключения/короткой истории", "низкий", "да"),
    ("traffic_change_1m_gb", "использование", "numeric", "Изменение трафика M−1 к M−2", "traffic(M−1) − traffic(M−2)", "billing_clean", "M−2, M−1", "ГБ",
     "пусто, если любой из двух месяцев неизвестен; предыдущий не подставляется", "0 — без изменений", "направление 2 EDA", "низкий", "да"),
    ("voice_avg_3m_min", "использование", "numeric", "Средние минуты за 3 месяца по известным месяцам", "mean(voice_minutes) по известным из M−3…M−1", "billing_clean", "3 мес.", "мин.",
     "пусто, если известных месяцев нет (отрицательные исключены на этапе 2)", "—", "связь слабая (Spearman −0,02), контроль использования", "низкий", "да"),
    ("support_count_3m", "поддержка", "numeric", "Обращения за 3 месяца, известные на T", "#{created_at ∈ [M−3, T), available_at ≤ T}", "support_clean", "3 мес.", "шт.",
     "0 — нет доступных корректно связанных обращений", "0 ≠ доказательство отсутствия реальных обращений", "H2: связь слабая; окно 3 мес. — проверка", "средний: пересчёт на T включает поздно поступившие, но известные к T", "да"),
    ("support_billing_count_3m", "поддержка", "numeric", "Обращения об оплате за 3 месяца", "то же, topic = billing", "support_clean", "3 мес.", "шт.",
     "как support_count_3m", "как support_count_3m", "обращения об оплате могут дополнять сигнал долга", "средний", "да"),
    ("support_network_count_3m", "поддержка", "numeric", "Обращения о качестве связи за 3 месяца", "то же, topic = network", "support_clean", "3 мес.", "шт.",
     "как support_count_3m", "как support_count_3m", "связь не найдена (churn_by_network_ticket.csv) — предварительный", "средний", "да"),
    ("support_count_last", "поддержка", "numeric", "Обращения за предыдущий месяц, известные на T", "#{created_at ∈ [M−1, T), available_at ≤ T}", "support_clean", "M−1", "шт.",
     "0 — нет доступных записей", "как support_count_3m", "соответствует support_count EDA", "низкий", "да"),
    ("digital_status_last", "цифровой источник", "categorical", "Состояние отчёта кабинета за M−1 на T", "observed / not_enabled / export_gap / not_published (отчёт ещё не опубликован к T)",
     "digital_usage_monthly", "M−1", "—", "not_published — отдельная категория, не нулевая активность", "—", "новая гипотеза: отсутствие наблюдения ≠ отсутствие активности", "высокий без фильтра: статус известен только после available_at", "да"),
    ("app_sessions_last", "цифровой источник", "numeric", "Сессии личного кабинета в M−1", "app_sessions при observed и available_at ≤ T", "digital_usage_monthly", "M−1", "сессии",
     "пусто при not_enabled, export_gap, not_published", "0 — измеренная нулевая активность (только observed)", "предварительная: снижение вовлечённости", "высокий без фильтра доступности", "да"),
    ("sessions_per_observed_day_last", "цифровой источник", "numeric", "Сессии на наблюдаемый день в M−1", "app_sessions / observed_days при observed_days > 0", "digital_usage_monthly", "M−1", "сессии/день",
     "пусто, если отчёт не observed или observed_days = 0", "0 — сессий не было при наблюдении", "нормировка неполного месяца", "как app_sessions_last", "да"),
    ("payment_failure_rate_last", "цифровой источник", "numeric", "Доля неуспешных попыток оплаты в M−1", "payment_failures / payment_attempts при attempts > 0", "digital_usage_monthly", "M−1", "доля",
     "пусто при 0 попыток (счётчик — payment_attempts_last) и при неизвестном отчёте", "0 — попытки были, ошибок нет", "предварительная: проблемы оплаты дополняют долг", "как app_sessions_last", "да"),
    ("payment_attempts_last", "цифровой источник", "numeric", "Попытки оплаты в кабинете в M−1", "payment_attempts при observed", "digital_usage_monthly", "M−1", "попытки",
     "пусто, если отчёт не observed", "0 — измеренный ноль попыток", "знаменатель доли ошибок", "как app_sessions_last", "да"),
    ("selfservice_actions_last", "цифровой источник", "numeric", "Действия самообслуживания в M−1", "selfservice_actions при observed", "digital_usage_monthly", "M−1", "действия",
     "пусто, если отчёт не observed", "0 — измеренный ноль", "предварительная", "как app_sessions_last", "да"),
    ("app_sessions_avg_3m", "цифровой источник", "numeric", "Средние сессии за 3 месяца по известным observed-отчётам", "mean(app_sessions) по observed, available_at ≤ T", "digital_usage_monthly", "3 мес.", "сессии",
     "пусто, если известных observed-месяцев нет; число — digital_observed_months_3m", "—", "сглаживает задержку отчёта M−1", "средний", "да"),
    ("app_sessions_latest_observed", "цифровой источник", "numeric", "Сессии в последнем известном на T observed-отчёте", "app_sessions месяца digital_latest_observed_month", "digital_usage_monthly", "≤ M−1", "сессии",
     "пусто, если известных observed-отчётов нет", "0 — измеренный ноль", "альтернатива «последний доступный отчёт» с явным возрастом", "средний: обязательно вместе с возрастом", "да"),
    ("digital_latest_observed_age_months", "цифровой источник", "numeric", "Возраст последнего известного observed-отчёта", "months(T) − months(usage_month); 1 = M−1", "digital_usage_monthly", "≤ M−1", "мес.",
     "пусто, если отчётов нет", "—", "делает видимой подмену месяца", "низкий", "да"),
    # служебная полнота
    ("billing_months_3m", "полнота", "numeric", "Доступных строк биллинга в окне 3 месяца", "#строк M−3…M−1 с available_at ≤ T", "billing_clean", "3 мес.", "мес.",
     "0 при отсутствии", "0–3: короткая история не дополняется нулями", "короткая история (H4)", "низкий", "да"),
    ("traffic_known_months_3m", "полнота", "numeric", "Месяцев с известным трафиком в окне", "count(traffic_gb) по M−3…M−1", "billing_clean", "3 мес.", "мес.",
     "0 при отсутствии", "—", "полнота среднего трафика", "низкий", "да"),
    ("digital_observed_months_3m", "полнота", "numeric", "Известных на T observed-отчётов в окне", "#месяцев M−3…M−1 со статусом observed и available_at ≤ T", "digital_usage_monthly", "3 мес.", "мес.",
     "0 при отсутствии", "—", "полнота цифровых средних", "низкий", "да"),
    # дополнительный региональный показатель — кандидат, в X не входит
    ("region_fail_rate_latest_known", "сеть (доп.)", "numeric", "Доля неуспешных проб региона в последнем опубликованном к T отчёте", "failed/probes отчёта с report_available_at ≤ T", "network_quality_monthly", "≤ M−1", "доля",
     "не бывает пустым на T ≥ 01.12.2025", "—", "EDA: связь с оттоком не подтверждена (16 известных пар, Spearman −0,24)", "средний: отчёт M−1 часто опубликован позже T", "нет"),
    ("region_fail_report_age_months", "сеть (доп.)", "numeric", "Возраст использованного регионального отчёта", "months(T) − months(measurement_month)", "network_quality_monthly", "≤ M−1", "мес.",
     "—", "—", "делает видимой задержку публикации", "низкий", "нет"),
]
CATALOG_COLS = ["feature_name", "group", "x_type", "meaning", "formula", "source", "window", "unit", "missing_policy", "zero_policy",
                "eda_reason", "leakage_risk", "include_in_X"]
AVAILABLE_WHEN = {
    "subscribers_clean": "постоянное поле / дата подключения < T",
    "billing_clean": "строка с available_at ≤ T (для M−1 — ровно T 00:00)",
    "support_clean": "available_at ≤ T",
    "digital_usage_monthly": "available_at строки ≤ T (статус и значения — вместе)",
    "network_quality_monthly": "report_available_at ≤ T",
}


# --------------------------------------------------------------------------- данные

def load_inputs(cfg: dict) -> dict:
    subs = read_prepared(STAGE2_OUT / "subscribers_clean.csv")
    events = read_prepared(STAGE2_OUT / "contract_events_clean.csv")
    b = read_prepared(STAGE2_OUT / "billing_clean.csv")
    b["billing_month"] = pd.to_datetime(b.billing_month)
    b["available_at"] = to_msk(b.available_at)[0]
    for c in ["charged_rub", "paid_rub", "closing_debt_rub", "traffic_gb", "voice_minutes", "active_days",
              "payment_delay_days", "tariff_changes_count"]:
        b[c] = pd.to_numeric(b[c]).astype(float)
    sp = read_prepared(STAGE2_OUT / "support_clean.csv")
    sp["created_at"] = to_msk(sp.created_at)[0]
    sp["available_at"] = to_msk(sp.available_at)[0]
    dg = read_raw_csv(cfg["abs_paths"]["digital_usage"]).replace("", np.nan)
    dg["usage_month"] = pd.to_datetime(dg.usage_month)
    dg["available_at"] = to_msk(dg.available_at)[0]
    for c in ["observed_days", "app_sessions", "payment_attempts", "payment_failures", "selfservice_actions"]:
        dg[c] = pd.to_numeric(dg[c]).astype(float)
    net = read_raw_csv(cfg["abs_paths"]["network_quality"])
    net["measurement_month"] = pd.to_datetime(net.measurement_month)
    net["report_available_at"] = to_msk(net.report_available_at)[0]
    net["failed_probe_rate"] = pd.to_numeric(net.failed_probe_count) / pd.to_numeric(net.probe_count)
    return {"subs": subs, "events": events, "contracts": contract_dates(subs, events), "billing": b,
            "support": sp, "digital": dg, "network": net}


def _months_between(T: pd.Timestamp, m: pd.Series) -> pd.Series:
    return (T.year - m.dt.year) * 12 + (T.month - m.dt.month)


# --------------------------------------------------------------------------- признаки на одну дату

def build_features_at(T: pd.Timestamp, D: dict, active_mode: str = "before_events") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Признаки и аудит на дату T. Используются только записи с моментом доступности ≤ T."""
    base = active_base(D["contracts"], T, active_mode)
    ids = base.customer_id
    M1, M2, M3 = add_months(T, -1), add_months(T, -2), add_months(T, -3)
    f = base.copy()
    a = f.copy()  # аудит
    subs = D["subs"].set_index("customer_id")
    act = D["contracts"].set_index("customer_id").activation_date
    f["tenure_days"] = (T - ids.map(act)).dt.days.astype(float).values
    for c in ["region", "acquisition_channel", "customer_segment"]:
        f[c] = ids.map(subs[c]).values

    # биллинг: окно M−3…M−1, каждая строка доступна к T
    b = D["billing"]
    bw = b[b.customer_id.isin(ids) & (b.billing_month >= M3) & (b.billing_month < T) & (b.available_at <= T)]
    b1 = bw[bw.billing_month == M1].set_index("customer_id")
    b2 = bw[bw.billing_month == M2].set_index("customer_id")
    g = bw.groupby("customer_id")
    m = lambda s: ids.map(s).values  # noqa: E731
    f["tariff_last"] = m(b1.tariff)
    f["debt_last_rub"] = m(b1.closing_debt_rub)
    f["debt_months_3m"] = m(g.closing_debt_rub.apply(lambda v: float((v > 0).sum())))
    f["debt_change_1m_rub"] = m(b1.closing_debt_rub - b2.closing_debt_rub.reindex(b1.index))
    f["payment_delay_last_days"] = m(b1.payment_delay_days)
    f["paid_to_charged_last"] = m(b1.paid_rub / b1.charged_rub.where(b1.charged_rub > 0))
    f["tariff_changes_3m"] = m(g.tariff_changes_count.sum(min_count=1))
    f["traffic_last_gb"] = m(b1.traffic_gb)
    f["traffic_avg_3m_gb"] = m(g.traffic_gb.mean())
    known_tr = bw[bw.traffic_gb.notna()].groupby("customer_id")
    days = known_tr.active_days.sum()
    f["traffic_per_active_day_3m"] = m(known_tr.traffic_gb.sum() / days.where(days > 0))
    f["traffic_change_1m_gb"] = m(b1.traffic_gb - b2.traffic_gb.reindex(b1.index))
    f["voice_avg_3m_min"] = m(g.voice_minutes.mean())

    # поддержка: пересчёт на T по support_clean
    sp = D["support"]
    sw = sp[sp.customer_id.isin(ids) & (sp.created_at >= M3) & (sp.created_at < T) & (sp.available_at <= T)]
    cnt = lambda d: ids.map(d.groupby("customer_id").size()).fillna(0).astype(float).values  # noqa: E731
    f["support_count_3m"] = cnt(sw)
    f["support_billing_count_3m"] = cnt(sw[sw.topic == "billing"])
    f["support_network_count_3m"] = cnt(sw[sw.topic == "network"])
    f["support_count_last"] = cnt(sw[sw.created_at >= M1])

    # цифровой источник
    dg = D["digital"]
    dc = dg[dg.customer_id.isin(ids) & (dg.usage_month >= M3) & (dg.usage_month < T)]
    d1 = dc[dc.usage_month == M1].set_index("customer_id")
    known1 = d1.available_at <= T
    status = d1.observation_status.where(known1, "not_published")
    f["digital_status_last"] = ids.map(status).fillna("no_row").values
    obs1 = d1[known1 & (d1.observation_status == "observed")]
    f["app_sessions_last"] = m(obs1.app_sessions)
    f["sessions_per_observed_day_last"] = m(obs1.app_sessions / obs1.observed_days.where(obs1.observed_days > 0))
    f["payment_failure_rate_last"] = m(obs1.payment_failures / obs1.payment_attempts.where(obs1.payment_attempts > 0))
    f["payment_attempts_last"] = m(obs1.payment_attempts)
    f["selfservice_actions_last"] = m(obs1.selfservice_actions)
    dobs = dc[(dc.available_at <= T) & (dc.observation_status == "observed")]
    f["app_sessions_avg_3m"] = m(dobs.groupby("customer_id").app_sessions.mean())
    hist = dg[dg.customer_id.isin(ids) & (dg.usage_month < T) & (dg.available_at <= T) & (dg.observation_status == "observed")]
    latest = hist.sort_values("usage_month").groupby("customer_id").tail(1).set_index("customer_id")
    f["app_sessions_latest_observed"] = m(latest.app_sessions)
    f["digital_latest_observed_age_months"] = m(_months_between(T, latest.usage_month).astype(float))

    # полнота
    f["billing_months_3m"] = ids.map(g.size()).fillna(0).astype(float).values
    f["traffic_known_months_3m"] = ids.map(g.traffic_gb.count()).fillna(0).astype(float).values
    f["digital_observed_months_3m"] = ids.map(dobs.groupby("customer_id").size()).fillna(0).astype(float).values

    # регион: последний опубликованный к T отчёт
    net = D["network"]
    nk = net[(net.measurement_month < T) & (net.report_available_at <= T)].sort_values("measurement_month").groupby("region").tail(1).set_index("region")
    f["region_fail_rate_latest_known"] = f.region.map(nk.failed_probe_rate).values
    f["region_fail_report_age_months"] = f.region.map(_months_between(T, nk.measurement_month).astype(float)).values

    # аудит: какие периоды и моменты доступности реально использованы
    a["window_start"] = M3
    a["window_end_exclusive"] = T
    a["billing_months_used"] = m(g.billing_month.apply(lambda v: "|".join(sorted(x.strftime("%Y-%m") for x in v))))
    a["billing_max_available_at"] = m(g.available_at.max())
    a["billing_last_month_present"] = ids.isin(b1.index).astype(int).values
    a["support_tickets_used_3m"] = f.support_count_3m
    a["support_max_available_at"] = m(sw.groupby("customer_id").available_at.max())
    a["digital_m1_available_at"] = m(d1.available_at)
    a["digital_m1_published_by_T"] = ids.map(known1.astype(int)).fillna(0).astype(int).values
    a["digital_latest_observed_month"] = m(latest.usage_month)
    a["digital_max_available_at_used"] = m(dc[dc.available_at <= T].groupby("customer_id").available_at.max())
    a["network_report_month_used"] = f.region.map(nk.measurement_month).values
    a["network_report_available_at"] = f.region.map(nk.report_available_at).values
    a["short_history_lt3m"] = (f.billing_months_3m < 3).astype(int)
    return f, a


def build_all(cfg: dict, D: dict, dates: list[pd.Timestamp]):
    fs, aus, ls = [], [], []
    for T in dates:
        f, a = build_features_at(T, D, cfg["active_mode"])
        fs.append(f)
        aus.append(a)
        ls.append(make_labels(f[KEYS], D["contracts"], cfg["horizon"], pd.Timestamp(cfg["observation_end_exclusive"])))
    sort = lambda d: d.sort_values(KEYS).reset_index(drop=True)  # noqa: E731
    return sort(pd.concat(fs, ignore_index=True)), sort(pd.concat(aus, ignore_index=True)), sort(pd.concat(ls, ignore_index=True))


# --------------------------------------------------------------------------- baseline

def rank_by_debt(f: pd.DataFrame) -> pd.DataFrame:
    """Сортировка: долг M−1 по убыванию, неизвестный долг — после известных, при равенстве customer_id по возрастанию."""
    d = f[KEYS + ["debt_last_rub"]].copy()
    d["debt_unknown"] = d.debt_last_rub.isna().astype(int)
    d = d.sort_values(["snapshot_at", "debt_unknown", "debt_last_rub", "customer_id"], ascending=[True, True, False, True])
    d["rank"] = d.groupby("snapshot_at").cumcount() + 1
    return d


def baseline_metrics(f: pd.DataFrame, labels: pd.DataFrame, fraction: float) -> pd.DataFrame:
    r = rank_by_debt(f).merge(labels[KEYS + ["churn_target"]], on=KEYS, validate="one_to_one")
    rows = []
    for T, g in r.groupby("snapshot_at"):
        n = len(g)
        k = int(np.floor(fraction * n))
        top = g[g["rank"] <= k]
        pos = int(g.churn_target.sum())
        tp = int(top.churn_target.sum())
        rows.append(dict(snapshot_at=T, n_active=n, K=k, positives=pos, tp_at_k=tp,
                         recall_at_k=tp / pos if pos > 0 else np.nan, precision_at_k=tp / k if k > 0 else np.nan,
                         random_expected_recall=k / n if n else np.nan, best_possible_recall=min(pos, k) / pos if pos else np.nan,
                         debt_unknown_rows=int(g.debt_unknown.sum()), debt_positive_rows=int((g.debt_last_rub > 0).sum()),
                         threshold_debt_rub_at_k=float(top.debt_last_rub.min()) if k > 0 else np.nan))
    m = pd.DataFrame(rows)
    total = dict(snapshot_at="ИТОГО train", n_active=int(m.n_active.sum()), K=int(m.K.sum()), positives=int(m.positives.sum()),
                 tp_at_k=int(m.tp_at_k.sum()), recall_at_k=m.tp_at_k.sum() / m.positives.sum(),
                 precision_at_k=m.tp_at_k.sum() / m.K.sum(), random_expected_recall=m.K.sum() / m.n_active.sum(),
                 best_possible_recall=np.nan, debt_unknown_rows=int(m.debt_unknown_rows.sum()),
                 debt_positive_rows=int(m.debt_positive_rows.sum()), threshold_debt_rub_at_k=np.nan)
    macro = dict(snapshot_at="среднее по датам (macro)", recall_at_k=m.recall_at_k.mean(), precision_at_k=m.precision_at_k.mean(),
                 random_expected_recall=m.random_expected_recall.mean())
    m["snapshot_at"] = fmt_date(m.snapshot_at)
    res = pd.concat([m, pd.DataFrame([total, macro])], ignore_index=True)
    for c in ["n_active", "K", "positives", "tp_at_k", "debt_unknown_rows", "debt_positive_rows"]:
        res[c] = res[c].astype("Int64")
    return res


# --------------------------------------------------------------------------- проверки

def _perturb_future(D: dict, T: pd.Timestamp) -> dict:
    """Копия данных, где изменено и добавлено только будущее относительно T."""
    E = {k: v.copy() for k, v in D.items()}
    b = E["billing"]
    fut = b.billing_month >= T
    b.loc[fut, ["closing_debt_rub", "traffic_gb", "payment_delay_days"]] = b.loc[fut, ["closing_debt_rub", "traffic_gb", "payment_delay_days"]] * 7 + 13
    sp = E["support"]
    extra = sp.sample(200, random_state=1).copy()
    extra["ticket_id"] = "FUT" + extra.ticket_id
    extra["created_at"] = T + pd.Timedelta(hours=5)
    extra["available_at"] = T + pd.Timedelta(hours=6)
    late = sp.sample(200, random_state=2).copy()
    late["ticket_id"] = "LATE" + late.ticket_id
    late["created_at"] = T - pd.Timedelta(days=3)
    late["available_at"] = T + pd.Timedelta(seconds=1)
    E["support"] = pd.concat([sp, extra, late], ignore_index=True)
    dg = E["digital"]
    fut = dg.available_at > T
    dg.loc[fut, "observation_status"] = "observed"
    dg.loc[fut, ["observed_days", "app_sessions", "payment_attempts", "payment_failures", "selfservice_actions"]] = 0.0
    c = E["contracts"]
    later = c.termination_date >= T
    c.loc[later, "termination_date"] = c.loc[later, "termination_date"] + pd.Timedelta(days=17)
    n = E["network"]
    n.loc[n.report_available_at > T, "failed_probe_rate"] = 0.99
    return E


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def synthetic_feature_checks() -> list[dict]:
    T = pd.Timestamp("2026-07-01")
    cid = ["NEW", "GAP", "OLD"]
    contracts = pd.DataFrame({"customer_id": cid, "activation_date": pd.to_datetime(["2026-06-10", "2025-01-01", "2025-01-01"]),
                              "termination_date": pd.NaT})
    subs = pd.DataFrame({"customer_id": cid, "region": "R", "acquisition_channel": "online", "customer_segment": "individual"})

    def brow(c, mth, traffic, days=30, debt=0.0):
        mth = pd.Timestamp(mth)
        return dict(customer_id=c, billing_month=mth, available_at=add_months(mth, 1), tariff="Basic", charged_rub=450.0,
                    paid_rub=450.0, closing_debt_rub=debt, traffic_gb=traffic, voice_minutes=100.0, active_days=days,
                    payment_delay_days=0.0, tariff_changes_count=0.0)
    billing = pd.DataFrame([brow("NEW", "2026-06-01", 10.0, days=21),
                            brow("GAP", "2026-04-01", 30.0), brow("GAP", "2026-05-01", np.nan), brow("GAP", "2026-06-01", 20.0),
                            brow("OLD", "2026-04-01", 30.0), brow("OLD", "2026-05-01", 30.0), brow("OLD", "2026-06-01", 30.0)])
    support = pd.DataFrame({"ticket_id": ["S1", "S2"], "customer_id": ["OLD", "OLD"], "topic": ["billing", "network"],
                            "created_at": [T - pd.Timedelta(days=2), T - pd.Timedelta(days=1)],
                            "available_at": [T, T + pd.Timedelta(seconds=1)]})

    def drow(c, status, avail, sessions=np.nan, attempts=np.nan, failures=np.nan, days=np.nan):
        return dict(customer_id=c, usage_month=pd.Timestamp("2026-06-01"), available_at=avail, observation_status=status,
                    observed_days=days, app_sessions=sessions, payment_attempts=attempts, payment_failures=failures, selfservice_actions=sessions)
    digital = pd.DataFrame([drow("NEW", "observed", T, sessions=0.0, attempts=0.0, failures=0.0, days=21.0),
                            drow("GAP", "not_enabled", T),
                            drow("OLD", "observed", T + pd.Timedelta(seconds=1), sessions=5.0, attempts=2.0, failures=1.0, days=30.0)])
    network = pd.DataFrame({"region": ["R"], "measurement_month": [pd.Timestamp("2026-06-01")], "report_available_at": [T],
                            "failed_probe_rate": [0.02]})
    D = dict(subs=subs, contracts=contracts, billing=billing, support=support, digital=digital, network=network)
    f, _ = build_features_at(T, D)
    f = f.set_index("customer_id")
    res = [
        check_row("synthetic: история из одного месяца не превращается в три нулевых", "billing_months_3m=1, avg=10.0",
                  f"billing_months_3m={f.loc['NEW', 'billing_months_3m']:.0f}, avg={f.loc['NEW', 'traffic_avg_3m_gb']}",
                  f.loc["NEW", "billing_months_3m"] == 1 and f.loc["NEW", "traffic_avg_3m_gb"] == 10.0),
        check_row("synthetic: трафик на активный день учитывает неполный месяц", round(10 / 21, 6), round(f.loc["NEW", "traffic_per_active_day_3m"], 6),
                  abs(f.loc["NEW", "traffic_per_active_day_3m"] - 10 / 21) < 1e-9),
        check_row("synthetic: пропущенный месяц не подменяется предыдущим в разности", "NaN", f.loc["GAP", "traffic_change_1m_gb"],
                  pd.isna(f.loc["GAP", "traffic_change_1m_gb"])),
        check_row("synthetic: среднее по известным месяцам и их число", "avg=25.0, known=2",
                  f"avg={f.loc['GAP', 'traffic_avg_3m_gb']}, known={f.loc['GAP', 'traffic_known_months_3m']:.0f}",
                  f.loc["GAP", "traffic_avg_3m_gb"] == 25.0 and f.loc["GAP", "traffic_known_months_3m"] == 2),
        check_row("synthetic: обращение с available_at = T входит, T + 1 с — нет", 1, f.loc["OLD", "support_count_3m"],
                  f.loc["OLD", "support_count_3m"] == 1 and f.loc["OLD", "support_billing_count_3m"] == 1 and f.loc["OLD", "support_network_count_3m"] == 0),
        check_row("synthetic: ноль попыток оплаты -> неизвестная доля ошибок", "rate=NaN, attempts=0",
                  f"rate={f.loc['NEW', 'payment_failure_rate_last']}, attempts={f.loc['NEW', 'payment_attempts_last']}",
                  pd.isna(f.loc["NEW", "payment_failure_rate_last"]) and f.loc["NEW", "payment_attempts_last"] == 0),
        check_row("synthetic: observed с нулём сессий остаётся измеренным нулём", 0.0, f.loc["NEW", "app_sessions_last"], f.loc["NEW", "app_sessions_last"] == 0),
        check_row("synthetic: not_enabled не становится нулевыми сессиями", "not_enabled, NaN",
                  f"{f.loc['GAP', 'digital_status_last']}, {f.loc['GAP', 'app_sessions_last']}",
                  f.loc["GAP", "digital_status_last"] == "not_enabled" and pd.isna(f.loc["GAP", "app_sessions_last"])),
        check_row("synthetic: отчёт, опубликованный через 1 с после T, неизвестен", "not_published, NaN",
                  f"{f.loc['OLD', 'digital_status_last']}, {f.loc['OLD', 'app_sessions_last']}",
                  f.loc["OLD", "digital_status_last"] == "not_published" and pd.isna(f.loc["OLD", "app_sessions_last"])),
    ]
    return res


# --------------------------------------------------------------------------- запуск

def run(cfg: dict) -> dict:
    out = STAGE4_OUT
    out.mkdir(parents=True, exist_ok=True)
    parts = snapshot_dates(cfg)
    dates = parts["train"] + parts["validation"] + parts["test"]
    D = load_inputs(cfg)
    feats, audit, labels = build_all(cfg, D, dates)
    split_of = {d: name for name, ds in parts.items() for d in ds}
    split = feats[KEYS].copy()
    split["split"] = split.snapshot_at.map(split_of)

    catalog = pd.DataFrame(FEATURES, columns=CATALOG_COLS)
    catalog.insert(catalog.columns.get_loc("missing_policy"), "available_when", catalog.source.map(AVAILABLE_WHEN))
    feature_cols = list(catalog.feature_name)
    assert list(feats.columns) == KEYS + feature_cols, "Порядок столбцов features.csv расходится с паспортом"
    in_x = catalog[catalog.include_in_X == "да"]
    num_cols = list(in_x[in_x.x_type == "numeric"].feature_name)
    cat_cols = list(in_x[in_x.x_type == "categorical"].feature_name)

    write_csv(feats.assign(snapshot_at=fmt_date(feats.snapshot_at)), out / "features.csv")
    write_csv(labels.assign(snapshot_at=fmt_date(labels.snapshot_at), target_end_exclusive=fmt_date(labels.target_end_exclusive)), out / "labels.csv")
    write_csv(split.assign(snapshot_at=fmt_date(split.snapshot_at)), out / "split.csv")
    aud = audit.copy()
    for c in ["snapshot_at", "window_start", "window_end_exclusive", "digital_latest_observed_month", "network_report_month_used"]:
        aud[c] = fmt_date(aud[c])
    for c in ["billing_max_available_at", "support_max_available_at", "digital_m1_available_at", "digital_max_available_at_used", "network_report_available_at"]:
        aud[c] = fmt_ts(aud[c])
    write_csv(aud, out / "feature_audit.csv")
    write_csv(catalog, out / "feature_catalog.csv")

    # Преобразование: fit только на train
    is_tr = split.split == "train"
    X_tr_df, X_va_df, X_te_df = (feats.loc[split.split == s, num_cols + cat_cols] for s in ["train", "validation", "test"])
    prep = make_preprocessor(num_cols, cat_cols)
    X_tr = prep.fit_transform(X_tr_df)
    X_va = prep.transform(X_va_df)
    X_te = prep.transform(X_te_df)
    names = list(prep.get_feature_names_out())
    write_csv(pd.DataFrame({"position": range(len(names)), "x_column": names}), out / "x_columns.csv")
    imp = prep.named_transformers_["numeric"].named_steps["impute"]
    medians_train = X_tr_df[num_cols].median()
    enc = prep.named_transformers_["categorical"].named_steps["encode"]
    report = {
        "fit_on": "train (T = 01.12.2025–01.07.2026)", "transform_only": ["validation", "test"],
        "numeric_columns": num_cols, "categorical_columns": cat_cols,
        "shapes": {"train": list(X_tr.shape), "validation": list(X_va.shape), "test": list(X_te.shape)},
        "row_order": "строки X идут в порядке features.csv внутри части split (сортировка snapshot_at, customer_id); ключи — split.csv",
        "train_medians": {c: (None if pd.isna(v) else round(float(v), 6)) for c, v in medians_train.items()},
        "categories": {c: [str(x) for x in cats] for c, cats in zip(cat_cols, enc.categories_)},
        "missing_indicator_columns": [n for n in names if "missingindicator" in n],
    }
    write_json(report, out / "preprocessing_report.json")

    # Baseline на train
    tr_keys = split[is_tr][KEYS]
    base = baseline_metrics(feats.merge(tr_keys, on=KEYS), labels.merge(tr_keys, on=KEYS), cfg["campaign_fraction"])
    write_csv(base, out / "baseline_train_metrics.csv")

    # Диагностика нового источника
    dg = D["digital"]
    sub_ids = set(D["subs"].customer_id)
    bkeys = set(zip(D["billing"].customer_id, D["billing"].billing_month))
    dchk = pd.DataFrame([
        dict(check="ключ (customer_id, usage_month) уникален", value=int(dg.duplicated(["customer_id", "usage_month"]).sum()), expected=0),
        dict(check="ссылки на абонентов без пары в реестре", value=int((~dg.customer_id.isin(sub_ids)).sum()), expected=0),
        dict(check="пары (абонент, месяц) совпадают с billing_clean", value=int(set(zip(dg.customer_id, dg.usage_month)) == bkeys), expected=1),
        dict(check="статусы вне observed/not_enabled/export_gap", value=int((~dg.observation_status.isin(["observed", "not_enabled", "export_gap"])).sum()), expected=0),
        dict(check="available_at раньше конца месяца", value=int((dg.available_at < dg.usage_month + pd.offsets.MonthBegin(1)).sum()), expected=0),
        dict(check="failures > attempts", value=int((dg.payment_failures > dg.payment_attempts).sum()), expected=0),
        dict(check="не-observed с заполненными показателями", value=int(((dg.observation_status != "observed") & dg.app_sessions.notna()).sum()), expected=0),
        dict(check="observed с пустыми показателями", value=int(((dg.observation_status == "observed") & dg.app_sessions.isna()).sum()), expected=0),
        dict(check="observed_days ≠ active_days биллинга", value=int((dg[dg.observation_status == "observed"].merge(
            D["billing"][["customer_id", "billing_month", "active_days"]], left_on=["customer_id", "usage_month"], right_on=["customer_id", "billing_month"])
            .pipe(lambda d: d.observed_days != d.active_days)).sum()), expected=0),
    ] + [dict(check=f"строк со статусом {s}", value=int((dg.observation_status == s).sum()), expected="—") for s in ["observed", "not_enabled", "export_gap"]]
      + [dict(check=f"задержка публикации {h:g} ч после конца месяца", value=int(n), expected="—") for h, n in
         ((dg.available_at - (dg.usage_month + pd.offsets.MonthBegin(1))).dt.total_seconds() / 3600).value_counts().sort_index().items()])
    write_csv(dchk, out / "digital_checks.csv")

    # Проверки
    checks = []
    checks.append(check_row("ключи features уникальны", 0, int(feats.duplicated(KEYS).sum()), not feats.duplicated(KEYS).any()))
    indep = {d.date().isoformat(): independent_active_count(D["events"], d) for d in dates}
    got = {d.date().isoformat(): int(n) for d, n in feats.groupby("snapshot_at").size().items()}
    checks.append(check_row("строк на T = независимо рассчитанная активная база", indep, got, indep == got))
    same_keys = feats[KEYS].equals(labels[KEYS]) and feats[KEYS].equals(split[KEYS]) and feats[KEYS].equals(audit[KEYS])
    checks.append(check_row("features/labels/split/audit: одинаковые ключи в одном порядке", True, same_keys, same_keys))
    leak = [c for c in feats.columns if any(w in c for w in ["churn", "termination", "target", "split"])]
    checks.append(check_row("в features нет метки, дат расторжения и split", [], leak, not leak))
    checks.append(check_row("все использованные строки биллинга доступны к T", 0, int((audit.billing_max_available_at > audit.snapshot_at).sum()),
                            not (audit.billing_max_available_at > audit.snapshot_at).any()))
    checks.append(check_row("все использованные обращения доступны к T", 0, int((audit.support_max_available_at > audit.snapshot_at).sum()),
                            not (audit.support_max_available_at > audit.snapshot_at).any()))
    checks.append(check_row("все использованные цифровые отчёты доступны к T", 0, int((audit.digital_max_available_at_used > audit.snapshot_at).sum()),
                            not (audit.digital_max_available_at_used > audit.snapshot_at).any()))
    checks.append(check_row("все метки наблюдаемы (до 01.10.2026)", 0, int(labels.churn_target.isna().sum()), labels.churn_target.notna().all()))
    T0 = pd.Timestamp("2026-07-01")
    f_ref, _ = build_features_at(T0, D)
    f_fut, _ = build_features_at(T0, _perturb_future(D, T0))
    checks.append(check_row("признаки на T не меняются после изменения/добавления будущих записей", True, _frames_equal(f_ref, f_fut),
                            _frames_equal(f_ref, f_fut), "биллинг ≥ T, обращения после T и с доступностью T + 1 с, цифровые отчёты после T, сеть, сдвиг будущих расторжений"))
    E = {k: v.copy() for k, v in D.items()}
    churners = labels[(labels.snapshot_at == T0) & (labels.churn_target == 1)].customer_id
    c = E["contracts"]
    c.loc[c.customer_id.isin(churners), "termination_date"] = pd.Timestamp("2026-09-15")
    f_lab, _ = build_features_at(T0, E)
    checks.append(check_row("изменение будущей метки не меняет признаки", True, _frames_equal(f_ref, f_lab), _frames_equal(f_ref, f_lab),
                            f"{len(churners)} расторжений перенесены за горизонт"))
    nonobs = feats.digital_status_last != "observed"
    checks.append(check_row("not_enabled/export_gap/not_published -> сессии пусты (не 0)", 0, int(feats.loc[nonobs, "app_sessions_last"].notna().sum()),
                            feats.loc[nonobs, "app_sessions_last"].isna().all()))
    zero_att = feats.payment_attempts_last == 0
    checks.append(check_row("ноль попыток оплаты -> доля ошибок пуста", 0, int(feats.loc[zero_att, "payment_failure_rate_last"].notna().sum()),
                            feats.loc[zero_att, "payment_failure_rate_last"].isna().all(), f"строк с 0 попыток: {int(zero_att.sum())}"))
    short = feats.billing_months_3m < 3
    ok_short = (feats.loc[short, "traffic_known_months_3m"] <= feats.loc[short, "billing_months_3m"]).all()
    checks.append(check_row("короткая история: число известных месяцев ≤ числу строк", True, ok_short, ok_short, f"строк с историей < 3 мес.: {int(short.sum())}"))
    checks += synthetic_feature_checks()
    # преобразование
    ok_median = all((pd.isna(medians_train[c]) and pd.isna(s)) or abs(medians_train[c] - s) < 1e-9 for c, s in zip(num_cols, imp.statistics_))
    checks.append(check_row("медианы импутера = медианы train", True, ok_median, ok_median))
    checks.append(check_row("преобразователь обучен на числе строк train", int(is_tr.sum()), int(X_tr.shape[0]), X_tr.shape[0] == is_tr.sum()))
    va_new = X_va_df.copy()
    va_new["region"] = "Новый регион"
    va_new.loc[va_new.index[:5], "tariff_last"] = "Ultra"
    X_new = prep.transform(va_new)
    region_cols = [i for i, n in enumerate(names) if n.startswith("categorical__region_")]
    ok_new = X_new.shape == X_va.shape and np.allclose(X_new[:, region_cols], 0)
    checks.append(check_row("новая категория на validation: transform без fit, нули в one-hot", X_va.shape, X_new.shape, ok_new))
    checks.append(check_row("в X нет пропусков после преобразования", 0, int(np.isnan(X_tr).sum() + np.isnan(X_va).sum() + np.isnan(X_te).sum()),
                            not (np.isnan(X_tr).any() or np.isnan(X_va).any() or np.isnan(X_te).any())))
    checks.append(check_row("в features.csv пропуски сохранены", ">0", int(feats[num_cols].isna().sum().sum()), feats[num_cols].isna().sum().sum() > 0))
    f2, a2, l2 = build_all(cfg, D, dates)
    rep_ok = f2.equals(feats) and l2.equals(labels)
    checks.append(check_row("повторный расчёт даёт те же признаки и метки", True, rep_ok, rep_ok))
    ch = pd.DataFrame(checks)
    write_csv(ch, out / "checks_stage4.csv")

    split_stats = pd.DataFrame([dict(split=s, dates=", ".join(fmt_date(pd.Series(parts[s]))), rows=int((split.split == s).sum()),
                                     customers=int(split[split.split == s].customer_id.nunique())) for s in ["train", "validation", "test"]])
    write_csv(split_stats, out / "split_summary.csv")
    tot = base[base.snapshot_at == "ИТОГО train"].iloc[0]
    mac = base[base.snapshot_at == "среднее по датам (macro)"].iloc[0]
    summary = {
        "dates": [d.date().isoformat() for d in dates],
        "rows": len(feats), "features_total": len(feature_cols), "features_in_X": len(num_cols) + len(cat_cols),
        "x_shape": report["shapes"], "split": split_stats.to_dict("records"),
        "baseline_train_micro": {"recall_at_k": round(float(tot.recall_at_k), 4), "precision_at_k": round(float(tot.precision_at_k), 4)},
        "baseline_train_macro": {"recall_at_k": round(float(mac.recall_at_k), 4), "precision_at_k": round(float(mac.precision_at_k), 4)},
        "missing_share_in_features": {c: round(float(feats[c].isna().mean()), 4) for c in feature_cols if feats[c].isna().any()},
        "digital_status_last_counts": feats.digital_status_last.value_counts().to_dict(),
        "checks_failed": int((ch.passed == "FAIL").sum()),
    }
    write_json(summary, out / "features_summary.json")
    return summary
