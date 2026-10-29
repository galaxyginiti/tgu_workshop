"""Этап 3. Разведочный анализ на датах обучения.

Вход: таблицы этапа 2 и новые источники 3_eda/data.
Выход (solution/stage3_eda/output):
  eda_sample.csv (ключ customer_id + snapshot_at), eda_fields.csv,
  network_availability.csv, figures/fig1..fig6.png, tables/*.csv,
  analysis_plan.csv (журнал гипотез), feature_directions.csv, checks_stage3.csv.
Метки валидации (01.08.2026) и теста (01.09.2026) здесь не строятся и не смотрятся.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import (STAGE2_OUT, STAGE3_OUT, add_months, check_row, fmt_date, fmt_ts, read_prepared,  # noqa: E402
                     read_raw_csv, snapshot_dates, to_msk, write_csv, write_json)
from .snapshots import active_base, contract_dates, independent_active_count, make_labels  # noqa: E402

NUMERIC = ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb",
           "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count", "support_count",
           "support_network_count", "support_billing_count", "support_unknown_topic_count", "support_score_mean"]
COLORS = {"main": "#2F6B6A", "accent": "#C8553D", "muted": "#8A8F98", "light": "#D9E3E2"}
REGION_COLORS = ["#2F6B6A", "#C8553D", "#E0A458", "#5B7DB1", "#8E6C8A"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- исследовательская таблица

def load_prepared() -> dict:
    d = {name: read_prepared(STAGE2_OUT / f"{name}.csv") for name in
         ["subscribers_clean", "contract_events_clean", "customer_month"]}
    cm = d["customer_month"]
    cm["billing_month"] = pd.to_datetime(cm.billing_month)
    cm["available_at"] = to_msk(cm.available_at)[0]
    for c in NUMERIC + ["support_resolved_count", "support_score_count"]:
        cm[c] = pd.to_numeric(cm[c])
    return d


def tenure_group(days: pd.Series, bins: list[int]) -> pd.Series:
    lo, hi = bins
    return pd.Series(np.select([days < lo, days < hi], [f"<{lo} дн.", f"{lo}–{hi - 1} дн."], f"≥{hi} дн."),
                     index=days.index).where(days.notna(), "неизвестно")


def build_eda_sample(cfg: dict, data: dict, dates: list[pd.Timestamp]) -> pd.DataFrame:
    contracts = contract_dates(data["subscribers_clean"], data["contract_events_clean"])
    subs = data["subscribers_clean"].set_index("customer_id")
    cm = data["customer_month"]
    parts = []
    for T in dates:
        base = active_base(contracts, T, cfg["active_mode"])
        prev = add_months(T, -1)
        rows = cm[(cm.billing_month == prev) & (cm.available_at <= T)]
        x = base.merge(rows.drop(columns=["region", "acquisition_channel", "customer_segment", "activation_date"]),
                       on="customer_id", how="left", validate="one_to_one")
        x = x.rename(columns={"billing_month": "source_billing_month", "available_at": "billing_available_at"})
        for c in ["region", "acquisition_channel", "customer_segment"]:
            x[c] = x.customer_id.map(subs[c])
        act = pd.to_datetime(x.customer_id.map(subs.activation_date))
        x["tenure_days"] = (T - act).dt.days
        labels = make_labels(base, contracts, cfg["horizon"], pd.Timestamp(cfg["observation_end_exclusive"]))
        x = x.merge(labels, on=["customer_id", "snapshot_at"], how="left", validate="one_to_one")
        parts.append(x)
    s = pd.concat(parts, ignore_index=True)
    s["active_rule"] = "activation < T и (нет termination или termination ≥ T)"
    s["tenure_group"] = tenure_group(s.tenure_days, cfg["tenure_bins_days"])
    s["debt_group"] = np.select([s.closing_debt_rub.isna(), s.closing_debt_rub > 0], ["неизвестно", "долг > 0"], "долг = 0")
    s["support_group"] = np.select([s.support_count.isna(), s.support_count == 0, s.support_count == 1],
                                   ["неизвестно", "0", "1"], "2+")
    s["paid_zero"] = np.where(s.paid_rub.isna(), np.nan, (s.paid_rub == 0).astype(float))
    cols = ["customer_id", "snapshot_at", "active_rule", "source_billing_month", "billing_available_at",
            "region", "acquisition_channel", "customer_segment", "tenure_days", "tenure_group", "tariff"] + \
           [c for c in NUMERIC if c in s.columns] + ["support_resolved_count", "support_score_count", "support_system",
                                                       "debt_group", "support_group", "paid_zero",
                                                       "churn_target", "target_end_exclusive", "label_observed"]
    return s[cols].sort_values(["snapshot_at", "customer_id"]).reset_index(drop=True)


EDA_FIELDS = [
    ("customer_id, snapshot_at", "ключ", "Абонент и дата расчёта T (00:00 МСК первого числа)", "—"),
    ("active_rule", "служебное", "Записанное правило активной базы", "—"),
    ("source_billing_month, billing_available_at", "служебное", "Месяц биллинга, из которого взяты показатели, и момент его доступности (≤ T)", "—"),
    ("region, acquisition_channel, customer_segment", "признак", "Постоянные поля справочника", "—"),
    ("tenure_days, tenure_group", "признак", "Стаж на T = T − activation_date; группы заданы заранее: <90, 90–364, ≥365 дней", "дни"),
    ("tariff … tariff_changes_count", "признак", "Показатели биллинга предыдущего календарного месяца (customer_month)", "руб./ГБ/мин./дни"),
    ("support_*", "признак", "Агрегаты поддержки предыдущего месяца, доступные на T (из customer_month)", "шт./балл"),
    ("support_system", "контекст", "legacy/current — система учёта обращений в месяце показателей", "—"),
    ("debt_group, support_group, paid_zero", "группировка", "Группы для сравнения; неизвестные значения — отдельная группа", "—"),
    ("churn_target", "метка", "1 — termination в [T, target_end_exclusive); пусто — исход не наблюдаем", "0/1"),
    ("target_end_exclusive", "метка", "Исключённый конец горизонта (первое число следующего месяца)", "дата"),
    ("label_observed", "метка", "1 — горизонт полностью внутри журнала (до 01.10.2026)", "0/1"),
]


# --------------------------------------------------------------------------- статистика групп

def wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def group_stats(df: pd.DataFrame, by: list[str] | str, small: int) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else by
    d = df[df.churn_target.notna()]
    g = d.groupby(by, observed=True).agg(n_observations=("customer_id", "size"), n_customers=("customer_id", "nunique"),
                                         n_churn=("churn_target", "sum")).reset_index()
    g["n_churn"] = g.n_churn.astype(int)
    g["churn_rate"] = (g.n_churn / g.n_observations).round(5)
    lo, hi = wilson(g.n_churn, g.n_observations)
    g["ci95_low"], g["ci95_high"] = lo.round(5), hi.round(5)
    g["small_group"] = np.where(g.n_observations < small, "да (<30 строк)", "")
    return g


def numeric_profile(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC + ["tenure_days"]:
        v = s[c]
        k = v.dropna()
        rows.append(dict(field=c, n_rows=len(v), n_known=len(k), n_missing=int(v.isna().sum()),
                         share_missing=round(v.isna().mean(), 5), share_zero=round(float((k == 0).mean()), 5) if len(k) else np.nan,
                         min=k.min(), q25=k.quantile(.25), median=k.median(), q75=k.quantile(.75), q90=k.quantile(.9),
                         q99=k.quantile(.99), max=k.max(), mean=round(k.mean(), 4)))
    return pd.DataFrame(rows)


def category_profile(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["tariff", "region", "acquisition_channel", "customer_segment", "tenure_group", "debt_group", "support_group", "support_system"]:
        vc = s[c].fillna("неизвестно").value_counts()
        for k, n in vc.items():
            rows.append(dict(field=c, value=k, n_rows=int(n), share=round(n / len(s), 5)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- сеть и контекст

def load_network(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    net = read_raw_csv(cfg["abs_paths"]["network_quality"])
    net["measurement_month"] = pd.to_datetime(net.measurement_month)
    net["report_available_at"] = to_msk(net.report_available_at)[0]
    for c in ["probe_count", "failed_probe_count"]:
        net[c] = pd.to_numeric(net[c]).astype(int)
    for c in ["download_mbps_p50", "download_mbps_p90"]:
        net[c] = pd.to_numeric(net[c])
    net["failed_probe_rate"] = net.failed_probe_count / net.probe_count
    ctx = read_raw_csv(cfg["abs_paths"]["business_context"])
    ctx["announced_at"] = to_msk(ctx.announced_at)[0]
    return net, ctx


def network_availability(net: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Ключ (region, measurement_month, snapshot_at): известен ли отчёт на T. Значение — только если известен."""
    rows = []
    for T in dates:
        part = net[net.measurement_month < T].copy()
        part["snapshot_at"] = T
        part["is_previous_month"] = (part.measurement_month == add_months(T, -1)).astype(int)
        part["known_at_T"] = (part.report_available_at <= T).astype(int)
        part["failed_probe_rate_at_T"] = part.failed_probe_rate.where(part.known_at_T == 1)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[["region", "measurement_month", "snapshot_at", "report_available_at", "known_at_T", "is_previous_month",
                "probe_count", "failed_probe_count", "failed_probe_rate_at_T"]].sort_values(
        ["snapshot_at", "region", "measurement_month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- рисунки

def _bar_labels(ax, bars, texts, dy=0.002):
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, t, ha="center", va="bottom", fontsize=8.5)


def fig1_missing(s: pd.DataFrame, net_known: pd.Series, path) -> pd.DataFrame:
    fields = {"tariff": "tariff", "closing_debt_rub": "closing_debt_rub", "paid_rub": "paid_rub",
              "payment_delay_days": "payment_delay_days", "traffic_gb": "traffic_gb", "voice_minutes": "voice_minutes",
              "support_count": "support_count", "support_score_mean": "support_score_mean",
              "tenure_days": "tenure_days", "регион. отчёт сети за M−1 на T": None}
    rows = []
    for label, col in fields.items():
        miss = (~net_known).astype(bool) if col is None else s[col].isna()
        reason = {
            "traffic_gb": "пропуск трафика в биллинге (неизвестно, не 0)",
            "voice_minutes": "исключены отрицательные минуты + пропуски",
            "support_score_mean": "нет доступной оценки (нет обращений, не закрыто или оценка не дана)",
            "регион. отчёт сети за M−1 на T": "отчёт опубликован позже T (задержка 24–48 ч)",
        }.get(label, "")
        rows.append(dict(indicator=label, n_rows=len(s), n_missing=int(miss.sum()), share_missing=round(float(miss.mean()), 5), reason=reason))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    order = t.sort_values("share_missing")
    bars = ax.barh(order.indicator, order.share_missing * 100,
                   color=[COLORS["accent"] if v > 0.5 else COLORS["main"] for v in order.share_missing])
    for b, (_, r) in zip(bars, order.iterrows()):
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{r.share_missing * 100:.1f}%  ({r.n_missing:,} из {r.n_rows:,})".replace(",", " "),
                va="center", fontsize=8.5)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Доля пропусков, % строк «абонент × T»")
    ax.set_title("Рис. 1. Пропуски исследуемых показателей")
    ax.text(0, -0.24, f"T = 01.12.2025–01.07.2026 (8 дат); {len(s):,} строк; {s.customer_id.nunique():,} абонентов.".replace(",", " ")
            + "\nПоказатели — предыдущий календарный месяц, доступный на T.", transform=ax.transAxes, fontsize=8, color=COLORS["muted"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return t


def fig2_distributions(s: pd.DataFrame, path) -> pd.DataFrame:
    tr = s.traffic_gb.dropna()
    debt = s.closing_debt_rub.dropna()
    pos = debt[debt > 0]
    stats = []
    for name, v, unit in [("traffic_gb", tr, "ГБ"), ("closing_debt_rub", debt, "руб."), ("closing_debt_rub (>0)", pos, "руб.")]:
        stats.append(dict(indicator=name, unit=unit, n_known=len(v), n_missing=int(len(s) - len(v)) if "(>0)" not in name else "",
                          share_zero=round(float((v == 0).mean()), 5), mean=round(v.mean(), 2), median=v.median(), q75=v.quantile(.75),
                          q90=v.quantile(.9), q99=v.quantile(.99), max=v.max()))
    t = pd.DataFrame(stats)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    bins = np.geomspace(max(tr.min(), 0.05), tr.max() * 1.05, 45)
    ax.hist(tr, bins=bins, color=COLORS["main"], alpha=.9)
    ax.set_xscale("log")
    for (q, lab, ls), yy in zip([(tr.median(), "медиана", "-"), (tr.quantile(.9), "p90", "--"), (tr.quantile(.99), "p99", ":")], [.97, .86, .75]):
        ax.axvline(q, color=COLORS["accent"], ls=ls, lw=1.2)
        ax.text(q, ax.get_ylim()[1] * yy, f" {lab} {q:.1f}", color=COLORS["accent"], fontsize=8, va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=.8))
    ax.set_xlabel("Трафик M−1, ГБ (лог. шкала)")
    ax.set_ylabel("Строк «абонент × T»")
    ax.set_title("Трафик")
    ax.text(0.01, -0.22, f"известно {len(tr):,}; пропусков {len(s) - len(tr)}; среднее {tr.mean():.1f} > медианы {tr.median():.1f} — длинный правый хвост".replace(",", " "),
            transform=ax.transAxes, fontsize=8, color=COLORS["muted"])
    ax = axes[1]
    ax.hist(pos, bins=np.geomspace(pos.min(), pos.max() * 1.05, 40), color=COLORS["accent"], alpha=.9)
    ax.set_xscale("log")
    for (q, lab, ls), yy in zip([(pos.median(), "медиана", "-"), (pos.quantile(.9), "p90", "--"), (pos.quantile(.99), "p99", ":")], [.97, .86, .75]):
        ax.axvline(q, color=COLORS["main"], ls=ls, lw=1.2)
        ax.text(q, ax.get_ylim()[1] * yy, f" {lab} {q:.0f}", color=COLORS["main"], fontsize=8, va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=.8))
    ax.set_xlabel("Долг на конец M−1 > 0, руб. (лог. шкала)")
    ax.set_title("Задолженность")
    ax.text(0.01, -0.22, f"известно {len(debt):,}; долг = 0 у {(debt == 0).mean() * 100:.1f}%; > 0 у {len(pos):,} строк".replace(",", " "),
            transform=ax.transAxes, fontsize=8, color=COLORS["muted"])
    fig.suptitle("Рис. 2. Распределения трафика и долга (T = 01.12.2025–01.07.2026)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return t


def fig3_dynamics(by_t: pd.DataFrame, ctx: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    x = np.arange(len(by_t))
    bars = ax.bar(x, by_t.n_observations, color=COLORS["light"], width=.6, label="Активная база N (лев. ось)")
    for b, n in zip(bars, by_t.n_observations):
        ax.text(b.get_x() + b.get_width() / 2, 40, f"N={n:,}".replace(",", " "), ha="center", fontsize=8, color=COLORS["main"])
    ax.set_ylim(0, by_t.n_observations.max() * 1.25)
    ax.set_ylabel("Активных абонентов на T")
    ax.set_xticks(x, [d.strftime("%d.%m.%y") for d in by_t.snapshot_at])
    ax2 = ax.twinx()
    ax2.plot(x, by_t.churn_rate * 100, color=COLORS["accent"], marker="o", lw=2, label="Доля оттока, % (прав. ось)")
    ax2.fill_between(x, by_t.ci95_low * 100, by_t.ci95_high * 100, color=COLORS["accent"], alpha=.12)
    for xi, (_, r) in zip(x, by_t.iterrows()):
        ax2.text(xi, r.churn_rate * 100 + .12, f"{r.n_churn}/{r.n_observations}", ha="center", fontsize=8, color=COLORS["accent"])
    ax2.set_ylim(0, by_t.ci95_high.max() * 100 * 1.25)
    ax2.set_ylabel("Доля расторжений в месяце T, %")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=True, facecolor="white", edgecolor="none", ncol=2)
    ax.text(0, -0.2, "Контекст (business_context.csv): B001 — переход поддержки на новую систему 01.11.2025 (признаки поддержки всех T ≥ 01.12 — "
                     "из новой системы);\nB002 — праздники 01–11.01.2026; B005 — летний период с июня. Сообщения — пояснения, не причины. Заливка — 95% Wilson.",
            transform=ax.transAxes, fontsize=7.5, color=COLORS["muted"])
    ax.set_title("Рис. 3. Доля оттока и размер активной базы по датам T (числитель/знаменатель над точками)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _rate_panel(ax, g, label_col, title, color):
    x = np.arange(len(g))
    bars = ax.bar(x, g.churn_rate * 100, color=color, width=.6,
                  yerr=[(g.churn_rate - g.ci95_low) * 100, (g.ci95_high - g.churn_rate) * 100], capsize=3, ecolor=COLORS["muted"])
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r.ci95_high * 100 + .15, f"{r.churn_rate * 100:.1f}%\n{r.n_churn}/{r.n_observations:,}".replace(",", " "),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, g[label_col])
    ax.set_ylim(0, g.ci95_high.max() * 100 * 1.35)
    ax.set_ylabel("Доля расторжений в следующем месяце, %")
    ax.set_title(title)


def fig4_debt_support(g_debt, g_sup, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _rate_panel(axes[0], g_debt, "debt_group", "По долгу на конец M−1", COLORS["accent"])
    _rate_panel(axes[1], g_sup, "support_group", "По числу обращений в M−1", COLORS["main"])
    axes[1].set_xlabel("Обращений, доступных на T")
    fig.suptitle("Рис. 4. Доля оттока по долгу и обращениям (усы — 95% Wilson, без поправки на повторы абонентов)", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig5_tenure(g_ten, g_ten_debt, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
    _rate_panel(axes[0], g_ten, "tenure_group", "Стаж на T", COLORS["main"])
    ax = axes[1]
    groups = list(g_ten.tenure_group)
    w = .38
    for i, (dg, col) in enumerate([("долг = 0", COLORS["main"]), ("долг > 0", COLORS["accent"])]):
        part = g_ten_debt[g_ten_debt.debt_group == dg].set_index("tenure_group").reindex(groups)
        x = np.arange(len(groups)) + (i - .5) * w
        bars = ax.bar(x, part.churn_rate * 100, width=w, color=col, label=dg,
                      yerr=[(part.churn_rate - part.ci95_low) * 100, (part.ci95_high - part.churn_rate) * 100], capsize=3, ecolor=COLORS["muted"])
        for b, (_, r) in zip(bars, part.iterrows()):
            ax.text(b.get_x() + b.get_width() / 2, r.ci95_high * 100 + .15, f"{r.churn_rate * 100:.1f}%\nN={r.n_observations:,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylim(0, g_ten_debt.ci95_high.max() * 100 * 1.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Стаж × долг (проверка состава групп)")
    fig.suptitle("Рис. 5. Сегмент «стаж»: доля оттока и размеры групп", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig6_network(net: pd.DataFrame, ctx: pd.DataFrame, eda_months: list[pd.Timestamp], path) -> pd.DataFrame:
    t = net.copy()
    t["snapshot_using_it"] = t.measurement_month.map(lambda m: add_months(m, 1))
    t["known_at_next_T"] = (t.report_available_at <= t.snapshot_using_it).astype(int)
    t["delay_hours"] = ((t.report_available_at - t.snapshot_using_it).dt.total_seconds() / 3600).round(1)
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    lo, hi = min(eda_months), max(eda_months)
    for a in (ax, axb):
        a.axvspan(lo - pd.Timedelta(days=12), hi + pd.Timedelta(days=12), color=COLORS["light"], alpha=.5, lw=0)
    for color, (region, g) in zip(REGION_COLORS, t.groupby("region")):
        g = g.sort_values("measurement_month")
        ax.plot(g.measurement_month, g.failed_probe_rate * 100, color=color, lw=1.6, label=region)
        k = g[g.known_at_next_T == 1]
        u = g[g.known_at_next_T == 0]
        ax.scatter(k.measurement_month, k.failed_probe_rate * 100, color=color, s=28, zorder=3)
        ax.scatter(u.measurement_month, u.failed_probe_rate * 100, facecolor="white", edgecolor=color, s=28, zorder=3, lw=1.3)
    total = t.groupby("measurement_month")[["failed_probe_count", "probe_count"]].sum()
    ax.plot(total.index, total.failed_probe_count / total.probe_count * 100, color="black", lw=1, ls="--", label="все регионы: Σfailed/Σprobes")
    for _, c in ctx[ctx.event_type == "measurement_alert"].iterrows():
        m0 = pd.Timestamp(c.event_start)
        ax.annotate(f"{c.context_id}: {c.region}\nопубликовано {c.announced_at:%d.%m.%Y}", (m0, 6.35), fontsize=7.5, color=COLORS["muted"])
    ax.set_ylabel("Доля неуспешных проб, %")
    ax.set_ylim(0, 7.2)
    ax.scatter([], [], color="grey", s=28, label="отчёт известен на T = 1-е след. месяца")
    ax.scatter([], [], facecolor="white", edgecolor="grey", s=28, label="на эту T ещё не опубликован")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Рис. 6. Неуспешные технические пробы по регионам и месяцам измерений")
    for color, (region, g) in zip(REGION_COLORS, t.groupby("region")):
        g = g.sort_values("measurement_month")
        axb.plot(g.measurement_month, g.probe_count, color=color, lw=1.2, marker=".")
    axb.set_ylabel("Число проб")
    axb.set_xlabel("Месяц измерений (серая зона — месяцы, использованные в T основного EDA)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return t[["region", "measurement_month", "probe_count", "failed_probe_count", "failed_probe_rate", "download_mbps_p50",
              "download_mbps_p90", "report_available_at", "snapshot_using_it", "known_at_next_T", "delay_hours"]]


# --------------------------------------------------------------------------- проверки

def synthetic_boundary_check(cfg: dict) -> list[dict]:
    T = pd.Timestamp("2026-03-01")
    contracts = pd.DataFrame({
        "customer_id": ["act_on_T", "act_before_T", "term_on_T", "term_last_day", "term_on_end", "term_before_T"],
        "activation_date": pd.to_datetime(["2026-03-01", "2026-02-28", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01"]),
        "termination_date": pd.to_datetime([None, None, "2026-03-01", "2026-03-31", "2026-04-01", "2026-02-28"]),
    })
    base = active_base(contracts, T, cfg["active_mode"])
    lab = make_labels(base, contracts, cfg["horizon"], pd.Timestamp(cfg["observation_end_exclusive"])).set_index("customer_id")
    exp_base = {"act_before_T", "term_on_T", "term_last_day", "term_on_end"}
    exp_lab = {"act_before_T": 0, "term_on_T": 1, "term_last_day": 1, "term_on_end": 0}
    got_lab = lab.churn_target.to_dict()
    late = make_labels(pd.DataFrame({"customer_id": ["act_before_T"], "snapshot_at": [pd.Timestamp("2026-10-01")]}),
                       contracts, cfg["horizon"], pd.Timestamp(cfg["observation_end_exclusive"]))
    return [
        check_row("synthetic: граница T (подключение в T — вне базы, расторжение в T — в базе)", sorted(exp_base), sorted(base.customer_id),
                  set(base.customer_id) == exp_base),
        check_row("synthetic: метка на T и на конце горизонта", exp_lab, {k: int(v) for k, v in got_lab.items()},
                  {k: int(v) for k, v in got_lab.items()} == exp_lab, "расторжение 01.04 при T=01.03 -> 0 (конец исключён)"),
        check_row("synthetic: горизонт за пределами журнала -> неизвестная метка", "NaN", late.churn_target.iloc[0],
                  pd.isna(late.churn_target.iloc[0])),
    ]


def run(cfg: dict) -> dict:
    out = STAGE3_OUT
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    dates = snapshot_dates(cfg)["train"]
    small = cfg["small_group_rows"]
    data = load_prepared()
    s = build_eda_sample(cfg, data, dates)
    net, ctx = load_network(cfg)
    avail = network_availability(net, dates)

    # сетевой показатель M−1 на T: только если отчёт опубликован к T
    prev = avail[avail.is_previous_month == 1][["region", "snapshot_at", "known_at_T", "failed_probe_rate_at_T"]]
    s_net = s[["customer_id", "snapshot_at", "region", "churn_target"]].merge(prev, on=["region", "snapshot_at"], how="left", validate="many_to_one")

    write_csv(s.assign(snapshot_at=fmt_date(s.snapshot_at), source_billing_month=fmt_date(s.source_billing_month),
                       billing_available_at=fmt_ts(s.billing_available_at), target_end_exclusive=fmt_date(s.target_end_exclusive)),
              out / "eda_sample.csv")
    write_csv(pd.DataFrame(EDA_FIELDS, columns=["field", "role", "description", "unit"]), out / "eda_fields.csv")
    write_csv(avail.assign(measurement_month=fmt_date(avail.measurement_month), snapshot_at=fmt_date(avail.snapshot_at),
                           report_available_at=fmt_ts(avail.report_available_at)), out / "network_availability.csv")

    T = out / "tables"
    F = out / "figures"
    write_csv(numeric_profile(s), T / "describe_numeric.csv")
    write_csv(category_profile(s), T / "describe_categories.csv")
    t1 = fig1_missing(s, s_net.known_at_T.eq(1), F / "fig1_missing.png")
    write_csv(t1, T / "fig1_missing.csv")
    t2 = fig2_distributions(s, F / "fig2_distributions.png")
    write_csv(t2, T / "fig2_distributions.csv")
    by_t = group_stats(s, "snapshot_at", small)
    fig3_dynamics(by_t, ctx, F / "fig3_churn_dynamics.png")
    write_csv(by_t.assign(snapshot_at=fmt_date(by_t.snapshot_at)), T / "fig3_churn_by_T.csv")
    g_debt = group_stats(s, "debt_group", small)
    g_debt = g_debt.set_index("debt_group").reindex([x for x in ["долг = 0", "долг > 0", "неизвестно"] if x in set(g_debt.debt_group)]).reset_index()
    g_sup = group_stats(s, "support_group", small)
    g_sup = g_sup.set_index("support_group").reindex([x for x in ["0", "1", "2+", "неизвестно"] if x in set(g_sup.support_group)]).reset_index()
    fig4_debt_support(g_debt, g_sup, F / "fig4_debt_support.png")
    write_csv(g_debt, T / "fig4_churn_by_debt.csv")
    write_csv(g_sup, T / "fig4_churn_by_support.csv")
    order = [f"<{cfg['tenure_bins_days'][0]} дн.", f"{cfg['tenure_bins_days'][0]}–{cfg['tenure_bins_days'][1] - 1} дн.", f"≥{cfg['tenure_bins_days'][1]} дн."]
    g_ten = group_stats(s, "tenure_group", small).set_index("tenure_group").reindex(order).reset_index()
    g_ten_debt = group_stats(s, ["tenure_group", "debt_group"], small)
    fig5_tenure(g_ten, g_ten_debt, F / "fig5_segment_tenure.png")
    write_csv(g_ten, T / "fig5_churn_by_tenure.csv")
    write_csv(g_ten_debt, T / "fig5_tenure_x_debt.csv")
    t6 = fig6_network(net, ctx, [add_months(d, -1) for d in dates], F / "fig6_network_quality.png")
    write_csv(t6.assign(measurement_month=fmt_date(t6.measurement_month), report_available_at=fmt_ts(t6.report_available_at),
                        snapshot_using_it=fmt_date(t6.snapshot_using_it)), T / "fig6_network_rates.csv")

    # дополнительные сравнения и проверка альтернативных объяснений
    debt_month = group_stats(s, ["snapshot_at", "debt_group"], small)
    piv = debt_month.pivot(index="snapshot_at", columns="debt_group", values="churn_rate")
    piv["rate_ratio_debt_vs_no_debt"] = (piv["долг > 0"] / piv["долг = 0"]).round(3)
    write_csv(debt_month.assign(snapshot_at=fmt_date(debt_month.snapshot_at)), T / "debt_by_month.csv")
    write_csv(piv.reset_index().assign(snapshot_at=lambda d: fmt_date(d.snapshot_at)), T / "debt_ratio_by_month.csv")
    debt_tariff = group_stats(s, ["tariff", "debt_group"], small)
    write_csv(debt_tariff, T / "debt_by_tariff.csv")
    write_csv(group_stats(s, "tariff", small), T / "churn_by_tariff.csv")
    write_csv(group_stats(s, "region", small), T / "churn_by_region.csv")
    write_csv(group_stats(s.assign(paid_zero=s.paid_zero.map({1.0: "оплата = 0", 0.0: "оплата > 0"})), "paid_zero", small), T / "churn_by_paid_zero.csv")
    s_delay = s.assign(delay_group=np.select([s.payment_delay_days.isna(), s.payment_delay_days == 0, s.payment_delay_days <= 10],
                                             ["неизвестно", "0 дн.", "1–10 дн."], "> 10 дн."))
    write_csv(group_stats(s_delay, "delay_group", small), T / "churn_by_payment_delay.csv")
    s_topic = s.assign(network_ticket=np.where(s.support_network_count > 0, "есть обращение о связи", "нет"))
    write_csv(group_stats(s_topic, "network_ticket", small), T / "churn_by_network_ticket.csv")
    sup_debt = group_stats(s, ["debt_group", "support_group"], small)
    write_csv(sup_debt, T / "support_x_debt.csv")
    sup_system = group_stats(s, ["support_system", "support_group"], small)
    write_csv(sup_system, T / "support_by_system.csv")
    # корреляции числовых показателей (без идентификаторов и кодов категорий)
    corr_cols = ["closing_debt_rub", "paid_rub", "charged_rub", "payment_delay_days", "traffic_gb", "voice_minutes",
                 "active_days", "support_count", "tenure_days", "churn_target"]
    corr = s[corr_cols].astype(float).corr(method="spearman").round(3)
    write_csv(corr.reset_index().rename(columns={"index": "field"}), T / "spearman_correlations.csv")
    # сеть и отток на уровне регион × T (только известные на T отчёты)
    rm = s_net.groupby(["region", "snapshot_at"]).agg(n_observations=("customer_id", "size"), n_churn=("churn_target", "sum"),
                                                       known_at_T=("known_at_T", "max"), failed_probe_rate_prev_month_known=("failed_probe_rate_at_T", "max")).reset_index()
    rm["churn_rate"] = (rm.n_churn / rm.n_observations).round(5)
    write_csv(rm.assign(snapshot_at=fmt_date(rm.snapshot_at)), T / "network_vs_churn_region_T.csv")
    known_pairs = rm[rm.known_at_T == 1]
    net_corr = known_pairs[["failed_probe_rate_prev_month_known", "churn_rate"]].corr(method="spearman").iloc[0, 1] if len(known_pairs) > 2 else np.nan

    # --------------------------------------------------------------- гипотезы
    r = lambda g, k, col: g.set_index(col).loc[k]  # noqa: E731
    d0, d1 = r(g_debt, "долг = 0", "debt_group"), r(g_debt, "долг > 0", "debt_group")
    s0, s1, s2 = (r(g_sup, k, "support_group") for k in ["0", "1", "2+"])
    t_lo, t_mid, t_hi = (r(g_ten, k, "tenure_group") for k in order)
    first, last = by_t.iloc[0], by_t.iloc[-1]
    ratio_min, ratio_max = piv.rate_ratio_debt_vs_no_debt.min(), piv.rate_ratio_debt_vs_no_debt.max()
    ten_debt = g_ten_debt.set_index(["tenure_group", "debt_group"]).churn_rate
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    plan = pd.DataFrame([
        dict(hypothesis_id="H1", question="Отличается ли доля расторжений в месяце T у абонентов с долгом на конец M−1?",
             indicator="churn_rate = n_churn / n_observations (строки абонент × T)", groups="долг = 0; долг > 0; неизвестно (отдельно)",
             period="T = 01.12.2025–01.07.2026, горизонт — календарный месяц", expected_direction="не задавалось",
             limitations="Повторные наблюдения одних абонентов; долг связан с задержкой и нулевой оплатой; связь не доказывает причину",
             result="поддерживается",
             evidence=f"долг > 0: {pct(d1.churn_rate)} ({d1.n_churn}/{d1.n_observations}); долг = 0: {pct(d0.churn_rate)} ({d0.n_churn}/{d0.n_observations}); "
                      f"отношение долей {d1.churn_rate / d0.churn_rate:.1f}; внутри каждой из 8 дат T отношение {ratio_min:.1f}–{ratio_max:.1f}; "
                      f"внутри групп стажа и тарифа направление сохраняется (рис. 4, 5; debt_by_month.csv, debt_by_tariff.csv)",
             next_action="Признаки: долг M−1, число месяцев с долгом за 3 мес., задержка оплаты, оплата/начисление; baseline — сортировка по долгу"),
        dict(hypothesis_id="H2", question="Связано ли число доступных на T обращений за M−1 с расторжением в месяце T?",
             indicator="churn_rate по группам support_count", groups="0; 1; 2+ обращений (неизвестно — отдельно)",
             period="T = 01.12.2025–01.07.2026", expected_direction="не задавалось",
             limitations="Учитываются только доступные к T и корректно связанные обращения; 0 ≠ отсутствие реальных обращений; смена системы учёта в ноябре 2025",
             result="данных недостаточно",
             evidence=f"0: {pct(s0.churn_rate)} ({s0.n_churn}/{s0.n_observations}); 1: {pct(s1.churn_rate)} ({s1.n_churn}/{s1.n_observations}); "
                      f"2+: {pct(s2.churn_rate)} ({s2.n_churn}/{s2.n_observations}); доверительные интервалы пересекаются (рис. 4, fig4_churn_by_support.csv)",
             next_action="Проверить окно 3 месяца и темы (billing/network) как кандидатов; не ожидать сильного вклада"),
        dict(hypothesis_id="H3", question="Меняется ли доля оттока и размер активной базы по датам T?",
             indicator="n_churn / N_active на каждой T", groups="8 дат T", period="T = 01.12.2025–01.07.2026",
             expected_direction="не задавалось",
             limitations="8 точек; рост может быть связан с составом базы и сезонностью; валидация/тест не смотрелись",
             result="поддерживается",
             evidence=f"доля растёт монотонно с {pct(first.churn_rate)} ({first.n_churn}/{first.n_observations}) до {pct(last.churn_rate)} "
                      f"({last.n_churn}/{last.n_observations}); база почти стабильна: {by_t.n_observations.min()}–{by_t.n_observations.max()} (рис. 3)",
             next_action="Не смешивать даты при оценке: K и метрики считать по каждой T; учесть дрейф при сравнении train и validation"),
        dict(hypothesis_id="H4", question="Отличается ли доля оттока по стажу (<90, 90–364, ≥365 дней)?",
             indicator="churn_rate по tenure_group", groups=" / ".join(order), period="T = 01.12.2025–01.07.2026",
             expected_direction="не задавалось",
             limitations=f"Малая группа новичков ({t_lo.n_observations} строк); новички чаще имеют долг — состав групп",
             result="данных недостаточно (направление: новички уходят чаще)",
             evidence=f"{order[0]}: {pct(t_lo.churn_rate)} ({t_lo.n_churn}/{t_lo.n_observations}); {order[1]}: {pct(t_mid.churn_rate)}; {order[2]}: {pct(t_hi.churn_rate)}; "
                      f"интервалы пересекаются; внутри «долг = 0» {pct(ten_debt[(order[0], 'долг = 0')])} vs {pct(ten_debt[(order[2], 'долг = 0')])}, "
                      f"внутри «долг > 0» {pct(ten_debt[(order[0], 'долг > 0')])} vs {pct(ten_debt[(order[2], 'долг > 0')])} (рис. 5)",
             next_action="Включить стаж как числовой признак; эффект слабее долга, проверить на модели"),
    ])
    write_csv(plan, out / "analysis_plan.csv")
    directions = pd.DataFrame([
        dict(direction=1, meaning="Платёжная дисциплина: долг и его история, задержка оплаты, оплата/начисление",
             expected_value=f"Сильнейшая связь в EDA (H1): {d1.churn_rate / d0.churn_rate:.1f}× по доле оттока",
             source="billing_clean / customer_month", availability_at_T="месяцы ≤ M−1, available_at = 1-е число следующего месяца ≤ T",
             leakage_risk="Биллинг месяца T (прогнозного) и месяца расторжения недоступны на T — не использовать"),
        dict(direction=2, meaning="Изменение использования: трафик M−1, средний за 3 мес., изменение M−1 к M−2, трафик на активный день",
             expected_value="Низкий трафик чаще у уходящих (квинтили трафика), нужна нормировка на active_days",
             source="billing_clean", availability_at_T="как биллинг; пропуск трафика не заменяется нулём",
             leakage_risk="Неполный месяц подключения искажает сумму; разность считать только при обоих месяцах"),
        dict(direction=3, meaning="Стаж и обращения за 3 месяца (по темам), пересчёт по support_clean на T",
             expected_value="Слабая связь (H2, H4); полезны как контекст и для модели с нелинейностями",
             source="subscribers_clean, support_clean", availability_at_T="created_at в окне и available_at ≤ T",
             leakage_risk="Поздно поступившие обращения и закрытия после T; флаги карантина — не признаки"),
    ])
    write_csv(directions, out / "feature_directions.csv")

    # --------------------------------------------------------------- проверки
    checks = []
    checks.append(check_row("ключ (customer_id, snapshot_at) уникален", 0, int(s.duplicated(["customer_id", "snapshot_at"]).sum()),
                            not s.duplicated(["customer_id", "snapshot_at"]).any()))
    ev = data["contract_events_clean"]
    indep = {d: independent_active_count(ev, d) for d in dates}
    got = s.groupby("snapshot_at").size().to_dict()
    checks.append(check_row("строк на T = независимый расчёт базы по журналу событий", {k.date().isoformat(): v for k, v in indep.items()},
                            {k.date().isoformat(): v for k, v in got.items()}, indep == got))
    checks.append(check_row("billing_available_at ≤ snapshot_at", 0, int((s.billing_available_at > s.snapshot_at).sum()),
                            (s.billing_available_at <= s.snapshot_at).all()))
    exp_m = s.snapshot_at.map(lambda t: add_months(t, -1))
    checks.append(check_row("показатели взяты из предыдущего месяца", 0, int((s.source_billing_month != exp_m).sum()),
                            (s.source_billing_month == exp_m).all()))
    checks.append(check_row("у всех строк базы есть строка биллинга M−1", 0, int(s.source_billing_month.isna().sum()), s.source_billing_month.notna().all()))
    leak_cols = [c for c in s.columns if "termination" in c or c == "event_date"]
    checks.append(check_row("дата будущего расторжения не входит в таблицу", [], leak_cols, not leak_cols))
    checks.append(check_row("все метки EDA наблюдаемы (конец горизонта ≤ 01.10.2026)", 0, int(s.churn_target.isna().sum()), s.churn_target.notna().all()))
    checks += synthetic_boundary_check(cfg)
    syn = pd.DataFrame({"region": ["R", "R"], "measurement_month": pd.to_datetime(["2026-02-01", "2026-02-01"]),
                        "report_available_at": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-01 00:00:01")],
                        "probe_count": [100, 100], "failed_probe_count": [5, 5]})
    syn["failed_probe_rate"] = .05
    sa = network_availability(syn, [pd.Timestamp("2026-03-01")])
    checks.append(check_row("synthetic: сетевой отчёт ровно на T известен, на 1 с позже — нет", [1, 0], list(sa.known_at_T),
                            list(sa.known_at_T) == [1, 0] and pd.isna(sa.failed_probe_rate_at_T.iloc[1])))
    late = avail[(avail.report_available_at > avail.snapshot_at)]
    checks.append(check_row("реальные отчёты, опубликованные после T, не имеют значения на T", 0,
                            int(late.failed_probe_rate_at_T.notna().sum()), late.failed_probe_rate_at_T.isna().all(),
                            f"пар (регион, месяц, T) с поздним отчётом: {len(late)}"))
    for name, g in [("T", by_t), ("долг", g_debt), ("обращения", g_sup), ("стаж", g_ten)]:
        ok = g.n_observations.sum() == s.churn_target.notna().sum() and g.n_churn.sum() == s.churn_target.sum()
        checks.append(check_row(f"суммы групп «{name}» восстанавливают таблицу", (int(s.churn_target.notna().sum()), int(s.churn_target.sum())),
                                (int(g.n_observations.sum()), int(g.n_churn.sum())), ok))
    s_again = build_eda_sample(cfg, load_prepared(), dates)
    checks.append(check_row("повторное построение eda_sample даёт ту же таблицу", True, s_again.equals(s), s_again.equals(s)))
    ch = pd.DataFrame(checks)
    write_csv(ch, out / "checks_stage3.csv")

    prev_av = avail[avail.is_previous_month == 1]
    summary = {
        "eda_dates": [d.date().isoformat() for d in dates],
        "rows": len(s), "customers": int(s.customer_id.nunique()), "churn_events": int(s.churn_target.sum()),
        "churn_rate_overall": round(float(s.churn_target.mean()), 5),
        "network_prev_month_reports_known_at_T": int(prev_av.known_at_T.sum()),
        "network_prev_month_reports_unknown_at_T": int((prev_av.known_at_T == 0).sum()),
        "network_vs_churn_spearman_known_pairs": None if pd.isna(net_corr) else round(float(net_corr), 3),
        "network_known_pairs": int(len(known_pairs)),
        "checks_failed": int((ch.passed == "FAIL").sum()),
    }
    write_json(summary, out / "eda_summary.json")
    return summary
