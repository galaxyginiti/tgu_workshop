"""Этапы 6–7. Итоговая проверка сохранённой модели и baseline на test, без обучения.

Вход: features.csv, labels.csv, split.csv этапа 4 и candidate.joblib этапа 5.
Выход (solution/stage6_final/output):
  test_metrics.csv, test_candidate_list.csv, test_baseline_list.csv, quality_table.csv,
  test_overlap_with_baseline.csv, segment_debt_test.csv, three_features.csv, traffic_quintiles_train.csv,
  checks_stage6.csv, final_summary.json.
Расчёт на test идёт в отдельном процессе, где любой fit — ошибка; второй такой процесс
повторяет расчёт для проверки совпадения списков и метрик.
"""
from __future__ import annotations

import filecmp
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import (SOLUTION_DIR, STAGE2_OUT, STAGE3_OUT, STAGE4_OUT, STAGE5_OUT, STAGE6_OUT, check_row, read_prepared, read_raw_csv,
                     sha256, write_csv, write_json)
from .preprocessing import FORBIDDEN
from .snapshots import independent_active_count
from .stage5_train import (INPUT_FILES, KEYS, MODEL_PATH, baseline_score, compare_with_baseline, forbid_fit, rank_and_measure,
                           read_stage4, risk_score, write_predictions)

RESULT_FILES = ["test_candidate_list.csv", "test_baseline_list.csv", "test_metrics.csv"]


def load_holdout(tables: dict[str, pd.DataFrame], bundle: dict, part: str, observation_end: str) -> dict:
    """Одна отложенная часть по ключу; исход должен быть полностью наблюдаем, неизвестный исход не становится нулём."""
    features, labels, split = tables["features"], tables["labels"], tables["split"]
    key_set = lambda d: set(map(tuple, d[KEYS].values))  # noqa: E731
    if key_set(features) != key_set(labels) or key_set(features) != key_set(split):
        raise ValueError("Множества ключей features/labels/split различаются")
    numeric, categorical = bundle["numeric_columns"], bundle["categorical_columns"]
    columns = numeric + categorical
    if len(set(columns)) != len(columns) or set(columns) & FORBIDDEN:
        raise ValueError("Недопустимая схема входа модели")
    wanted = split.loc[split.split == part, KEYS]
    frame = features[KEYS + columns].merge(wanted, on=KEYS, validate="one_to_one")
    frame = frame.merge(labels[KEYS + ["churn_target", "target_end_exclusive"]], on=KEYS, validate="one_to_one")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Пустая часть {part}")
    target = pd.to_numeric(frame.churn_target.replace("", np.nan), errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Неизвестный исход нельзя превращать в ноль")
    t, end = pd.to_datetime(frame.snapshot_at), pd.to_datetime(frame.target_end_exclusive)
    if not ((end > t) & (end <= pd.Timestamp(observation_end))).all():
        raise ValueError("Горизонт метки наблюдается не полностью")
    X = frame[columns].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col].replace("", np.nan), errors="raise").astype(float)
    for col in categorical:
        X[col] = X[col].replace("", np.nan).astype(object)
    return {"X": X, "y": target.to_numpy(dtype=int), "keys": frame[KEYS].copy()}


def evaluate_saved_model(out: Path, observation_end: str) -> dict:
    """Запускается в отдельном процессе: загрузка кандидата и расчёт на test. Обучения здесь нет."""
    forbid_fit()
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    tables = read_stage4()
    test = load_holdout(tables, bundle, "test", observation_end)
    fraction = bundle["config"]["campaign_fraction"]
    _, table_m, ranked_m = rank_and_measure(test["keys"], test["y"], risk_score(bundle["pipeline"], test["X"]), fraction)
    _, table_b, ranked_b = rank_and_measure(test["keys"], test["y"], baseline_score(test["X"], bundle["config"]["baseline_feature"]), fraction)
    write_predictions(ranked_m, out / "test_candidate_list.csv")
    write_predictions(ranked_b, out / "test_baseline_list.csv")
    metrics = pd.concat([table_b.assign(approach="baseline"), table_m.assign(approach=bundle["selected_experiment"])], ignore_index=True)
    write_csv(metrics[["approach", "snapshot_at", "n_active", "K", "n_churn", "hits", "recall_at_K", "precision_at_K", "average_precision"]],
              out / "test_metrics.csv")
    return dict(selected_experiment=bundle["selected_experiment"], fit_forbidden_in_process=True, rows=len(ranked_m))


def _run_evaluation(out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(SOLUTION_DIR / "evaluate_test.py"), "--out", str(out)],
                          capture_output=True, encoding="utf-8", env=env)


def _ranked_from_files(list_path: Path, keys: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    saved = read_raw_csv(list_path)
    frame = keys.assign(target=y).merge(saved, on=KEYS, validate="one_to_one")
    frame["selected"] = frame.selected == "True"
    return frame


def segment_debt(keys: pd.DataFrame, y: np.ndarray, debt: pd.Series, model: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Отбор взят из общего top-K: заново 10% внутри группы не выбираются."""
    frame = keys.assign(target=y, group=np.select([debt.isna(), debt > 0], ["неизвестно", "долг > 0"], "долг = 0"))
    frame = frame.merge(model[KEYS + ["selected"]].rename(columns={"selected": "m"}), on=KEYS).merge(
        base[KEYS + ["selected"]].rename(columns={"selected": "b"}), on=KEYS)
    rows = []
    for group, g in frame.groupby("group", sort=False):
        rows.append(dict(group=group, n=len(g), n_churn=int(g.target.sum()),
                         churn_rate=g.target.mean(),
                         model_selected=int(g.m.sum()), model_hits=int((g.m & (g.target == 1)).sum()),
                         baseline_selected=int(g.b.sum()), baseline_hits=int((g.b & (g.target == 1)).sum())))
    order = {"долг > 0": 0, "долг = 0": 1, "неизвестно": 2}
    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.map(order)).reset_index(drop=True)


def traffic_quintiles_train() -> pd.DataFrame:
    """Отток по пятым долям трафика M−1 на датах train (неизвестный трафик не входит)."""
    eda = read_raw_csv(STAGE3_OUT / "eda_sample.csv")
    traffic = pd.to_numeric(eda.traffic_gb.replace("", np.nan))
    known = eda[traffic.notna()].assign(traffic=traffic.dropna(), target=lambda d: pd.to_numeric(d.churn_target))
    known["quintile"] = pd.qcut(known.traffic, 5, labels=False) + 1
    g = known.groupby("quintile").agg(traffic_min=("traffic", "min"), traffic_max=("traffic", "max"), n=("target", "size"), n_churn=("target", "sum")).reset_index()
    g["n_churn"] = g.n_churn.astype(int)
    g["churn_rate"] = g.n_churn / g.n
    return g


def _ru_pct(v: float) -> str:
    return f"{v * 100:.1f}%".replace(".", ",")


def _ru_int(v) -> str:
    return f"{int(v):,}".replace(",", " ")


def three_features(quintiles: pd.DataFrame) -> pd.DataFrame:
    """Основание — выводы EDA на train и коэффициенты exp04 (не выбранная модель): важность для бустинга не считалась."""
    coef = read_raw_csv(STAGE5_OUT / "lr_coefficients.csv")
    value = lambda name: float(coef.loc[coef.x_column == f"numeric__{name}", "coefficient"].iloc[0])  # noqa: E731
    tables = STAGE3_OUT / "tables"
    debt = read_raw_csv(tables / "fig4_churn_by_debt.csv").set_index("debt_group")
    delay = read_raw_csv(tables / "churn_by_payment_delay.csv").set_index("delay_group")
    row = lambda t, k: f"{_ru_pct(float(t.loc[k, 'churn_rate']))} ({_ru_int(t.loc[k, 'n_churn'])} из {_ru_int(t.loc[k, 'n_observations'])})"  # noqa: E731
    low, rest = quintiles.iloc[0], quintiles.iloc[1:]
    return pd.DataFrame([
        dict(feature="debt_last_rub", meaning="Долг на конец прошлого месяца: сколько абонент не оплатил к дате расчёта",
             possible_link="Неоплата может отражать снижение интереса к услуге или финансовые трудности",
             eda_basis=f"Долг > 0: {row(debt, 'долг > 0')} оттока; долг = 0: {row(debt, 'долг = 0')}; направление на всех 8 датах T",
             lr_coefficient=value("debt_last_rub")),
        dict(feature="payment_delay_last_days", meaning="Задержка оплаты в прошлом месяце, дни",
             possible_link="Систематическая задержка может предшествовать отказу от услуги",
             eda_basis=f"Задержка > 10 дней: {row(delay, '> 10 дн.')} оттока; без задержки: {row(delay, '0 дн.')}",
             lr_coefficient=value("payment_delay_last_days")),
        dict(feature="traffic_last_gb", meaning="Трафик в прошлом месяце, ГБ; пропуск означает «неизвестно», а не ноль",
             possible_link="Низкое использование может означать, что абонент уже пользуется услугами реже",
             eda_basis=f"Нижние 20% по трафику: {_ru_pct(low.churn_rate)} ({_ru_int(low.n_churn)} из {_ru_int(low.n)}) оттока; "
                       f"остальные группы: {_ru_pct(rest.churn_rate.min())}–{_ru_pct(rest.churn_rate.max())}; связь слабая",
             lr_coefficient=value("traffic_last_gb")),
    ])


def run(cfg: dict) -> dict:
    out = STAGE6_OUT
    out.mkdir(parents=True, exist_ok=True)
    observation_end = cfg["observation_end_exclusive"]
    hashes_before = {f: sha256(STAGE4_OUT / f) for f in INPUT_FILES}
    model_hash_before = sha256(MODEL_PATH)

    first = _run_evaluation(out)
    if first.returncode != 0:
        raise RuntimeError(f"Расчёт на test не выполнен: {first.stderr[-800:]}")
    with tempfile.TemporaryDirectory() as tmp:
        repeat = _run_evaluation(Path(tmp))
        repeat_ok = repeat.returncode == 0 and all(filecmp.cmp(out / f, Path(tmp) / f, shallow=False) for f in RESULT_FILES)

    bundle = joblib.load(MODEL_PATH)
    tables = read_stage4()
    test = load_holdout(tables, bundle, "test", observation_end)
    sel = bundle["selected_experiment"]
    metrics = read_raw_csv(out / "test_metrics.csv")
    tm, tb = metrics[metrics.approach == sel].iloc[0], metrics[metrics.approach == "baseline"].iloc[0]
    ranked_m = _ranked_from_files(out / "test_candidate_list.csv", test["keys"], test["y"])
    ranked_b = _ranked_from_files(out / "test_baseline_list.csv", test["keys"], test["y"])

    # Единая таблица качества: validation перенесена из этапа 5, test — из текущего расчёта
    stage5 = read_raw_csv(STAGE5_OUT / "experiment_metrics.csv")
    rows = []
    for period, approach, name, source in [("Validation", "baseline", "Baseline", None), ("Validation", sel, "Модель", None),
                                           ("Test", "baseline", "Baseline", tb), ("Test", sel, "Модель", tm)]:
        if source is None:
            m = stage5[(stage5.approach == approach) & (stage5.part == "validation")].iloc[0]
            n, k, hits, churn, recall, precision, ap = m.n_active, m.K, m.hits, m.n_churn, m.macro_recall_at_K, m.macro_precision_at_K, m.macro_average_precision
        else:
            n, k, hits, churn, recall, precision, ap = source.n_active, source.K, source.hits, source.n_churn, source.recall_at_K, source.precision_at_K, source.average_precision
        rows.append(dict(period=period, approach=name, experiment=approach, n_active=int(n), K=int(k), hits=int(hits), n_churn=int(churn),
                         recall_at_K=float(recall), precision_at_K=float(precision), average_precision=float(ap)))
    quality = pd.DataFrame(rows)
    write_csv(quality, out / "quality_table.csv")

    overlap, overlap_info = compare_with_baseline(ranked_m, ranked_b)
    write_csv(overlap, out / "test_overlap_with_baseline.csv")
    debt = test["X"][bundle["config"]["baseline_feature"]]
    segment = segment_debt(test["keys"], test["y"], debt, ranked_m, ranked_b)
    write_csv(segment, out / "segment_debt_test.csv")
    quintiles = traffic_quintiles_train()
    write_csv(quintiles, out / "traffic_quintiles_train.csv")
    features = three_features(quintiles)
    write_csv(features, out / "three_features.csv")

    # Три обязательные проверки задания и их составные части
    checks = []
    events = read_prepared(STAGE2_OUT / "contract_events_clean.csv")
    t = sorted(test["keys"].snapshot_at.unique())
    indep = {d: independent_active_count(events, pd.Timestamp(d)) for d in t}
    got = {d: int(n) for d, n in test["keys"].groupby("snapshot_at").size().items()}
    checks.append(check_row("сопоставимость: строк на T test = независимая активная база", indep, got, indep == got))
    same_nk = int(tm.n_active) == int(tb.n_active) and int(tm.K) == int(tb.K)
    checks.append(check_row("сопоставимость: у модели и baseline одинаковые N и K", f"{tb.n_active}/{tb.K}", f"{tm.n_active}/{tm.K}", same_nk))
    same_keys = set(map(tuple, ranked_m[KEYS].values)) == set(map(tuple, ranked_b[KEYS].values)) == set(map(tuple, test["keys"].values))
    checks.append(check_row("сопоставимость: ключи списков модели и baseline совпадают с базой test", True, same_keys, same_keys))
    k = int(tm.K)
    expected_k = int(np.floor(cfg["campaign_fraction"] * int(tm.n_active) + 1e-9))
    checks.append(check_row("K = floor(0,10 × N), в списке ровно K выбранных", f"{expected_k}, {expected_k}, {expected_k}",
                            f"{k}, {int(ranked_m.selected.sum())}, {int(ranked_b.selected.sum())}",
                            k == expected_k == int(ranked_m.selected.sum()) == int(ranked_b.selected.sum())))
    checks.append(check_row("исход test полностью наблюдаем (горизонт не позже конца журнала)", observation_end, str(pd.to_datetime(tables["labels"].merge(test["keys"], on=KEYS).target_end_exclusive).max().date()),
                            pd.to_datetime(tables["labels"].merge(test["keys"], on=KEYS).target_end_exclusive).max() <= pd.Timestamp(observation_end)))
    card = json.loads((STAGE5_OUT / "model" / "model_card.json").read_text(encoding="utf-8"))
    checks.append(check_row("корректность: загружена модель, выбранная на этапе 5", card["selected_experiment"], sel, card["selected_experiment"] == sel))
    checks.append(check_row("корректность: модель обучена только на train этапа 5", card["fit_rows"], int(tables["split"].split.eq("train").sum()),
                            card["fit_rows"] == int(tables["split"].split.eq("train").sum()) and not card["test_used"]))
    checks.append(check_row("корректность: расчёт на test в процессе, где любой fit — ошибка", 0, first.returncode, first.returncode == 0,
                            "подготовка и модель только применяются"))
    model_same = sha256(MODEL_PATH) == model_hash_before
    checks.append(check_row("корректность: файл модели не изменился при расчёте (sha256)", "равны", "равны" if model_same else "различаются", model_same))
    data_same = all(sha256(STAGE4_OUT / f) == hashes_before[f] == bundle["data_sha256"][f] for f in INPUT_FILES)
    checks.append(check_row("корректность: входы — те же таблицы этапа 4, что при обучении (sha256)", "равны", "равны" if data_same else "различаются", data_same))
    checks.append(check_row("повторный запуск: второй процесс дал те же списки контактов и метрики", True, repeat_ok, repeat_ok, "побайтовое сравнение трёх файлов"))
    ch = pd.DataFrame(checks)
    write_csv(ch, out / "checks_stage6.csv")

    summary = dict(selected_experiment=sel, test_date=t[0],
                   quality=quality.to_dict("records"), test_extra_hits_vs_baseline=int(tm.hits) - int(tb.hits),
                   test_overlap_with_baseline=overlap_info, segment_debt_test=segment.to_dict("records"),
                   checks_failed=int((ch.passed == "FAIL").sum()))
    write_json(summary, out / "final_summary.json")
    return summary
