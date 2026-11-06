"""Этап 5. Обучение двух методов, сравнение с baseline и сохранение кандидата.

Вход: features.csv, labels.csv, split.csv этапа 4 и план solution/config/experiments.json.
Выход (solution/stage5_models/output):
  experiments.csv, experiment_metrics.csv, metrics_by_T.csv, stability_folds.csv,
  stability_summary.csv, validation_predictions.csv, validation_baseline_ranking.csv,
  lr_coefficients.csv, model/candidate.joblib, model/model_card.json, reload_check.json,
  checks_stage5.csv, training_summary.json.
Время обучения меняется от запуска к запуску, поэтому лежит отдельно: run_log/ и журнал экспериментов.
Test (01.09.2026) отбрасывается при загрузке: метрики по нему здесь не считаются.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from math import comb
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import (SOLUTION_DIR, STAGE2_OUT, STAGE4_OUT, STAGE5_OUT, check_row, read_prepared, read_raw_csv,
                     sha256, write_csv, write_json)
from .preprocessing import FORBIDDEN, make_preprocessor, make_tree_preprocessor
from .snapshots import independent_active_count

KEYS = ["customer_id", "snapshot_at"]
EXPERIMENTS_PATH = SOLUTION_DIR / "config" / "experiments.json"
MODEL_PATH = STAGE5_OUT / "model" / "candidate.joblib"
PREDICTIONS_PATH = STAGE5_OUT / "validation_predictions.csv"
METRICS_PATH = STAGE5_OUT / "experiment_metrics.csv"
RELOAD_PATH = STAGE5_OUT / "reload_check.json"
RUN_LOG_PATH = SOLUTION_DIR / "stage5_models" / "run_log" / "experiment_runtime.csv"
JOURNAL_PATH = SOLUTION_DIR / "stage5_models" / "Журнал_экспериментов.md"
INPUT_FILES = ["features.csv", "labels.csv", "split.csv"]
METHOD_NAMES = {"logistic_regression": "Логистическая регрессия", "hist_gradient_boosting": "Градиентный бустинг"}


def load_plan() -> dict:
    with open(EXPERIMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_x_columns() -> tuple[list[str], list[str]]:
    with open(STAGE4_OUT / "preprocessing_report.json", encoding="utf-8") as f:
        report = json.load(f)
    return report["numeric_columns"], report["categorical_columns"]


def read_stage4() -> dict[str, pd.DataFrame]:
    return {name: read_raw_csv(STAGE4_OUT / f"{name}.csv") for name in ["features", "labels", "split"]}


# --------------------------------------------------------------------------- вход

def _check_keys(frame: pd.DataFrame, name: str) -> None:
    if not set(KEYS) <= set(frame.columns):
        raise ValueError(f"{name}: нет ключей")
    if frame[KEYS].eq("").any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name}: пустые или повторные ключи")


def load_parts(tables: dict[str, pd.DataFrame], numeric: list[str], categorical: list[str]) -> dict:
    """Train и validation, соединённые по ключу. Test отбрасывается до разбора значений и меток."""
    features, labels, split = tables["features"], tables["labels"], tables["split"]
    for name, frame in tables.items():
        _check_keys(frame, name)
    key_set = lambda d: set(map(tuple, d[KEYS].values))  # noqa: E731
    if key_set(features) != key_set(labels) or key_set(features) != key_set(split):
        raise ValueError("Множества ключей features/labels/split различаются")
    selected = numeric + categorical
    if len(set(selected)) != len(selected) or set(selected) & FORBIDDEN or not set(selected) <= set(features.columns):
        raise ValueError("Список X должен быть явным, без ключей и метки")
    wanted = split[split.split.isin(["train", "validation"])]
    frame = features[KEYS + selected].merge(wanted, on=KEYS, how="inner", validate="one_to_one")
    frame = frame.merge(labels[KEYS + ["churn_target", "target_end_exclusive"]], on=KEYS, how="left", validate="one_to_one")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    target = pd.to_numeric(frame.churn_target.replace("", np.nan), errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Метки train/validation должны быть известны и бинарны")
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col].replace("", np.nan), errors="raise").astype(float)
        if np.isinf(frame[col]).any():
            raise ValueError(f"Бесконечность в признаке {col}")
    for col in categorical:
        frame[col] = frame[col].replace("", np.nan).astype(object)
    frame["churn_target"] = target.astype(int)
    parts = {}
    for part in ["train", "validation"]:
        p = frame[frame.split == part].reset_index(drop=True)
        parts[part] = {"X": p[selected].copy(), "y": p.churn_target.to_numpy(), "keys": p[KEYS].copy(),
                       "target_end": p.target_end_exclusive.copy()}
    train_t = pd.to_datetime(parts["train"]["keys"].snapshot_at)
    valid_t = pd.to_datetime(parts["validation"]["keys"].snapshot_at)
    train_end = pd.to_datetime(parts["train"]["target_end"])
    if train_t.max() >= valid_t.min() or train_end.max() > valid_t.min():
        raise ValueError("Даты или горизонты train заходят за начало validation")
    if set(parts["train"]["y"]) != {0, 1}:
        raise ValueError("В train нужны оба класса")
    return parts


# --------------------------------------------------------------------------- метрики кампании

def campaign_k(n: int, fraction: float) -> int:
    return int((Decimal(str(fraction)) * n).to_integral_value(rounding=ROUND_FLOOR))


def rank_and_measure(keys: pd.DataFrame, y: np.ndarray, score: np.ndarray, fraction: float):
    """Вся база T сортируется по score (при равенстве — customer_id), берутся первые K. Порог 0,5 не используется."""
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("Score должен быть конечным")
    frame = keys.reset_index(drop=True).copy()
    frame["target"] = np.asarray(y)
    frame["score"] = score
    frame = frame.sort_values(["snapshot_at", "score", "customer_id"], ascending=[True, False, True], kind="stable")
    tables, ranked = [], []
    for t, part in frame.groupby("snapshot_at", sort=True):
        part = part.copy()
        n = len(part)
        k = campaign_k(n, fraction)
        part["rank"] = np.arange(1, n + 1)
        part["selected"] = part["rank"] <= k
        pos = int(part.target.sum())
        hits = int(part.loc[part.selected, "target"].sum())
        tables.append(dict(snapshot_at=t, n_active=n, K=k, n_churn=pos, hits=hits,
                           recall_at_K=hits / pos if pos else np.nan,
                           precision_at_K=hits / k if k else np.nan,
                           average_precision=float(average_precision_score(part.target, part.score)) if pos else np.nan))
        ranked.append(part)
    table = pd.DataFrame(tables)
    summary = dict(macro_recall_at_K=float(table.recall_at_K.mean()), macro_precision_at_K=float(table.precision_at_K.mean()),
                   macro_average_precision=float(table.average_precision.mean()),
                   pooled_recall_at_K=int(table.hits.sum()) / int(table.n_churn.sum()) if table.n_churn.sum() else np.nan,
                   pooled_precision_at_K=int(table.hits.sum()) / int(table.K.sum()) if table.K.sum() else np.nan,
                   n_active=int(table.n_active.sum()), K=int(table.K.sum()), n_churn=int(table.n_churn.sum()),
                   hits=int(table.hits.sum()), n_dates=len(table))
    return summary, table, pd.concat(ranked, ignore_index=True)


def baseline_score(X: pd.DataFrame, feature: str) -> np.ndarray:
    """Долг M−1; неизвестный долг ставится после известных. Это ранжирование, а не восстановление значения."""
    debt = X[feature]
    return debt.fillna(debt.min() - 1 if debt.notna().any() else -1).to_numpy()


def risk_score(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    positive = int(np.flatnonzero(pipeline.classes_ == 1)[0])
    return pipeline.predict_proba(X)[:, positive]


# --------------------------------------------------------------------------- обучение

def build_pipeline(experiment: dict, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    params = dict(experiment.get("params", {}))
    family = experiment["family"]
    if family == "logistic_regression":
        options = dict(max_iter=2000, solver="lbfgs", random_state=seed) | params
        return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", LogisticRegression(**options))])
    if family == "hist_gradient_boosting":
        # Внутренняя ранняя остановка случайна по строкам и нарушила бы порядок времени.
        options = dict(early_stopping=False, random_state=seed) | params
        return Pipeline([("preprocess", make_tree_preprocessor(numeric, categorical)),
                         ("model", HistGradientBoostingClassifier(**options))])
    raise ValueError(f"Неизвестный метод: {family}")


def fit_with_log(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[float, str]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    seconds = time.perf_counter() - started
    texts = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})
    return seconds, "; ".join(texts)


def select_candidate(rows: list[dict]) -> dict:
    """Правило из experiments.json: Recall@K на validation, затем AP, затем порядок в плане."""
    return sorted(rows, key=lambda r: (-r["valid_recall"], -r["valid_ap"], r["order"]))[0]


def stability(experiment: dict, folds: list[dict], parts: dict, numeric: list[str], categorical: list[str],
              plan: dict) -> list[dict]:
    """Расширяющееся окно внутри train: весь Pipeline обучается заново в каждом фолде."""
    X, y, keys = parts["train"]["X"], parts["train"]["y"], parts["train"]["keys"]
    t = keys.snapshot_at
    rows = []
    for fold in folds:
        tr = (t <= fold["train_until_inclusive"]).to_numpy()
        ev = (t == fold["evaluate_at"]).to_numpy()
        ends = pd.to_datetime(parts["train"]["target_end"][tr])
        assert ends.max() <= pd.Timestamp(fold["evaluate_at"]), "Метки обучения фолда не наблюдаемы к дате оценки"
        pipe = build_pipeline(experiment, numeric, categorical, plan["seed"])
        fit_with_log(pipe, X[tr], y[tr])
        _, table, _ = rank_and_measure(keys[ev], y[ev], risk_score(pipe, X[ev]), plan["campaign_fraction"])
        rows.append(table.assign(approach=experiment["id"], fold=fold["fold"],
                                 train_until_inclusive=fold["train_until_inclusive"], train_rows=int(tr.sum())).iloc[0].to_dict())
    return rows


def baseline_stability(folds: list[dict], parts: dict, plan: dict) -> list[dict]:
    X, y, keys = parts["train"]["X"], parts["train"]["y"], parts["train"]["keys"]
    rows = []
    for fold in folds:
        ev = (keys.snapshot_at == fold["evaluate_at"]).to_numpy()
        _, table, _ = rank_and_measure(keys[ev], y[ev], baseline_score(X[ev], plan["baseline_feature"]), plan["campaign_fraction"])
        rows.append(table.assign(approach="baseline", fold=fold["fold"], train_until_inclusive=fold["train_until_inclusive"],
                                 train_rows=0).iloc[0].to_dict())
    return rows


def debt_signal_by_t(parts: dict, feature: str) -> pd.DataFrame:
    """Сила сигнала долга на каждой T: почему правило и модели могут проседать на отдельной дате."""
    rows = []
    for part, data in parts.items():
        frame = data["keys"].assign(debt=data["X"][feature].to_numpy(), target=data["y"])
        for t, g in frame.groupby("snapshot_at", sort=True):
            has_debt = g.debt > 0
            n_debt, n_churn = int(has_debt.sum()), int(g.target.sum())
            churn_debt, churn_clean = int(g.target[has_debt].sum()), int(g.target[~has_debt].sum())
            rows.append(dict(part=part, snapshot_at=t, n_active=len(g), n_churn=n_churn, churn_rate=n_churn / len(g),
                             n_debt_positive=n_debt, share_debt_positive=n_debt / len(g),
                             churn_with_debt=churn_debt, churn_without_debt=churn_clean,
                             churn_rate_with_debt=churn_debt / n_debt if n_debt else np.nan,
                             churn_rate_without_debt=churn_clean / (len(g) - n_debt) if len(g) > n_debt else np.nan,
                             share_churners_with_debt=churn_debt / n_churn if n_churn else np.nan))
    return pd.DataFrame(rows)


def compare_with_baseline(model_ranked: pd.DataFrame, baseline_ranked: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Кого из ушедших нашли модель и правило. Точный парный тест знаков — только по расхождениям."""
    m = model_ranked[KEYS + ["target", "selected"]].merge(baseline_ranked[KEYS + ["selected"]], on=KEYS, validate="one_to_one",
                                                          suffixes=("_model", "_baseline"))
    churned = m[m.target == 1]
    both = int((churned.selected_model & churned.selected_baseline).sum())
    only_model = int((churned.selected_model & ~churned.selected_baseline).sum())
    only_base = int((~churned.selected_model & churned.selected_baseline).sum())
    neither = int((~churned.selected_model & ~churned.selected_baseline).sum())
    discordant = only_model + only_base
    tail = sum(comb(discordant, i) for i in range(min(only_model, only_base) + 1)) / 2 ** discordant if discordant else 1.0
    p_value = min(1.0, 2 * tail)
    top_both = int((m.selected_model & m.selected_baseline).sum())
    table = pd.DataFrame([
        dict(group="ушедшие: нашли оба", n=both), dict(group="ушедшие: нашла только модель", n=only_model),
        dict(group="ушедшие: нашло только правило", n=only_base), dict(group="ушедшие: не нашёл никто", n=neither),
        dict(group="top-K: общие абоненты", n=top_both), dict(group="top-K: размер", n=int(m.selected_model.sum())),
    ])
    return table, dict(both=both, only_model=only_model, only_baseline=only_base, neither=neither, top_k_common=top_both,
                       sign_test_p_two_sided=round(p_value, 4))


def write_predictions(ranked: pd.DataFrame, path: Path) -> None:
    out = ranked[KEYS + ["score", "rank", "selected"]].copy()
    out["score"] = out.score.map(lambda v: format(float(v), ".17g"))  # без округления: нужно для сверки после загрузки
    write_csv(out, path)


# --------------------------------------------------------------------------- проверка загрузки

def forbid_fit() -> None:
    """В процессе проверки любое обучение становится ошибкой."""
    def _forbidden(self, *args, **kwargs):
        raise RuntimeError(f"Запрещённый fit в {type(self).__name__}")
    for cls in [Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, LogisticRegression,
                HistGradientBoostingClassifier]:
        cls.fit = _forbidden
        if hasattr(cls, "fit_transform"):
            cls.fit_transform = _forbidden


def reload_check() -> dict:
    """Запускается в новом процессе: загрузка кандидата, validation-оценки без fit и сверка с сохранёнными."""
    forbid_fit()
    bundle = joblib.load(MODEL_PATH)
    tables = read_stage4()
    data = load_parts(tables, bundle["numeric_columns"], bundle["categorical_columns"])["validation"]
    fraction = bundle["config"]["campaign_fraction"]
    summary, _, ranked = rank_and_measure(data["keys"], data["y"], risk_score(bundle["pipeline"], data["X"]), fraction)
    saved = read_raw_csv(PREDICTIONS_PATH)
    if saved.duplicated(KEYS).any():
        raise ValueError("Повторные ключи в сохранённых прогнозах")
    paired = ranked.merge(saved, on=KEYS, how="outer", suffixes=("_new", "_old"), validate="one_to_one", indicator=True)
    keys_equal = bool((paired._merge == "both").all())
    score_old = pd.to_numeric(paired.score_old)
    scores_close = bool(np.allclose(paired.score_new, score_old, atol=1e-12, rtol=1e-12))
    rank_equal = bool((paired.rank_new == pd.to_numeric(paired.rank_old)).all())
    top_old = set(saved.loc[saved.selected == "True", "customer_id"])
    top_equal = set(ranked.loc[ranked.selected, "customer_id"]) == top_old
    metrics = read_raw_csv(METRICS_PATH)
    row = metrics[(metrics.approach == bundle["selected_experiment"]) & (metrics.part == "validation")].iloc[0]
    hits_equal = summary["hits"] == int(row.hits) and summary["K"] == int(row.K) and summary["n_churn"] == int(row.n_churn)
    hashes_equal = all(sha256(STAGE4_OUT / f) == bundle["data_sha256"][f] for f in INPUT_FILES)
    result = dict(selected_experiment=bundle["selected_experiment"], fit_forbidden_in_process=True, rows=len(ranked),
                  keys_equal=keys_equal, scores_close_1e_12=scores_close, rank_equal=rank_equal, top_k_equal=top_equal,
                  metrics_equal=hits_equal, input_data_hashes_equal=hashes_equal,
                  validation_hits=summary["hits"], validation_n_churn=summary["n_churn"], validation_K=summary["K"],
                  validation_recall_at_K=round(summary["macro_recall_at_K"], 6))
    result["passed"] = all([keys_equal, scores_close, rank_equal, top_equal, hits_equal, hashes_equal])
    write_json(result, RELOAD_PATH)
    return result


def synthetic_metric_checks() -> list[dict]:
    res = []
    keys = pd.DataFrame({"customer_id": [f"C{i:02d}" for i in range(25)], "snapshot_at": "2026-01-01"})
    y = np.zeros(25, dtype=int)
    y[[1, 2]] = 1
    score = np.full(25, 0.5)
    summary, table, ranked = rank_and_measure(keys, y, score, 0.1)
    top = list(ranked[ranked.selected].customer_id)
    res.append(check_row("synthetic: K = floor(0,10 × 25), равные score — по customer_id", "K=2, C00+C01", f"K={table.K[0]}, {'+'.join(top)}",
                         table.K[0] == 2 and top == ["C00", "C01"]))
    res.append(check_row("synthetic: Recall@K и Precision@K считаются по top-K", "hits=1, recall=0.5, precision=0.5",
                         f"hits={summary['hits']}, recall={table.recall_at_K[0]}, precision={table.precision_at_K[0]}",
                         summary["hits"] == 1 and table.recall_at_K[0] == .5 and table.precision_at_K[0] == .5))
    _, t0, _ = rank_and_measure(keys, np.zeros(25, dtype=int), score, 0.1)
    res.append(check_row("synthetic: без ушедших Recall@K не определён (не 0)", "NaN", t0.recall_at_K[0], pd.isna(t0.recall_at_K[0])))
    small = pd.DataFrame({"customer_id": ["A", "B", "C"], "snapshot_at": "2026-01-01"})
    _, t1, _ = rank_and_measure(small, np.array([1, 0, 0]), np.array([.9, .1, .1]), 0.1)
    res.append(check_row("synthetic: при K=0 Precision@K не определена (не 0)", "K=0, NaN", f"K={t1.K[0]}, {t1.precision_at_K[0]}",
                         t1.K[0] == 0 and pd.isna(t1.precision_at_K[0])))
    x = pd.DataFrame({"debt": [100.0, np.nan, 0.0, 50.0]})
    bs = baseline_score(x, "debt")
    order = list(np.argsort(-bs, kind="stable"))
    res.append(check_row("synthetic: неизвестный долг идёт после известных и нулём не считается", [0, 3, 2, 1], [int(i) for i in order], order == [0, 3, 2, 1]))
    return res


# --------------------------------------------------------------------------- журнал

def _pct(v) -> str:
    return "не определён" if pd.isna(v) else f"{v * 100:.1f}%".replace(".", ",")


def write_journal(plan: dict, data_hash: str, git_rev: str, log_rows: list[dict], metrics: pd.DataFrame, selected: dict,
                  baseline_valid: dict, warnings_text: str, runtime: pd.DataFrame) -> None:
    lines = ["# Журнал экспериментов (этап 5)", "",
             "Журнал пишется кодом при запуске: `python solution/run_all.py --stage 5`.", "",
             "Команда: Анишко Руслан, Шоймардонов Ботир, Адылбаев Камал.",
             f"Версия кода: `{git_rev}` (коммит перед запуском). Версия признаков: {plan['x_version']}, sha256 features.csv `{data_hash[:12]}`.", "",
             "Периоды: train — T = 01.12.2025–01.07.2026 (8 дат), validation — T = 01.08.2026; test на этом этапе не загружается.",
             "Основная метрика: Recall@K, K = ⌊0,10 × N_active⌋ на каждой T. Равные оценки — по `customer_id` по возрастанию.", "",
             "Правило выбора зафиксировано до обучения (отдельный коммит с `solution/config/experiments.json`): " + "; ".join(plan["selection_rule"]) + ".", "",
             "| ID | Метод | Параметры | Seed | Версия X | Fit, с | Статус |", "|---|---|---|---:|---|---:|---|",
             f"| baseline | Правило: долг M−1 по убыванию, неизвестный долг после известных | — | — | {plan['baseline_feature']} | — | ok |"]
    secs = runtime.set_index("experiment_id")
    for r in log_rows:
        lines.append(f"| {r['experiment_id']} | {r['method']} | `{r['params']}` | {r['seed']} | {r['x_version']} | "
                     f"{secs.loc[r['experiment_id'], 'fit_seconds']:.2f} | {r['status']} |")
    lines += ["", "Train — macro по 8 датам (pooled отдельно); validation — одна дата, поэтому macro = pooled. AP — `average_precision_score`, для train среднее по датам.", "",
              "| ID | Часть | Recall@K | Precision@K | AP | Найдено / ушли | Комментарий |", "|---|---|---:|---:|---:|---|---|"]
    for approach in ["baseline"] + [r["experiment_id"] for r in log_rows]:
        for part in ["train", "validation"]:
            m = metrics[(metrics.approach == approach) & (metrics.part == part)].iloc[0]
            comment = f"pooled Recall@K {_pct(m.pooled_recall_at_K)}" if part == "train" else (
                "выбранный кандидат" if approach == selected["experiment_id"] else "")
            lines.append(f"| {approach} | {part} | {_pct(m.macro_recall_at_K)} | {_pct(m.macro_precision_at_K)} | "
                         f"{m.macro_average_precision:.4f} | {m.hits} / {m.n_churn} | {comment} |".replace("0.", "0,"))
    extra = selected["valid_hits"] - baseline_valid["hits"]
    lines += ["", "Таблица по каждой T: `output/metrics_by_T.csv`. Устойчивость по временным фолдам: `output/stability_summary.csv`.", "",
              f"Ошибки и предупреждения: {warnings_text or 'нет: все опыты сошлись без предупреждений.'}", "",
              f"Выбранный кандидат: **{selected['experiment_id']}**. Дополнительных уходов против baseline на validation: **{extra:+d}** "
              f"({selected['valid_hits']} против {baseline_valid['hits']} из {baseline_valid['n_churn']}).", "",
              "Сохранение: `output/model/candidate.joblib`, схема входа и версии — `output/model/model_card.json`. "
              "Загрузка проверена в новом процессе без fit: `python solution/check_saved_model.py`, результат — `output/reload_check.json`."]
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- запуск

def run(cfg: dict) -> dict:
    out = STAGE5_OUT
    (out / "model").mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    numeric, categorical = load_x_columns()
    selected_cols = numeric + categorical
    hashes_before = {f: sha256(STAGE4_OUT / f) for f in INPUT_FILES}
    tables = read_stage4()
    parts = load_parts(tables, numeric, categorical)
    train, valid = parts["train"], parts["validation"]
    fraction = plan["campaign_fraction"]
    assert fraction == cfg["campaign_fraction"], "Лимит кампании расходится с конфигурацией проекта"

    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=SOLUTION_DIR).stdout.strip() or "неизвестна"

    # Baseline на тех же данных
    results = {}
    b_tr = rank_and_measure(train["keys"], train["y"], baseline_score(train["X"], plan["baseline_feature"]), fraction)
    b_va = rank_and_measure(valid["keys"], valid["y"], baseline_score(valid["X"], plan["baseline_feature"]), fraction)
    results["baseline"] = {"train": b_tr, "validation": b_va}

    # Эксперименты: новый Pipeline на каждый опыт, fit только на train
    log_rows, runtime_rows, select_rows, pipelines, warn_lines = [], [], [], {}, []
    for order, exp in enumerate(plan["experiments"]):
        pipe = build_pipeline(exp, numeric, categorical, plan["seed"])
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seconds, warn = fit_with_log(pipe, train["X"], train["y"])
        pipelines[exp["id"]] = pipe
        results[exp["id"]] = {
            "train": rank_and_measure(train["keys"], train["y"], risk_score(pipe, train["X"]), fraction),
            "validation": rank_and_measure(valid["keys"], valid["y"], risk_score(pipe, valid["X"]), fraction),
        }
        va = results[exp["id"]]["validation"][0]
        select_rows.append(dict(experiment_id=exp["id"], order=order, valid_recall=va["macro_recall_at_K"],
                                valid_ap=va["macro_average_precision"], valid_hits=va["hits"]))
        log_rows.append(dict(experiment_id=exp["id"], method=METHOD_NAMES[exp["family"]], params=json.dumps(exp["params"], ensure_ascii=False),
                             goal=exp["goal"], seed=plan["seed"], x_version=plan["x_version"], fit_rows=len(train["y"]),
                             classes=str([int(c) for c in pipe.classes_]), status=f"предупреждение: {warn}" if warn else "ok"))
        runtime_rows.append(dict(experiment_id=exp["id"], started_at=started_at, fit_seconds=round(seconds, 3)))
        if warn:
            warn_lines.append(f"{exp['id']} — {warn}")

    selected = select_candidate(select_rows)
    sel_id = selected["experiment_id"]
    sel_exp = next(e for e in plan["experiments"] if e["id"] == sel_id)
    for row in log_rows:
        row["selected"] = "да" if row["experiment_id"] == sel_id else ""

    # Устойчивость по временным фолдам (в выборе не участвует)
    fold_rows = baseline_stability(plan["stability_folds"], parts, plan)
    for exp in plan["experiments"]:
        fold_rows += stability(exp, plan["stability_folds"], parts, numeric, categorical, plan)
    folds = pd.DataFrame(fold_rows)[["approach", "fold", "train_until_inclusive", "snapshot_at", "train_rows", "n_active", "K",
                                     "n_churn", "hits", "recall_at_K", "precision_at_K", "average_precision"]]
    write_csv(folds, out / "stability_folds.csv")
    fs = folds.groupby("approach", sort=False).agg(folds=("fold", "size"), hits=("hits", "sum"), n_churn=("n_churn", "sum"),
                                                   K=("K", "sum"), macro_recall_at_K=("recall_at_K", "mean"),
                                                   macro_average_precision=("average_precision", "mean")).reset_index()
    fs["pooled_recall_at_K"] = fs.hits / fs.n_churn
    base_hits = int(fs.loc[fs.approach == "baseline", "hits"].iloc[0])
    fs["extra_hits_vs_baseline"] = fs.hits - base_hits
    write_csv(fs, out / "stability_summary.csv")

    # Таблицы результатов
    metrics_rows, by_t = [], []
    for approach, res in results.items():
        for part, (summary, table, _) in res.items():
            metrics_rows.append(dict(approach=approach, part=part) | summary)
            by_t.append(table.assign(approach=approach, part=part))
    metrics = pd.DataFrame(metrics_rows)
    write_csv(metrics, METRICS_PATH)
    by_t_df = pd.concat(by_t, ignore_index=True)[["approach", "part", "snapshot_at", "n_active", "K", "n_churn", "hits", "recall_at_K",
                                                  "precision_at_K", "average_precision"]]
    write_csv(by_t_df, out / "metrics_by_T.csv")
    gap = metrics.pivot(index="approach", columns="part", values=["macro_recall_at_K", "macro_average_precision"])
    write_csv(pd.DataFrame({"approach": gap.index,
                            "recall_train": gap["macro_recall_at_K"]["train"].values, "recall_validation": gap["macro_recall_at_K"]["validation"].values,
                            "recall_gap": (gap["macro_recall_at_K"]["train"] - gap["macro_recall_at_K"]["validation"]).values,
                            "ap_train": gap["macro_average_precision"]["train"].values, "ap_validation": gap["macro_average_precision"]["validation"].values,
                            "ap_gap": (gap["macro_average_precision"]["train"] - gap["macro_average_precision"]["validation"]).values}),
              out / "train_validation_gap.csv")
    write_csv(pd.DataFrame(log_rows), out / "experiments.csv")
    runtime = pd.DataFrame(runtime_rows)
    write_csv(runtime, RUN_LOG_PATH)

    # Коэффициенты лучшей логистической регрессии (масштабированные признаки, обучение на train)
    lr_ids = {e["id"] for e in plan["experiments"] if e["family"] == "logistic_regression"}
    best_lr = select_candidate([r for r in select_rows if r["experiment_id"] in lr_ids])["experiment_id"]
    lr_pipe = pipelines[best_lr]
    coef = pd.DataFrame({"experiment_id": best_lr, "x_column": lr_pipe[:-1].get_feature_names_out(), "coefficient": lr_pipe[-1].coef_[0]})
    coef = coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)
    write_csv(coef, out / "lr_coefficients.csv")

    # Диагностика вывода: сила сигнала долга по датам и расхождение с baseline на validation
    write_csv(debt_signal_by_t(parts, plan["baseline_feature"]), out / "debt_signal_by_T.csv")
    overlap, overlap_info = compare_with_baseline(results[sel_id]["validation"][2], results["baseline"]["validation"][2])
    write_csv(overlap, out / "validation_overlap_with_baseline.csv")

    # Прогнозы validation и сохранение кандидата
    write_predictions(results[sel_id]["validation"][2], PREDICTIONS_PATH)
    write_predictions(results["baseline"]["validation"][2], out / "validation_baseline_ranking.csv")
    write_csv(pd.DataFrame({"position": range(len(selected_cols)), "column": selected_cols,
                            "type": ["numeric"] * len(numeric) + ["categorical"] * len(categorical)}), out / "model" / "input_schema.csv")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    card = dict(selected_experiment=sel_id, family=sel_exp["family"], params=sel_exp["params"], seed=plan["seed"], x_version=plan["x_version"],
                numeric_columns=numeric, categorical_columns=categorical, fit_on="train: T = 01.12.2025–01.07.2026", fit_rows=len(train["y"]),
                not_refit_on_validation=True, test_used=False, selection_rule=plan["selection_rule"], versions=versions,
                data_sha256=hashes_before, metrics={part: results[sel_id][part][0] for part in ["train", "validation"]},
                baseline_metrics={part: results["baseline"][part][0] for part in ["train", "validation"]})
    write_json(card, out / "model" / "model_card.json")
    joblib.dump(dict(pipeline=pipelines[sel_id], numeric_columns=numeric, categorical_columns=categorical, selected_experiment=sel_id,
                     data_sha256=hashes_before, versions=versions,
                     config=dict(campaign_fraction=fraction, baseline_feature=plan["baseline_feature"], seed=plan["seed"], x_version=plan["x_version"])),
                MODEL_PATH, compress=3)

    # Проверка загрузки в новом процессе
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(SOLUTION_DIR / "check_saved_model.py")], capture_output=True, encoding="utf-8", env=env)
    reload_ok = proc.returncode == 0
    reload_info = json.loads(RELOAD_PATH.read_text(encoding="utf-8")) if RELOAD_PATH.exists() else {}

    # Проверки этапа
    checks = []
    checks.append(check_row("ключи features/labels/split уникальны и без пустых", 0,
                            sum(int(t.duplicated(KEYS).sum()) for t in tables.values()), all(not t.duplicated(KEYS).any() for t in tables.values())))
    same = len({frozenset(map(tuple, t[KEYS].values)) for t in tables.values()}) == 1
    checks.append(check_row("множества ключей трёх таблиц совпадают", True, same, same))
    sp = tables["split"].split.value_counts()
    ok_rows = len(train["y"]) == sp["train"] and len(valid["y"]) == sp["validation"]
    checks.append(check_row("строки не потеряны после объединения по ключу", (int(sp["train"]), int(sp["validation"])), (len(train["y"]), len(valid["y"])), ok_rows))
    events = read_prepared(STAGE2_OUT / "contract_events_clean.csv")
    dates = sorted(pd.concat([train["keys"], valid["keys"]]).snapshot_at.unique())
    indep = {d: independent_active_count(events, pd.Timestamp(d)) for d in dates}
    got = pd.concat([train["keys"], valid["keys"]]).groupby("snapshot_at").size().to_dict()
    checks.append(check_row("в каждом снимке сохранена активная база (независимый расчёт по журналу)", indep, got, indep == {k: int(v) for k, v in got.items()}))
    labels_ok = set(np.unique(train["y"])) == {0, 1} and set(np.unique(valid["y"])) <= {0, 1}
    checks.append(check_row("метки train/validation известны и бинарны, в train оба класса", True, labels_ok, labels_ok))
    catalog = read_raw_csv(STAGE4_OUT / "feature_catalog.csv")
    in_x = set(catalog[catalog.include_in_X == "да"].feature_name)
    checks.append(check_row("список X совпадает с паспортом, ключи и метка в него не входят", len(in_x), len(selected_cols),
                            in_x == set(selected_cols) and not (set(selected_cols) & FORBIDDEN)))
    train_end = pd.to_datetime(train["target_end"]).max()
    valid_start = pd.to_datetime(valid["keys"].snapshot_at).min()
    checks.append(check_row("горизонты train заканчиваются не позже начала validation", str(valid_start.date()), str(train_end.date()), train_end <= valid_start))
    loaded = set(dates)
    test_dates = set(tables["split"][tables["split"].split == "test"].snapshot_at)
    checks.append(check_row("test не загружен: дат test нет в обучении и оценке", sorted(test_dates), f"пересечение: {sorted(loaded & test_dates)}", not (loaded & test_dates)))
    spoiled = {k: v.copy() for k, v in tables.items()}
    test_mask = spoiled["split"].split == "test"
    test_keys = set(map(tuple, spoiled["split"].loc[test_mask, KEYS].values))
    for name, col, value in [("features", plan["baseline_feature"], "99999"), ("labels", "churn_target", "1")]:
        frame = spoiled[name]
        frame.loc[[k in test_keys for k in map(tuple, frame[KEYS].values)], col] = value
    parts2 = load_parts(spoiled, numeric, categorical)
    same_parts = all(parts2[p]["X"].equals(parts[p]["X"]) and np.array_equal(parts2[p]["y"], parts[p]["y"]) for p in parts)
    checks.append(check_row("изменение строк и меток test не меняет данные обучения и validation", True, same_parts, same_parts))
    fit_rows_ok = all(r["fit_rows"] == len(train["y"]) for r in log_rows)
    checks.append(check_row("все опыты обучены только на строках train", len(train["y"]), sorted({r["fit_rows"] for r in log_rows}), fit_rows_ok))
    medians = train["X"][numeric].median()
    lr_imp = pipelines[best_lr].named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    med_ok = bool(np.allclose(lr_imp.statistics_, medians.to_numpy()))
    checks.append(check_row("медианы подготовки логистической регрессии = медианы train", True, med_ok, med_ok))
    classes_ok = all(list(p.classes_) == [0, 1] for p in pipelines.values())
    checks.append(check_row("порядок классов [0, 1]: score берётся из столбца класса ухода", True, classes_ok, classes_ok))
    base_t = by_t_df[by_t_df.approach == "baseline"][["part", "snapshot_at", "n_active", "K"]].reset_index(drop=True)
    same_k = all(by_t_df[by_t_df.approach == r["experiment_id"]][["part", "snapshot_at", "n_active", "K"]].reset_index(drop=True).equals(base_t) for r in log_rows)
    checks.append(check_row("база и K одинаковы для baseline и всех опытов на каждой T", True, same_k, same_k))
    st4 = read_raw_csv(STAGE4_OUT / "baseline_train_metrics.csv").iloc[:8]
    tr_t = by_t_df[(by_t_df.approach == "baseline") & (by_t_df.part == "train")].reset_index(drop=True)
    st4_ok = (st4.K.astype(int).values == tr_t.K.values).all() and (st4.tp_at_k.astype(int).values == tr_t.hits.values).all()
    checks.append(check_row("baseline на train совпадает с таблицей этапа 4 (K и найденные по датам)", int(st4.tp_at_k.astype(int).sum()), int(tr_t.hits.sum()), bool(st4_ok)))
    n_exp = len(plan["experiments"])
    families = {e["family"] for e in plan["experiments"]}
    checks.append(check_row("в плане 3–6 конфигураций и два метода", "3–6, 2", f"{n_exp}, {len(families)}", 3 <= n_exp <= 6 and len(families) == 2))
    first = select_rows[0]
    picked = select_candidate([dict(first, valid_ap=first["valid_ap"] - 1, order=0), dict(first, experiment_id="other", order=1)])
    checks.append(check_row("synthetic: при равном Recall@K выбирается больший AP", "other", picked["experiment_id"], picked["experiment_id"] == "other"))
    checks += synthetic_metric_checks()
    again = build_pipeline(sel_exp, numeric, categorical, plan["seed"])
    fit_with_log(again, train["X"], train["y"])
    rep_ok = bool(np.array_equal(risk_score(again, valid["X"]), risk_score(pipelines[sel_id], valid["X"])))
    checks.append(check_row("повторное обучение кандидата с тем же seed даёт те же оценки", True, rep_ok, rep_ok))
    checks.append(check_row("загрузка в новом процессе без fit: ключи, score, rank, top-K и метрики совпали", True, reload_info.get("passed"), reload_ok and bool(reload_info.get("passed")),
                            proc.stderr.strip()[-300:] if not reload_ok else f"кандидат {reload_info.get('selected_experiment')}, {reload_info.get('rows')} строк"))
    hashes_after = {f: sha256(STAGE4_OUT / f) for f in INPUT_FILES}
    checks.append(check_row("таблицы этапа 4 не изменены (sha256 до/после)", "равны", "равны" if hashes_before == hashes_after else "различаются", hashes_before == hashes_after))
    ch = pd.DataFrame(checks)
    write_csv(ch, out / "checks_stage5.csv")

    write_journal(plan, hashes_before["features.csv"], git_rev, log_rows, metrics, selected, b_va[0], "; ".join(warn_lines), runtime)

    summary = dict(selected_experiment=sel_id, selected_family=sel_exp["family"], train_rows=len(train["y"]), validation_rows=len(valid["y"]),
                   baseline_validation=b_va[0], candidate_validation=results[sel_id]["validation"][0],
                   candidate_train=results[sel_id]["train"][0], extra_hits_vs_baseline_validation=selected["valid_hits"] - b_va[0]["hits"],
                   validation_overlap_with_baseline=overlap_info,
                   stability_hits_candidate=int(fs.loc[fs.approach == sel_id, "hits"].iloc[0]), stability_hits_baseline=base_hits,
                   stability_n_churn=int(fs.loc[fs.approach == "baseline", "n_churn"].iloc[0]),
                   checks_failed=int((ch.passed == "FAIL").sum()))
    write_json(summary, out / "training_summary.json")
    return summary
