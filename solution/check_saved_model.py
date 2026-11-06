"""Проверка сохранённого кандидата этапа 5 в отдельном процессе, без обучения.

    python solution/check_saved_model.py

Загружает solution/stage5_models/output/model/candidate.joblib, считает оценки на validation
и сверяет ключи, score, rank, top-K и метрики с сохранёнными. Любой fit в этом процессе — ошибка.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from churn.stage5_train import reload_check  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = reload_check()
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
