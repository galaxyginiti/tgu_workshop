"""Итоговый расчёт на test в отдельном процессе, без обучения: сохранённая модель и baseline при одном K.

    python solution/run_all.py --stage 6       # полный этап: расчёт, повтор, разбор, проверки
    python solution/evaluate_test.py --out DIR # только списки контактов и метрики в указанную папку

Любой fit в этом процессе — ошибка.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from churn.common import STAGE6_OUT, load_config  # noqa: E402
from churn.stage6_evaluate import evaluate_saved_model  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=STAGE6_OUT, help="папка для списков контактов и метрик")
    args = parser.parse_args()
    result = evaluate_saved_model(args.out, load_config()["observation_end_exclusive"])
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
