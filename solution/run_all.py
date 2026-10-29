"""Единый запуск проекта.

    python solution/run_all.py            # этапы 2, 3, 4 подряд
    python solution/run_all.py --stage 2  # только один этап

Вход: исходные файлы репозитория (1_introduction, 2_preprocessing, 3_eda, 4_features),
они только читаются. Выход: solution/stage*/output. В конце пишется
solution/output_manifest.csv с sha256 всех результатов — по нему проверяется,
что повторный запуск даёт те же файлы.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from churn.common import SOLUTION_DIR, load_config, sha256, write_csv  # noqa: E402

MODULES = {2: "stage2_prepare", 3: "stage3_eda", 4: "stage4_features"}
STAGES = {n: m for n, m in MODULES.items() if (SOLUTION_DIR / "src" / "churn" / f"{m}.py").exists()}


def write_manifest() -> Path:
    rows = []
    for path in sorted(SOLUTION_DIR.glob("stage*/output/**/*")):
        if path.is_file():
            rows.append({"file": path.relative_to(SOLUTION_DIR).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return write_csv(pd.DataFrame(rows), SOLUTION_DIR / "output_manifest.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), action="append",
                        help="запустить только указанные этапы (можно несколько раз)")
    parser.add_argument("--config", type=Path, default=None, help="путь к project.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    failed = 0
    for n in args.stage or sorted(STAGES):
        t0 = time.time()
        module = importlib.import_module(f"churn.{STAGES[n]}")
        print(f"== {module.__doc__.splitlines()[0]}")
        summary = module.run(cfg)
        failed += summary.get("checks_failed", 0)
        print(f"   готово за {time.time() - t0:.1f} с; непройденных проверок: {summary.get('checks_failed', 0)}")
    print("Манифест результатов:", write_manifest().relative_to(SOLUTION_DIR.parent))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
