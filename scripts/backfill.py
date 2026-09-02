"""历史回填：为指定日期区间逐工作日生成历史报告，建立存档与回测样本。

用法::
    python scripts/backfill.py 2026-08-01 2026-08-31            # strict 真实数据
    python scripts/backfill.py 2026-08-01 2026-08-31 --demo     # 演示合成数据
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oilcast.pipeline.main import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="YYYY-MM-DD")
    ap.add_argument("end", help="YYYY-MM-DD")
    ap.add_argument("--demo", "--offline", dest="demo", action="store_true",
                    help="演示模式（全部合成数据）")
    args = ap.parse_args()
    for d in pd.bdate_range(args.start, args.end):
        print(f"==== backfill {d.date()} ====")
        run(as_of=d.tz_localize("Asia/Shanghai"), demo=args.demo)


if __name__ == "__main__":
    main()
