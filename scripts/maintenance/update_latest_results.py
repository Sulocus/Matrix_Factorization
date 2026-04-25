#!/usr/bin/env python3
"""Rebuild the lightweight results/latest display directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matrix_factorization.modules.outputs.latest import (  # noqa: E402
    DEFAULT_LATEST_COUNT,
    refresh_latest_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="results/alpha_scan")
    parser.add_argument("--latest-dir", default=None)
    parser.add_argument("--count", type=int, default=DEFAULT_LATEST_COUNT)
    args = parser.parse_args()

    summaries = refresh_latest_results(
        source_dir=Path(args.source_dir),
        latest_dir=Path(args.latest_dir) if args.latest_dir else None,
        count=args.count,
    )
    for summary in summaries:
        print(summary["run_id"])


if __name__ == "__main__":
    main()
