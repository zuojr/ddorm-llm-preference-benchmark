
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics JSON files over seeds.")
    parser.add_argument("--glob", required=True, help='Example: "outputs/*/metrics/*.json"')
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in Path().glob(args.glob):
        data = json.loads(Path(path).read_text())
        data["path"] = str(path)
        rows.append(data)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No metrics files matched the pattern.")
    numeric_cols = [c for c in df.columns if df[c].dtype != object]
    summary = df[numeric_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(args.output_csv, index=False)
    print(summary)


if __name__ == "__main__":
    main()
