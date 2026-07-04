from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphsum.aggregate import aggregate_result_files
from graphsum.evaluate import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate graph summarization result CSVs.")
    parser.add_argument("inputs", nargs="+", help="Input result CSV files.")
    parser.add_argument("--output", default="runs/summary_results.csv")
    args = parser.parse_args()

    output_rows = aggregate_result_files([Path(input_path) for input_path in args.inputs])

    write_csv(Path(args.output), output_rows)
    for row in output_rows:
        print(row)


if __name__ == "__main__":
    main()
