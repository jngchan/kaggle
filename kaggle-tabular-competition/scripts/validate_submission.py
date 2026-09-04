#!/usr/bin/env python3
"""Validate a Kaggle submission against sample_submission.csv using the stdlib."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path, help="Path to sample_submission.csv")
    parser.add_argument("submission", type=Path, help="Path to candidate submission.csv")
    parser.add_argument("--id-column", help="ID column; defaults to the first column")
    parser.add_argument("--min", dest="minimum", type=float, help="Minimum numeric prediction")
    parser.add_argument("--max", dest="maximum", type=float, help="Maximum numeric prediction")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Allowed non-numeric labels for every prediction column",
    )
    parser.add_argument(
        "--row-sum",
        type=float,
        help="Required sum across numeric prediction columns, e.g. 1 for multiclass probabilities",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for --row-sum (default: 1e-6)",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def main() -> int:
    args = parse_args()
    sample_header, sample_rows = read_csv(args.sample)
    candidate_header, candidate_rows = read_csv(args.submission)
    errors: list[str] = []

    if candidate_header != sample_header:
        errors.append(f"header mismatch: expected {sample_header}, got {candidate_header}")
    if len(candidate_rows) != len(sample_rows):
        errors.append(f"row-count mismatch: expected {len(sample_rows)}, got {len(candidate_rows)}")

    id_column = args.id_column or sample_header[0]
    if id_column not in sample_header or id_column not in candidate_header:
        errors.append(f"ID column {id_column!r} is missing")
    else:
        sample_ids = [row.get(id_column, "") for row in sample_rows]
        candidate_ids = [row.get(id_column, "") for row in candidate_rows]
        if any(value == "" for value in candidate_ids):
            errors.append("blank ID found")
        if len(set(candidate_ids)) != len(candidate_ids):
            errors.append("duplicate ID found")
        if candidate_ids != sample_ids:
            errors.append("candidate IDs or their order do not match the sample submission")

    prediction_columns = [column for column in candidate_header if column != id_column]
    allowed_labels = set(args.labels) if args.labels else None
    for row_number, row in enumerate(candidate_rows, start=2):
        numeric_values: list[float] = []
        for column in prediction_columns:
            value = row.get(column, "").strip()
            if not value:
                errors.append(f"row {row_number}, {column}: blank prediction")
                continue
            if allowed_labels is not None:
                if value not in allowed_labels:
                    errors.append(
                        f"row {row_number}, {column}: label {value!r} is not allowed"
                    )
                continue
            try:
                number = float(value)
            except ValueError:
                errors.append(f"row {row_number}, {column}: expected numeric value, got {value!r}")
                continue
            if not math.isfinite(number):
                errors.append(f"row {row_number}, {column}: non-finite value")
                continue
            numeric_values.append(number)
            if args.minimum is not None and number < args.minimum:
                errors.append(f"row {row_number}, {column}: {number} < {args.minimum}")
            elif args.maximum is not None and number > args.maximum:
                errors.append(f"row {row_number}, {column}: {number} > {args.maximum}")
        if args.row_sum is not None and len(numeric_values) == len(prediction_columns):
            if not math.isclose(
                sum(numeric_values), args.row_sum, rel_tol=0.0, abs_tol=args.atol
            ):
                errors.append(
                    f"row {row_number}: prediction sum {sum(numeric_values)} "
                    f"!= {args.row_sum} within atol={args.atol}"
                )

    if errors:
        print("FAILED")
        for message in errors[:50]:
            print(f"- {message}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} additional error(s)")
        return 1

    print(
        f"PASS: {len(candidate_rows)} rows, {len(candidate_header)} columns, "
        f"ID column {id_column!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
