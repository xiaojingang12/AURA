#!/usr/bin/env python3
"""Aggregate Overall Winner counts from eval_results_h2h_eval200."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

SAMPLE_ID_PATTERN = re.compile(r"(?:^|_)(S_(\d+))(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan head-to-head evaluation result folders, count Overall Winner "
            "for each method, and write a summary JSON file."
        )
    )
    parser.add_argument(
        "--root-dir",
        default="eval_results_h2h_eval200",
        help="Root directory that contains comparison subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="eval_results_h2h_eval200/overall_winner_summary.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    parser.add_argument(
        "--write-participate-comparison",
        action="store_true",
        help=(
            "Write Participate_comparison.json in each comparison directory with "
            "the filtered CSV filenames that participate in aggregation."
        ),
    )
    parser.add_argument(
        "--exclude-diff-json",
        default=None,
        help="Optional diff JSON whose missing_in_left/extra_in_left sample IDs are excluded from aggregation.",
    )
    parser.add_argument(
        "--participate-filename",
        default="Participate_comparison.json",
        help="Filename used for per-directory participate comparison JSON output.",
    )
    return parser.parse_args()


def percentage(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def iter_result_csvs(comparison_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in comparison_dir.glob("*.csv")
        if not path.name.endswith("_metrics.csv")
    )


def normalize_sample_id(raw_sample_id: str | None) -> str | None:
    if not raw_sample_id:
        return None
    match = SAMPLE_ID_PATTERN.search(raw_sample_id.strip())
    if not match:
        return None
    return f"S_{int(match.group(2))}"


def extract_sample_id_from_path(csv_path: Path) -> str | None:
    return normalize_sample_id(csv_path.stem)


def load_excluded_sample_ids(diff_json_path: Path | None) -> set[str]:
    if diff_json_path is None:
        return set()

    data = json.loads(diff_json_path.read_text(encoding="utf-8"))
    excluded_sample_ids: set[str] = set()
    for key in ("missing_in_left", "extra_in_left"):
        for raw_sample_id in data.get(key, []):
            sample_id = normalize_sample_id(raw_sample_id)
            if sample_id is not None:
                excluded_sample_ids.add(sample_id)
    return excluded_sample_ids


def iter_participating_csvs(
    comparison_dir: Path, excluded_sample_ids: set[str]
) -> list[Path]:
    return [
        csv_file
        for csv_file in iter_result_csvs(comparison_dir)
        if extract_sample_id_from_path(csv_file) not in excluded_sample_ids
    ]


def parse_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def collect_resource_usage(csv_files: list[Path]) -> dict[str, int | float]:
    totals = {
        "total_sample_time_seconds": 0.0,
        "total_api_time_seconds": 0.0,
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                totals["total_sample_time_seconds"] += parse_float(
                    row.get("sample_time_seconds")
                )
                totals["total_api_time_seconds"] += parse_float(
                    row.get("api_time_seconds")
                )
                totals["total_tokens"] += parse_int(row.get("total_tokens"))
                totals["total_prompt_tokens"] += parse_int(row.get("prompt_tokens"))
                totals["total_completion_tokens"] += parse_int(
                    row.get("completion_tokens")
                )

    totals["total_sample_time_seconds"] = round(
        totals["total_sample_time_seconds"], 4
    )
    totals["total_api_time_seconds"] = round(totals["total_api_time_seconds"], 4)
    return totals


def normalize_winner(raw_winner: str | None, method1: str, method2: str) -> str | None:
    winner = (raw_winner or "").strip()
    if winner == "Answer 1":
        return method1
    if winner == "Answer 2":
        return method2
    if winner in {method1, method2}:
        return winner
    if winner in {"", "N/A"}:
        return None
    return winner


def aggregate_comparison(comparison_dir: Path, excluded_sample_ids: set[str]) -> dict:
    method1, method2 = comparison_dir.name.split("_vs_", 1)
    counts = Counter({method1: 0, method2: 0})
    unexpected = Counter()
    total_rows = 0
    csv_files = iter_participating_csvs(comparison_dir, excluded_sample_ids)

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "Overall Winner" not in reader.fieldnames:
                unexpected[f"missing Overall Winner column in {csv_file.name}"] += 1
                continue

            for row in reader:
                total_rows += 1
                winner = normalize_winner(row.get("Overall Winner"), method1, method2)
                if winner in counts:
                    counts[winner] += 1
                else:
                    unexpected[winner or "UNKNOWN"] += 1

    method1_rate = percentage(counts[method1], total_rows)
    method2_rate = percentage(counts[method2], total_rows)

    if counts[method1] > counts[method2]:
        overall_winner = method1
        overall_winner_rate = method1_rate
    elif counts[method2] > counts[method1]:
        overall_winner = method2
        overall_winner_rate = method2_rate
    else:
        overall_winner = "Tie"
        overall_winner_rate = method1_rate

    result = {
        method1: counts[method1],
        f"{method1}_win_rate": method1_rate,
        method2: counts[method2],
        f"{method2}_win_rate": method2_rate,
        "winner": overall_winner,
        "winner_rate": overall_winner_rate,
        "total_samples": total_rows,
        "result_csv_files": len(csv_files),
    }

    if unexpected:
        result["other_outcomes"] = dict(unexpected)

    return result


def write_participate_comparison(
    comparison_dir: Path,
    csv_files: list[Path],
    output_filename: str,
    indent: int,
) -> None:
    output_path = comparison_dir / output_filename
    payload = {
        "comparison_dir": comparison_dir.name,
        "total_files": len(csv_files),
        **collect_resource_usage(csv_files),
        "files": [csv_file.name for csv_file in csv_files],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )


def aggregate_root(
    root_dir: Path,
    excluded_sample_ids: set[str],
    *,
    write_participate_comparison_files: bool,
    participate_filename: str,
    indent: int,
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for comparison_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        if "_vs_" not in comparison_dir.name:
            continue
        csv_files = iter_participating_csvs(comparison_dir, excluded_sample_ids)
        if write_participate_comparison_files:
            write_participate_comparison(
                comparison_dir,
                csv_files,
                participate_filename,
                indent,
            )
        summary[comparison_dir.name] = aggregate_comparison(
            comparison_dir, excluded_sample_ids
        )
    return summary


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_path = Path(args.output)
    exclude_diff_path = Path(args.exclude_diff_json) if args.exclude_diff_json else None

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if exclude_diff_path is not None and not exclude_diff_path.exists():
        raise FileNotFoundError(
            f"Exclude diff JSON does not exist: {exclude_diff_path}"
        )

    excluded_sample_ids = load_excluded_sample_ids(exclude_diff_path)
    summary = aggregate_root(
        root_dir,
        excluded_sample_ids,
        write_participate_comparison_files=args.write_participate_comparison,
        participate_filename=args.participate_filename,
        indent=args.indent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=args.indent) + "\n",
        encoding="utf-8",
    )

    print(f"Processed {len(summary)} comparison directories.")
    print(f"Excluded {len(excluded_sample_ids)} sample IDs.")
    if args.write_participate_comparison:
        print(f"Per-directory file lists written as: {args.participate_filename}")
    print(f"Summary written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
