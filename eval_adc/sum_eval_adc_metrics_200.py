import argparse
import json
import os
from pathlib import Path


DEFAULT_FIELDS = [
    "total_tokens",
    "sample_time_seconds_sum",
]


def load_selected_rel_dirs(list_json_path: Path) -> set[str]:
    with list_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = payload.get("selected", [])
    rel_dirs = set()
    for item in selected:
        rel_dir = item.get("rel_dir")
        if isinstance(rel_dir, str) and rel_dir.strip():
            rel_dirs.add(rel_dir.strip())
    return rel_dirs


def collect_eval_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file()
    )


def extract_method_and_rel_dir(response_path: str) -> tuple[str | None, str | None]:
    path_obj = Path(response_path)
    parts = path_obj.parts

    if path_obj.name != "results.json":
        return None, None

    try:
        collected_idx = parts.index("Collected")
    except ValueError:
        return None, None

    if len(parts) <= collected_idx + 5:
        return None, None

    method = parts[collected_idx + 1]
    rel_parts = parts[collected_idx + 2:-1]
    if len(rel_parts) < 3:
        return None, None

    return method, "/".join(rel_parts)


def parse_eval_file(file_path: Path) -> dict | None:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    summary = payload.get("summary", {})
    if not isinstance(config, dict) or not isinstance(summary, dict):
        return None

    response_path = config.get("response_path")
    if not isinstance(response_path, str) or not response_path.strip():
        return None

    method, rel_dir = extract_method_and_rel_dir(response_path)
    if method is None or rel_dir is None:
        return None

    return {
        "file_path": str(file_path),
        "method": method,
        "rel_dir": rel_dir,
        "summary": summary,
    }


def sum_eval_metrics(
    eval_records: list[dict],
    selected_rel_dirs: set[str],
    fields: list[str],
    method: str | None = None,
) -> tuple[dict, dict]:
    totals = {field: 0 for field in fields}
    matched_files = []
    matched_rel_dirs = set()
    skipped_files = []
    duplicate_rel_dirs: dict[str, list[str]] = {}

    seen_rel_dir_to_file: dict[str, str] = {}

    for record in eval_records:
        if method and record["method"] != method:
            continue

        rel_dir = record["rel_dir"]
        if rel_dir not in selected_rel_dirs:
            continue

        file_path = record["file_path"]
        summary = record["summary"]

        invalid_field = False
        for field in fields:
            value = summary.get(field, 0)
            if not isinstance(value, (int, float)):
                skipped_files.append(file_path)
                invalid_field = True
                break

        if invalid_field:
            continue

        if rel_dir in seen_rel_dir_to_file:
            duplicate_rel_dirs.setdefault(rel_dir, [seen_rel_dir_to_file[rel_dir]]).append(file_path)
            continue

        seen_rel_dir_to_file[rel_dir] = file_path
        matched_files.append(file_path)
        matched_rel_dirs.add(rel_dir)

        for field in fields:
            totals[field] += summary.get(field, 0)

    missing_rel_dirs = sorted(selected_rel_dirs - matched_rel_dirs)

    metadata = {
        "matched_files": matched_files,
    }
    return totals, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum eval_adc metrics for the 200-sample rel_dir list used by batch_h2h_eval_200.sh."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing eval_adc JSON outputs.",
    )
    parser.add_argument(
        "--list-json",
        type=Path,
        default=Path(os.environ["AURA_EVAL_200_LIST_JSON"]) if os.getenv("AURA_EVAL_200_LIST_JSON") else None,
        help="Path to the 200-sample list JSON. Defaults to AURA_EVAL_200_LIST_JSON.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Only include files whose response_path belongs to this method under Collected/<METHOD>/...",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Metric fields to sum. Defaults to total_tokens prompt_tokens sample_time_seconds_sum api_time_seconds_sum",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.list_json is None:
        raise SystemExit("Missing list JSON. Use --list-json or set AURA_EVAL_200_LIST_JSON.")

    list_json_path = args.list_json.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not list_json_path.is_file():
        raise SystemExit(f"List JSON does not exist: {list_json_path}")

    selected_rel_dirs = load_selected_rel_dirs(list_json_path)
    eval_files = collect_eval_files(input_dir)

    parsed_records = []
    ignored_files = []
    for file_path in eval_files:
        try:
            record = parse_eval_file(file_path)
        except Exception:
            ignored_files.append(str(file_path))
            continue

        if record is None:
            ignored_files.append(str(file_path))
            continue
        parsed_records.append(record)

    totals, metadata = sum_eval_metrics(
        eval_records=parsed_records,
        selected_rel_dirs=selected_rel_dirs,
        fields=args.fields,
        method=args.method,
    )

    result = {
        "input_dir": str(input_dir),
        "list_json": str(list_json_path),
        "selected_rel_dir_count": len(selected_rel_dirs),
        "method_filter": args.method,
        "parsed_record_count": len(parsed_records),
        "ignored_file_count": len(ignored_files),
        "ignored_files": ignored_files,
        "fields": args.fields,
        "totals": totals,
        **metadata,
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
