#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTED_ROOT = Path(os.getenv("AURA_COLLECTED_ROOT", REPO_ROOT / "Collected"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("AURA_EVAL_OUTPUT_ROOT", REPO_ROOT / "eval_aura" / "batch_results"))
EVAL_SCRIPT = REPO_ROOT / "eval_aura" / "eval_adc.py"
VALID_DATASETS = ["Single-Sum", "Pair-Comp", "Multi-Comp", "Enumeration", "Temporal"]
VALID_DIFFICULTIES = {
    "simple": "simple_QA",
    "middle": "middle_QA",
    "hard": "hard_QA",
}



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval_aura/eval_aura.py over collected method outputs for selected datasets and difficulties."
    )
    parser.add_argument("method", help="Method name, for example RAPTOR, HippoRAG, or LGraphRAG.")
    parser.add_argument(
        "arg2",
        help="Difficulty when two positional arguments are provided; dataset when three are provided.",
    )
    parser.add_argument(
        "arg3",
        nargs="?",
        default=None,
        help="Optional difficulty when arg2 is a dataset name.",
    )
    parser.add_argument(
        "--collected-root",
        default=str(COLLECTED_ROOT),
        help="Root directory containing collected method outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for batch evaluation outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run eval_aura/eval_adc.py.",
    )
    parser.add_argument(
        "--method-subdir",
        default="",
        help="Optional method subdirectory such as is_data.",
    )
    return parser.parse_args()



def normalize_difficulty(value):
    if value not in VALID_DIFFICULTIES:
        valid_values = ", ".join(sorted(VALID_DIFFICULTIES.keys()))
        raise ValueError(f"Invalid difficulty: {value}. Expected one of: {valid_values}")
    return VALID_DIFFICULTIES[value]



def resolve_targets(method, arg2, arg3):
    if arg3 is None:
        dataset_names = VALID_DATASETS
        difficulty = normalize_difficulty(arg2)
    else:
        if arg2 not in VALID_DATASETS:
            valid_datasets = ", ".join(VALID_DATASETS)
            raise ValueError(f"Invalid dataset: {arg2}. Expected one of: {valid_datasets}")
        dataset_names = [arg2]
        difficulty = normalize_difficulty(arg3)
    return dataset_names, difficulty



def derive_source_name_from_result_path(response_path):
    response_path_obj = Path(response_path)
    if response_path_obj.is_dir():
        anchor_path = response_path_obj
    elif response_path_obj.parent.name == "output" and response_path_obj.name.startswith("answer-"):
        anchor_path = response_path_obj.parent.parent
    else:
        anchor_path = response_path_obj.parent

    path_parts = [part for part in anchor_path.parts if part not in {"", "/"}]
    dataset_index = next((index for index, part in enumerate(path_parts) if part in VALID_DATASETS), None)

    if dataset_index is not None:
        source_parts = path_parts[dataset_index : dataset_index + 3]
        if dataset_index > 0 and path_parts[dataset_index - 1] in {"is_data", "result"}:
            source_parts = [path_parts[dataset_index - 1], *source_parts]
    else:
        source_parts = path_parts[-4:]

    if not source_parts:
        source_parts = [anchor_path.name or response_path_obj.stem or "unknown_source"]
    return "_".join(source_parts)


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]


def resolve_result_path(question_dir):
    rerun_path = question_dir / "results.rerun.json"
    result_path = question_dir / "results.json"
    result1_path = question_dir / "Result.json"
    hirag_candidates = sorted(question_dir.glob("hirag_result_q*.json"), key=natural_sort_key)
    answer_candidates = sorted((question_dir / "output").glob("answer-*.json"))
    if rerun_path.is_file():
        return rerun_path
    if result_path.is_file():
        return result_path
    if result1_path.is_file():
        return result1_path
    if hirag_candidates:
        return question_dir if len(hirag_candidates) > 1 else hirag_candidates[0]
    if answer_candidates:
        return question_dir if len(answer_candidates) > 1 else answer_candidates[0]
    return None



def build_search_roots(collected_root, method, dataset_name, difficulty, method_subdir):
    method_root = Path(collected_root) / method
    normalized_subdir = method_subdir.strip().strip("/")
    if normalized_subdir:
        return [method_root / normalized_subdir / dataset_name / difficulty]
    return [
        method_root / dataset_name / difficulty,
        method_root / "result" / dataset_name / difficulty,
        method_root / "is_data" / dataset_name / difficulty,
    ]


def collect_response_paths(collected_root, method, dataset_names, difficulty, method_subdir=""):
    response_paths = []
    missing_tasks = []
    for dataset_name in dataset_names:
        search_roots = build_search_roots(collected_root, method, dataset_name, difficulty, method_subdir)
        search_root = next((path for path in search_roots if path.is_dir()), None)
        if search_root is None:
            searched_locations = ", ".join(str(path) for path in search_roots)
            print(f"[WARN] skip missing directory. searched: {searched_locations}")
            continue
        question_dirs = sorted(path for path in search_root.iterdir() if path.is_dir())
        if not question_dirs:
            print(f"[WARN] no question directories found under: {search_root}")
            continue
        for question_dir in question_dirs:
            resolved_result = resolve_result_path(question_dir)
            if resolved_result is None:
                missing_tasks.append(
                    {
                        "method": method,
                        "dataset": dataset_name,
                        "difficulty": difficulty,
                        "question_id": question_dir.name,
                        "source_name": derive_source_name_from_result_path(question_dir),
                        "question_dir": str(question_dir),
                    }
                )
                continue
            response_paths.append(resolved_result)
    return response_paths, missing_tasks



def write_missing_tasks(output_root, method, missing_tasks):
    if not missing_tasks:
        return None
    err_dir = Path(output_root) / method
    err_dir.mkdir(parents=True, exist_ok=True)
    err_path = err_dir / "err.json"
    payload = {
        "method": method,
        "missing_count": len(missing_tasks),
        "missing_tasks": missing_tasks,
    }
    with open(err_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return err_path



def run_one(python_bin, method, response_path, output_root, difficulty):
    output_base = Path(output_root) / method
    command = [
        python_bin,
        str(EVAL_SCRIPT),
        "--response_path",
        str(response_path),
        "--save_path",
        str(output_base),
        "--difficulty",
        difficulty,
    ]

    if response_path.is_dir():
        question_path_candidates = [response_path / "Question.json"]
    else:
        question_path_candidates = [response_path.parent / "Question.json"]
        if response_path.parent.name == "output":
            question_path_candidates.append(response_path.parent.parent / "Question.json")

    for question_path in question_path_candidates:
        if question_path.is_file():
            command.extend(["--question_path", str(question_path)])
            break

    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)



def main():
    args = parse_args()
    dataset_names, difficulty = resolve_targets(args.method, args.arg2, args.arg3)
    response_paths, missing_tasks = collect_response_paths(
        args.collected_root,
        args.method,
        dataset_names,
        difficulty,
        args.method_subdir,
    )
    err_path = write_missing_tasks(args.output_root, args.method, missing_tasks)

    if not response_paths:
        if err_path is not None:
            raise FileNotFoundError(
                f"No available result files found. Missing tasks have been recorded in {err_path}"
            )
        raise FileNotFoundError(
            f"No result files found for method={args.method}, datasets={dataset_names}, difficulty={difficulty}"
        )

    print(f"method={args.method}")
    print(f"datasets={','.join(dataset_names)}")
    print(f"difficulty={difficulty}")
    if args.method_subdir:
        print(f"method_subdir={args.method_subdir}")
    print(f"matched_results={len(response_paths)}")
    print(f"missing_results={len(missing_tasks)}")
    if err_path is not None:
        print(f"err_path={err_path}")

    success_count = 0
    failure_count = 0
    for response_path in response_paths:
        try:
            run_one(args.python_bin, args.method, response_path, args.output_root, difficulty)
            success_count += 1
        except subprocess.CalledProcessError as exc:
            failure_count += 1
            print(f"[ERROR] failed for {response_path}: {exc}")

    print(f"success_count={success_count}")
    print(f"failure_count={failure_count}")
    if failure_count > 0:
        sys.exit(1)



if __name__ == "__main__":
    main()
