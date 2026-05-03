#!/bin/bash

set -euo pipefail

LIST_JSON="${LIST_JSON:-}"
COLLECTED_ROOT="${COLLECTED_ROOT:-./Collected}"


METHOD1="${METHOD1:-HiRAG}"
METHOD2="${METHOD2:-KETRAG}"

ENGINE="${ENGINE:-gpt-4o}"
API_BASE="${API_BASE:-${OPENAI_BASE_URL:-}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

NUM_WORKERS=1
MAX_RETRIES=3
FORCE="false"


OUT_ROOT="./eval_results_h2h_eval200"
OUT_ROOT_PAIR="${OUT_ROOT}/${METHOD1}_vs_${METHOD2}"
TMP_ROOT="${OUT_ROOT_PAIR}/tmp_inputs"

# ---------------------------------

if [ -z "$LIST_JSON" ] || [ ! -f "$LIST_JSON" ]; then
  echo "[FATAL] list json not found. Set LIST_JSON or edit this script." >&2
  exit 1
fi

if [ -z "$API_KEY" ]; then
  echo "[FATAL] OPENAI_API_KEY is empty. Please run: export OPENAI_API_KEY=..." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "[FATAL] jq not found. Please install jq." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT_PAIR" "$TMP_ROOT"

MISMATCH_LOG="${OUT_ROOT_PAIR}/question_mismatch.tsv"
# header
printf "rel_dir\tmethod1_json\tmethod2_json\tq1\tq2\n" > "$MISMATCH_LOG"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/summary_eval_with_metrics.py"

if [ ! -f "$EVAL_PY" ]; then
  echo "[FATAL] evaluator not found: $EVAL_PY" >&2
  exit 1
fi

resolve_input_info() {
  local method="$1"
  local rel_dir="$2"

  python3 - "$COLLECTED_ROOT" "$method" "$rel_dir" << 'PY'
import glob
import json
import os
import re
import sys

collected_root, method, rel_dir = sys.argv[1:4]
method_root = os.path.join(collected_root, method)


def natural_sort_key(path: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", os.path.basename(path))]


def emit(input_path: str, case_dir: str):
    question_path = os.path.join(case_dir, "Question.json")
    print(
        json.dumps(
            {
                "input_path": input_path,
                "case_dir": case_dir,
                "question_path": question_path if os.path.isfile(question_path) else "",
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0)


candidate_case_dirs = [
    os.path.join(method_root, rel_dir),
    os.path.join(method_root, "is_data", rel_dir),
    os.path.join(method_root, "result", rel_dir),
]

for case_dir in candidate_case_dirs:
    if not os.path.isdir(case_dir):
        continue

    for candidate_name in ["results.rerun.json", "results.json", "Result.json", "hirag_result_q1.json"]:
        candidate_path = os.path.join(case_dir, candidate_name)
        if os.path.isfile(candidate_path):
            emit(candidate_path, case_dir)

    answer_candidates = sorted(
        glob.glob(os.path.join(case_dir, "output", "answer-*.json")),
        key=natural_sort_key,
    )
    if answer_candidates:
        emit(answer_candidates[0], case_dir)

raise SystemExit(
    f"No supported input file found for method={method}, rel_dir={rel_dir}. "
    f"Checked: {candidate_case_dirs}"
)
PY
}


extract_first_pair_json() {
  local in_path="$1"
  local question_path="${2:-}"

  python3 - "$in_path" "$question_path" << 'PY'
import json
import os
import sys

in_path = sys.argv[1]
question_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""


def load_raw_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().lstrip("\ufeff").strip()

    if not raw:
        raise SystemExit(f"Empty file: {path}")

    first_line = raw.splitlines()[0].strip()

    if first_line.startswith("{") and not raw.lstrip().startswith("[") and "\n" in raw:
        try:
            return json.loads(first_line)
        except Exception:
            pass

    try:
        return json.loads(raw)
    except Exception as exc:
        try:
            return json.loads(first_line)
        except Exception:
            raise SystemExit(f"Failed to parse JSON/JSONL: {path}: {exc}")


def to_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "results", "list", "items", "qa_list"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    raise SystemExit(f"Unsupported json type: {type(data)} in {in_path}")


def load_question_context(path: str):
    if not path or not os.path.isfile(path):
        return {"single_question": "", "questions_by_index": {}}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        question = str(data.get("question", "")).strip()
        return {
            "single_question": question,
            "questions_by_index": {1: question} if question else {},
        }

    if isinstance(data, list):
        questions_by_index = {}
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if question:
                questions_by_index[idx] = question
        return {
            "single_question": questions_by_index.get(1, ""),
            "questions_by_index": questions_by_index,
        }

    return {"single_question": "", "questions_by_index": {}}


records = to_records(load_raw_json(in_path))
if not records:
    raise SystemExit(f"Empty record list: {in_path}")

item = records[0]
if not isinstance(item, dict):
    raise SystemExit(f"First record is not a JSON object: {in_path}")

question_context = load_question_context(question_path)
question = str(item.get("question", "")).strip()

if not question:
    question_index = item.get("question_index")
    try:
        question_index = int(question_index)
    except (TypeError, ValueError):
        question_index = None
    if question_index is not None:
        question = question_context["questions_by_index"].get(question_index, "")

if not question:
    index_value = item.get("index")
    try:
        index_value = int(index_value)
    except (TypeError, ValueError):
        index_value = None
    if index_value is not None:
        question = question_context["questions_by_index"].get(index_value + 1, "")

if not question:
    question = question_context["questions_by_index"].get(1, "") or question_context["single_question"]

output = item.get("output", None)
if output is None:
    output = item.get("answer", None)

if output is None and isinstance(item.get("results"), list) and item["results"]:
    nested = item["results"][0]
    if isinstance(nested, dict):
        if not question:
            question = str(nested.get("question", "")).strip() or question
        output = nested.get("output", nested.get("answer", ""))

if output is None:
    output = ""

print(
    json.dumps(
        {
            "question": str(question).strip(),
            "output": str(output),
        },
        ensure_ascii=False,
    )
)
PY
}


make_one_csv() {
  local in_path="$1"
  local question_path="$2"
  local out_csv="$3"
  local pair_json

  pair_json="$(extract_first_pair_json "$in_path" "$question_path")"

  python3 - "$out_csv" "$pair_json" << 'PY'
import csv
import json
import sys

out_csv, pair_json = sys.argv[1], sys.argv[2]
pair = json.loads(pair_json)

with open(out_csv, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['question', 'output'])
    w.writeheader()
    w.writerow(
        {
            'question': str(pair.get('question', '')).strip(),
            'output': str(pair.get('output', '')),
        }
    )
PY
}

TOTAL=$(jq '.selected | length' "$LIST_JSON")
echo "[INFO] Loaded $TOTAL selected samples from $LIST_JSON"
echo "[INFO] METHOD1=$METHOD1 METHOD2=$METHOD2 ENGINE=$ENGINE workers=$NUM_WORKERS retries=$MAX_RETRIES"

i=0
jq -r '.selected[].rel_dir' "$LIST_JSON" | while read -r rel_dir; do
  i=$((i+1))
  case_name=$(echo "$rel_dir" | tr '/' '_')

  out_dir="$OUT_ROOT_PAIR"
  mkdir -p "$out_dir"
  # include both method names in every output file prefix
  out_name="${case_name}_${METHOD1}-vs-${METHOD2}"
  out_csv="${out_dir}/${out_name}-${ENGINE}.csv"
  out_log="${out_dir}/${out_name}-${ENGINE}.log"

  info1="$(resolve_input_info "$METHOD1" "$rel_dir" 2>/dev/null || true)"
  info2="$(resolve_input_info "$METHOD2" "$rel_dir" 2>/dev/null || true)"

  if [ -z "$info1" ]; then
    echo "[WARN] ($i/$TOTAL) missing input1 for ${METHOD1}: ${rel_dir}" >&2
    continue
  fi
  if [ -z "$info2" ]; then
    echo "[WARN] ($i/$TOTAL) missing input2 for ${METHOD2}: ${rel_dir}" >&2
    continue
  fi

  in1="$(printf '%s' "$info1" | jq -r '.input_path')"
  case_dir1="$(printf '%s' "$info1" | jq -r '.case_dir')"
  question_path1="$(printf '%s' "$info1" | jq -r '.question_path // ""')"
  in2="$(printf '%s' "$info2" | jq -r '.input_path')"
  case_dir2="$(printf '%s' "$info2" | jq -r '.case_dir')"
  question_path2="$(printf '%s' "$info2" | jq -r '.question_path // ""')"

  selected_in1="$in1"
  selected_in2="$in2"
  force_this="$FORCE"

  if [ -f "$out_csv" ]; then
    rerun_found="false"

    if [ -f "${case_dir1}/results.rerun.json" ]; then
      selected_in1="${case_dir1}/results.rerun.json"
      rerun_found="true"
    fi

    if [ -f "${case_dir2}/results.rerun.json" ]; then
      selected_in2="${case_dir2}/results.rerun.json"
      rerun_found="true"
    fi

    if [ "$FORCE" = "true" ]; then
      force_this="true"
      echo "[INFO] ($i/$TOTAL) FORCE=true, rerunning existing output: $rel_dir"
    elif [ "$rerun_found" = "false" ]; then
      echo "[INFO] ($i/$TOTAL) skip existing output without rerun: $rel_dir"
      continue
    else
      force_this="true"
      echo "[INFO] ($i/$TOTAL) existing output found; rerunning with rerun file when available: $rel_dir"
    fi
  fi

  pair_json1="$(extract_first_pair_json "$selected_in1" "$question_path1")"
  pair_json2="$(extract_first_pair_json "$selected_in2" "$question_path2")"
  q1="$(printf '%s' "$pair_json1" | jq -r '.question // ""')"
  q2="$(printf '%s' "$pair_json2" | jq -r '.question // ""')"

  if [ "$q1" != "$q2" ]; then
    echo "[WARN] ($i/$TOTAL) question mismatch, skip: $rel_dir" >&2
    printf "%s\t%s\t%s\t%s\t%s\n" "$rel_dir" "$selected_in1" "$selected_in2" "$q1" "$q2" >> "$MISMATCH_LOG"
    continue
  fi

  csv1="${TMP_ROOT}/${case_name}_${METHOD1}.csv"
  csv2="${TMP_ROOT}/${case_name}_${METHOD2}.csv"
  make_one_csv "$selected_in1" "$question_path1" "$csv1"
  make_one_csv "$selected_in2" "$question_path2" "$csv2"

  echo "[INFO] ($i/$TOTAL) eval: $rel_dir"
  echo "[INFO] ($i/$TOTAL) input1: $selected_in1"
  echo "[INFO] ($i/$TOTAL) input2: $selected_in2"

  python3 -u "$EVAL_PY" \
    --input_file1 "$csv1" \
    --input_file2 "$csv2" \
    --output_dir "$out_dir" \
    --output_file_name "$out_name" \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --engine "$ENGINE" \
    --num_workers "$NUM_WORKERS" \
    --max_retries "$MAX_RETRIES" \
    --force "$force_this" \
    > "$out_log" 2>&1

done

echo "[DONE] Batch Head-to-Head eval finished. Outputs under: $OUT_ROOT_PAIR"
echo "[DONE] Question mismatches (if any) recorded in: $MISMATCH_LOG"
