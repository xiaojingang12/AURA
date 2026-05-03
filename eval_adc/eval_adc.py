import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("AURA_EVAL_MODEL", "gpt-4o")
METADATA_SOURCE_PATH = os.getenv("AURA_METADATA_SOURCE_PATH", "")
DATASET_NAMES = {"Single-Sum", "Pair-Comp", "Multi-Comp", "Enumeration", "Temporal"}

SYS_PROMPT = '''
You are an expert tasked with extracting topic lists from response of the question.
Task: Read the response and return the complete predicted topics list.
Topic normalization: When you extract a topic, check whether it is semantically equivalent to an existing topic in the Common Errors List or Ground_truth list.
- If it matches a topic in the Common Errors List or Ground_truth List, use the Common Errors List or Ground_truth List wording.
- Otherwise, write a concise topic label that accurately reflects the response.
Requirements:
- Extract all distinct topics supported by the response.
- Extract all the topics contained within the response.
- Use semantic matching, not literal matching.
- Keep each topic concise and complete.
- Merge duplicates and near-duplicates.
- The purpose of extraction is to analyze the coverage of the topic in the response. Therefore, you must be very strict and cannot simply assume that the meanings are the same.
- Return only JSON.
Output format: { "predicted_topics": ["topic1", "topic2", "topic3"] }
'''

#- Do not infer unsupported topics.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract predicted topics from response records and evaluate them with ground truth and common errors."
    )
    parser.add_argument(
        "--response_path",
        type=str,
        required=True,
        help="Response file path, or a question directory containing results.rerun.json/results.json/Result.json/hirag_result_q*.json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the evaluation results as JSON",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["simple_QA", "middle_QA", "hard_QA"],
        default=None,
        help="Difficulty type. simple_QA only evaluates the first record, middle_QA and hard_QA evaluate all records.",
    )
    parser.add_argument(
        "--question_path",
        type=str,
        default=None,
        help="Optional question file path. Supports both single-question dict JSON and multi-question list JSON.",
    )
    parser.add_argument(
        "--metadata-source-path",
        type=str,
        default=METADATA_SOURCE_PATH,
        help="Path to metadata JSON/JSONL records containing question, answer, and common error fields.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=BASE_URL,
        help="OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=API_KEY,
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model used for topic extraction. Defaults to AURA_EVAL_MODEL or gpt-4o.",
    )
    return parser.parse_args()


def read_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line in {file_path}:{line_no}: {exc}") from exc
    return records


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "results", "list", "items", "qa_list"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]

    raise ValueError(f"Unsupported JSON structure in {file_path}: {type(data)}")


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]


def load_single_file_records(file_path):
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        return read_jsonl(file_path)
    if file_path.suffix == ".json":
        try:
            return read_json(file_path)
        except Exception as json_error:
            print(f"[Warning] JSON read failed for {file_path}: {json_error}")
            print(f"[Info] Falling back to JSONL read for {file_path}")
            return read_jsonl(file_path)
    try:
        return read_jsonl(file_path)
    except Exception as jsonl_error:
        print(f"[Warning] JSONL read failed for {file_path}: {jsonl_error}")
        print(f"[Info] Falling back to JSON read for {file_path}")
        return read_json(file_path)


def resolve_response_files(response_path):
    response_path = Path(response_path)
    if response_path.is_dir():
        for candidate_name in ["results.rerun.json", "results.json", "Result.json"]:
            candidate_path = response_path / candidate_name
            if candidate_path.is_file():
                return [candidate_path]

        hirag_candidates = sorted(response_path.glob("hirag_result_q*.json"), key=natural_sort_key)
        if hirag_candidates:
            return hirag_candidates

        answer_candidates = sorted((response_path / "output").glob("answer-*.json"), key=natural_sort_key)
        if answer_candidates:
            return answer_candidates

        raise FileNotFoundError(f"No supported response files found under directory: {response_path}")

    return [response_path]


def load_records(file_path):
    all_records = []
    for resolved_path in resolve_response_files(file_path):
        all_records.extend(load_single_file_records(resolved_path))
    return all_records


def load_question_context(question_path):
    if not question_path:
        return {"single_question": "", "questions_by_index": {}}

    with open(question_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        question = str(data.get("question", "")).strip()
        questions_by_index = {1: question} if question else {}
        return {
            "single_question": question,
            "questions_by_index": questions_by_index,
        }

    if isinstance(data, list):
        questions_by_index = {}
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if question:
                questions_by_index[index] = question
        return {
            "single_question": questions_by_index.get(1, ""),
            "questions_by_index": questions_by_index,
        }

    raise ValueError(f"Unsupported question JSON structure in {question_path}: {type(data)}")


def normalize_topic_text(text):
    topic = text.strip().strip('"').strip("'")
    if ":" in topic:
        topic = topic.split(":", 1)[1].strip()
    return topic



def split_malformed_topic_text(text):
    text = text.strip()
    if not text:
        return []

    split_parts = re.split(r"\]\s*\[", text)
    cleaned_parts = []
    for index, part in enumerate(split_parts):
        part = part.strip()
        if index > 0:
            part = part.lstrip("[").strip()
        if index < len(split_parts) - 1:
            part = part.rstrip("]").strip()
        cleaned_parts.extend(
            [normalize_topic_text(item) for item in part.split(",") if normalize_topic_text(item)]
        )
    return cleaned_parts



def parse_topic_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        topics = []
        for item in value:
            topics.extend(split_malformed_topic_text(str(item)))
        return topics
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                topics = []
                for item in parsed:
                    topics.extend(split_malformed_topic_text(str(item)))
                return topics
        except Exception:
            pass
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return split_malformed_topic_text(text)
    return split_malformed_topic_text(str(value))


def normalize_response_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_usage_value(usage, field_name):
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field_name)
    else:
        value = getattr(usage, field_name, None)
    return int(value) if value is not None else 0


def build_prompt(question, output, answer, common_errors):
    return f'''
Question:
{question}

Response:
{output}

Ground_truth List:
{answer}

Common Errors List:
{common_errors if common_errors else "N/A"}

Extract the complete predicted topics list.'''


def extract_predicted_topics(question, output, answer, common_errors):
    sample_start_time = time.time()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": build_prompt(question, output, answer, common_errors)},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    api_elapsed_seconds = 0.0
    usage_total_tokens = 0
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    retries = 0

    while retries < 3:
        try:
            api_call_start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            api_elapsed_seconds += time.time() - api_call_start
            response.raise_for_status()

            data = response.json()
            usage = data.get("usage")
            usage_total_tokens += get_usage_value(usage, "total_tokens")
            usage_prompt_tokens += get_usage_value(usage, "prompt_tokens")
            usage_completion_tokens += get_usage_value(usage, "completion_tokens")

            content = data["choices"][0]["message"]["content"].strip()
            content = re.sub(r"\n+", "\n", content)
            parsed = json.loads(content)
            predicted_topics = parse_topic_list(parsed.get("predicted_topics", []))

            total_elapsed_seconds = time.time() - sample_start_time
            return {
                "predicted_topics": predicted_topics,
                "raw_response": parsed,
                "api_error": None,
                "metrics": {
                    "sample_time_seconds": round(total_elapsed_seconds, 4),
                    "api_time_seconds": round(api_elapsed_seconds, 4),
                    "total_tokens": usage_total_tokens,
                    "prompt_tokens": usage_prompt_tokens,
                    "completion_tokens": usage_completion_tokens,
                    "retry_count": retries,
                    "success": True,
                },
            }
        except Exception as exc:
            last_error = str(exc)
            retries += 1
            time.sleep(2)

    total_elapsed_seconds = time.time() - sample_start_time
    return {
        "predicted_topics": [],
        "raw_response": None,
        "api_error": last_error or "Unknown API error",
        "metrics": {
            "sample_time_seconds": round(total_elapsed_seconds, 4),
            "api_time_seconds": round(api_elapsed_seconds, 4),
            "total_tokens": usage_total_tokens,
            "prompt_tokens": usage_prompt_tokens,
            "completion_tokens": usage_completion_tokens,
            "retry_count": retries,
            "success": False,
        },
    }


def evaluate_topics(predicted_topics, ground_truth, common_errors):
    predicted_set = set(predicted_topics)
    ground_truth_set = set(ground_truth)
    common_errors_set = set(common_errors)

    covered_topics = [topic for topic in predicted_topics if topic in ground_truth_set]
    hallucinated_topics = [topic for topic in predicted_topics if topic in common_errors_set]
    unmatched_predicted_topics = [
        topic for topic in predicted_topics if topic not in ground_truth_set and topic not in common_errors_set
    ]
    missed_ground_truth = [topic for topic in ground_truth if topic not in predicted_set]

    coverage = len(covered_topics)
    recall = coverage / len(ground_truth) if ground_truth else 0.0
    precision = coverage / len(predicted_topics) if predicted_topics else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hallucination = len(hallucinated_topics) / len(common_errors) if common_errors else 0.0

    return {
        "coverage": coverage,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "hallucination": round(hallucination, 4),
        "covered_topics": covered_topics,
        "missed_ground_truth": missed_ground_truth,
        "hallucinated_topics": hallucinated_topics,
        "unmatched_predicted_topics": unmatched_predicted_topics,
    }


def remove_common_errors_from_ground_truth(ground_truth, common_errors):
    common_errors_set = set(common_errors)
    cleaned_ground_truth = []
    removed_topics = []
    for topic in ground_truth:
        if topic in common_errors_set:
            removed_topics.append(topic)
            continue
        cleaned_ground_truth.append(topic)
    return cleaned_ground_truth, removed_topics



def derive_source_name_from_response_path(response_path):
    response_path_obj = Path(response_path)
    if response_path_obj.is_dir():
        anchor_path = response_path_obj
    elif response_path_obj.parent.name == "output" and response_path_obj.name.startswith("answer-"):
        anchor_path = response_path_obj.parent.parent
    else:
        anchor_path = response_path_obj.parent

    path_parts = [part for part in anchor_path.parts if part not in {"", "/"}]
    dataset_index = next((index for index, part in enumerate(path_parts) if part in DATASET_NAMES), None)

    if dataset_index is not None:
        source_parts = path_parts[dataset_index : dataset_index + 3]
        if dataset_index > 0 and path_parts[dataset_index - 1] in {"is_data", "result"}:
            source_parts = [path_parts[dataset_index - 1], *source_parts]
    else:
        source_parts = path_parts[-4:]

    if not source_parts:
        source_parts = [anchor_path.name or response_path_obj.stem or "unknown_source"]
    return "_".join(source_parts)



def build_metadata_lookup(metadata_records):
    lookup = {}
    for item in metadata_records:
        question = str(item.get("question", "")).strip()
        if not question or question in lookup:
            continue
        lookup[question] = {
            "ground_truth": parse_topic_list(item.get("answer", [])),
            "common_errors": parse_topic_list(item.get("com_err", [])),
        }
    return lookup



def extract_question_and_response(response_item, question_context=None, record_position=None):
    question_context = question_context or {"single_question": "", "questions_by_index": {}}
    question = str(response_item.get("question", "")).strip()
    if not question:
        question_index = response_item.get("question_index")
        try:
            question_index = int(question_index)
        except (TypeError, ValueError):
            question_index = None
        if question_index is not None:
            question = question_context["questions_by_index"].get(question_index, "")
    if not question and record_position is not None:
        question = question_context["questions_by_index"].get(record_position, "")
    if not question:
        question = question_context["single_question"]
    response = response_item.get("output", "")

    if not response and "answer" in response_item:
        response = response_item.get("answer", "")

    if not question and isinstance(response_item.get("results"), list) and response_item["results"]:
        nested_item = response_item["results"][0]
        question = str(nested_item.get("question", "")).strip() or question_context["single_question"]
        response = nested_item.get("answer", "")

    return question, normalize_response_text(response)



def align_records(response_records, metadata_lookup, question_context=None):
    aligned_records = []
    missing_questions = []
    for record_position, response_item in enumerate(response_records, start=1):
        question, normalized_response = extract_question_and_response(
            response_item,
            question_context,
            record_position,
        )
        metadata = metadata_lookup.get(question)
        if metadata is None:
            missing_questions.append(question)
            continue

        common_errors = metadata["common_errors"]
        ground_truth = metadata["ground_truth"]
        ground_truth, removed_overlap_topics = remove_common_errors_from_ground_truth(ground_truth, common_errors)
        aligned_records.append(
            {
                "record_position": record_position,
                "question_index": response_item.get("question_index"),
                "question": question,
                "ground_truth": ground_truth,
                "common_errors": common_errors,
                "removed_overlap_topics": removed_overlap_topics,
                "response": normalized_response,
            }
        )
    return aligned_records, missing_questions


def build_summary(results):
    sample_count = len(results)
    success_count = sum(1 for item in results if item["usage_metrics"]["success"])
    failure_count = sample_count - success_count
    total_tokens = sum(item["usage_metrics"]["total_tokens"] for item in results)
    prompt_tokens = sum(item["usage_metrics"]["prompt_tokens"] for item in results)
    completion_tokens = sum(item["usage_metrics"]["completion_tokens"] for item in results)
    sample_time_sum = sum(item["usage_metrics"]["sample_time_seconds"] for item in results)
    api_time_sum = sum(item["usage_metrics"]["api_time_seconds"] for item in results)

    return {
        "sample_count": sample_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "sample_time_seconds_sum": round(sample_time_sum, 4),
        "api_time_seconds_sum": round(api_time_sum, 4),
        "avg_total_tokens_per_sample": round(total_tokens / sample_count, 4) if sample_count else 0.0,
        "avg_prompt_tokens_per_sample": round(prompt_tokens / sample_count, 4) if sample_count else 0.0,
        "avg_completion_tokens_per_sample": round(completion_tokens / sample_count, 4) if sample_count else 0.0,
        "avg_sample_time_seconds": round(sample_time_sum / sample_count, 4) if sample_count else 0.0,
        "avg_api_time_seconds": round(api_time_sum / sample_count, 4) if sample_count else 0.0,
    }


def build_output_save_path(base_save_path, source_name):
    base_path = Path(base_save_path)
    source_name = source_name + ".json"
    return base_path / source_name



def filter_response_records_by_difficulty(response_records, difficulty):
    if difficulty == "simple_QA":
        return response_records[:1]
    return response_records



def main():
    global BASE_URL, API_KEY, MODEL

    args = parse_arguments()
    batch_start_time = time.time()
    BASE_URL = args.api_base
    API_KEY = args.api_key
    MODEL = args.model

    if not args.metadata_source_path:
        raise SystemExit("Missing metadata source path. Use --metadata-source-path or set AURA_METADATA_SOURCE_PATH.")
    if not API_KEY:
        raise SystemExit("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    response_records = load_records(args.response_path)
    response_records = filter_response_records_by_difficulty(response_records, args.difficulty)
    question_context = load_question_context(args.question_path)
    metadata_records = load_records(args.metadata_source_path)
    metadata_lookup = build_metadata_lookup(metadata_records)
    aligned_records, missing_questions = align_records(response_records, metadata_lookup, question_context)

    results = []
    for item in aligned_records:
        extraction_result = extract_predicted_topics(
            question=item["question"],
            output=item["response"],
            answer=item["ground_truth"],
            common_errors=item["common_errors"],
        )
        metric_result = evaluate_topics(
            predicted_topics=extraction_result["predicted_topics"],
            ground_truth=item["ground_truth"],
            common_errors=item["common_errors"],
        )

        results.append(
            {
                "record_position": item["record_position"],
                "question_index": item["question_index"],
                "question": item["question"],
                "response": item["response"],
                "ground_truth": item["ground_truth"],
                "common_errors": item["common_errors"],
                "removed_overlap_topics": item["removed_overlap_topics"],
                "predicted_topics": extraction_result["predicted_topics"],
                "coverage": metric_result["coverage"],
                "recall": metric_result["recall"],
                "precision": metric_result["precision"],
                "f1": metric_result["f1"],
                "hallucination": metric_result["hallucination"],
                "covered_topics": metric_result["covered_topics"],
                "missed_ground_truth": metric_result["missed_ground_truth"],
                "hallucinated_topics": metric_result["hallucinated_topics"],
                "unmatched_predicted_topics": metric_result["unmatched_predicted_topics"],
                "raw_response": extraction_result["raw_response"],
                "api_error": extraction_result["api_error"],
                "usage_metrics": extraction_result["metrics"],
            }
        )

    batch_wall_clock_seconds = time.time() - batch_start_time
    summary = build_summary(results)
    summary["batch_wall_clock_seconds"] = round(batch_wall_clock_seconds, 4)
    summary["missing_question_count"] = len(missing_questions)

    source_name = derive_source_name_from_response_path(args.response_path)
    final_save_path = build_output_save_path(args.save_path, source_name)
    output_payload = {
        "config": {
            "response_path": args.response_path,
            "question_path": args.question_path,
            "save_path": str(final_save_path),
            "save_path_base": args.save_path,
            "metadata_source_path": args.metadata_source_path,
            "source_name": source_name,
            "difficulty": args.difficulty,
            "model": MODEL,
            "base_url": BASE_URL,
        },
        "summary": summary,
        "missing_questions": missing_questions,
        "results": results,
    }

    save_path = final_save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} records to {save_path}")
    print(f"Source name: {source_name}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Missing questions: {len(missing_questions)}")
    print(f"Total token usage: {summary['total_tokens']}")
    print(f"Total sample time: {summary['sample_time_seconds_sum']:.4f} seconds")
    print(f"Total API time: {summary['api_time_seconds_sum']:.4f} seconds")


if __name__ == "__main__":
    main()
