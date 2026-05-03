# This code is based on the Summary Evaluation in LightRAG codebase.
# Extended version for token/time comparison with topic-set evaluation.

import re
import os
import time
import json
import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
from openai import OpenAI


INCLUDE_COL = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Directness",
    "Overall Winner",
]


def load_input_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".json"):
        return pd.read_json(file_path, orient="records", lines=True)
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path}")


def normalize_answer_columns(df: pd.DataFrame, answer_tag: str) -> pd.DataFrame:
    renamed_df = df.copy()
    answer_col = None
    for candidate in ["output", "pred", "answer"]:
        if candidate in renamed_df.columns:
            answer_col = candidate
            break

    if answer_col is None:
        raise ValueError(
            f"No supported answer column found for {answer_tag}. "
            f"Expected one of: output, pred, answer. Existing columns: {list(renamed_df.columns)}"
        )

    required_cols = ["question", answer_col]
    missing_cols = [col for col in required_cols if col not in renamed_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {answer_tag}: {missing_cols}")

    renamed_df = renamed_df[["question", answer_col]].copy()
    renamed_df = renamed_df.rename(columns={answer_col: f"answer_{answer_tag}"})
    return renamed_df


def get_usage_value(usage, field_name: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, field_name, None)
    return int(value) if value is not None else 0


def build_failure_result(error_message: str):
    return {
        "Comprehensiveness": "N/A",
        "Diversity": "N/A",
        "Empowerment": "N/A",
        "Directness": "N/A",
        "Overall Winner": "N/A",
        "ori_json_res": "N/A",
        "error_message": error_message,
    }


def eval_single(i, query, answer1, answer2, args):
    if i % 20 == 0:
        print(f"Processing {i}.")

    sample_start_time = time.time()
    client_kwargs = {"api_key": args.api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """

    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**,**Empowerment**, and **Directness**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
    - **Directness**. How specifically and clearly does the answer address the question?
    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these four categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence explanation here]"
        }},
        "Diversity": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Directness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Briefly summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    parameters = {
        "model": args.engine,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": 8000,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    }

    max_retries = args.max_retries
    retries = 0
    result = None
    usage_token_total = 0
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    api_elapsed_seconds = 0.0
    last_error_message = "Unknown error"

    while retries < max_retries:
        try:
            api_call_start = time.time()
            response = client.chat.completions.create(**parameters)
            api_elapsed_seconds += time.time() - api_call_start

            result = response.choices[0].message.content
            usage_token_total += get_usage_value(response.usage, "total_tokens")
            usage_prompt_tokens += get_usage_value(response.usage, "prompt_tokens")
            usage_completion_tokens += get_usage_value(response.usage, "completion_tokens")
        except Exception as e:
            retries += 1
            last_error_message = f"OpenAI error: {e}"
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)
            continue

        try:
            result = re.sub(r"\n+", "\n", result)
            json_res = json.loads(result)

            col_check = True
            for col in INCLUDE_COL:
                if col not in json_res:
                    print(f"Error parsing JSON response from OpenAI. not include col: {col}")
                    col_check = False
                    break
                if "Winner" not in json_res[col]:
                    print(
                        f"Error parsing JSON response from OpenAI. not include winner in col: {col}"
                    )
                    col_check = False
                    break

            if not col_check:
                retries += 1
                last_error_message = "Missing required fields in JSON response"
                continue

            res_dict = {
                "Comprehensiveness": json_res["Comprehensiveness"]["Winner"],
                "Diversity": json_res["Diversity"]["Winner"],
                "Empowerment": json_res["Empowerment"]["Winner"],
                "Directness": json_res["Directness"]["Winner"],
                "Overall Winner": json_res["Overall Winner"]["Winner"],
                "ori_json_res": json_res,
                "error_message": "",
            }
            total_elapsed_seconds = time.time() - sample_start_time
            return i, res_dict, {
                "sample_time_seconds": round(total_elapsed_seconds, 4),
                "api_time_seconds": round(api_elapsed_seconds, 4),
                "total_tokens": usage_token_total,
                "prompt_tokens": usage_prompt_tokens,
                "completion_tokens": usage_completion_tokens,
                "retry_count": retries,
                "success": True,
            }
        except Exception as e:
            print("Error parsing JSON response from OpenAI.")
            print(e)
            retries += 1
            last_error_message = f"JSON parsing error: {e}"
            continue

    print("Failed to get response from OpenAI.")
    total_elapsed_seconds = time.time() - sample_start_time
    return i, build_failure_result(last_error_message), {
        "sample_time_seconds": round(total_elapsed_seconds, 4),
        "api_time_seconds": round(api_elapsed_seconds, 4),
        "total_tokens": usage_token_total,
        "prompt_tokens": usage_prompt_tokens,
        "completion_tokens": usage_completion_tokens,
        "retry_count": retries,
        "success": False,
    }


def batch_eval(df_1, df_2, args):
    norm_df_1 = normalize_answer_columns(df_1, "1")
    norm_df_2 = normalize_answer_columns(df_2, "2")

    merged_df = pd.merge(norm_df_1, norm_df_2, on="question", how="inner")
    print(f"Merged shape: {merged_df.shape}")

    queries = merged_df["question"].tolist()
    answers1 = merged_df["answer_1"].tolist()
    answers2 = merged_df["answer_2"].tolist()
    eval_tuples = list(zip(queries, answers1, answers2))

    with mp.Pool(processes=args.num_workers) as pool:
        process_func = partial(eval_single, args=args)
        results = pool.starmap(
            process_func, [(i, *eval_tuple) for i, eval_tuple in enumerate(eval_tuples)]
        )

    results_all_list = []
    metrics_rows = []
    total_tokens_all = 0
    prompt_tokens_all = 0
    completion_tokens_all = 0
    sample_time_all = 0.0
    api_time_all = 0.0
    success_count = 0
    failure_count = 0

    for num_i, result, metrics in results:
        total_tokens_all += metrics["total_tokens"]
        prompt_tokens_all += metrics["prompt_tokens"]
        completion_tokens_all += metrics["completion_tokens"]
        sample_time_all += metrics["sample_time_seconds"]
        api_time_all += metrics["api_time_seconds"]
        success_count += 1 if metrics["success"] else 0
        failure_count += 0 if metrics["success"] else 1

        query = queries[num_i]
        answer1 = answers1[num_i]
        answer2 = answers2[num_i]

        result["query"] = query
        result["answer1"] = answer1
        result["answer2"] = answer2
        result["id"] = num_i
        result["sample_time_seconds"] = metrics["sample_time_seconds"]
        result["api_time_seconds"] = metrics["api_time_seconds"]
        result["total_tokens"] = metrics["total_tokens"]
        result["prompt_tokens"] = metrics["prompt_tokens"]
        result["completion_tokens"] = metrics["completion_tokens"]
        result["retry_count"] = metrics["retry_count"]
        result["success"] = metrics["success"]
        results_all_list.append(result)

        metrics_rows.append(
            {
                "id": num_i,
                "question": query,
                "sample_time_seconds": metrics["sample_time_seconds"],
                "api_time_seconds": metrics["api_time_seconds"],
                "total_tokens": metrics["total_tokens"],
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "retry_count": metrics["retry_count"],
                "success": metrics["success"],
            }
        )

    aggregate_metrics = {
        "sample_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "total_tokens": total_tokens_all,
        "prompt_tokens": prompt_tokens_all,
        "completion_tokens": completion_tokens_all,
        "sample_time_seconds_sum": round(sample_time_all, 4),
        "api_time_seconds_sum": round(api_time_all, 4),
        "avg_total_tokens_per_sample": round(total_tokens_all / len(results), 4) if results else 0.0,
        "avg_prompt_tokens_per_sample": round(prompt_tokens_all / len(results), 4) if results else 0.0,
        "avg_completion_tokens_per_sample": round(completion_tokens_all / len(results), 4) if results else 0.0,
        "avg_sample_time_seconds": round(sample_time_all / len(results), 4) if results else 0.0,
        "avg_api_time_seconds": round(api_time_all / len(results), 4) if results else 0.0,
    }

    print(f"Total token usage: {total_tokens_all}")
    print(f"Total sample time: {sample_time_all:.4f} seconds")
    print(f"Total API time: {api_time_all:.4f} seconds")

    return pd.DataFrame(results_all_list), pd.DataFrame(metrics_rows), aggregate_metrics


def print_win_statistics(res_df: pd.DataFrame):
    sample_count = len(res_df)
    if sample_count == 0:
        print("No evaluation samples found.")
        return

    for col in INCLUDE_COL:
        answer1_wins = res_df[col].value_counts().get("Answer 1", 0)
        answer2_wins = res_df[col].value_counts().get("Answer 2", 0)
        print(
            f"{col}: Answer 1 wins {answer1_wins} / {sample_count} times, {100 * (answer1_wins / sample_count):.2f}",
            end=" ",
        )
        print(
            f"Answer 2 wins {answer2_wins} / {sample_count} times, {100 * (answer2_wins / sample_count):.2f}"
        )


def save_summary_json(summary_path: str, aggregate_metrics: dict, args, save_file_path: str):
    summary_payload = {
        "engine": args.engine,
        "input_file1": args.input_file1,
        "input_file2": args.input_file2,
        "output_csv": save_file_path,
        "num_workers": args.num_workers,
        "max_retries": args.max_retries,
        "aggregate_metrics": aggregate_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"Summary metrics saved to {summary_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file1",
        type=str,
        default="",
        help="Path to the first input file containing the questions and answers",
    )

    parser.add_argument(
        "--input_file2",
        type=str,
        default="",
        help="Path to the second input file containing the questions and answers",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save evaluation outputs",
    )

    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Output CSV filename",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key",
    )

    parser.add_argument(
        "--api_base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI API base URL",
    )

    parser.add_argument(
        "--force",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to force re-evaluation of all samples",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI engine to use",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel worker processes",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for each sample",
    )

    args = parser.parse_args()
    if not args.input_file1 or not args.input_file2:
        raise SystemExit("Missing input files. Use --input_file1 and --input_file2.")
    if not args.output_file_name:
        raise SystemExit("Missing output filename. Use --output_file_name.")
    if not args.api_key:
        raise SystemExit("Missing API key. Use --api_key or set OPENAI_API_KEY.")

    eval_file1 = load_input_file(args.input_file1)
    eval_file2 = load_input_file(args.input_file2)

    print(f"Reading files:{args.input_file1}")
    print(f"Reading files:{args.input_file2}")
    print(f"shape1:{eval_file1.shape}, shape2:{eval_file2.shape}")

    save_path_dir = args.output_dir
    os.makedirs(save_path_dir, exist_ok=True)
    if not args.output_file_name.endswith(".csv"):
        args.output_file_name += f"-{args.engine}.csv"
    save_file_path = os.path.join(save_path_dir, args.output_file_name)

    metrics_file_path = save_file_path.replace(".csv", "_metrics.csv")
    summary_file_path = save_file_path.replace(".csv", "_summary.json")

    force = args.force
    print(f"running with force:{force}")

    if os.path.exists(save_file_path) and not force:
        print(f"File {save_file_path} already exists. Reading.")
        res_df = pd.read_csv(save_file_path)
        if os.path.exists(summary_file_path):
            print(f"Existing summary metrics file found: {summary_file_path}")
        else:
            print("Existing evaluation CSV found, but summary metrics file is missing.")
    else:
        if force:
            print("Forcing re-evaluation of all samples.")

        batch_start_time = time.time()
        res_df, metrics_df, aggregate_metrics = batch_eval(eval_file1, eval_file2, args)
        batch_wall_clock_seconds = time.time() - batch_start_time
        aggregate_metrics["batch_wall_clock_seconds"] = round(batch_wall_clock_seconds, 4)

        res_df.to_csv(save_file_path, index=False)
        metrics_df.to_csv(metrics_file_path, index=False)
        save_summary_json(summary_file_path, aggregate_metrics, args, save_file_path)

        print(f"Results saved to {save_file_path}.")
        print(f"Metrics saved to {metrics_file_path}.")
        print(f"Time taken: {batch_wall_clock_seconds:.2f} seconds.")

    print_win_statistics(res_df)
