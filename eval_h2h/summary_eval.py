# This code is based on the Summary Evaluation in LightRAG codebase.

import re
import os
import time
import json
import argparse
import pandas as pd
from openai import OpenAI
import multiprocessing as mp
from functools import partial


INCLUDE_COL = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Directness",
    "Overall Winner",
]


def eval_single(i, query, answer1, answer2, args):
    if i % 20 == 0:
        print(f"Processing {i}.")

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
        "max_tokens": 8000,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    }
    max_retries = 3
    retries = 0
    success = False
    result = None
    res_dict = {}
    token_usage = 0
    while not success and retries < max_retries:
        try:
            response = client.chat.completions.create(**parameters)
            result = response.choices[0].message.content
            token_usage += response.usage.total_tokens
        except Exception as e:
            retries += 1
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)

        try:
            result = re.sub(r"\n+", "\n", result)

            json_res = json.loads(result)

            col_check = True
            for col in INCLUDE_COL:
                if col not in json_res:
                    print(
                        f"Error parsing JSON response from OpenAI. not include col: {col}"
                    )
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
                continue
            res_dict = {
                "Comprehensiveness": json_res["Comprehensiveness"]["Winner"],
                "Diversity": json_res["Diversity"]["Winner"],
                "Empowerment": json_res["Empowerment"]["Winner"],
                "Directness": json_res["Directness"]["Winner"],
                "Overall Winner": json_res["Overall Winner"]["Winner"],
                "ori_json_res": json_res,
            }

        except Exception as e:
            print("Error parsing JSON response from OpenAI.")
            print(e)
            retries += 1
            continue

        success = True

    if not success:
        print("Failed to get response from OpenAI.")
        tmp_dict = {
            "Comprehensiveness": "N/A",
            "Diversity": "N/A",
            "Empowerment": "N/A",
            "Directness": "N/A",
            "Overall Winner": "N/A",
            "ori_json_res": "N/A",
        }
        return i, tmp_dict, token_usage

    return i, res_dict, token_usage


def batch_eval(df_1, df_2, args):

    col_1 = "output" if "output" in df_1.columns else "pred"
    col_2 = "output" if "output" in df_2.columns else "pred"
    print(f"col1:{col_1}, col2:{col_2}")
    merged_df = pd.merge(df_1, df_2, on='question')
    queries = merged_df["question"].tolist()
    if col_1 in merged_df.columns:
        answers1 = merged_df[col_1].tolist()
        answers2 = merged_df[col_2].tolist()
    else:
        answers1 = merged_df[f"{col_1}_x"].tolist()
        answers2 = merged_df[f"{col_2}_y"].tolist()
    # queries = df_1["question"].tolist()
    # answers1 = df_1[col_1].tolist()
    # answers2 = df_2[col_2].tolist()

    eval_tuples = list(zip(queries, answers1, answers2))

    with mp.Pool(processes=10) as pool:
        process_func = partial(eval_single, args=args)
        results = pool.starmap(
            process_func, [(i, *eval_tuple) for i, eval_tuple in enumerate(eval_tuples)]
        )

    results_all_list = []
    usage_token_all = 0
    for num_i, result, token in results:
        usage_token_all += token
        query = queries[num_i]
        answer1 = answers1[num_i]
        answer2 = answers2[num_i]
        result["query"] = query
        result["answer1"] = answer1
        result["answer2"] = answer2
        result["id"] = num_i
        results_all_list.append(result)

    print(f"Total token usage: {usage_token_all}")
    res_df = pd.DataFrame(results_all_list)
    return res_df


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
        help="Path to the output file to write the evaluation prompts",
    )

    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Path to the output file to write the evaluation prompts",
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

    args = parser.parse_args()
    if not args.input_file1 or not args.input_file2:
        raise SystemExit("Missing input files. Use --input_file1 and --input_file2.")
    if not args.output_file_name:
        raise SystemExit("Missing output filename. Use --output_file_name.")
    if not args.api_key:
        raise SystemExit("Missing API key. Use --api_key or set OPENAI_API_KEY.")

    if args.input_file1.endswith(".json"):
        eval_file1 = pd.read_json(args.input_file1, orient="records", lines=True)
    elif args.input_file1.endswith(".csv"):
        eval_file1 = pd.read_csv(args.input_file1)

    if args.input_file2.endswith(".json"):
        eval_file2 = pd.read_json(args.input_file2, orient="records", lines=True)
    elif args.input_file2.endswith(".csv"):
        eval_file2 = pd.read_csv(args.input_file2)

    print(f"Reading files:{args.input_file1}")
    print(f"Reading files:{args.input_file2}")
    print(f"shape1:{eval_file1.shape}, shape2:{eval_file2.shape}")

    save_path_dir = args.output_dir
    os.makedirs(save_path_dir, exist_ok=True)
    if not args.output_file_name.endswith(".csv"):
        args.output_file_name += f"-{args.engine}.csv"
    save_file_path = os.path.join(save_path_dir, args.output_file_name)

    force = args.force
    print(f"running with force:{force}")
    if os.path.exists(save_file_path) and not force:
        print(f"File {save_file_path} already exists. Reading.")
        res_df = pd.read_csv(save_file_path)
    else:
        if force:
            print("Forcing re-evaluation of all samples.")
        import time

        start_time = time.time()
        res_df = batch_eval(eval_file1, eval_file2, args)
        res_df.to_csv(save_file_path, index=False)
        print(f"Results saved to {save_file_path}.")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds.")

    for col in INCLUDE_COL:
        win1_times = res_df[col].value_counts().get("Answer 1", 0)
        print(
            f"{col}: Answer 1 wins {win1_times} / 125 times, { 100 * (win1_times / 125) :.2f}",
            end=" ",
        )
        print(
            f"Answer 2 wins {125-win1_times} / 125 times, {100 * ((125-win1_times)/125):.2f}"
        )
