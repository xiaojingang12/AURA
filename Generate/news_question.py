import argparse
import json
import os
import time

import requests


def load_news(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_qa_single_topic(news_list, base_url, api_key, model):
    """Generate one single-topic enumeration QA pair from a list of news articles."""
    news_prompt_part = "Here is the list of news articles:\n\n"
    for index, news in enumerate(news_list):
        news_prompt_part += (
            f"[News {index}]\n"
            f"Title: {news['title']}\n"
            f"Description: {news.get('description', 'N/A')}\n"
            f"Published At: {news['published_at']}\n"
            f"Source: {news['source']}\n"
            "---\n"
        )

    prompt = f"""
{news_prompt_part}

Task:
1. Scan the list of news articles above.
2. Identify a single, specific, and concrete topic or subject that is mentioned or discussed across multiple articles.
3. Formulate a question that asks for a list or enumeration of key points, methods, features, impacts, or other relevant details specifically related to the chosen topic.
4. Provide the answer as a list of concise keywords or short phrases. Format the answer strictly like: [Point 1, brief description; Point 2, brief description].
5. Briefly explain in 1-2 sentences why you chose this topic and how the selected articles contribute information to answer your question.
6. List the titles of the relevant news articles separated by semicolons.

Please respond in JSON format with this structure:
{{
  "question": "<generated enumeration question>",
  "answer": "[<Point 1, brief description; Point 2, brief description; ...>]",
  "reason": "<reason for choosing this topic>",
  "titles": "<Relevant News Title 1; Relevant News Title 2; ...>"
}}

Only output the final JSON object.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant tasked with analyzing news articles. "
                    "Identify a single clear topic discussed across multiple articles, create a list-style question, "
                    "and provide a concise keyword-based answer. Strictly follow the output format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  Attempting API call ({attempt + 1}/{max_retries})...")
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            parsed_result = json.loads(result_text)
            required_keys = ["question", "answer", "reason", "titles"]
            if all(key in parsed_result for key in required_keys):
                q_lower = parsed_result["question"].lower()
                enumeration_words = [
                    "what are",
                    "list",
                    "enumerate",
                    "approaches",
                    "methods",
                    "types",
                    "ways",
                    "features",
                    "challenges",
                    "impacts",
                    "strategies",
                ]
                if any(word in q_lower for word in enumeration_words):
                    print("  API call successful and JSON parsed.")
                    return parsed_result
                print(f"  Generated question does not look like an enumeration: {parsed_result['question']}")
            else:
                print(f"  API response missing required keys: {result_text}")
        except requests.exceptions.Timeout:
            print(f"  API call timed out on attempt {attempt + 1}.")
        except requests.exceptions.RequestException as exc:
            print(f"  API request failed on attempt {attempt + 1}: {exc}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  Failed to parse API response on attempt {attempt + 1}: {exc}")
        except Exception as exc:
            print(f"  Unexpected error during API call {attempt + 1}: {exc}")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print(f"  Failed to get a valid response after {max_retries} attempts.")
    return {
        "question": "Error generating question.",
        "answer": "[Error]",
        "reason": "Failed to receive or parse a valid API response after multiple attempts.",
        "titles": "N/A",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate single-topic QA pairs from news records.")
    parser.add_argument("--news-path", default="news_api.json", help="Input news JSON file.")
    parser.add_argument("--output-file", default="newsqa_single_topic.json", help="Output QA JSON file.")
    parser.add_argument("--num-pairs", type=int, default=20, help="Number of QA pairs to generate.")
    parser.add_argument("--delay", type=float, default=10.0, help="Delay between API calls in seconds.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=os.getenv("AURA_GENERATE_MODEL", "gpt-4o"), help="Model name.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    all_news_data = load_news(args.news_path)
    print(f"Starting single-topic QA generation for {len(all_news_data)} news items...")
    print(f"Target number of QA pairs: {args.num_pairs}")

    if not os.path.exists(args.output_file):
        existing_qa_list = []
        print(f"Creating new output file: {args.output_file}")
    else:
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                existing_qa_list = json.load(f)
            print(f"Loaded existing QA pairs from {args.output_file}. Current count: {len(existing_qa_list)}")
        except (json.JSONDecodeError, FileNotFoundError):
            existing_qa_list = []
            print(f"Could not load {args.output_file}; starting fresh.")

    generated_count = 0
    for index in range(args.num_pairs):
        print(f"\n--- Generating single-topic QA pair {index + 1}/{args.num_pairs} ---")
        if index > 0 and args.delay > 0:
            print(f"  Waiting for {args.delay} seconds before next API call...")
            time.sleep(args.delay)

        qa_dict = generate_qa_single_topic(all_news_data, args.api_base, args.api_key, args.model)
        existing_qa_list.append(qa_dict)
        generated_count += 1

        if (index + 1) % 5 == 0 or (index + 1) == args.num_pairs:
            try:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(existing_qa_list, f, ensure_ascii=False, indent=2)
                print(f"  Progress saved to {args.output_file}. Total QA pairs: {len(existing_qa_list)}")
            except Exception as exc:
                print(f"  Error saving to file: {exc}")

    print("\n" + "=" * 50)
    print("Single-topic generation complete.")
    print(f"Newly generated QA pairs: {generated_count}")
    print(f"Total QA pairs in {args.output_file}: {len(existing_qa_list)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
