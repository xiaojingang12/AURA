import argparse
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_common_errors(qa_records, keyword_mapping):
    keyword_dict = {}
    for item in keyword_mapping:
        truncated_question = item["question"].rstrip(".").strip()
        keyword_dict[truncated_question] = item["keywords"]

    for item in qa_records:
        question = item["question"]

        if question in keyword_dict:
            item["common_errors"] = keyword_dict[question]
            continue

        found_match = False
        for truncated_q in keyword_dict:
            if question.startswith(truncated_q):
                item["common_errors"] = keyword_dict[truncated_q]
                found_match = True
                break

        if found_match:
            continue

        best_match = None
        max_common_prefix = 0
        for truncated_q in keyword_dict:
            common_length = 0
            min_len = min(len(question), len(truncated_q))
            for index in range(min_len):
                if question[index] == truncated_q[index]:
                    common_length += 1
                else:
                    break
            if common_length > max_common_prefix and common_length >= len(truncated_q) * 0.8:
                max_common_prefix = common_length
                best_match = truncated_q

        if best_match:
            item["common_errors"] = keyword_dict[best_match]

    return qa_records


def parse_args():
    parser = argparse.ArgumentParser(description="Attach common error keywords to QA records.")
    parser.add_argument("--qa-path", required=True, help="Input QA JSON file.")
    parser.add_argument("--mapping-path", required=True, help="QA keyword mapping JSON file.")
    parser.add_argument("--output-path", required=True, help="Output QA JSON file.")
    return parser.parse_args()


def main():
    args = parse_args()
    qa_records = load_json(args.qa_path)
    keyword_mapping = load_json(args.mapping_path)
    updated_records = add_common_errors(qa_records, keyword_mapping)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, ensure_ascii=False, indent=2)

    matched_count = sum(1 for item in updated_records if "common_errors" in item)
    print(f"Output file: {args.output_path}")
    print(f"Matched records: {matched_count}")
    print(f"Unmatched records: {len(updated_records) - matched_count}")

    print("\nSample results:")
    for index, item in enumerate(updated_records[:3]):
        print(f"\nQuestion {index + 1}: {item['question'][:80]}...")
        if "common_errors" in item:
            print(f"common_errors: {item['common_errors']}")
        else:
            print("common_errors: no matching keywords found")


if __name__ == "__main__":
    main()
