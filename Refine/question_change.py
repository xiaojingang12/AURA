import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_file(filepath: str) -> list:

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"Expected a list in {filepath}, but got {type(data)}.")
            raise TypeError(f"Data in {filepath} is not a list.")
        logger.info(f"Successfully loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}")
        raise

def save_json_file(filepath: str, data: list):

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved modified data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        raise

def replace_questions(a_data: list, b_data: list) -> list:


    replacement_map = {}
    for item_a in a_data:
        before_q = item_a.get('before_question')
        after_q = item_a.get('question')
        if before_q is not None and after_q is not None:

            replacement_map[before_q] = after_q
        elif before_q is None:
             logger.warning(f"Item in a.json missing 'before_question': {item_a}")


    logger.info(f"Created replacement map with {len(replacement_map)} entries.")


    modified_b_data = []
    replaced_count = 0
    for item_b in b_data:
        original_question = item_b.get('question')
        if original_question in replacement_map:

            item_b['question'] = replacement_map[original_question]
            replaced_count += 1
            logger.debug(f"Replaced question: '{original_question}' -> '{replacement_map[original_question]}'")

        modified_b_data.append(item_b) 

    logger.info(f"Finished processing b.json. Replaced {replaced_count} questions.")
    return modified_b_data

def parse_args():
    parser = argparse.ArgumentParser(description="Replace questions in one QA file using a before/after mapping file.")
    parser.add_argument("--mapping-path", required=True, help="JSON file containing before_question and question fields.")
    parser.add_argument("--qa-path", required=True, help="QA JSON file whose question fields will be replaced.")
    parser.add_argument("--output-path", required=True, help="Output JSON file.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        logger.info("Starting the question replacement process...")
        a_data = load_json_file(args.mapping_path)
        b_data = load_json_file(args.qa_path)
        modified_b_data = replace_questions(a_data, b_data)
        save_json_file(args.output_path, modified_b_data)
        logger.info("Process completed successfully.")
    except Exception as e:
        logger.critical(f"An error occurred during the process: {e}")


if __name__ == "__main__":
    main()



