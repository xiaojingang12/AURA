import argparse
import json

def load_json_file(file_path: str) -> any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_qa_keyword_mapping(errors_file_path: str, output_file_path: str):
    """Generate a QA-to-keyword mapping from keyword validation errors."""
    errors_data = load_json_file(errors_file_path)
    
    errors_by_qa = errors_data.get("errors_by_qa", {})
    
    qa_keyword_mapping = []
    
    for qa_id, errors_list in errors_by_qa.items():
        if errors_list:  
            question_text = errors_list[0]["question"]
            
            keywords = [error["keyword"] for error in errors_list]
            
            mapping_item = {
                "question": question_text,
                "keywords": keywords
            }
            
            qa_keyword_mapping.append(mapping_item)
    
    print(f"Saving mapping data to {output_file_path}...")
    save_json_file(qa_keyword_mapping, output_file_path)
    
    print(f"Generated mappings for {len(qa_keyword_mapping)} QA pairs")
    
    total_keywords = sum(len(item["keywords"]) for item in qa_keyword_mapping)
    print(f"Total keywords: {total_keywords}")

def main():
    parser = argparse.ArgumentParser(description="Generate QA keyword mappings from validation errors.")
    parser.add_argument("--errors-path", required=True, help="Input validation errors JSON file.")
    parser.add_argument("--output-path", required=True, help="Output mapping JSON file.")
    args = parser.parse_args()
    
    generate_qa_keyword_mapping(args.errors_path, args.output_path)

if __name__ == "__main__":
    main()



