import argparse
import os
import re
import time
import json
import requests

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("AURA_REFINE_MODEL", "gpt-4o")

def refined(question):
    sys_prompt = '''
        You are an expert topic analyzer. Your primary skill is to analyze given questions and generate a comprehensive list of relevant topics that a complete answer should cover. Your goal is to identify all key aspects, themes, and dimensions that need to be addressed to provide a thorough and well-structured answer to the question. You will return your analysis as a single, valid JSON object.

        **Core Instructions:**

        1.  **Role**: Act as a Topic Analysis expert who identifies all relevant topics that should be covered when answering a given question comprehensively.

        2.  **Input**: You will be provided with:
            -   `Question`: The user's question that needs topic analysis.

        3.  **Analytical & Generation Logic**:
            -   **Step 1: Analyze the Question.**
                *   **Identify Core Subject**: Determine what the question is fundamentally asking about.
                *   **Identify Question Type**: Recognize whether the question is asking for definition, explanation, comparison, process, evaluation, solution, or analysis.
                *   **Determine Expected Scope**: Based on the question type and subject, infer what aspects a complete answer should cover.
                *   **Consider Multiple Dimensions**: Think about the question from different angles - theoretical, practical, technical, conceptual, comparative, historical, future-oriented, etc.
            
            -   **Step 2: Generate Comprehensive Topics List.**
                *   Generate a comprehensive list of topics that a complete answer should address.
                *   **Topic Generation Principles**:
                    *   **Completeness**: The topics list should cover all major aspects needed for a thorough answer. Consider what would be missing if any topic were omitted.
                    *   **Logical Organization**: Topics should follow a logical flow (e.g., definition → components → principles → applications → challenges).
                    *   **Appropriate Granularity**: Topics should be neither too broad (e.g., "Everything about X") nor too narrow (e.g., "The third step in sub-process Y").
                    *   **Direct Relevance**: Each topic must be directly relevant to answering the question, not tangentially related.
                *   **Common Topic Categories to Consider** (adapt based on question type):
                    *   **Foundational**: Definitions, basic concepts, background information
                    *   **Structural**: Components, elements, categories, classifications
                    *   **Functional**: How it works, processes, mechanisms, principles
                    *   **Comparative**: Similarities, differences, alternatives, comparisons
                    *   **Applied**: Applications, use cases, examples, implementations
                    *   **Evaluative**: Advantages, disadvantages, limitations, challenges
                    *   **Contextual**: History, current state, trends, future directions
                *   **Generate 5-10 topics** that collectively provide a complete framework for answering the question.
            
            -   **Step 3: Generate Justification (Reason).**
                *   Explain **why** you generated this specific set of topics.
                *   Your justification should:
                    *   Clearly state what type of question this is and what a complete answer requires.
                    *   Explain the logic behind the topics you selected - how they collectively address the question.
                    *   Describe the organizational principle or framework you used to structure the topics list.
                    *   Justify why these topics are sufficient and necessary for a comprehensive answer.

        4.  **Language Consistency Rule**:
            -   **All generated topics MUST be in the SAME language as the input `Question`.**
            -   If the question is in Chinese, generate all topics and reason in Chinese.
            -   If the question is in English, generate all topics and reason in English.
            -   Do NOT mix languages within the output.

        5.  **Output Format (Strict)**:
            -   Your entire output **MUST BE a single, valid JSON object**. Do not include any text outside the JSON structure.
            -   The JSON object MUST contain exactly three keys:
                1.  `"question"`: The value must be the original question from the input.
                2.  `"topics_list"`: The value must be an array of strings, each representing a topic that should be covered in the answer.
                3.  `"reason"`: The value must be a string explaining why you generated this specific topics list and how it addresses the question comprehensively.

        **Output Format:**

        Your JSON output must conform to the following structure:

        {
            "Current_Question": String, // The original question text from the input.
            "Related_Topics": Array of Strings, // The list of topics you generated that a complete answer should cover.
            "Reason": String // Your justification explaining the logic and completeness of the topics list.
        }

        **Quality Standards:**

        -   **Completeness**: The topics list should enable a comprehensive answer that leaves no major aspects unaddressed.
        -   **Coherence**: Topics should form a logically connected set that flows naturally.
        -   **Relevance**: Every topic must be necessary for answering the question; avoid including tangential or unnecessary topics.
        -   **Clarity**: Each topic should be clearly stated and immediately understandable.
        -   **Justification Quality**: The reason should clearly demonstrate your analytical thinking and explain the completeness of your topics list.
        '''

    prompt = f'''
        Analyze the following question and generate relevant topics strictly in the JSON format defined in your instructions.

        ---

        ### Question:
        {question}
    '''

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
        "response_format": { "type": "json_object" } 
    }
    
    max_retries = 3
    retries = 0
    success = False
    result = None
    res_dict = {}
    while not success and retries < max_retries:
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            result = response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            retries += 1
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)

        try:
            result = re.sub(r"\n+", "\n", result)

            json_res = json.loads(result)

            res_dict = {
                "Current_Question": json_res["Current_Question"],
                "Related_Topics": json_res["Related_Topics"],
                "Reason": json_res["Reason"],
                "ori_json_res": json_res,
            }

        except Exception as e:
            print(f"Error parsing JSON response from OpenAI, Error: {e}.")
            retries += 1
            continue

        success = True

    if not success:
        print("Failed to get response from OpenAI.")
        return (
            {
                "Current_Question": "N/A",
                "Related_Topics": "N/A",
                "Reason": "N/A",
                "ori_json_res": "N/A",
            },
        )
    
    return res_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate possible answer topics for QA records.")
    parser.add_argument("--qa-path", required=True, help="Input QA JSON file.")
    parser.add_argument("--output-path", required=True, help="Output JSON file.")
    parser.add_argument("--api-base", default=BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=API_KEY, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=MODEL, help="Model name.")
    return parser.parse_args()


def main():
    global BASE_URL, API_KEY, MODEL
    args = parse_args()
    BASE_URL = args.api_base
    API_KEY = args.api_key
    MODEL = args.model

    if not API_KEY:
        raise SystemExit("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    qa_path = args.qa_path

    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: '{qa_path}'")
        exit(1) 
    except json.JSONDecodeError:
        print(f"Error: '{qa_path}' is not a valid JSON file.")
        exit(1)

    save_path = args.output_path
    results_for_json = [] 

    for i, item in enumerate(qa_list):

        # if i >= 3: 
        #     print(f"Broke loop after processing {i} items (first 10).")
        #     break

        current_question = item.get("question")
        current_answer = item.get("answer")

        if current_question is None:
            print(f"Warning: No question found, Skipping {i}th question.")
            continue
        
        res_dict = refined(
            question=current_question, 
        )

        results_for_json.append({
            'question': current_question,
            'answer': current_answer,
            'possible_topics': res_dict.get("Related_Topics"),
            'refined_reason': res_dict.get("Reason"),
        })

    if results_for_json:
        try:
            with open(save_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(results_for_json, jsonfile, indent=4, ensure_ascii=False)
            
            print(f"\nSuccessfully saved results to {save_path}")

        except IOError as e:
            print(f"\nError writing to JSON file: {e}")


if __name__ == "__main__":
    main()
