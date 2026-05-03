import argparse
import os
import re
import time
import json
import requests

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("AURA_REFINE_MODEL", "gpt-4o")

def refined(question, answer):
    sys_prompt = '''
        You are an expert in pedagogy and Question-Answer design. Your primary skill is to analyze an existing question in conjunction with a list of expected answer topics, in order to clarify the question's underlying purpose and scope. Your goal is to refine the question so that it elicits a comprehensive and well-structured answer that implicitly covers all the provided topics, while maintaining a single, holistic, purpose-driven, and non-leading form. You will return your analysis as a single, valid JSON object.

**Core Instructions:**

1. **Role**  
   Act as an analytical expert who refines questions by clarifying their inherent purpose, guided by a list of target answer topics.

2. **Input**  
   You will be provided with two pieces of information:
   - `Original Question`: The initial question that needs evaluation and refinement.
   - `Topics_List`: A list of key topics or aspects that a comprehensive answer should cover. This list serves as a guide to determine the expected scope and depth of the answer.

3. **Analytical & Refinement Logic**

   **Step 1: Analyze the Original Question and Evaluate It Against Topics_List.**
   - Analyze the `Original Question`: identify its central subject(s), current implied purpose, and initial scope.
   - Analyze the `Topics_List`: understand the full breadth and depth of information expected in a complete answer by considering all items in the list. Infer the relationships between these topics (for example: components, stages, causes, effects, comparisons, applications, or evaluations).
   - Identify Gaps: compare the original question’s implied purpose and scope with the comprehensive coverage implied by the `Topics_List`. Determine where the original question falls short in implicitly drawing out all necessary topics.

   **Step 2: Determine the Singular, Holistic Purpose That Encompasses the Topics.**
   - Based on your analysis in Step 1, identify the most appropriate and singularly focused holistic purpose for the refined question.
   - This holistic purpose must implicitly guide an answerer to cover all items in the `Topics_List` without explicitly mentioning each topic within the question itself.
   - Your goal is to transform the original question from any ambiguous phrasing into a clear, unified statement that expresses one overarching inquiry, ensuring this inquiry’s scope naturally includes the provided `Topics_List`.

   **Step 3: Enforce the Non-Leading Constraint.**
   The `Refined_Question` must not contain prompt-like, hinting, or leading content that reveals the expected answer structure or the hidden topics list.

   Specifically:
   - Do **not** explicitly list, enumerate, or restate the items from `Topics_List`.
   - Do **not** use checklist-like phrasing that signals what the answer should cover.
   - Do **not** include overt guiding phrases such as:
     - “including ...”
     - “such as ...”
     - “from the perspectives of ...”
     - “covering ...”
     - “in terms of ...”
     - “with emphasis on ...”
     - “discuss X, Y, and Z”
   - Do **not** make the question sound like an instruction for answer organization.
   - The question should sound natural, concise, neutral, and user-like, rather than like a meta-prompt or rubric.
   - The question may be broader or clearer than the original, but it must remain a single, standalone question with no visible hints about the expected subtopics.

   **Step 4: Generate Refined Question.**
   Construct a `Refined_Question` that:
   - embodies the singular, holistic purpose identified in Step 2,
   - remains a single, unified question,
   - does not explicitly mention the individual items in `Topics_List`,
   - and does not contain leading or suggestive wording that gives away the answer structure.

   **Step 5: Generate Justification.**
   Create a concise `Reason` that explains:
   - why the original question was insufficient in scope, clarity, or purpose relative to the `Topics_List`,
   - and how the refined question improves it by forming a unified, overarching, and non-leading inquiry that implicitly covers the required topics.

4. **Crucial Rules**
   - The `Refined_Question` must be a **single question**, not multiple questions.
   - The `Refined_Question` must **not** explicitly list the answer dimensions or subtopics.
   - The `Refined_Question` must **not** contain hinting or prompt-like scaffolding.
   - The `Reason` may refer to scope, clarity, completeness, and implicit coverage, but should remain concise.
   - Preserve the original subject and intent as much as possible while improving the question.

5. **Output Format (Strict)**
   - Your entire output must be a **single, valid JSON object**.
   - Do not include any text outside the JSON structure.
   - The JSON object must contain exactly the following four keys:
     1. `"Current_Question"`: The original question text from the input.
     2. `"Topics_List"`: The original topics list from the input.
     3. `"Refined_Question"`: The improved question text you generated.
     4. `"Reason"`: The justification you generated.

**Output Format:**

{
    "Current_Question": "String",
    "Topics_List": ["String", "String"],
    "Refined_Question": "String",
    "Reason": "String"
}
    '''

    prompt = f'''
        Refine the following question based on the provided reason and return the result strictly in the JSON format defined in your instructions.

        ---

        ### Original Question:
        {question}
        ### Topics_List:
        {answer}
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
                "Refined_Question": json_res["Refined_Question"],
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
                "Refined_Question": "N/A",
                "Reason": "N/A",
                "ori_json_res": "N/A",
            },
        )
    
    return res_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Refine questions based on expected answer topics.")
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
            answer=current_answer, 
        )

        results_for_json.append({
            'question': res_dict.get("Refined_Question"),
            'before_question': res_dict.get("Current_Question"),
            'answer': item.get("answer"),
            'reason': item.get("reason"),
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
