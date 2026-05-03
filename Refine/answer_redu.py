import argparse
import os
import re
import time
import json
import requests
from tqdm import tqdm

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("AURA_REFINE_MODEL", "gpt-4o")

def refined(question, answer, evidence):
    sys_prompt = '''
        You are an expert in information relevance analysis and content refinement. Your primary skill is to evaluate a given topics list against provided evidence and corpus, identifying and removing only those topics that are genuinely redundant, irrelevant, or unsupported. Your goal is to produce a refined, high-quality topics list that maintains maximum coverage while eliminating true redundancies. You will return your analysis as a single, valid JSON object.

        **Core Instructions:**

        1.  **Role**: Act as a Topics Refinement expert who carefully evaluates topics lists against evidence and corpus to remove only genuinely problematic topics while preserving valuable content.

        2.  **Input**: You will be provided with:
            -   `Question`: The original question being addressed.
            -   `Topics_List`: The initial list of topics that should be covered when answering the question.
            -   `Evidence`: Specific evidence, facts, or information relevant to the question.
           
        3.  **Important Guiding Principle - Conservative Refinement**:
            -   **Default Assumption**: The original `Topics_List` is generally well-constructed and accurate. Most topics are likely relevant and should be retained.
            -   **High Bar for Removal**: Only remove topics that meet strict criteria for removal (detailed below). When in doubt, **keep the topic**.
            -   **Preservation Priority**: Your goal is refinement, not aggressive reduction. It is better to keep a marginally relevant topic than to remove a potentially valuable one.
            -   **Expected Outcome**: In most cases, you should retain the majority of the original topics. Removing more than 30% of topics should be rare and only when there are clear, significant issues.

        4.  **Analytical & Refinement Logic**:
            -   **Step 1: Understand the Context.**
                *   **Analyze the Question**: Understand what is being asked and what scope of answer is expected.
                *   **Review Evidence**: Examine the provided evidence to understand what specific information is available.
                *   **Review Topics_List**: Understand the coverage and intent of each topic in the original list.
            
            -   **Step 2: Evaluate Each Topic Against Strict Removal Criteria.**
                
                For each topic in `Topics_List`, evaluate whether it should be **removed** based on the following strict criteria. A topic should ONLY be removed if it clearly meets at least one of these criteria:
                
                **Removal Criterion 1: Complete Irrelevance**
                -   The topic has **no logical connection** to the question whatsoever.
                -   The topic addresses a completely different subject matter that does not help answer the question in any way.
                -   **Standard**: The topic is obviously off-topic to any reasonable person.
                
                **Removal Criterion 2: Zero Support in Evidence**
                -   The topic **cannot be addressed at all** based on the provided evidence.
                -   There is **no information available** in the evidence that would allow discussing this topic, even briefly.
                -   **Standard**: It would be impossible to write even a single meaningful sentence about this topic using the available information.
                -   **Note**: If there is even minimal information available, keep the topic.
                
                **Removal Criterion 3: True Redundancy (Duplicate)**
                -   The topic is **essentially identical** to another topic in the list, just worded differently.
                -   The two topics would result in **completely overlapping content** with no distinct value.
                -   **Standard**: The topics are near-perfect synonyms in this context, not just related or overlapping.
                -   **Important**: Topics that are related but address different aspects (e.g., "Advantages" vs "Applications") are NOT redundant.
                
                **Removal Criterion 4: Out of Scope**
                -   The topic goes **significantly beyond** what the question is asking for.
                -   Including this topic would make the answer unfocused or tangential.
                -   **Standard**: The topic clearly belongs to a different, broader question.

            -   **Step 3: Generate Refined Topics List.**
                -   Create the `Refined_Topics_List` containing all topics that **passed** the evaluation (were NOT removed).
                -   Maintain the original order of topics where possible.
                -   The refined list should typically contain **70-100% of the original topics** unless there are significant quality issues.
            
            -   **Step 4: Generate Justification (Reason).**
                -   Explain your refinement decisions clearly and concisely.
                -   Your justification should:
                    *   State how many topics were retained vs. removed (e.g., "Retained 7 out of 8 topics").
                    *   Specifically identify which topics (if any) were removed and **exactly why** they met the strict removal criteria.
                    *   If no topics were removed, explain why the original list was already well-constructed and aligned with the evidence.
                    *   Acknowledge the overall quality of the original topics list.
                -   **Tone**: Your reason should be analytical and respectful of the original topics list, not overly critical.

        5.  **Language Consistency Rule**:
            -   **All output content (refined topics and reason) MUST be in the SAME language as the input `Question` and `Topics_List`.**
            -   Maintain language consistency throughout the JSON output.

        6.  **Output Format (Strict)**:
            -   Your entire output **MUST BE a single, valid JSON object**. Do not include any text outside the JSON structure.
            -   The JSON object MUST contain exactly four keys:
                1.  `"question"`: The value must be the original question from the input.
                2.  `"original_topics_list"`: The value must be the original topics list from the input.
                3.  `"refined_topics_list"`: The value must be the refined topics list you generated after evaluation.
                4.  `"reason"`: The value must be your justification explaining the refinement decisions.

        **Output Format:**

        Your JSON output must conform to the following structure:

        {
            "Current_Question": String, // The original question text from the input.
            "Current_Answer": Array of Strings, // The original topics list from input.
            "Refined_Answer": Array of Strings, // The refined topics list after your evaluation.
            "Deleted_Answer": Array of Strings, // The list of removed topics.
            "Reason": String // Your justification explaining what was refined and why, with specific reference to the removal criteria.
        }

        **Quality Standards:**

        -   **Preservation**: The refined list should retain all valuable and relevant topics from the original list.
        -   **Precision**: Only topics that clearly meet strict removal criteria should be removed.
        -   **Justification Clarity**: The reason must clearly explain any removals with specific reference to which removal criterion was met.
        -   **Respect for Input**: Acknowledge that the original topics list is generally well-constructed; your role is refinement, not reconstruction.
        -   **Consistency**: Maintain language consistency and logical coherence in the output.
        '''

    prompt = f'''
        Refine the following topics list by evaluating it against the provided question, evidence. Remove only those topics that clearly meet the strict removal criteria defined in your instructions. Remember: the original topics list is generally well-constructed, so retain most topics unless there are clear issues. Return the result strictly in the JSON format defined in your instructions.

        ---

        ### Question:
        {question}

        ### Original Topics List:
        {answer}

        ### Evidence:
        {evidence}
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
        # "max_tokens": 2048,
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

            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                retries += 1
                time.sleep(2)
                continue

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
                "Current_Answer": json_res["Current_Answer"],
                "Refined_Answer": json_res["Refined_Answer"],
                "Deleted_Answer": json_res["Deleted_Answer"],
                "Reason": json_res["Reason"],
                "ori_json_res": json_res,
            }

            success = True

        except Exception as e:
            print(f"Error parsing JSON response from OpenAI, Error: {e}.")
            print(f"Raw result: {result}") 
            retries += 1
            time.sleep(2)
            continue

    if not success:
        print("Failed to get response from OpenAI.")
        return {
            "Current_Question": question,
            "Current_Answer": answer,
            "Refined_Answer": answer,
            "Deleted_Answer": [],
            "Reason": "API call failed after retries",
            "ori_json_res": {},
        }
    
    return res_dict



def parse_args():
    parser = argparse.ArgumentParser(description="Remove unsupported or redundant answer topics using evidence.")
    parser.add_argument("--qa-path", required=True, help="Input QA JSON file.")
    parser.add_argument("--evidence-path", required=True, help="Evidence/corpus JSON file.")
    parser.add_argument("--output-path", required=True, help="Output JSON file for refined QA records.")
    parser.add_argument("--reason-output-path", required=True, help="Output JSON file for refinement reasons.")
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
    evidence_path = args.evidence_path

    save_path = args.output_path
    results_for_json = [] 

    reson_json = []
    save_path_reason = args.reason_output_path


    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: '{qa_path}'")
        exit(1) 
    except json.JSONDecodeError:
        print(f"Error: '{qa_path}' is not a valid JSON file.")
        exit(1)

    try:
        with open(evidence_path, 'r', encoding='utf-8') as f:
            evidence_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Corpus file not found: '{evidence_path}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{evidence_path}' is not a valid json file.")
        exit(1)

    corpus_map = {}
    for doc in evidence_data:
        doc_id = doc.get('total_id')
        if doc_id is not None:
            corpus_map[doc_id] = doc.get('context', '')

    for i, item in tqdm(enumerate(qa_list), total=len(qa_list), desc="questions"):

        # if i >= 3: 
        #     print(f"Broke loop after processing {i} items (first 10).")
        #     break

        current_question = item.get("question")
        current_answer = item.get("answer")

        if current_question is None:
            print(f"Warning: No question found, Skipping {i}th question.")
            continue

        evidence_list = item.get("evidence_list", [])

        evidence_texts = []
        for evidence in evidence_list:
            evidence_id = evidence.get("total_id")
            if evidence_id in corpus_map:
                evidence_texts.append(corpus_map[evidence_id])
        
        res_dict = refined(
            question=current_question, 
            answer=current_answer, 
            evidence=evidence_texts
        )

        reson_json.append({
            'question': current_question,
            'answer': current_answer,
            'refined_answer': res_dict.get("Refined_Answer"),
            'deleted_answer': res_dict.get("Deleted_Answer"),
            'refined_reason': res_dict.get("Reason"),
        })

        results_for_json.append({
            'question': current_question,
            'answer': res_dict.get("Refined_Answer"),
            'evidence_list': item.get("evidence_list"),
            'common_errors': list(set(item.get("common_errors", []) + res_dict.get("Deleted_Answer"))),
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

    if reson_json:
        try:
            with open(save_path_reason, 'w', encoding='utf-8') as jsonfile:
                json.dump(reson_json, jsonfile, indent=4, ensure_ascii=False)
            
            print(f"\nSuccessfully saved results to {save_path_reason}")

        except IOError as e:
            print(f"\nError writing to JSON file: {e}")
