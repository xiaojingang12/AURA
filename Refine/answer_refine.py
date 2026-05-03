import argparse
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import requests
from sklearn.cluster import KMeans
import warnings
import re
import os


QA_FILE = os.getenv("AURA_QA_FILE", "")
CORPUS_FILE = os.getenv("AURA_CORPUS_FILE", "")
OUTPUT_FILE = os.getenv("AURA_OUTPUT_FILE", "topic_gap_results.json")
CLUSTER_OUTPUT_FILE = os.getenv("AURA_CLUSTER_OUTPUT_FILE", "cluster_to_questions_mapping.json")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_EMBED_BASE_URL", ""))
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "")
NUM_CLUSTERS = 10
TOP_K_EVIDENCE = 5
CHUNK_SIZE = 600


def get_ollama_embedding(text: str) -> List[float]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_EMBED_MODEL,
        "input": text
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        embeddings = result.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        else:
            print(f"Warning: unexpected Ollama embedding API response structure: {result}")
            raise ValueError("Failed to get embeddings from the Ollama API response")
    except requests.exceptions.RequestException as e:
        print(f"Error while fetching embeddings: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while processing the Ollama embedding response: {e}")
        raise

def cluster_questions(qa_data: List[Dict], num_clusters: int) -> Tuple[np.ndarray, Dict[str, int], Dict[int, List[Dict]]]:

    
    questions = [item['question'] for item in qa_data]
    qa_ids = []
    for item in qa_data:
        ev_list = item.get('evidence_list', [])
        if ev_list and isinstance(ev_list, list) and len(ev_list) > 0:
            first_ev = ev_list[0]
            if isinstance(first_ev, dict):
                qid = first_ev.get('id')
                if qid is not None:
                    qa_ids.append(str(qid))
                else:
                    print("  Warning: first evidence item is missing an 'id' field; using the QA index.")
                    qa_ids.append(f"qa_idx_{len(qa_ids)}")
            else:
                print("  Warning: first evidence item is not a dict; using the QA index.")
                qa_ids.append(f"qa_idx_{len(qa_ids)}")
        else:
            print("  Warning: evidence_list is empty or missing; using the QA index.")
            qa_ids.append(f"qa_idx_{len(qa_ids)}")

    print(f"Generating embeddings for {len(questions)} questions...")
    embeddings_list = []
    for i, question in enumerate(questions):
        try:
            embedding = get_ollama_embedding(question)
            embeddings_list.append(embedding)
            if (i + 1) % 50 == 0 or (i + 1) == len(questions):
                 print(f"Encoded {i+1}/{len(questions)} questions...")
        except Exception as e:
             print(f"Failed to encode question {i+1} (id: {qa_ids[i]}): {e}")
             raise

    embeddings_array = np.array(embeddings_list)
    print(f"Embedding generation complete. Shape: {embeddings_array.shape}")

    print(f"Running standard KMeans clustering with {num_clusters} clusters...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans_model.fit_predict(embeddings_array)
    
    qa_id_to_cluster = {qa_id: int(label) for qa_id, label in zip(qa_ids, cluster_labels)}
    cluster_to_qa_items = defaultdict(list)
    for qa_item, cluster_label, qid in zip(qa_data, cluster_labels, qa_ids):
        cluster_to_qa_items[int(cluster_label)].append(qa_item)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))
    print(f"Cluster sizes: {cluster_stats}")
    
    print("Question clustering complete.")
    return cluster_labels, qa_id_to_cluster, dict(cluster_to_qa_items)

def load_corpus(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        print(f"Loaded corpus with {len(corpus_data)} documents")
        
        total_id_to_doc = {}
        for doc in corpus_data:
            total_id = doc.get('total_id')
            if total_id is not None:
                total_id_to_doc[str(total_id)] = doc
            else:
                print(f"Warning: corpus document is missing 'total_id': {doc.get('title', 'Unknown Title')[:50]}...")
                
        print(f"Built {len(total_id_to_doc)} total_id-to-document mappings.")
        return total_id_to_doc
    except Exception as e:
        print(f"Failed to load corpus file {file_path}: {e}")
        raise

def find_relevant_evidence_for_cluster(cluster_qa_items: List[Dict], total_id_to_doc: Dict[str, Any]) -> List[Dict]:
    all_evidence_total_ids = set()
    for qa_item in cluster_qa_items:
        evidence_list = qa_item.get('evidence_list', [])
        for evidence in evidence_list:
            tid = evidence.get('total_id')
            if tid is not None:
                all_evidence_total_ids.add(str(tid))
    
    relevant_evidence = []
    for total_id in all_evidence_total_ids:
        doc = total_id_to_doc.get(total_id)
        if doc:
            text_content = doc.get('context', '') 
            if not text_content:
                 print(f"  Warning: corpus document total_id='{total_id}' has an empty 'context' field.")
                 continue
            if len(text_content) > CHUNK_SIZE:
                 num_chunks = (len(text_content) + CHUNK_SIZE - 1) // CHUNK_SIZE
                 for i in range(num_chunks):
                     start_idx = i * CHUNK_SIZE
                     end_idx = min((i + 1) * CHUNK_SIZE, len(text_content))
                     chunk_text = text_content[start_idx:end_idx]
                     relevant_evidence.append({
                         'total_id': total_id,
                         'chunk_index': i,
                         'text': chunk_text,
                         'source_document': doc
                     })
            else:
                 relevant_evidence.append({
                     'total_id': total_id,
                     'text': text_content,
                     'source_document': doc
                 })
        else:
            print(f"  Warning: total_id='{total_id}' was not found in the corpus.")

    return relevant_evidence[:TOP_K_EVIDENCE]

def query_ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_CHAT_MODEL) -> str:

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get('message', {}).get('content', '')
    except Exception as e:
        print(f"Failed to call the Ollama Chat API: {e}")
        return ""

def detect_topic_gaps_using_cluster_and_evidence(
    qa_data: List[Dict], 
    qa_id_to_cluster: Dict[str, int], 
    cluster_to_qa_items: Dict[int, List[Dict]], 
    total_id_to_doc: Dict[str, Any]
) -> List[Dict]:
    results = []
    topic_patterns = {
        "asset_creation": [
            r"asset.*creation", r"3d.*asset", r"model.*creation", r"texturing", r"mesh.*generation",
            r"procedural.*generation", r"content.*creation", r"asset.*pipeline"
        ],
        "data_sourcing": [
            r"data.*sourc", r"real.*world.*scan", r"lidar.*data", r"sensor.*data", 
            r"photogrammetry", r"scanned.*data", r"synthetic.*data", r"real.*data"
        ],
        "scene_construction": [
            r"scene.*construction", r"scene.*building", r"environment.*creation", 
            r"world.*building", r"scene.*generation", r"level.*design"
        ],
        "simulation_approaches": [
            r"game.*based", r"world.*based", r"physics.*engine", r"simulator.*type",
            r"simulation.*method", r"approach.*simulation"
        ],
        "realism": [
            r"realism", r"fidelity", r"real.*world.*rep", r"visual.*quality", 
            r"photo.*realistic", r"photorealism", r"authenticity"
        ],
        "transfer": [
            r"sim.*to.*real", r"real.*to.*sim", r"domain.*transfer", r"sim2real",
            r"generalization", r"cross.*domain"
        ],
        "performance": [
            r"performance", r"efficiency", r"optimization", r"speed", r"latency",
            r"computational.*cost", r"resource.*usage", r"rendering.*time"
        ],
        "accuracy": [
            r"accuracy", r"precision", r"error.*rate", r"measurement.*accuracy",
            r"spatial.*accuracy", r"detection.*accuracy"
        ]
    }

    def extract_topics_from_text(text: str) -> List[str]:
        if isinstance(text, list):
            text_str = ' '.join([str(item) for item in text])
        elif not isinstance(text, str):
            text_str = str(text)
        else:
            text_str = text
        
        topics = set()
        text_lower = text_str.lower()
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    topics.add(topic)
                    break
        return list(topics)

    print(f"Detecting topic gaps for {len(qa_data)} QA pairs using clusters and evidence...")
    for i, qa in enumerate(qa_data):
        current_qa_id = None
        ev_list = qa.get('evidence_list', [])
        if ev_list and isinstance(ev_list, list) and len(ev_list) > 0:
            first_ev = ev_list[0]
            if isinstance(first_ev, dict):
                current_qa_id = first_ev.get('id')
        
        if current_qa_id is None:
            print(f"Processing QA pair {i+1}/{len(qa_data)}... skipped because no ID could be determined")
            continue
            
        print(f"Processing QA pair {i+1}/{len(qa_data)} (ID: {current_qa_id})...")
        
        target_question = qa['question']
        target_answer = qa['answer']
        
        cluster_label = qa_id_to_cluster.get(str(current_qa_id))
        if cluster_label is None:
            print(f"  Warning: no cluster label found for QA ID {current_qa_id}; skipping.")
            continue

        cluster_qa_items = cluster_to_qa_items.get(cluster_label, [])
        if len(cluster_qa_items) <= 1:
            print(f"  Warning: QA ID {current_qa_id} is the only item in its cluster; skipping.")
            continue
        target_topics = set(extract_topics_from_text(target_answer))

        all_cluster_topics = set()
        cluster_topic_info = defaultdict(list) 
        for cluster_qa in cluster_qa_items:

            cluster_qa_id_val = None
            cluster_ev_list = cluster_qa.get('evidence_list', [])
            if cluster_ev_list and isinstance(cluster_ev_list, list) and len(cluster_ev_list) > 0:
                first_cluster_ev = cluster_ev_list[0]
                if isinstance(first_cluster_ev, dict):
                    cluster_qa_id_val = first_cluster_ev.get('id')
            
            if cluster_qa_id_val is not None and str(cluster_qa_id_val) != str(current_qa_id):
                cluster_qa_answer = cluster_qa.get('answer', '')
                topics_in_cluster_qa = extract_topics_from_text(cluster_qa_answer)
                all_cluster_topics.update(topics_in_cluster_qa)
                
  
                if isinstance(cluster_qa_answer, list):
                    cluster_qa_answer_str = ' '.join([str(item) for item in cluster_qa_answer])
                elif not isinstance(cluster_qa_answer, str):
                    cluster_qa_answer_str = str(cluster_qa_answer)
                else:
                    cluster_qa_answer_str = cluster_qa_answer
                
                sentences = re.split(r'[.!?]+', cluster_qa_answer_str)
                for sentence in sentences:
                    sentence = sentence.strip()
                    for topic in topics_in_cluster_qa:
                         if any(re.search(pattern, sentence.lower()) for pattern in topic_patterns[topic]):
                             cluster_topic_info[topic].append(sentence)


        missing_topics = all_cluster_topics - target_topics
        
        if missing_topics:
            print(f"  Missing topics found: {missing_topics}")
            

            relevant_evidence_for_cluster = find_relevant_evidence_for_cluster(cluster_qa_items, total_id_to_doc)
            

            prompt_parts = [
                f"""You are an AI assistant tasked with analyzing the informational completeness of a question-answer pair.
                  Current question: {target_question}
                  Current answer: {target_answer}
                  The evidence documents related to this question contain the following topics, which are not mentioned in the current answer: {list(missing_topics)}.
                  Relevant evidence excerpts from the corpus: {json.dumps(relevant_evidence_for_cluster, ensure_ascii=False, indent=2)[:1000]}...
                  Please carefully analyze the current answer and the missing thematic information in the associated evidence.
                  Determine whether the current answer genuinely needs to be supplemented with important information related to these missing topics.
                  Consider the following points:
                  Are the missing topics directly relevant to the current question?
                  Does the current answer already address the core requirements of the question?
                  Are the missing topics essential components of the question, or are they merely related but non-essential information?
                  Is the current answer already sufficiently complete and accurate?
                  Respond with 'YES' if the current answer indeed requires supplementation with important information from the missing topics.
                  Otherwise, respond with 'NO' to indicate that the current answer is already sufficiently complete.
                  If you answer 'YES', please briefly explain why the information from these missing topics is important for the current answer.
                  If you answer 'NO', please briefly explain why these missing topics are not essential components of the current answer. """   
            ]
            prompt = "\n".join(prompt_parts)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                ollama_response = query_ollama_chat(messages)
                print(f"  Ollama response: {ollama_response[:100]}...")
                
                response_lower = ollama_response.lower()
                needs_addition = False
                if "yes" in response_lower and "no" not in response_lower:
                    needs_addition = True
                elif "no" in response_lower and "yes" not in response_lower:
                    needs_addition = False
                elif ("yes" in response_lower and "no" in response_lower) or ("yes" not in response_lower and "no" not in response_lower):
                    if any(keyword in response_lower for keyword in ["need", "important", "should", "indeed", "necessary"]):
                        needs_addition = True
                    elif any(keyword in response_lower for keyword in ["not need", "already", "sufficient", "not", "unnecessary"]):
                        needs_addition = False
                    else:
                        needs_addition = False
                
                result = {
                    'index': i,
                    'id': str(current_qa_id), 
                    'question': target_question,
                    'answer': target_answer,
                    'cluster': cluster_label,
                    'cluster_size': len(cluster_qa_items),
                    'missing_topics': list(missing_topics),
                    'cluster_topic_info': {topic: info_list[:3] for topic, info_list in cluster_topic_info.items()},
                    'relevant_evidence': relevant_evidence_for_cluster,
                    'ollama_response': ollama_response,
                    'needs_topic_addition': needs_addition
                }
                

                updated_answer = target_answer
                insertion_info = None
                if needs_addition:

                    update_prompt_parts = [
                       f"Current question: {target_question}",
f"Current answer: {target_answer}",
f"Missing topics: {list(missing_topics)}",
f"Relevant information from other answers in the same cluster: {json.dumps(cluster_topic_info, ensure_ascii=False, indent=2)[:1000]}...",
f"Relevant evidence from the corpus: {json.dumps(relevant_evidence_for_cluster, ensure_ascii=False, indent=2)[:1000]}...",
f"Please analyze the structure of the current answer and determine where the missing thematic information should be inserted.",
f"Provide the following information:",
f"1. Insertion position: Which answer list(s) should the new content be added to (e.g., one or multiple lists)?",
f"2. Insertion reason: Briefly explain why this position was chosen.",
f"3. New content: Generate the exact snippet to insert, based on the missing topics and relevant information.",
f"4. Updated full answer: The complete answer after inserting the new content into the appropriate position(s).",
f"Return the result strictly in the following JSON format:",
f'{{"insertion_position": "...", "reason": "...", "new_content": "...", "updated_answer": "..."}}'
                    ]
                    update_prompt = "\n".join(update_prompt_parts)
                    
                    update_messages = [
                        {"role": "user", "content": update_prompt}
                    ]
                    
                    update_response = query_ollama_chat(update_messages)
                    print(f"  Answer update response: {update_response[:100]}...")
                    

                    try:
                        json_start = update_response.find('{')
                        json_end = update_response.rfind('}') + 1
                        if json_start != -1 and json_end != 0:
                            json_str = update_response[json_start:json_end]
                            insertion_info = json.loads(json_str)
 
                            updated_answer = insertion_info.get('updated_answer', target_answer)
                        else:
                            print(f"  Could not parse JSON from the answer update response: {update_response}")
                    except json.JSONDecodeError as e:
                        print(f"  Failed to parse answer update JSON: {e}")
                        print(f"  Response content: {update_response}")

                result['updated_answer'] = updated_answer
                result['insertion_info'] = insertion_info
                
                results.append(result)
            except Exception as e:
                print(f"  Error while calling Ollama for the topic-gap decision: {e}")
                result = {
                    'index': i,
                    'id': str(current_qa_id), 
                    'question': target_question,
                    'answer': target_answer,
                    'cluster': cluster_label,
                    'cluster_size': len(cluster_qa_items),
                    'missing_topics': list(missing_topics),
                    'cluster_topic_info': {topic: info_list[:3] for topic, info_list in cluster_topic_info.items()},
                    'relevant_evidence': relevant_evidence_for_cluster,
                    'ollama_response': f"Error calling Ollama: {e}",
                    'needs_topic_addition': False, 
                    'updated_answer': target_answer,
                    'insertion_info': None 
                }
                results.append(result)
        else:
            print("  No missing topics found; skipping.")
            continue

    return results

def save_results(results: List[Dict], output_path: str):
    try:
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")
        raise

def save_cluster_mapping(cluster_to_qa_items: Dict[int, List[Dict]], output_file: str):
    try:
        cluster_to_qa_ids = {}
        for cluster_id, qa_items in cluster_to_qa_items.items():
            qa_ids = []
            for qa_item in qa_items:
                evidence_list = qa_item.get("evidence_list", [])
                qid = None
                if evidence_list and isinstance(evidence_list[0], dict):
                    qid = evidence_list[0].get("id")
                qa_ids.append(str(qid) if qid is not None else qa_item.get("id", "unknown"))

            cluster_to_qa_ids[int(cluster_id)] = qa_ids

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_to_qa_ids, f, indent=2, ensure_ascii=False)
        print(f"Cluster-to-question mapping saved to {output_file}")
    except Exception as e:
        print(f"Failed to save cluster mapping to {output_file}: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and optionally update topic gaps in QA answers.")
    parser.add_argument("--qa-file", default=QA_FILE, help="Input QA JSON file.")
    parser.add_argument("--corpus-file", default=CORPUS_FILE, help="Input corpus JSON file.")
    parser.add_argument("--output-file", default=OUTPUT_FILE, help="Output JSON file.")
    parser.add_argument("--cluster-output-file", default=CLUSTER_OUTPUT_FILE, help="Output cluster mapping JSON file.")
    parser.add_argument("--ollama-base-url", default=OLLAMA_BASE_URL, help="Ollama API base URL.")
    parser.add_argument("--ollama-api-key", default=OLLAMA_API_KEY, help="Ollama API key, if required.")
    parser.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL, help="Ollama embedding model.")
    parser.add_argument("--chat-model", default=OLLAMA_CHAT_MODEL, help="Ollama chat model.")
    parser.add_argument("--num-clusters", type=int, default=NUM_CLUSTERS, help="Number of KMeans clusters.")
    parser.add_argument("--top-k-evidence", type=int, default=TOP_K_EVIDENCE, help="Maximum evidence chunks per cluster.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Evidence chunk size in characters.")
    return parser.parse_args()


def main():
    global OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL
    global TOP_K_EVIDENCE, CHUNK_SIZE

    args = parse_args()
    OLLAMA_BASE_URL = args.ollama_base_url
    OLLAMA_API_KEY = args.ollama_api_key
    OLLAMA_EMBED_MODEL = args.embed_model
    OLLAMA_CHAT_MODEL = args.chat_model
    TOP_K_EVIDENCE = args.top_k_evidence
    CHUNK_SIZE = args.chunk_size

    if not args.qa_file or not args.corpus_file:
        raise SystemExit("Missing input files. Use --qa-file and --corpus-file.")
    if not OLLAMA_BASE_URL:
        raise SystemExit("Missing Ollama base URL. Use --ollama-base-url or set OLLAMA_BASE_URL.")

    print("1. Loading QA data...")
    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"   Loaded {len(qa_data)} QA pairs")

    print("\n2. Clustering questions...")
    cluster_labels, qa_id_to_cluster, cluster_to_qa_items = cluster_questions(qa_data, args.num_clusters)
    save_cluster_mapping(cluster_to_qa_items, args.cluster_output_file)

    print("\n3. Loading corpus and building mappings...")
    total_id_to_doc = load_corpus(args.corpus_file)

    print("\n4. Detecting topic gaps and linking evidence...")
    results = detect_topic_gaps_using_cluster_and_evidence(qa_data, qa_id_to_cluster, cluster_to_qa_items, total_id_to_doc)

    print(f"\n5. Saving results to {args.output_file}...")
    save_results(results, args.output_file)

    total_qas = len(qa_data)
    gap_qas = len(results)  
    need_addition_count = sum(1 for res in results if res.get('needs_topic_addition', False))

    print("\n--- Detection Complete ---")
    print(f"Total QA pairs: {total_qas}")
    print(f"QA pairs with potential topic gaps: {gap_qas}")
    print(f"QA pairs requiring additional information: {need_addition_count}")
    print(f"Detailed results saved to: {args.output_file}")
    print(f"Cluster mapping saved to: {args.cluster_output_file}")


    if results:
        print("\n--- Sample Results (first 3) ---")
        for i, result in enumerate(results[:3]):
            print(f"\nQA pair {i+1} (index: {result['index']}, ID: {result['id']}):")
            print(f"  Question: {result['question'][:100]}...")
            print(f"  Cluster: {result['cluster']} (size: {result['cluster_size']})")
            print(f"  Missing topics: {result['missing_topics']}")
            print(f"  Needs addition: {result['needs_topic_addition']}")
            print(f"  Original answer: {result['answer'][:100]}...")
            print(f"  Updated answer: {result['updated_answer'][:100]}...")
            print(f"  Insertion info: {result['insertion_info']}")
            print(f"  Ollama decision: {result['ollama_response'][:200]}...")
            print(f"  Relevant evidence (first 2): {result['relevant_evidence'][:2]}")
    else:
        print("\nNo QA pairs with topic gaps or required additions were found.")

if __name__ == "__main__":
    main()



