import argparse
import json
import numpy as np
import hnswlib
import logging
import os
import requests
from typing import List, Dict, Any, Tuple, Optional

QA_FILE = os.getenv("AURA_QA_FILE", "")
CORPUS_FILE = os.getenv("AURA_CORPUS_FILE", "")
HNSW_SPACE = 'cosine'
TOP_K = 10


OLLAMA_EMBED_BASE_URL = os.getenv("OLLAMA_EMBED_BASE_URL", "")
OLLAMA_EMBED_API_KEY = os.getenv("OLLAMA_EMBED_API_KEY", "")
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"   

OLLAMA_LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("OLLAMA_LLM_BASE_URL", "https://api.openai.com/v1"))
OLLAMA_LLM_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OLLAMA_LLM_API_KEY", ""))
OLLAMA_LLM_MODEL = os.getenv("AURA_REFINE_MODEL", "gpt-4o")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(qa_file: str, corpus_file: str) -> Tuple[List[Dict], List[Dict]]:
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        logger.info(f"Loaded {len(qa_data)} QA pairs from {qa_file}")
    except Exception as e:
        logger.error(f"Failed to load QA data from {qa_file}: {e}")
        raise

    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        logger.info(f"Loaded {len(corpus_data)} evidence documents from {corpus_file}")
    except Exception as e:
        logger.error(f"Failed to load Corpus data from {corpus_file}: {e}")
        raise

    return qa_data, corpus_data

def get_ollama_embedding(text: str, base_url: str, model: str, api_key: str = None) -> List[float]:

    url = f"{base_url.rstrip('/')}/api/embed"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": model,
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
            logger.error(f"Unexpected response structure from Ollama embedding API: {result}")
            raise ValueError("Failed to get embedding from Ollama API response")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting embedding from Ollama: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing Ollama embedding response: {e}")
        raise

def prepare_embeddings_and_index_ollama(corpus_data: List[Dict], model: str, base_url: str, api_key: str, space: str, ef_construction: int = 200, M: int = 16) -> Tuple[np.ndarray, Any, Dict, Dict]:
    logger.info("Preparing embeddings (via Ollama) and HNSW index...")
    texts = [f"Title: {item['title']}. Context:{item.get('description') or item.get('body', 'N/A')}" for item in corpus_data]
    ids = [item['id'] for item in corpus_data]

    logger.info(f"Loading embeddings from Ollama model '{model}'...")
    corpus_embeddings_list = []
    for i, text in enumerate(texts):
        try:
            embedding = get_ollama_embedding(text, base_url, model, api_key)
            corpus_embeddings_list.append(embedding)
            if (i + 1) % 50 == 0 or (i + 1) == len(texts):
                 logger.info(f"Encoded {i+1}/{len(texts)} corpus documents...")
        except Exception as e:
             logger.error(f"Failed to encode document {i+1} (ID: {ids[i]}): {e}")
             raise 

    corpus_embeddings = np.array(corpus_embeddings_list)
    dim = corpus_embeddings.shape[1]
    num_elements = len(corpus_embeddings)
    logger.info(f"Encoding complete. Dimension: {dim}, Number of elements: {num_elements}")

    id_to_index_map = {id_: i for i, id_ in enumerate(ids)}
    index_to_id_map = {i: id_ for i, id_ in enumerate(ids)}

    hnsw_index = hnswlib.Index(space=space, dim=dim)
    hnsw_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    logger.info("Initializing HNSW index...")
    
    labels = np.arange(num_elements)
    hnsw_index.add_items(corpus_embeddings, labels)
    logger.info("HNSW index populated.")
    hnsw_index.set_ef(max(ef_construction, 50))

    return corpus_embeddings, hnsw_index, id_to_index_map, index_to_id_map

def find_top_k_most_similar_efficient(hnsw_index: Any, query_embedding: np.ndarray, index_to_id_map: Dict, k: int) -> List:
    """Find the top-k most similar evidence IDs with HNSW."""
    labels, _ = hnsw_index.knn_query(query_embedding, k=k)
    top_k_ids = [index_to_id_map[label] for label in labels[0]]
    return top_k_ids

def query_ollama(question: str, context: str, base_url: str, api_key: str, model: str, confidence_threshold: float = 0.7) -> Tuple[str, float, bool]:

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Based ONLY on the provided context, answer the question concisely. "
        f"Then, on a new line, provide your confidence level as a number between 0 and 1 "
        f"(e.g., 'Confidence: 0.95'). "
        f"If the context does not contain enough information, say 'I cannot answer based on the context' "
        f"and provide a low confidence (e.g., 'Confidence: 0.1')."
        f"\nAnswer:"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [ 
             {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        answer_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not answer_content:
             logger.warning("Ollama LLM returned empty content.")
             return "No response from model", 0.0, False
        lines = answer_content.split('\n')
        answer = lines[0].strip() if lines else "Could not parse answer"
        confidence = 0.0
        for line in lines:
            if line.lower().startswith("confidence:"):
                try:
                    confidence_str = line.split(":")[1].strip()
                    confidence = float(confidence_str)
                    break
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse confidence from line: {line}")
                    pass 
        
        logger.info(f"Ollama LLM Response: {answer_content} | Parsed Confidence: {confidence}")
        is_answerable = confidence >= confidence_threshold
        return answer, confidence, is_answerable

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama LLM: {e}")
        return f"Error calling model: {e}", 0.0, False
    except Exception as e:
        logger.error(f"Unexpected error parsing Ollama LLM response: {e}")
        return f"Error parsing response: {e}", 0.0, False

def refine_question_with_ollama(
    original_question: str, answer: str, negative_context: str,
    base_url: str, api_key: str, model: str
) -> Optional[str]:

    prompt = (
        f"Original Question:\n{original_question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Negative Context (information that should NOT easily answer the new question):\n{negative_context}\n\n"
        f"Task:\n"
        f"Generate a new question that:\n"
        f"- Can still be correctly answered by the provided 'Answer'.\n"
        f"- Is significantly different from the 'Original Question'.\n"
        f"- Cannot be easily answered using only the 'Negative Context'.\n"
        f"- Is specific and clear.\n\n"
        f"New Question:"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7 
    }

    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        modified_question = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if modified_question:
            logger.info(f"Refined Question Generated: {modified_question}")
            return modified_question
        else:
            logger.warning("Ollama LLM returned empty content for refined question.")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating refined question with Ollama LLM: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing refined question response: {e}")
        return None


def analyze_evidence_relevance_with_ollama(
    qa_data: List[Dict], corpus_data: List[Dict], corpus_id_to_item_map: Dict, hnsw_index: Any,
    id_to_index_map: Dict, index_to_id_map: Dict, top_k: int,
    ollama_embed_base_url: str, ollama_embed_api_key: str, ollama_embed_model: str, 
    ollama_llm_base_url: str, ollama_llm_api_key: str, ollama_llm_model: str       
):

    logger.info("Starting evidence relevance analysis with Ollama evaluation...")
    stats = {"total_qa_pairs": len(qa_data), "details": []}
    confidence_threshold = 0.7 

    for i, qa_item in enumerate(qa_data):
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '') 
        evidence_list = qa_item.get('evidence_list', [])
        positive_ids = set(ev['id'] for ev in evidence_list)

        if not positive_ids:
            logger.warning(f"QA pair {i+1} has no evidence IDs.")
            stats["details"].append({"qa_index": i+1, "question": question, "status": "No evidence IDs provided"})
            continue


        positive_context_parts = []
        missing_evidence = []
        for eid in positive_ids:
            evidence_item = corpus_id_to_item_map.get(eid)
            if evidence_item:
                combined_text = f"Title: {evidence_item['title']}. Context: {evidence_item.get('description') or evidence_item.get('body', 'N/A')}"
                positive_context_parts.append(combined_text)
            else:
                missing_evidence.append(eid)
                logger.warning(f"Evidence ID {eid} not found in corpus for QA pair {i+1}.")
        if missing_evidence:
             stats["details"].append({
                "qa_index": i+1, "question": question,
                "status": f"Missing evidence IDs in corpus: {missing_evidence}"
            })
             continue
        positive_context = " ".join(positive_context_parts)


        try:
            query_embedding_list = get_ollama_embedding(positive_context, ollama_embed_base_url, ollama_embed_model, ollama_embed_api_key)
            query_embedding = np.array([query_embedding_list])
        except Exception as e:
            logger.error(f"Error getting embedding for QA pair {i+1} query: {e}")
            stats["details"].append({
                "qa_index": i+1, "question": question,
                "status": f"Error during query embedding: {e}"
            })
            continue

        try:
            top_k_most_similar_ids = find_top_k_most_similar_efficient(hnsw_index, query_embedding, index_to_id_map, top_k)
        except Exception as e:
            logger.error(f"Error finding most similar evidence for QA pair {i+1}: {e}")
            stats["details"].append({
                "qa_index": i+1, "question": question,
                "status": f"Error during similarity search: {e}"
            })
            continue

        found_in_top_k = positive_ids.intersection(set(top_k_most_similar_ids))
        other_similar_ids = [eid for eid in top_k_most_similar_ids if eid not in positive_ids]

        logger.info(f"[QA {i+1}] Querying Ollama LLM with POSITIVE context...")
        pos_answer, pos_confidence, pos_is_answerable = query_ollama(
            question, positive_context, ollama_llm_base_url, ollama_llm_api_key, ollama_llm_model, confidence_threshold
        )


        negative_context_parts = []
        for eid in other_similar_ids:
             evidence_item = corpus_id_to_item_map.get(eid)
             if evidence_item:
                 combined_text = f"Title: {evidence_item['title']}. Context: {evidence_item.get('description') or evidence_item.get('body', 'N/A')}"
                 negative_context_parts.append(combined_text)
        negative_context = " ".join(negative_context_parts) if negative_context_parts else "[No relevant context found in other similar documents.]"

        logger.info(f"[QA {i+1}] Querying Ollama LLM with NEGATIVE context (Other Similar IDs)...")
        neg_answer, neg_confidence, neg_is_answerable = query_ollama(
            question, negative_context, ollama_llm_base_url, ollama_llm_api_key, ollama_llm_model, confidence_threshold
        )

        potentially_redundant = neg_is_answerable

        modified_question = None
        if potentially_redundant:
            logger.info(f"[QA {i+1}] Potentially redundant. Generating modified question...")
            modified_question = refine_question_with_ollama(
                question, answer, negative_context,
                ollama_llm_base_url, ollama_llm_api_key, ollama_llm_model
            )
            if modified_question:
                 logger.info(f"[QA {i+1}] Modified question generated successfully.")
            else:
                 logger.warning(f"[QA {i+1}] Failed to generate a modified question.")


        detail = {
            "qa_index": i+1,
            "original_question": question, 
            "modified_question": modified_question, 
            "answer": answer, 
            "evidence_list": evidence_list, 
            "specified_evidence_ids": list(positive_ids),
            "top_k_most_similar_ids": top_k_most_similar_ids,
            "found_specified_in_top_k_count": len(found_in_top_k),
            "other_similar_ids_in_top_k": other_similar_ids,
            "other_similar_ids_count": len(other_similar_ids),
            "ollama_positive": {
                "context": positive_context,
                "answer": pos_answer,
                "confidence": pos_confidence,
                "is_answerable": pos_is_answerable
            },
            "ollama_negative": {
                "context": negative_context,
                "answer": neg_answer,
                "confidence": neg_confidence,
                "is_answerable": neg_is_answerable
            },
            "potentially_redundant_based_on_negative": potentially_redundant
        }
        stats["details"].append(detail)
        logger.info(f"Analyzed QA pair {i+1}/{len(qa_data)}")

    logger.info("Evidence relevance analysis with Ollama evaluation and question refinement complete.")
    return stats

def print_summary_and_find_redundant(stats: Dict, top_k: int):
    print("\n--- Evidence Relevance Analysis & Ollama Evaluation Results ---")
    print(f"Total QA pairs processed: {stats['total_qa_pairs']}")
    
    if stats['total_qa_pairs'] == 0:
        print("No QA pairs to analyze.")
        return


    total_found_counts = [d['found_specified_in_top_k_count'] for d in stats['details'] if 'found_specified_in_top_k_count' in d]
    total_other_counts = [d['other_similar_ids_count'] for d in stats['details'] if 'other_similar_ids_count' in d]
    
    if total_found_counts:
        avg_found = np.mean(total_found_counts)
        print(f"Average number of specified evidence IDs found in Top-{top_k}: {avg_found:.2f}")
    if total_other_counts:
        avg_other = np.mean(total_other_counts)
        print(f"Average number of OTHER similar evidence IDs in Top-{top_k}: {avg_other:.2f}")


    redundant_qa_details = [
        d for d in stats['details']
        if d.get('potentially_redundant_based_on_negative', False) and d.get('modified_question')
    ]
    print(f"\nNumber of QA pairs successfully refined (potentially redundant & modified question generated): {len(redundant_qa_details)}")
    if redundant_qa_details:
        print("Indices of refined QA pairs:")
        for detail in redundant_qa_details:
            print(f"  - Index {detail['qa_index']}: Original: '{detail['original_question']}' -> Modified: '{detail['modified_question']}'")

    print("\n--- Interpretation ---")
    print("- High average 'found' count: Specified evidence is generally relevant.")
    print("- High average 'other' count: More similar docs found, potential for overlap.")
    print("- 'potentially_redundant_based_on_negative' = True: Model found answer in 'other' context with high confidence.")
    print("- 'modified_question' generated: A new question was created to address potential redundancy.")
    print("  This aims to make the original evidence list more specific/unique for the QA pair.")

    print("\n--- Detailed Results (first 2) ---")
    for detail in stats['details'][:2]:
        simplified_detail = {
            k: v for k, v in detail.items()
            if k not in ['ollama_positive', 'ollama_negative', 'top_k_most_similar_ids', 'other_similar_ids_in_top_k']
        }
        print(json.dumps(simplified_detail, indent=2, ensure_ascii=False))
        print("-" * 20)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evidence relevance and refine redundant QA questions.")
    parser.add_argument("--qa-file", default=QA_FILE, help="Input QA JSON file.")
    parser.add_argument("--corpus-file", default=CORPUS_FILE, help="Input corpus/evidence JSON file.")
    parser.add_argument("--output-file", default="evidence_analysis_with_refined_questions.json", help="Output JSON file.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of nearest evidence documents to retrieve.")
    parser.add_argument("--hnsw-space", default=HNSW_SPACE, help="HNSW distance space.")
    parser.add_argument("--embed-base-url", default=OLLAMA_EMBED_BASE_URL, help="Embedding API base URL.")
    parser.add_argument("--embed-api-key", default=OLLAMA_EMBED_API_KEY, help="Embedding API key, if required.")
    parser.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL, help="Embedding model name.")
    parser.add_argument("--llm-base-url", default=OLLAMA_LLM_BASE_URL, help="OpenAI-compatible LLM API base URL.")
    parser.add_argument("--llm-api-key", default=OLLAMA_LLM_API_KEY, help="LLM API key.")
    parser.add_argument("--llm-model", default=OLLAMA_LLM_MODEL, help="LLM model name.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.qa_file or not args.corpus_file:
        raise SystemExit("Missing input files. Use --qa-file and --corpus-file.")
    if not args.embed_base_url:
        raise SystemExit("Missing embedding base URL. Use --embed-base-url or set OLLAMA_EMBED_BASE_URL.")
    if not args.llm_base_url:
        raise SystemExit("Missing LLM base URL. Use --llm-base-url or set OPENAI_BASE_URL.")

    qa_data, corpus_data = load_data(args.qa_file, args.corpus_file)

    corpus_id_to_item_map = {item['id']: item for item in corpus_data}

    corpus_embeddings, hnsw_index, id_to_index_map, index_to_id_map = prepare_embeddings_and_index_ollama(
        corpus_data, args.embed_model, args.embed_base_url, args.embed_api_key, args.hnsw_space
    )

    stats = analyze_evidence_relevance_with_ollama(
        qa_data, corpus_data, corpus_id_to_item_map,
        hnsw_index, id_to_index_map, index_to_id_map, args.top_k,
        args.embed_base_url, args.embed_api_key, args.embed_model,
        args.llm_base_url, args.llm_api_key, args.llm_model
    )

    print_summary_and_find_redundant(stats, args.top_k)

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Full results (including refined questions) saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {args.output_file}: {e}")

if __name__ == "__main__":
    print("--- Prerequisites Check ---")
    print("1. Ensure `ollama serve` is running.")
    print("2. Ensure `ollama pull nomic-embed-text:latest` has been executed.")
    print("3. Ensure OLLAMA_EMBED_BASE_URL points to your local Ollama instance.")
    print("--------------------------\n")
    
    main()



