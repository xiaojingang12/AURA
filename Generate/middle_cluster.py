import argparse
import json
import numpy as np
from collections import defaultdict
import logging
import os
from typing import List, Dict, Any, Tuple
import requests
from sklearn.cluster import KMeans 
import warnings
import random 

QA_FILE = os.getenv("AURA_QA_FILE", "")
CORPUS_FILE = os.getenv("AURA_CORPUS_FILE", "")
CLASSIFIED_OUTPUT_FILE = os.getenv("AURA_CLASSIFIED_OUTPUT_FILE", "qa_with_all_evidence_lists.json")
CLUSTER_MAPPING_OUTPUT_FILE = os.getenv("AURA_CLUSTER_MAPPING_OUTPUT_FILE", "cluster_to_documents_mapping.json")

OLLAMA_EMBED_BASE_URL = os.getenv("OLLAMA_EMBED_BASE_URL", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


NUM_CLUSTERS = 10 
MIDDLE_EVIDENCE_LIMIT = 20 
MAX_BALANCE_ITERATIONS = 10 
BALANCE_TOLERANCE = 0.1 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(qa_file: str, corpus_file: str) -> Tuple[List[Dict], List[Dict]]:
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        logger.info(f"Loaded {len(qa_data)} QA pairs from {qa_file}")
        for i, qa_item in enumerate(qa_data):
            if 'id' not in qa_item:
                logger.warning(f"QA item at index {i} is missing 'id' field. Assigning default ID 'qa_{i}'.")
                qa_item['id'] = f"qa_{i}" 

    except Exception as e:
        logger.error(f"Failed to load QA data from {qa_file}: {e}")
        raise

    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        logger.info(f"Loaded {len(corpus_data)} evidence documents from {corpus_file}")
        for i, doc_item in enumerate(corpus_data):
             if 'total_id' not in doc_item:
                 logger.error(f"Corpus item at index {i} is missing 'total_id' field. This is critical.")
                 raise ValueError(f"Corpus item at index {i} is missing 'total_id' field.")
    except Exception as e:
        logger.error(f"Failed to load Corpus data from {corpus_file}: {e}")
        raise

    return qa_data, corpus_data

def get_ollama_embedding(text: str, base_url: str, model: str) -> List[float]:
    url = f"{base_url.rstrip('/')}/api/embed"
    headers = {"Content-Type": "application/json"}
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
        logger.error(f"Error getting embedding for text (truncated): {text[:50]}... Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing Ollama embedding response for text (truncated): {text[:50]}... Error: {e}")
        raise

def balanced_kmeans(X: np.ndarray, n_clusters: int, max_iter: int = 10, tol: float = 0.1) -> np.ndarray:

    n_samples = X.shape[0]
    if n_samples < n_clusters:
        logger.warning(f"Number of samples ({n_samples}) is less than number of clusters ({n_clusters}). Assigning each sample to its own cluster.")
        return np.arange(n_samples)

    logger.info(f"Initializing Balanced KMeans with standard KMeans (n_clusters={n_clusters})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        initial_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        initial_labels = initial_kmeans.fit_predict(X)
        cluster_centers = initial_kmeans.cluster_centers_

    labels = initial_labels.copy()
    target_size = n_samples / n_clusters
    upper_bound = int(target_size * (1 + tol))
    lower_bound = int(target_size * (1 - tol))
    logger.info(f"Target cluster size: ~{target_size:.2f}. Bounds: [{lower_bound}, {upper_bound}]")

    for iteration in range(max_iter):
        logger.debug(f"Balanced KMeans iteration {iteration + 1}/{max_iter}")
        changed = False
        unique_labels, counts = np.unique(labels, return_counts=True)
        current_sizes = dict(zip(unique_labels, counts))
        
        overloaded_clusters = [c for c, size in current_sizes.items() if size > upper_bound]
        underloaded_clusters = [c for c, size in current_sizes.items() if size < lower_bound]

        if not overloaded_clusters:
            logger.info(f"Balanced KMeans converged after {iteration + 1} iterations. No overloaded clusters.")
            break

        for over_c in overloaded_clusters:
            points_in_over_cluster = np.where(labels == over_c)[0]
            if len(points_in_over_cluster) <= upper_bound:
                continue
            

            distances_to_all_centers = np.linalg.norm(X[points_in_over_cluster, np.newaxis] - cluster_centers, axis=2) # (n_points_in_over, n_clusters)

            current_distances = distances_to_all_centers[:, over_c]
            sorted_indices_by_distance = np.argsort(-current_distances) 
            
            for point_idx_in_list in sorted_indices_by_distance:
                point_global_idx = points_in_over_cluster[point_idx_in_list]
                if len(np.where(labels == over_c)[0]) <= upper_bound:
                     break

                candidate_clusters = [c for c in underloaded_clusters if current_sizes.get(c, 0) < upper_bound]
                if not candidate_clusters:
                    continue 
                
                distances_to_candidates = distances_to_all_centers[point_idx_in_list, candidate_clusters]
                closest_candidate_idx = np.argmin(distances_to_candidates)
                closest_candidate_cluster = candidate_clusters[closest_candidate_idx]

                if closest_candidate_cluster != over_c:
                    old_cluster = labels[point_global_idx]
                    labels[point_global_idx] = closest_candidate_cluster
                    current_sizes[old_cluster] -= 1
                    current_sizes[closest_candidate_cluster] = current_sizes.get(closest_candidate_cluster, 0) + 1
                    changed = True
                    logger.debug(f"Moved point {point_global_idx} from cluster {old_cluster} to {closest_candidate_cluster}")
        
        if not changed:
            logger.info(f"Balanced KMeans stopped after {iteration + 1} iterations (no changes).")
            break
    
    if changed and max_iter > 0:
         logger.info(f"Balanced KMeans reached maximum iterations ({max_iter}).")
    return labels

def cluster_documents(corpus_data: List[Dict], embed_base_url: str, embed_model: str, num_clusters: int = None) -> Tuple[np.ndarray, Dict[str, int], Dict[int, List[str]]]:
    logger.info("Starting document clustering using 'total_id' and Balanced KMeans...")
    texts = [f"Title: {item['title']}. Context: {item['context']}" for item in corpus_data]
    doc_total_ids = [item['total_id'] for item in corpus_data]

    logger.info(f"Generating embeddings for {len(texts)} documents using Ollama '{embed_model}'...")
    embeddings_list = []
    for i, text in enumerate(texts):
        try:
            embedding = get_ollama_embedding(text, embed_base_url, embed_model)
            embeddings_list.append(embedding)
            if (i + 1) % 50 == 0 or (i + 1) == len(texts):
                 logger.info(f"Encoded {i+1}/{len(texts)} documents...")
        except Exception as e:
             logger.error(f"Failed to encode document {i+1} (total_id: {doc_total_ids[i]}): {e}")
             raise

    embeddings_array = np.array(embeddings_list)
    logger.info(f"Embeddings generated. Shape: {embeddings_array.shape}")

    logger.info(f"Performing Balanced KMeans clustering with {num_clusters} clusters...")
    
    if num_clusters is not None:
        cluster_labels = balanced_kmeans(embeddings_array, n_clusters=num_clusters, max_iter=MAX_BALANCE_ITERATIONS, tol=BALANCE_TOLERANCE)
    else:
        logger.error("Number of clusters (NUM_CLUSTERS) must be specified for Balanced KMeans.")
        raise ValueError("NUM_CLUSTERS not set for Balanced KMeans")
    
    total_id_to_cluster = {total_id: int(label) for total_id, label in zip(doc_total_ids, cluster_labels)}
    
    cluster_to_total_ids = defaultdict(list)
    for total_id, cluster_label in total_id_to_cluster.items():
        cluster_to_total_ids[int(cluster_label)].append(total_id)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))
    logger.info(f"Final cluster sizes after balancing: {cluster_stats}")
    
    logger.info("Document clustering complete using 'total_id' and Balanced KMeans.")
    return cluster_labels, total_id_to_cluster, dict(cluster_to_total_ids)

def generate_all_evidence_lists_for_all(
    qa_data: List[Dict], 
    total_id_to_cluster: Dict[str, int], 
    cluster_to_total_ids: Dict[int, List[str]], 
    corpus_data: List[Dict],
    middle_limit: int = 20
) -> Dict[str, Dict[str, List[Dict]]]:

    logger.info(f"Generating simple, middle (max {middle_limit}), and hard evidence lists for all {len(qa_data)} QA pairs using 'total_id'...")

    total_id_to_corpus_item = {item['total_id']: item for item in corpus_data}
    

    total_id_to_corpus_item_minimal = {
        item['total_id']: {k: v for k, v in item.items() if k in ['id', 'title', 'total_id']}
        for item in corpus_data
    }
    
    all_evidence_lists_dict = {}
    
    for qa_item in qa_data:
        qid = qa_item.get('id', 'unknown_qa_id_in_loop')
        original_evidence_list_raw = qa_item.get('evidence_list', [])
        original_evidence_total_ids = {ev['total_id'] for ev in original_evidence_list_raw if 'total_id' in ev} 

        simple_evidence_list = []
        middle_evidence_list = [] 
        hard_evidence_list = [] 

        for ev_item in original_evidence_list_raw:
             if 'total_id' in ev_item:
                 total_id = ev_item['total_id']
                 corpus_item_minimal = total_id_to_corpus_item_minimal.get(total_id)
                 if corpus_item_minimal:
                     simple_evidence_list.append(corpus_item_minimal)
                 else:
                     logger.warning(f"QA {qid} (Simple): Corpus item for total_id {total_id} not found.")
             else:
                 logger.warning(f"QA {qid} (Simple): Evidence item missing 'total_id' field.")

        cluster_labels_involved = set()
        missing_evidence_total_ids = []
        for ev_item in original_evidence_list_raw:
            if 'total_id' in ev_item:
                total_id = ev_item['total_id']
                cluster_label = total_id_to_cluster.get(total_id)
                if cluster_label is not None:
                    cluster_labels_involved.add(cluster_label)
                else:
                    missing_evidence_total_ids.append(total_id)
            else:
                logger.warning(f"QA {qid} (Middle): Evidence item missing 'total_id' field.")
        
        if missing_evidence_total_ids:
            logger.warning(f"QA {qid} (Middle): Evidence total_ids not found in clustering: {missing_evidence_total_ids}")


        involved_doc_total_ids = set()
        for cl_label in cluster_labels_involved:
            involved_doc_total_ids.update(cluster_to_total_ids.get(cl_label, []))

        involved_doc_total_ids.update(original_evidence_total_ids) 
        
        if len(involved_doc_total_ids) <= middle_limit:
            selected_doc_total_ids = list(involved_doc_total_ids)
        else:
            selected_doc_total_ids = list(original_evidence_total_ids)
            remaining_candidates = list(involved_doc_total_ids - original_evidence_total_ids)
            num_extra_needed = middle_limit - len(selected_doc_total_ids)
            
            if num_extra_needed > 0 and remaining_candidates:
                try:
                    selected_additional_ids = random.sample(remaining_candidates, num_extra_needed)
                    selected_doc_total_ids.extend(selected_additional_ids)
                except ValueError as e:
                    logger.error(f"QA {qid} (Middle): Error sampling additional evidence: {e}. Using all available.")
                    selected_doc_total_ids.extend(remaining_candidates) 
        

        random.shuffle(selected_doc_total_ids) 
        

        for doc_total_id in selected_doc_total_ids:
            corpus_item_minimal = total_id_to_corpus_item_minimal.get(doc_total_id)
            if corpus_item_minimal:
                middle_evidence_list.append(corpus_item_minimal)
            else:

                logger.warning(f"QA {qid} (Middle): Corpus item for selected doc total_id {doc_total_id} not found.")


   
        hard_cluster_labels_involved = set()
        for ev_item in original_evidence_list_raw:
            if 'total_id' in ev_item:
                total_id = ev_item['total_id']
                cluster_label = total_id_to_cluster.get(total_id)
                if cluster_label is not None:
                    hard_cluster_labels_involved.add(cluster_label)


        all_hard_doc_total_ids = set()
        for cl_label in hard_cluster_labels_involved:
            all_hard_doc_total_ids.update(cluster_to_total_ids.get(cl_label, []))

        for doc_total_id in all_hard_doc_total_ids:
             corpus_item_minimal = total_id_to_corpus_item_minimal.get(doc_total_id)
             if corpus_item_minimal:
                 hard_evidence_list.append(corpus_item_minimal)



        random.shuffle(hard_evidence_list)

        all_evidence_lists_dict[qid] = {
            "simple_evidence_list": simple_evidence_list,
            "middle_evidence_list": middle_evidence_list,
            "hard_evidence_list": hard_evidence_list
        }
        
    logger.info("Generation of all evidence lists (with middle limit applied & new hard definition) complete using 'total_id'.")
    return all_evidence_lists_dict

def integrate_all_evidence_lists(qa_data: List[Dict], all_evidence_lists_dict: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    logger.info("Integrating simple, middle, and hard evidence lists into QA data...")
    final_qa_data = []
    for qa_item in qa_data:
        qid = qa_item.get('id', 'unknown_qa_id_during_integration')
        evidence_lists = all_evidence_lists_dict.get(qid, {
            "simple_evidence_list": [],
            "middle_evidence_list": [],
            "hard_evidence_list": []
        })
        
        final_qa_item = {
            **qa_item,
            "simple_evidence_list": evidence_lists["simple_evidence_list"],
            "middle_evidence_list": evidence_lists["middle_evidence_list"],
            "hard_evidence_list": evidence_lists["hard_evidence_list"]
        }
        final_qa_data.append(final_qa_item)
    logger.info("Integration of all evidence lists complete.")
    return final_qa_data

def save_final_data(qa_data_with_lists: List[Dict], output_file: str):
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

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_data_with_lists, f, indent=2, ensure_ascii=False, default=convert)
        logger.info(f"Final QA data with all evidence lists saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final data to {output_file}: {e}")
        raise

def save_cluster_mapping(cluster_to_total_ids: Dict[int, List[str]], output_file: str):
    """Save the mapping from cluster IDs to document total_id values."""
    try:
        serializable_mapping = {int(k): v for k, v in cluster_to_total_ids.items()}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Cluster to document total_ids mapping saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save cluster mapping to {output_file}: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Build simple/middle/hard evidence lists with balanced document clustering.")
    parser.add_argument("--qa-file", default=QA_FILE, help="Input QA JSON file.")
    parser.add_argument("--corpus-file", default=CORPUS_FILE, help="Input corpus JSON file.")
    parser.add_argument("--output-file", default=CLASSIFIED_OUTPUT_FILE, help="Output QA JSON file.")
    parser.add_argument("--cluster-output-file", default=CLUSTER_MAPPING_OUTPUT_FILE, help="Output cluster mapping JSON file.")
    parser.add_argument("--embed-base-url", default=OLLAMA_EMBED_BASE_URL, help="Ollama embedding API base URL.")
    parser.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL, help="Ollama embedding model.")
    parser.add_argument("--num-clusters", type=int, default=NUM_CLUSTERS, help="Number of clusters.")
    parser.add_argument("--middle-evidence-limit", type=int, default=MIDDLE_EVIDENCE_LIMIT, help="Maximum middle evidence items per QA pair.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.qa_file or not args.corpus_file:
        raise SystemExit("Missing input files. Use --qa-file and --corpus-file.")
    if not args.embed_base_url:
        raise SystemExit("Missing embedding base URL. Use --embed-base-url or set OLLAMA_EMBED_BASE_URL.")

    qa_data, corpus_data = load_data(args.qa_file, args.corpus_file)
    _, total_id_to_cluster, cluster_to_total_ids = cluster_documents(corpus_data, args.embed_base_url, args.embed_model, args.num_clusters)

    all_evidence_lists_dict = generate_all_evidence_lists_for_all(
        qa_data, 
        total_id_to_cluster, 
        cluster_to_total_ids, 
        corpus_data,
        middle_limit=args.middle_evidence_limit
    )


    final_qa_data = integrate_all_evidence_lists(qa_data, all_evidence_lists_dict)

    save_final_data(final_qa_data, args.output_file)

    save_cluster_mapping(cluster_to_total_ids, args.cluster_output_file)

if __name__ == "__main__":

    print("--- Prerequisites Check ---")
    print("1. Ensure `ollama serve` is running.")
    print("2. Ensure `ollama pull nomic-embed-text:latest` has been executed.")
    print("3. Ensure OLLAMA_EMBED_BASE_URL points to your local Ollama instance.")
    print("4. Install required packages: `pip install scikit-learn`")
    print("--------------------------\n")
    
    main()



