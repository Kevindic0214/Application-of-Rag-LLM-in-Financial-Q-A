import os
import json
from ckip_transformers.nlp import CkipWordSegmenter  # CKIP word segmentation tool
from rank_bm25 import BM25Okapi  # BM25 algorithm for document retrieval
import numpy as np
from itertools import product

# Specify the path to the FAQ data
question_path = "dataset/preliminary/questions_example.json"  # Path to the FAQ questions file
source_path = "reference/faq/pid_map_content.json"  # Path to the FAQ reference data
output_path = "dataset/preliminary/faq_pred_retrieve.json"  # Path to the output file for FAQ prediction results

# Initialize CKIP word segmenter
ws_driver = CkipWordSegmenter(model="bert-base")

# Use CKIP for word segmentation
def ckip_tokenize(text):
    result = ws_driver([text])
    return result[0] if result else []

# FAQ retrieval function with adjustable k1 and b
def BM25_retrieve(qs, source, corpus_dict, k1=1.5, b=0.75):
    # Filter out FAQ documents related to the source
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # Use CKIP word segmentation instead of jieba
    tokenized_corpus = [ckip_tokenize(doc) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)  # Build the retrieval model with specified k1 and b
    tokenized_query = ckip_tokenize(qs)  # Tokenize the query sentence
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # Retrieve the most relevant document based on the query
    
    # Find the file name corresponding to the best matching text
    if ans:
        a = ans[0]
        res = [key for key, value in corpus_dict.items() if value == a]
        return res[0] if res else None  # Return the file name or None
    return None

# Define a function to extract text content
def extract_text(value):
    texts = []
    for item in value:
        if 'question' in item:
            texts.append(item['question'])
        if 'answers' in item:
            texts.extend(item['answers'])
    return ' '.join(texts)

# Calculate precision for given parameters
def calculate_precision(predictions, ground_truth):
    correct_count = 0
    for pred in predictions['answers']:
        qid = pred['qid']
        pred_retrieve = pred['retrieve']
        gt_retrieve = next((item['retrieve'] for item in ground_truth['ground_truths'] if item['qid'] == qid), None)
        if pred_retrieve == gt_retrieve:
            correct_count += 1
    precision = correct_count / len(predictions['answers'])
    return precision

# Main program with parameter tuning
if __name__ == "__main__":
    answer_dict = {"answers": []}

    # Load ground truth for precision calculation
    ground_truth_path = "dataset/preliminary/ground_truths_example.json"  # Ground truth file
    with open(ground_truth_path, 'r', encoding='utf-8') as f_gt:
        ground_truth = json.load(f_gt)

    # Read question data
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)
        faq_questions = [q for q in qs_ref['questions'] if q['category'] == 'faq' and 101 <= q['qid'] <= 150]

    # Read reference data
    with open(source_path, 'r', encoding='utf-8') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}  # Ensure keys are integers

    # Parameter ranges for k1 and b
    k1_values = np.arange(1.2, 2.1, 0.1)  # Adjust k1 in range 1.2 to 2.0
    b_values = np.arange(0.0, 1.1, 0.1)   # Adjust b in range 0.0 to 1.0
    best_precision = 0
    best_k1 = None
    best_b = None

    # Tune k1 and b
    for k1, b in product(k1_values, b_values):
        temp_answer_dict = {"answers": []}
        
        # Process filtered FAQ questions
        for q_dict in faq_questions:
            # Filter FAQ content
            corpus_dict_faq = {key: extract_text(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq, k1=k1, b=b)
            temp_answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        # Calculate precision for the current k1 and b
        precision = calculate_precision(temp_answer_dict, ground_truth)

        # Update best parameters if precision improves
        if precision > best_precision:
            best_precision = precision
            best_k1 = k1
            best_b = b

        print(f"k1: {k1}, b: {b}, Precision: {precision:.4f}")

    print(f"Best Precision: {best_precision:.4f} with k1: {best_k1} and b: {best_b}")

    # Save final results with best parameters
    answer_dict = {"answers": []}
    for q_dict in faq_questions:
        corpus_dict_faq = {key: extract_text(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
        retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq, k1=best_k1, b=best_b)
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)