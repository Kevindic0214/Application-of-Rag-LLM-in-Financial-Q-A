import os
import json
from rank_bm25 import BM25Okapi
import jieba

# Define a function to extract text content from FAQ data
def extract_text(value):
    texts = []
    for item in value:
        if 'question' in item:
            texts.append(item['question'])
        if 'answers' in item:
            texts.extend(item['answers'])
    return ' '.join(texts)

# Use jieba for tokenization with BM25
def tokenize(text):
    return list(jieba.cut(text))

# Main program
if __name__ == "__main__":
    answer_dict = {"answers": []}

    # Read question data
    question_path = "dataset/preliminary/questions_example.json"
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)
        faq_questions = [q for q in qs_ref['questions'] if q['category'] == 'faq' and 101 <= q['qid'] <= 150]

    # Read reference data and prepare the corpus
    source_path = "reference/faq/pid_map_content.json"
    with open(source_path, 'r', encoding='utf-8') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        corpus_dict_faq = {key: extract_text(value) for key, value in key_to_source_dict.items()}

    # Process each FAQ question
    for q_dict in faq_questions:
        query = q_dict['query']
        tokenized_query = tokenize(query)

        # Filter corpus based on `source` for the current question
        source_ids = q_dict['source']
        filtered_corpus = {key: corpus_dict_faq[key] for key in source_ids if key in corpus_dict_faq}
        tokenized_corpus = [tokenize(doc) for doc in filtered_corpus.values()]

        # Initialize BM25 model with filtered corpus
        bm25 = BM25Okapi(tokenized_corpus)

        # BM25 retrieval to find the top_k matching documents
        top_k = 5
        bm25_candidates = bm25.get_top_n(tokenized_query, list(filtered_corpus.values()), n=top_k)
        
        # Find the IDs and contents of the top_k results
        candidate_ids = [key for key, value in filtered_corpus.items() if value in bm25_candidates]
        
        # Print out the results for inspection
        print(f"\nQID: {q_dict['qid']}, Query: {query}")
        for i, (doc_id, doc_text) in enumerate(zip(candidate_ids, bm25_candidates), start=1):
            print(f"Rank {i}: ID = {doc_id}, Content = {doc_text[:100]}...")  # Print first 100 chars of content

        # Append the top-1 best match ID to the answer dictionary (baseline answer)
        if candidate_ids:
            best_match_id = candidate_ids[0]
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": best_match_id})

    # # Save results to JSON file
    # output_path = "dataset/preliminary/faq_pred_retrieve.json"
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(answer_dict, f, ensure_ascii=False, indent=4)
