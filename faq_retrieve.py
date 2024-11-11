import os
import json
from rank_bm25 import BM25Okapi
from ckip_transformers.nlp import CkipWordSegmenter

# Initialize CKIP Word Segmenter
ws_driver = CkipWordSegmenter()

# Define a function to extract text content from FAQ data
def extract_text(value):
    texts = []
    for item in value:
        if 'question' in item:
            texts.append(item['question'])
        if 'answers' in item:
            texts.extend(item['answers'])
    return ' '.join(texts)

# Use CKIP for tokenization with BM25
def tokenize(text):
    # CKIP expects a list of strings for tokenization, returning a list of lists
    tokens = ws_driver([text])[0]
    return tokens

# Main program
if __name__ == "__main__":
    answer_dict = {"answers": []}

    # Read question data
    question_path = "dataset/preliminary/questions_preliminary.json"
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)
        faq_questions = [q for q in qs_ref['questions'] if q['category'] == 'faq' and 601 <= q['qid'] <= 900]

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

        # BM25 retrieval to find the best matching document
        top_n = 1
        bm25_candidates = bm25.get_top_n(tokenized_query, list(filtered_corpus.values()), n=top_n)
        
        # Find the ID of the best matching document
        best_match_id = None
        if bm25_candidates:
            best_match_text = bm25_candidates[0]
            best_match_id = [key for key, value in filtered_corpus.items() if value == best_match_text][0]

        # Append best match to answer dictionary
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": best_match_id})

    # Save results to JSON file
    output_path = "dataset/preliminary/faq_pred_retrieve.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
