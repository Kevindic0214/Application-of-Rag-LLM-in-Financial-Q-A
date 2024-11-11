import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Function: Generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding)

# Function: Extract text from FAQ data
def extract_text(value):
    texts = []
    for item in value:
        if 'question' in item:
            texts.append(item['question'])
        if 'answers' in item:
            texts.extend(item['answers'])
    return ' '.join(texts)

# Main program
if __name__ == "__main__":
    answer_dict = {"answers": []}

    # Read question data
    question_path = "dataset/preliminary/questions_preliminary.json"
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)
        faq_questions = [q for q in qs_ref['questions'] if q['category'] == 'faq' and 601 <= q['qid'] <= 900]

    # Read reference data and prepare corpus
    source_path = "reference/faq/pid_map_content.json"
    with open(source_path, 'r', encoding='utf-8') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        corpus_dict_faq = {key: extract_text(value) for key, value in key_to_source_dict.items()}

    # Precompute embeddings for all documents in the corpus
    corpus_embeddings = {}
    print("Generating embeddings for the corpus documents...")
    for key, text in tqdm(corpus_dict_faq.items(), desc="Corpus Embeddings", unit="doc"):
        corpus_embeddings[key] = get_embedding(text)

    # Process each FAQ question with a progress bar
    print("\nProcessing FAQ questions...")
    for q_dict in tqdm(faq_questions, desc="FAQ Questions", unit="query"):
        query = q_dict['query']
        
        # Compute embedding for the query
        query_embedding = get_embedding(query)

        # Filter corpus based on `source`
        source_ids = q_dict['source']
        filtered_embeddings = {key: corpus_embeddings[key] for key in source_ids if key in corpus_embeddings}

        # Compute cosine similarity between the query and each document
        cos_scores = {key: cosine_similarity([query_embedding], [emb])[0][0] for key, emb in filtered_embeddings.items()}

        # Find the document ID with the highest similarity
        best_match_id = max(cos_scores, key=cos_scores.get) if cos_scores else None

        # Add the best match result to the answer dictionary
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": best_match_id})

    # Save results to a JSON file
    output_path = "dataset/preliminary/faq_pred_retrieve.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)