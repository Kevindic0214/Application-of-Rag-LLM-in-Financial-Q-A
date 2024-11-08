import os
import json
import pdfplumber
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from ckip_transformers.nlp import CkipWordSegmenter

# Initialize CKIP word segmenter with GPU support (assuming device ID 0)
ws_driver = CkipWordSegmenter(model="bert-base", device=0)

# Specify paths
question_path = "dataset/preliminary/questions_example.json"
source_path = "reference"
output_path = "dataset/preliminary/insurance_pred_retrieve.json"

# Load reference materials and return a dictionary with file names as keys and PDF content as values
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {
        int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) 
        for file in tqdm(masked_file_ls) if file.endswith('.pdf')
    }
    return corpus_dict

# Read a single PDF file and return its text content
def read_pdf(pdf_loc, page_infos=None):
    pdf = pdfplumber.open(pdf_loc)
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''.join([page.extract_text() or '' for page in pages])
    pdf.close()
    return pdf_text

# Tokenize text using CKIP word segmenter and cache results
token_cache = {}
def tokenize_with_ckip(text):
    if text not in token_cache:
        words = ws_driver([text])
        token_cache[text] = [word for word in words[0]]
    return token_cache[text]

# Retrieve the top 1 candidate document using BM25
def BM25_retrieve_top1(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source if int(file) in corpus_dict]
    tokenized_corpus = [tokenize_with_ckip(doc) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize_with_ckip(qs)
    best_match = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # Get the top 1 candidate
    
    # Find the file name corresponding to the best matching text
    res = [key for key, value in corpus_dict.items() if value == best_match[0]]
    return res[0] if res else None  # Return the file name or None if not found

if __name__ == "__main__":
    answer_dict = {"answers": []}

    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)

    # Load insurance reference materials
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)

    # Retrieve and save results
    for q_dict in tqdm(qs_ref['questions'], desc="Processing questions"):
        if q_dict['category'] == 'insurance':
            # Retrieve top 1 candidate with BM25
            best_match_file = BM25_retrieve_top1(
                q_dict['query'], q_dict['source'], corpus_dict_insurance
            )
            # Add the result to the answer dictionary
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": best_match_file})

    # Save to JSON
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
