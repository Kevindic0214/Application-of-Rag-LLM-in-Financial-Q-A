import os
import json
import jieba  # For Chinese text segmentation
import pdfplumber  # Tool for extracting text from PDF files
from tqdm import tqdm
from rank_bm25 import BM25Okapi  # Using BM25 algorithm for document retrieval


# Specify fixed paths
question_path = "dataset/preliminary/questions_example.json"  # Path to the question file
source_path = "reference"  # Root path of reference materials
output_path = "dataset/preliminary/pred_retrieve.json"  # Output path for prediction results


# Load reference materials, return a dictionary with file names as keys and PDF content as values
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # Get the list of files in the folder
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # Read the text of each PDF file and store it in a dictionary with the file name as the key and the text content as the value
    return corpus_dict


# Read a single PDF file and return its text content
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # Open the specified PDF file

    # TODO: You can use other methods to read data or process multimodal data (tables, images, etc.) in the PDF

    # If a page range is specified, extract only those pages, otherwise extract all pages
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # Loop through each page
        text = page.extract_text()  # Extract the text content of the page
        if text:
            pdf_text += text
    pdf.close()  # Close the PDF file

    return pdf_text  # Return the extracted text


# Retrieve answers based on the query and specified source
def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # [TODO] You can replace with other retrieval methods to improve performance

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # Tokenize each document
    bm25 = BM25Okapi(tokenized_corpus)  # Use BM25 algorithm to build the retrieval model
    tokenized_query = list(jieba.cut_for_search(qs))  # Tokenize the query
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # Retrieve the most relevant document based on the query
    a = ans[0]
    # Find the file name corresponding to the best matching text
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # Return the file name


if __name__ == "__main__":
    answer_dict = {"answers": []}  # Initialize dictionary

    with open(question_path, 'rb') as f:
        qs_ref = json.load(f)  # Read the question file

    # Load reference materials for insurance and finance types
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(source_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance)

    # Read reference materials for FAQ
    with open(os.path.join(source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # Read the reference material file
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    # Retrieve and save results based on the question type
    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # Retrieve finance-related questions
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # Add the result to the dictionary
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            # Retrieve insurance-related questions
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            # Retrieve FAQ-related questions
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Unknown category")  # Raise an error if the category is unknown

    # Save the answer dictionary as a JSON file
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # Save the file, ensuring format and non-ASCII characters