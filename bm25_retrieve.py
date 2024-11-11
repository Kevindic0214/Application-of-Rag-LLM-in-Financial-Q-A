import os
import json
import re
import logging
import jieba
import pdfplumber
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from ckip_transformers.nlp import CkipWordSegmenter
from multiprocessing import Pool, cpu_count, current_process
import numpy as np
import time
import traceback

# 初始化 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 路徑設置
question_path = "dataset/preliminary/questions_preliminary.json"
source_path = "reference"
output_path = "dataset/preliminary/pred_retrieve.json"

# 初始化 CkipWordSegmenter
def initialize_ckip():
    logging.info("Initializing CKIP Word Segmenter...")
    start_time = time.time()
    try:
        ws = CkipWordSegmenter(model="bert-base")
        logging.info(f"CKIP Word Segmenter initialized in {time.time() - start_time:.2f} seconds.")
        return ws
    except Exception as e:
        logging.error(f"Failed to initialize CKIP Word Segmenter: {e}")
        traceback.print_exc()
        raise

# 文本清洗函數，去除特殊字符但保留中文標點
def text_cleaning(text):
    original_length = len(text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：]', '', text)
    text = re.sub(r'\s+', ' ', text)
    cleaned_length = len(text)
    logging.debug(f"Cleaned text from {original_length} to {cleaned_length} characters.")
    return text

# 根據最大長度將文本拆分成多個塊
def split_text(text, max_length=500):
    sentences = re.split('。|！|？|\n', text)
    chunks = []
    chunk = ''
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + '。'
        else:
            if chunk:
                chunks.append(chunk)
                logging.debug(f"Created chunk of length {len(chunk)}.")
            chunk = sentence + '。'
    if chunk:
        chunks.append(chunk)
        logging.debug(f"Created final chunk of length {len(chunk)}.")
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

# 讀取單個 PDF 文件並進行處理
def process_pdf(pdf_path):
    process_name = current_process().name
    logging.debug(f"[{process_name}] Starting to process PDF: {pdf_path}")
    try:
        doc_id_str = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_id = int(doc_id_str)  # 假設 doc_id 是整數
    except ValueError:
        logging.error(f"[{process_name}] 文件名 {pdf_path} 無法獲取有效的 doc_id。跳過。")
        return []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace('\n', '') + '\n'
                    logging.debug(f"[{process_name}] Extracted text from page {page_num}.")
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = ''
                            for row in table:
                                row_text = ' '.join(str(cell) if cell is not None else '' for cell in row)
                                table_text += row_text + '\n'
                            text += table_text + '\n'
                            logging.debug(f"[{process_name}] Extracted table from page {page_num}.")
                except Exception as e:
                    logging.error(f"[{process_name}] 從 PDF {pdf_path} 的第 {page_num} 頁提取表格時出錯：{e}")
                    traceback.print_exc()
                    continue
    except Exception as e:
        logging.error(f"[{process_name}] 讀取 PDF 文件 {pdf_path} 時出錯：{e}")
        traceback.print_exc()
        return []
    
    # 文本清洗和切分
    text = text_cleaning(text)
    chunks = split_text(text, max_length=500)
    logging.info(f"[{process_name}] Document {doc_id} split into {len(chunks)} chunks.")
    return [(doc_id, chunk) for chunk in chunks if isinstance(chunk, str)]

# 並行處理所有 PDF 文件
def load_data_parallel(source_path, clean=False):
    logging.info(f"Loading data from {source_path}...")
    pdf_files = [file for file in os.listdir(source_path) if file.endswith('.pdf')]
    pdf_paths = [os.path.join(source_path, file) for file in pdf_files]
    start_time = time.time()
    with Pool(processes=max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap_unordered(process_pdf, pdf_paths), total=len(pdf_paths), desc="Processing PDFs"))
    logging.info(f"Completed processing PDFs in {time.time() - start_time:.2f} seconds.")
    
    # 扁平化並處理結果
    corpus_list = [item for sublist in results for item in sublist if sublist]
    logging.debug(f"Total chunks collected: {len(corpus_list)}.")
    
    corpus_dict = {}
    for doc_id, chunk in corpus_list:
        if doc_id not in corpus_dict:
            corpus_dict[doc_id] = []
        cleaned_chunk = text_cleaning(chunk) if clean else chunk
        corpus_dict[doc_id].append(cleaned_chunk)
        logging.debug(f"Added chunk to doc_id {doc_id}. Total chunks for this doc: {len(corpus_dict[doc_id])}.")
    
    # 檢查已載入的文件
    loaded_files = set(corpus_dict.keys())
    logging.info(f"已加載 {len(loaded_files)} 個文件，總共包含 {len(corpus_list)} 個塊。")
    return corpus_dict, loaded_files

# 使用 CkipWordSegmenter 進行斷詞
def ckip_tokenize(ws, text):
    logging.debug("Starting CKIP tokenization.")
    sentences = [text]
    try:
        start_time = time.time()
        ws_results = ws(sentences)
        tokens = []
        for ws_result in ws_results:
            tokens.extend(ws_result)
        logging.debug(f"CKIP tokenization completed in {time.time() - start_time:.2f} seconds. Number of tokens: {len(tokens)}.")
        return tokens
    except Exception as e:
        logging.error(f"CKIP 分詞時出現錯誤：{e}")
        traceback.print_exc()
        return []

# 通用檢索函數
def bm25_retrieve(query, source, corpus_dict, tokenizer, top_n=3):
    logging.debug("Starting BM25 retrieval.")
    # 過濾出在 corpus_dict 中的 source 文件ID
    filtered_corpus = {doc_id: corpus_dict[doc_id] for doc_id in source if doc_id in corpus_dict}
    
    # 若 filtered_corpus 為空，則返回空結果
    if not filtered_corpus:
        logging.warning(f"No documents found in corpus_dict for the given source IDs: {source}")
        return {}
    
    # 收集所有斷詞後的塊，並建立塊到 doc_id 的映射
    tokenized_chunks = []
    chunk_to_doc_id = []
    for doc_id, chunks in filtered_corpus.items():
        for chunk in chunks:
            tokens = tokenizer(chunk)
            if tokens:
                tokenized_chunks.append(tokens)
                chunk_to_doc_id.append(doc_id)
    logging.debug(f"Total tokenized chunks: {len(tokenized_chunks)}.")
    
    if not tokenized_chunks:
        logging.warning("No tokens available after tokenization.")
        return {}
    
    # 初始化 BM25 模型
    start_time = time.time()
    bm25 = BM25Okapi(tokenized_chunks)
    logging.debug(f"BM25 model initialized in {time.time() - start_time:.2f} seconds.")
    
    # Tokenize query
    tokenized_query = tokenizer(query)
    logging.debug(f"Tokenized query: {tokenized_query}")
    
    # 計算分數
    start_time = time.time()
    scores = bm25.get_scores(tokenized_query)
    logging.debug(f"BM25 scoring completed in {time.time() - start_time:.2f} seconds.")
    
    # 累計每個 doc_id 的分數
    doc_scores = {}
    for idx, score in enumerate(scores):
        doc_id = chunk_to_doc_id[idx]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
    
    # 排序並返回 top_n 結果
    top_n_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    logging.debug(f"Top {top_n} documents retrieved: {top_n_docs}")
    return {doc_id: score for doc_id, score in top_n_docs}

# 主程式
if __name__ == "__main__":
    start_time = time.time()
    answer_dict = {"answers": []}
    try:
        ws = initialize_ckip()
    except Exception as e:
        logging.critical("Cannot proceed without CKIP Word Segmenter. Exiting.")
        exit(1)
    
    try:
        with open(question_path, 'r', encoding='utf8') as f:
            qs_ref = json.load(f)
        logging.info(f"Loaded {len(qs_ref['questions'])} questions from {question_path}.")
    except Exception as e:
        logging.error(f"Failed to load questions from {question_path}: {e}")
        traceback.print_exc()
        exit(1)
    
    # 加載並清洗 finance 和 insurance 類別的數據
    try:
        corpus_dict_insurance, loaded_files_insurance = load_data_parallel(os.path.join(source_path, 'insurance'), clean=True)
        corpus_dict_finance, loaded_files_finance = load_data_parallel(os.path.join(source_path, 'finance'), clean=True)
    except Exception as e:
        logging.error(f"Failed to load and process insurance or finance data: {e}")
        traceback.print_exc()
        exit(1)
    
    # FAQ 類別不進行清洗
    try:
        with open(os.path.join(source_path, 'faq/pid_map_content.json'), 'r', encoding='utf8') as f_s:
            key_to_source_dict = {int(key): value for key, value in json.load(f_s).items()}
        logging.info(f"Loaded FAQ data with {len(key_to_source_dict)} entries.")
    except Exception as e:
        logging.error(f"Failed to load FAQ data: {e}")
        traceback.print_exc()
        key_to_source_dict = {}
    
    # 定義斷詞器
    def jieba_tokenizer(text):
        tokens = list(jieba.cut_for_search(text))
        logging.debug(f"Jieba tokenized text into {len(tokens)} tokens.")
        return tokens
    
    def ckip_tokenizer_func(text):
        tokens = ckip_tokenize(ws, text)
        logging.debug(f"CKIP tokenized text into {len(tokens)} tokens.")
        return tokens
    
    logging.info("Starting to process questions...")
    # 依照問題檢索文件
    for idx, q_dict in enumerate(tqdm(qs_ref['questions'], desc="Processing questions"), start=1):
        category, query, qid, source = q_dict['category'], q_dict['query'], q_dict['qid'], q_dict['source']
        
        logging.debug(f"Processing question {idx}/{len(qs_ref['questions'])}: QID={qid}, Category={category}")
        
        # 確保 source 是整數列表
        if not isinstance(source, list):
            source = [source]
        try:
            source = [int(s) for s in source]
            logging.debug(f"Converted source to integers: {source}")
        except ValueError:
            logging.error(f"問題 ID {qid} 的 source 包含非整數值。跳過。")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 選擇對應的參考數據集
        if category == 'finance':
            corpus_dict_current = corpus_dict_finance
            logging.debug(f"Selected finance corpus for QID={qid}.")
        elif category == 'insurance':
            corpus_dict_current = corpus_dict_insurance
            logging.debug(f"Selected insurance corpus for QID={qid}.")
        elif category == 'faq':
            # 假設 key_to_source_dict 的值是文本內容
            corpus_dict_current = {key: str(value) for key, value in key_to_source_dict.items() if key in source}
            logging.debug(f"Selected FAQ corpus for QID={qid}. Number of documents: {len(corpus_dict_current)}.")
        else:
            logging.error(f"Unknown category: {category} for question ID {qid}. Skipping.")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 定義使用的斷詞器
        if category in ['finance', 'insurance']:
            tokenizer_jieba = jieba_tokenizer
            tokenizer_ckip = ckip_tokenizer_func
            logging.debug(f"Using Jieba and CKIP tokenizers for QID={qid}.")
        elif category == 'faq':
            # 根據需要選擇斷詞器，這裡選擇使用 Jieba 和 CKIP
            tokenizer_jieba = jieba_tokenizer
            tokenizer_ckip = ckip_tokenizer_func
            logging.debug(f"Using Jieba and CKIP tokenizers for FAQ QID={qid}.")
        else:
            tokenizer_jieba = jieba_tokenizer
            tokenizer_ckip = ckip_tokenizer_func
            logging.debug(f"Using default tokenizers for QID={qid}.")

        # 進行BM25檢索
        if category in ['finance', 'insurance']:
            logging.debug(f"Retrieving using Jieba tokenizer for QID={qid}")
            results_jieba = bm25_retrieve(query, source, corpus_dict_current, tokenizer_jieba)
            
            logging.debug(f"Retrieving using CKIP tokenizer for QID={qid}")
            results_ckip = bm25_retrieve(query, source, corpus_dict_current, tokenizer_ckip)
        elif category == 'faq':
            logging.debug(f"Retrieving using Jieba tokenizer for FAQ QID={qid}")
            # FAQ 可能沒有分塊，直接檢索
            tokenized_corpus = [jieba_tokenizer(text) for text in corpus_dict_current.values()]
            # 濾除空的斷詞結果
            tokenized_corpus = [tokens for tokens in tokenized_corpus if tokens]
            logging.debug(f"Tokenized FAQ corpus into {len(tokenized_corpus)} token lists.")
            if not tokenized_corpus:
                logging.warning(f"No valid tokens in FAQ corpus for question ID {qid}.")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue
            # 初始化 BM25 模型
            bm25 = BM25Okapi(tokenized_corpus)
            logging.debug(f"Initialized BM25 model for FAQ QID={qid}.")
            tokenized_query = jieba_tokenizer(query)
            logging.debug(f"Tokenized query for FAQ QID={qid}: {tokenized_query}")
            scores = bm25.get_scores(tokenized_query)
            # 將分數映射回 doc_id
            doc_ids = list(corpus_dict_current.keys())
            results_jieba = {doc_ids[i]: score for i, score in enumerate(scores)}
            # 可以添加 CKIP 的結果
            logging.debug(f"Retrieving using CKIP tokenizer for FAQ QID={qid}")
            results_ckip = {}
        else:
            results_jieba = {}
            results_ckip = {}
            logging.debug(f"No retrieval performed for QID={qid} due to unknown category.")

        # 合併Jieba和CKIP的結果並加總分數
        logging.debug(f"Merging Jieba and CKIP results for QID={qid}")
        combined_results = results_jieba.copy()
        for doc_id, score in results_ckip.items():
            combined_results[doc_id] = combined_results.get(doc_id, 0) + score
            logging.debug(f"Added CKIP score for doc_id {doc_id}: {score:.4f}")

        # 獲取最高分文件ID
        if combined_results:
            best_doc_id = max(combined_results, key=combined_results.get)
            best_score = combined_results[best_doc_id]
            logging.info(f"QID={qid}: Best doc_id={best_doc_id} with score={best_score:.4f}")
        else:
            best_doc_id = None
            logging.info(f"QID={qid}: No relevant documents found.")
        
        # 添加結果
        answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})
    
    logging.info(f"Processed all questions in {time.time() - start_time:.2f} seconds.")
    
    # 保存結果到JSON文件
    try:
        logging.info(f"Saving retrieval results to {output_path}...")
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save results to {output_path}: {e}")
        traceback.print_exc()
    
    # 關閉 CKIP 斷詞器
    try:
        ws.close()
        logging.info("CKIP Word Segmenter closed.")
    except Exception as e:
        logging.error(f"Failed to close CKIP Word Segmenter: {e}")
        traceback.print_exc()
    
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")
