import os
import json
import re
import logging
import jieba
import pdfplumber
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from ckip_transformers.nlp import CkipWordSegmenter
from multiprocessing import Pool, cpu_count
import numpy as np

# 初始化 logging
logging.basicConfig(level=logging.INFO)

# 路徑設置
question_path = "dataset/preliminary/questions_preliminary.json"
source_path = "reference"
output_path = "dataset/preliminary/pred_retrieve.json"

# 初始化 CkipWordSegmenter
def initialize_ckip():
    return CkipWordSegmenter(model="bert-base")

# 文本清洗函數，去除特殊字符但保留中文標點
def text_cleaning(text):
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：]', '', text)
    text = re.sub(r'\s+', ' ', text)
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
            chunk = sentence + '。'
    if chunk:
        chunks.append(chunk)
    return chunks

# 讀取單個 PDF 文件並進行處理
def process_pdf(pdf_path):
    try:
        doc_id_str = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_id = int(doc_id_str)  # 假設 doc_id 是整數
    except ValueError:
        logging.error(f"文件名 {pdf_path} 無法獲取有效的 doc_id。跳過。")
        return []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace('\n', '') + '\n'
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = ''
                            for row in table:
                                row_text = ' '.join(str(cell) if cell is not None else '' for cell in row)
                                table_text += row_text + '\n'
                            text += table_text + '\n'
                except Exception as e:
                    logging.error(f"從 PDF {pdf_path} 的表格提取時出錯：{e}")
                    continue
    except Exception as e:
        logging.error(f"讀取 PDF 文件 {pdf_path} 時出錯：{e}")
        return []
    
    # 文本清洗和切分
    text = text_cleaning(text)
    chunks = split_text(text, max_length=500)
    return [(doc_id, chunk) for chunk in chunks if isinstance(chunk, str)]

# 並行處理所有 PDF 文件
def load_data_parallel(source_path, clean=False):
    pdf_files = [file for file in os.listdir(source_path) if file.endswith('.pdf')]
    pdf_paths = [os.path.join(source_path, file) for file in pdf_files]
    with Pool(processes=max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap_unordered(process_pdf, pdf_paths), total=len(pdf_paths), desc="Processing PDFs"))

    # 扁平化並處理結果
    # 將嵌套的列表轉換為單一列表
    corpus_list = [item for sublist in results for item in sublist if sublist]
    
    corpus_dict = {}
    for doc_id, chunk in corpus_list:
        if doc_id not in corpus_dict:
            corpus_dict[doc_id] = []
        corpus_dict[doc_id].append(text_cleaning(chunk) if clean else chunk)

    # 檢查已載入的文件
    loaded_files = set(corpus_dict.keys())
    logging.info(f"已加載 {len(loaded_files)} 個文件")
    return corpus_dict, loaded_files

# 使用 CkipWordSegmenter 進行斷詞
def ckip_tokenize(ws, text):
    sentences = [text]
    try:
        ws_results = ws(sentences)
        tokens = []
        for ws_result in ws_results:
            tokens.extend(ws_result)
        return tokens
    except Exception as e:
        logging.error(f"CKIP 分詞時出現錯誤：{e}")
        return []

# 通用檢索函數
def bm25_retrieve(query, source, corpus_dict, tokenizer, top_n=3):
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
    
    if not tokenized_chunks:
        logging.warning("No tokens available after tokenization.")
        return {}
    
    # 初始化 BM25 模型
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = tokenizer(query)
    scores = bm25.get_scores(tokenized_query)
    
    # 累計每個 doc_id 的分數
    doc_scores = {}
    for idx, score in enumerate(scores):
        doc_id = chunk_to_doc_id[idx]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
    
    # 排序並返回 top_n 結果
    top_n_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {doc_id: score for doc_id, score in top_n_docs}

# 主程式
if __name__ == "__main__":
    answer_dict = {"answers": []}
    ws = initialize_ckip()
    
    with open(question_path, 'r', encoding='utf8') as f:
        qs_ref = json.load(f)

    # 加載並清洗 finance 和 insurance 類別的數據
    corpus_dict_insurance, loaded_files_insurance = load_data_parallel(os.path.join(source_path, 'insurance'), clean=True)
    corpus_dict_finance, loaded_files_finance = load_data_parallel(os.path.join(source_path, 'finance'), clean=True)
    
    # FAQ 類別不進行清洗
    with open(os.path.join(source_path, 'faq/pid_map_content.json'), 'r', encoding='utf8') as f_s:
        key_to_source_dict = {int(key): value for key, value in json.load(f_s).items()}
    
    # 定義斷詞器
    def jieba_tokenizer(text):
        return list(jieba.cut_for_search(text))
    
    def ckip_tokenizer_func(text):
        return ckip_tokenize(ws, text)
    
    # 依照問題檢索文件
    for q_dict in tqdm(qs_ref['questions'], desc="Processing questions"):
        category, query, qid, source = q_dict['category'], q_dict['query'], q_dict['qid'], q_dict['source']
        
        # 確保 source 是整數列表
        if not isinstance(source, list):
            source = [source]
        try:
            source = [int(s) for s in source]
        except ValueError:
            logging.error(f"問題 ID {qid} 的 source 包含非整數值。跳過。")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 選擇對應的參考數據集
        if category == 'finance':
            corpus_dict_current = corpus_dict_finance
        elif category == 'insurance':
            corpus_dict_current = corpus_dict_insurance
        elif category == 'faq':
            # 假設 key_to_source_dict 的值是文本內容
            corpus_dict_current = {key: str(value) for key, value in key_to_source_dict.items() if key in source}
        else:
            logging.error(f"Unknown category: {category} for question ID {qid}. Skipping.")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 定義使用的斷詞器
        if category in ['finance', 'insurance']:
            tokenizer_jieba = jieba_tokenizer
            tokenizer_ckip = ckip_tokenizer_func
        elif category == 'faq':
            tokenizer_jieba = jieba_tokenizer  # 根據需要選擇斷詞器
            tokenizer_ckip = ckip_tokenizer_func
        else:
            tokenizer_jieba = jieba_tokenizer
            tokenizer_ckip = ckip_tokenizer_func

        # 進行BM25檢索
        if category in ['finance', 'insurance']:
            results_jieba = bm25_retrieve(query, source, corpus_dict_current, tokenizer_jieba)
            results_ckip = bm25_retrieve(query, source, corpus_dict_current, tokenizer_ckip)
        elif category == 'faq':
            # FAQ 可能沒有分塊，直接檢索
            tokenized_corpus = [jieba_tokenizer(text) for text in corpus_dict_current.values()]
            # 濾除空的斷詞結果
            tokenized_corpus = [tokens for tokens in tokenized_corpus if tokens]
            if not tokenized_corpus:
                logging.warning(f"No valid tokens in FAQ corpus for question ID {qid}.")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue
            # 初始化 BM25 模型
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = jieba_tokenizer(query)
            scores = bm25.get_scores(tokenized_query)
            # 將分數映射回 doc_id
            doc_ids = list(corpus_dict_current.keys())
            results_jieba = {doc_ids[i]: score for i, score in enumerate(scores)}
            # 可以添加 CKIP 的結果
            results_ckip = {}
        else:
            results_jieba = {}
            results_ckip = {}
        
        # 合併Jieba和CKIP的結果並加總分數
        combined_results = results_jieba.copy()
        for doc_id, score in results_ckip.items():
            combined_results[doc_id] = combined_results.get(doc_id, 0) + score

        # 獲取最高分文件ID
        best_doc_id = max(combined_results, key=combined_results.get) if combined_results else None

        # 添加結果
        answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})
    
    # 保存結果到JSON文件
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    
    # 關閉 CKIP 斷詞器
    ws.close()
