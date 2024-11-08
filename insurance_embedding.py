import os
import json
import logging
import hashlib
import pickle
import re
import numpy as np
from tqdm import tqdm
import pdfplumber
import openai
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_fixed, stop_after_attempt
from multiprocessing import Pool, cpu_count
from functools import partial

if not os.path.exists('embeddings'):
    os.makedirs('embeddings')
    logging.info("建立 'embeddings' 資料夾")

# 定義嵌入緩存路徑
EMBEDDING_CACHE_PATH = 'embeddings/insurance_embedding_cache.pkl'

# 加載嵌入緩存
if os.path.exists(EMBEDDING_CACHE_PATH):
    with open(EMBEDDING_CACHE_PATH, 'rb') as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

def get_text_hash(text):
    """生成文本的哈希值，用於緩存鍵值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def get_embedding(text, model="text-embedding-ada-002"):
    # 添加類型檢查，確保傳入的 text 是字符串
    if not isinstance(text, str):
        logging.error(f"預期 text 為字符串，但獲得 {type(text)}。Text: {text}")
        return None
    text_hash = get_text_hash(text)
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    try:
        response = openai.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        embedding_cache[text_hash] = np.array(embedding)
        return embedding_cache[text_hash]
    except Exception as e:
        logging.error(f"生成嵌入時出錯，文本哈希 {text_hash}：{e}")
        raise e  # 讓 tenacity 能夠捕捉到錯誤並重試

def save_embedding_cache():
    """保存嵌入緩存到文件"""
    with open(EMBEDDING_CACHE_PATH, 'wb') as f:
        pickle.dump(embedding_cache, f)

# 函數：清洗文本，去除特殊字符但保留中文標點
def text_cleaning(text):
    # 去除特殊字符和多餘的空白，但保留中文標點
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# 函數：根據最大長度將文本拆分成多個塊
def split_text(text, max_length):
    """
    將文本分割成多個片段，每個片段的長度為 max_length。
    
    :param text: 待分割的文本
    :param max_length: 每個片段的最大長度
    :return: 分割後的文本片段列表
    """
    sentences = re.split('。|！|？|\n', text)  # 使用中文標點進行句子分割
    chunks = []
    chunk = ''
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + '。'
        else:
            if chunk:  # 確保不添加空的 chunk
                chunks.append(chunk)
            chunk = sentence + '。'
    if chunk:
        chunks.append(chunk)
    return chunks


# 函數：讀取並處理 PDF
def read_pdf(pdf_loc, page_infos: list = None, temp_dir='temp_texts', epsilon=5.0):
    pdf_text = ''
    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        logging.error(f"打開 PDF {pdf_loc} 時出錯：{e}")
        return pdf_text

    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages

    for page_num, page in enumerate(pages):
        try:
            # 提取文本
            text = page.extract_text()
            if text:
                # 去掉每行結尾的換行符並合併所有行
                pdf_text += text.replace('\n', '') + '\n'

            # 提取表格
            try:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = ''
                        for row in table:
                            row_text = ' '.join(str(cell) if cell is not None else '' for cell in row)
                            table_text += row_text + '\n'
                        pdf_text += table_text + '\n'
            except Exception as e:
                logging.error(f"從 PDF {pdf_loc} 的第 {page_num+1} 頁提取表格時出錯：{e}")
                continue

        except Exception as e:
            logging.error(f"處理 PDF {pdf_loc} 的第 {page_num+1} 頁時出錯：{e}")
            continue

    pdf.close()

    # 文本清洗
    pdf_text = text_cleaning(pdf_text)

    # 保存 pdf_text 到臨時文件
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    pdf_filename = os.path.basename(pdf_loc)
    text_filename = os.path.splitext(pdf_filename)[0] + '.txt'
    text_file_path = os.path.join(temp_dir, text_filename)
    try:
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(pdf_text)
    except Exception as e:
        logging.error(f"保存臨時文本文件 {text_file_path} 時出錯：{e}")

    return pdf_text

# 新增的頂層函數，用於處理 PDF 文件並拆分成多個塊
def process_pdf(fp, temp_dir='temp_texts'):
    try:
        doc_id = os.path.splitext(os.path.basename(fp))[0]
    except ValueError:
        logging.error(f"文件名 {fp} 無法獲取 doc_id。跳過。")
        return []  # 返回空列表以便在後續過濾
    pdf_text = read_pdf(fp, temp_dir=temp_dir)
    if not pdf_text:
        return []
    chunks = split_text(pdf_text, max_length=384)
    # 確保每個 chunk 是字符串
    chunked_data = [(f"{doc_id}_{i}", chunk) for i, chunk in enumerate(chunks) if isinstance(chunk, str)]
    return chunked_data

# 函數：加載數據並處理 PDF（並行處理版本）
def load_data_parallel(source_path, temp_dir='temp_texts', num_processes=None):
    if num_processes is None:
        num_processes = max(cpu_count() - 1, 1)  # 使用所有可用 CPU 核心，保留一個核心
    masked_file_ls = [file for file in os.listdir(source_path) if file.endswith('.pdf')]
    file_paths = [os.path.join(source_path, file) for file in masked_file_ls]

    with Pool(processes=num_processes) as pool:
        # 使用 functools.partial 來固定 temp_dir 參數
        process_func = partial(process_pdf, temp_dir=temp_dir)
        results = list(tqdm(pool.imap_unordered(process_func, file_paths), total=len(file_paths)))

    # 扁平化列表並過濾掉空結果
    results = [chunk for sublist in results if sublist for chunk in sublist]
    corpus_dict = {chunk_id: txt for chunk_id, txt in results}
    return corpus_dict

# 函數：使用 OpenAI 嵌入進行語義檢索
def semantic_retrieve_openai(qs, source, corpus_dict, corpus_embeddings):
    # 計算查詢的嵌入向量
    if not isinstance(qs, str):
        logging.error(f"查詢不是字串。查詢：{qs}，類型：{type(qs)}")
        return None
    query_embedding = get_embedding(qs)
    if query_embedding is None:
        logging.error(f"無法為查詢生成嵌入。查詢：{qs}")
        return None

    # 從來源中過濾相關的文檔塊
    relevant_embeddings = {}
    for doc_id in source:
        # 確保 doc_id 是字串
        doc_id_str = str(doc_id)
        pattern = re.compile(f"^{re.escape(doc_id_str)}_\\d+$")
        relevant_chunks = {cid: emb for cid, emb in corpus_embeddings.items() if pattern.match(cid)}
        if not relevant_chunks:
            logging.warning(f"在語料庫中找不到與來源文檔 ID {doc_id_str} 匹配的文本片段。")
            continue
        relevant_embeddings.update(relevant_chunks)

    if not relevant_embeddings:
        logging.error(f"沒有找到有效的嵌入向量，來源：{source}")
        return None  # 沒有有效的嵌入

    # 準備嵌入矩陣
    corpus_matrix = np.array(list(relevant_embeddings.values()))
    query_matrix = query_embedding.reshape(1, -1)

    # 計算相似度
    try:
        cos_sim = cosine_similarity(query_matrix, corpus_matrix)[0]
        # 找到最相關的文檔塊
        retrieved_index = np.argmax(cos_sim)
        retrieved_chunk_id = list(relevant_embeddings.keys())[retrieved_index]
        # 提取原始文檔 ID
        retrieved_doc_id_str = retrieved_chunk_id.split('_')[0]
        # 嘗試將 doc_id 轉換為整數
        try:
            retrieved_doc_id = int(retrieved_doc_id_str)
        except ValueError:
            logging.warning(f"文檔 ID {retrieved_doc_id_str} 不是數字，返回原始字串。")
            retrieved_doc_id = retrieved_doc_id_str
        return retrieved_doc_id
    except Exception as e:
        logging.error(f"計算餘弦相似度時出錯：{e}")
        return None

if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 您可以使用 argparse 來解析命令行參數，這裡為簡化，直接定義參數
    class Args:
        question_path = "dataset/preliminary/questions_example.json"
        source_path = "reference/"
        output_path = "dataset/preliminary/insurance_pred_retrieve.json"

    args = Args()

    answer_dict = {"answers": []}

    # 加載問題集
    try:
        with open(args.question_path, 'r', encoding='utf-8') as f:
            qs_ref = json.load(f)
    except Exception as e:
        logging.error(f"從 {args.question_path} 加載問題時出錯：{e}")
        exit(1)

    # 加載保險類別的資料
    source_path_insurance = os.path.join(args.source_path, 'insurance')
    if not os.path.exists(source_path_insurance):
        logging.error(f"保險資料來源路徑不存在：{source_path_insurance}")
        exit(1)
    logging.info("加載保險語料庫...")
    corpus_dict_insurance = load_data_parallel(source_path_insurance)

    # 預先計算保險語料庫的嵌入向量
    logging.info("計算保險語料庫的嵌入向量...")
    corpus_embeddings_insurance = {}
    for chunk_id, chunk_text in tqdm(corpus_dict_insurance.items(), desc="計算嵌入"):
        if not isinstance(chunk_text, str):
            logging.error(f"文檔塊 {chunk_id} 的文本不是字符串。文本：{chunk_text}，類型：{type(chunk_text)}")
            continue
        embedding = get_embedding(chunk_text)
        if embedding is not None:
            corpus_embeddings_insurance[chunk_id] = embedding
        else:
            logging.error(f"無法為文檔塊 {chunk_id} 生成嵌入。")

    # 處理每個問題
    for idx, q_dict in enumerate(tqdm(qs_ref.get('questions', []), desc="處理問題")):
        qid = q_dict.get('qid')
        category = q_dict.get('category')
        query = q_dict.get('query')
        source = q_dict.get('source', [])

        if not all([qid, category, query, source]):
            logging.warning(f"問題中缺少欄位：{q_dict}")
            continue

        if category == 'insurance':
            try:
                retrieved = semantic_retrieve_openai(query, source, corpus_dict_insurance, corpus_embeddings_insurance)
                # 將 qid 和 retrieve 轉換為整數（如果可能）
                try:
                    qid_int = int(qid)
                except ValueError:
                    qid_int = qid  # 如果 qid 不是數字，保留原始值
                answer_dict['answers'].append({"qid": qid_int, "retrieve": retrieved})
            except Exception as e:
                logging.error(f"處理問題 ID {qid} 時出錯：{e}")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
        else:
            logging.info(f"跳過非保險類別的問題，問題 ID：{qid}")

    # 將答案保存為 JSON 文件
    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"成功將答案保存到 {args.output_path}")
    except Exception as e:
        logging.error(f"保存答案到 {args.output_path} 時出錯：{e}")

    # 保存嵌入緩存
    save_embedding_cache()
    logging.info("嵌入緩存已成功保存。")