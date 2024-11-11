import os
import json
import openai
import numpy as np
import logging
import pickle
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, Lock
from functools import partial
from tenacity import retry, wait_fixed, stop_after_attempt
from tqdm import tqdm
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from ckip_transformers.nlp import CkipWordSegmenter
from llama_parse import LlamaParse
import nest_asyncio
from openai import OpenAI
from llama_index.core.schema import TextNode


# 配置日誌設置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設置 OpenAI API 密鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logging.error("未找到 OPENAI_API_KEY，請設置環境變量。")
    exit(1)
else:
    logging.info("OpenAI API 密鑰已設置")

# 初始化 nest_asyncio 以允許在現有事件循環中運行異步操作
nest_asyncio.apply()
logging.info("nest_asyncio 已應用")

# 初始化 LlamaParse
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
if not LLAMA_PARSE_API_KEY:
    logging.error("未找到 LLAMA_PARSE_API_KEY，請設置環境變量。")
    exit(1)
else:
    logging.info("LlamaParse API 金鑰已設置")

parser = LlamaParse(
    api_key=LLAMA_PARSE_API_KEY,
    result_type="markdown",
    verbose=True,
    language="ch_tra",
)
logging.info("LlamaParse 已初始化")

# 初始化 CKIP 分詞器
ws_driver = CkipWordSegmenter(model="bert-base")
logging.info("CKIP 分詞器已初始化")

# 初始化 MarkdownElementNodeParser
node_parser = MarkdownElementNodeParser(
    arbitrary_types_allowed=True  # 允許任意類型的節點
)
logging.info("MarkdownElementNodeParser 已初始化")

# 初始化嵌入模型（移除，因為使用 get_embedding 函數）
# embedding_model = OpenAIEmbeddings()
# logging.info("OpenAIEmbeddings 已初始化")

# 創建必要的文件夾
os.makedirs('contexts', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)
os.makedirs('temp_texts', exist_ok=True)

# 定義嵌入緩存路徑
EMBEDDING_CACHE_PATH = 'embeddings/finance_embedding_cache.pkl'

# 加載嵌入緩存
embedding_cache_lock = Lock()
if os.path.exists(EMBEDDING_CACHE_PATH):
    with open(EMBEDDING_CACHE_PATH, 'rb') as f:
        embedding_cache = pickle.load(f)
    logging.info("嵌入緩存已加載")
else:
    embedding_cache = {}
    logging.info("嵌入緩存不存在，已初始化空字典")

# 生成文本的哈希值，用於嵌入緩存
def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# 重試機制裝飾器，用於 OpenAI API 調用
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def get_embedding(text, model="text-embedding-ada-002"):
    if not isinstance(text, str):
        logging.error(f"預期 text 為字符串，但獲得 {type(text)}。Text: {text}")
        return None
    text_hash = get_text_hash(text)
    with embedding_cache_lock:
        if text_hash in embedding_cache:
            return embedding_cache[text_hash]
    try:
        response = openai.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        embedding_array = np.array(embedding)
        with embedding_cache_lock:
            embedding_cache[text_hash] = embedding_array
        return embedding_array
    except Exception as e:
        logging.error(f"生成嵌入時出錯，文本哈希 {text_hash}：{e}")
        raise e  # 讓 tenacity 能夠捕獲到錯誤並重試

# 保存嵌入緩存到文件
def save_embedding_cache():
    with embedding_cache_lock:
        with open(EMBEDDING_CACHE_PATH, 'wb') as f:
            pickle.dump(embedding_cache, f)
    logging.info("嵌入緩存已成功保存。")

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

def process_pdf_llama_parse(pdf_path):
    try:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    except ValueError:
        logging.error(f"文件名 {pdf_path} 無法獲取 doc_id。跳過。")
        return []  # 返回空列表以便在後續過濾

    try:
        # 使用 LlamaParse 將 PDF 轉換為 Markdown
        md_documents = parser.load_data(file_path=pdf_path)
    except Exception as e:
        logging.error(f"使用 LlamaParse 解析 PDF 文件 {pdf_path} 時出錯：{e}")
        return []

    cleaned_chunks = []
    for md_document in md_documents:
        text = md_document.text  # 假設 md_documents 是包含文本的對象
        elements = node_parser.extract_elements(text=text)
        for element in elements:
            # 將 Element 轉換為 TextNode
            text_node = TextNode(text=element.element, node_id=element.id)
            try:
                nodes = node_parser.get_nodes_from_node(text_node)
            except Exception as e:
                logging.error(f"將 Element 轉換為節點時出錯：{e}")
                continue
            for node in nodes:
                cleaned_text = text_cleaning(node.text)
                chunks = split_text(cleaned_text, max_length=500)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_{node.node_id}_{i}"
                    cleaned_chunks.append((chunk_id, chunk))
    return cleaned_chunks

# 並行處理所有 PDF 文件，使用 LlamaParse 解析
def load_data_parallel_llama_parse(source_path, num_processes=None):
    if num_processes is None:
        num_processes = max(cpu_count() - 1, 1)
    pdf_files = [file for file in os.listdir(source_path) if file.endswith('.pdf')]
    pdf_paths = [os.path.join(source_path, file) for file in pdf_files]

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_pdf_llama_parse, pdf_paths), total=len(pdf_paths), desc="處理 PDF 文件"))

    # 扁平化結果列表
    corpus_list = [item for sublist in results for item in sublist if sublist]
    corpus_dict = {chunk_id: text for chunk_id, text in corpus_list}
    logging.info(f"總共處理了 {len(corpus_dict)} 個文本塊")
    return corpus_dict

# 使用 CKIP 進行分詞
def ckip_tokenize(text):
    sentences = [text]
    try:
        ws_results = ws_driver(sentences)
        tokens = []
        for ws in ws_results:
            tokens.extend(ws)
        return tokens
    except Exception as e:
        logging.error(f"CKIP 分詞時出現錯誤：{e}")
        return []

# 建立 BM25 檢索器
def create_bm25_retriever(nodes, similarity_top_k=3):
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        tokenizer=ckip_tokenize,
        language="chinese"
    )
    logging.info("已建立 BM25 檢索器")
    return bm25_retriever

# 建立嵌入檢索器
def create_embedding_retriever(nodes, similarity_top_k=3):
    # 過濾掉沒有嵌入的節點
    nodes_with_embedding = [node for node in nodes if node.embedding is not None]
    if not nodes_with_embedding:
        logging.error("沒有具有嵌入的節點，無法創建索引。")
        return None

    vector_index = VectorStoreIndex(nodes_with_embedding)
    retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    logging.info("已建立嵌入檢索器")
    return retriever

# 使用 OpenAI 進行重排序
def rerank_with_openai(query, nodes, model="gpt-4"):
    scored_nodes = []
    nodes = list(nodes)[:5]  # 僅對前5個節點進行重排序
    for node in nodes:
        logging.info(f"正在處理節點：{node.ref_doc_id}")
        prompt = f"查詢：{query}\n\n文件內容：{node.text}\n\n請根據查詢和文件的相關性，從0到100進行評分。僅輸出一個數字，表示相關性得分。"
        try:
            time.sleep(1)  # 控制 API 調用頻率
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.choices[0].message.content.strip()
            score = int(re.sub(r'\D', '', score_text))
            logging.info(f"文件 {node.ref_doc_id} 的評分為 {score}")
        except Exception as e:
            logging.error(f"重排序時出現錯誤：{e}")
            score = 0
        scored_nodes.append((node, score))
    # 根據得分排序
    reranked_nodes = [node for node, score in sorted(scored_nodes, key=lambda x: x[1], reverse=True)]
    logging.info("已完成重排序")
    return reranked_nodes

# 綜合檢索與重排序
def retrieve_and_rerank(query, embedding_retriever, bm25_retriever, model="gpt-4"):
    # 嵌入檢索
    embedding_results = embedding_retriever.retrieve(query)
    logging.info(f"嵌入檢索獲得 {len(embedding_results)} 個結果")
    logging.info(f"嵌入檢索結果的文檔 IDs: {[doc.node.ref_doc_id for doc in embedding_results]}")
    # BM25 檢索
    bm25_results = bm25_retriever.retrieve(query)
    logging.info(f"BM25 檢索獲得 {len(bm25_results)} 個結果")
    logging.info(f"BM25 檢索結果的文檔 IDs: {[doc.node.ref_doc_id for doc in bm25_results]}")
    # 合併結果並去重
    combined_nodes = {doc.node.node_id: doc.node for doc in embedding_results + bm25_results}.values()
    logging.info(f"合併後共有 {len(combined_nodes)} 個結果")
    reranked_nodes = rerank_with_openai(query, combined_nodes, model)
    return reranked_nodes

# 主程序入口
def main():
    answer_dict = {"answers": []}
    question_path = "dataset/preliminary/questions_preliminary.json"
    source_path = "reference"
    output_path = "dataset/preliminary/finance_pred_retrieve.json"

    # 載入問題數據集
    with open(question_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    logging.info("已載入問題數據集")

    # 載入 PDF 文檔並生成上下文和嵌入
    source_path_finance = os.path.join(source_path, 'finance')
    logging.info("加載財務語料庫...")
    corpus_dict_finance = load_data_parallel_llama_parse(source_path_finance)

    # 創建 Document 對象列表
    documents = []
    for chunk_id, chunk_text in corpus_dict_finance.items():
        doc = Document(text=chunk_text, doc_id=chunk_id)  # 使用 chunk_id 作為 doc_id
        documents.append(doc)
    logging.info("已創建 Document 對象列表")

    # 解析節點
    all_nodes = node_parser.get_nodes_from_documents(documents)
    logging.info("已解析節點")

    # 為每個節點添加嵌入
    for node in tqdm(all_nodes, desc="生成嵌入向量"):
        if not isinstance(node.text, str):
            logging.error(f"節點 {node.node_id} 的文本不是字符串。")
            continue
        embedding = get_embedding(node.text)
        if embedding is not None:
            node.embedding = embedding
        else:
            logging.error(f"無法為節點 {node.node_id} 生成嵌入。")
    logging.info("已為節點添加嵌入")

    # 保存嵌入緩存
    save_embedding_cache()

    # 建立檢索器
    logging.info("建立檢索器...")
    embedding_retriever = create_embedding_retriever(all_nodes)
    bm25_retriever = create_bm25_retriever(all_nodes)
    logging.info("檢索器已建立")

    # 處理每個問題
    for q_dict in tqdm(questions_data['questions'], desc="Processing questions"):
        if q_dict['category'] == 'finance':
            query = q_dict['query']
            qid = q_dict['qid']
            sources = q_dict.get('source', [])
            doc_ids = set(map(str, sources))
            logging.info(f"處理問題 {qid}: {query}")

            # 篩選對應的節點
            nodes = [node for node in all_nodes if node.ref_doc_id.split('_')[0] in doc_ids]
            if not nodes:
                logging.warning(f"問題 {qid} 沒有可用的節點")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue  # 跳過此問題

            # 創建檢索器（針對特定問題）
            embedding_retriever_q = create_embedding_retriever(nodes)
            if embedding_retriever_q is None:
                logging.error(f"無法為問題 {qid} 創建嵌入檢索器，跳過該問題。")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue
            bm25_retriever_q = create_bm25_retriever(nodes)

            # 檢索並重新排序
            best_match_nodes = retrieve_and_rerank(
                query, embedding_retriever_q, bm25_retriever_q, model="gpt-4"
            )

            # 獲取得分最高的文檔ID
            if best_match_nodes:
                top_node = best_match_nodes[0]
                try:
                    retrieved = int(top_node.ref_doc_id.split('_')[0])  # 提取原始文檔ID
                except ValueError:
                    logging.error(f"無法從 ref_doc_id {top_node.ref_doc_id} 提取文檔ID。")
                    retrieved = None
                logging.info(f"問題 {qid} 的最佳匹配文件為 {retrieved}")
            else:
                retrieved = None
                logging.warning(f"問題 {qid} 未找到匹配的文件")
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

    # 保存檢索結果
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    logging.info("已保存檢索結果到文件")

if __name__ == "__main__":
    main()
