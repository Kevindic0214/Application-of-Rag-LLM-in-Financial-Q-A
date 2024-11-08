import os
import json
import openai
import numpy as np
import pdfplumber
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from ckip_transformers.nlp import CkipWordSegmenter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pickle

# 配置日誌設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設定 OpenAI API 密鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化 CKIP 分詞器
ws_driver = CkipWordSegmenter(model="bert-base")
logging.info("CKIP 分詞器已初始化")

# 建立 contexts 資料夾
if not os.path.exists('contexts'):
    os.makedirs('contexts')
    logging.info("建立 'contexts' 資料夾")
    
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')
    logging.info("建立 'embeddings' 資料夾")

# 生成嵌入
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    max_retries = 5
    for i in range(max_retries):
        try:
            response = openai.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            logging.info(f"成功生成嵌入")
            return np.array(embedding)
        except Exception as e:  # 通用的錯誤處理
            logging.error(f"生成嵌入時出現錯誤：{e}")
            time.sleep(2 ** i)
    raise Exception("API 呼叫失敗")

# 生成文件的上下文
def generate_context(text, model="gpt-4"):
    prompt = f"請為以下文本生成一個簡短的上下文摘要，用於改進搜索和檢索。\n\n文本：{text}\n\n摘要："
    max_retries = 5
    for i in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            context = response.choices[0].message.content.strip()
            logging.info(f"成功生成上下文摘要")
            return context
        except Exception as e:  # 通用的錯誤處理
            logging.error(f"生成上下文時出現錯誤：{e}")
            time.sleep(2 ** i)
    raise Exception("API 呼叫失敗")

def generate_contexts_parallel(corpus_dict, max_workers=5):
    contexts = {}
    context_files = {key: f"contexts/{key}_context.txt" for key in corpus_dict.keys()}
    # 先讀取已存在的上下文
    for key, context_file in context_files.items():
        if os.path.exists(context_file):
            with open(context_file, 'r', encoding='utf-8') as f:
                contexts[key] = f.read()
                logging.info(f"已載入上下文：{context_file}")

    # 剩餘需要生成上下文的文本
    keys_to_process = [key for key in corpus_dict.keys() if key not in contexts]

    def process_key(key):
        text = corpus_dict[key]
        context = generate_context(text)
        # 保存上下文到文件
        context_file = context_files[key]
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(context)
        logging.info(f"已保存上下文：{context_file}")
        return key, context

    # 使用多線程處理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_key, key): key for key in keys_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating contexts"):
            try:
                key, context = future.result()
                contexts[key] = context
            except Exception as e:
                logging.error(f"處理 key {futures[future]} 時出現錯誤：{e}")

    return contexts

# 讀取單個 PDF 文件
def read_pdf(pdf_loc, page_infos=None):
    try:
        with pdfplumber.open(pdf_loc) as pdf:
            pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
            pdf_text = ''.join([page.extract_text() or '' for page in pages])
        logging.info(f"成功讀取 PDF 文件：{pdf_loc}")
        return pdf_text
    except Exception as e:
        logging.error(f"讀取 PDF 文件時出現錯誤：{e}")
        return ""

# 載入 PDF 文件
def load_data(source_path):
    corpus_dict = {}
    for i in tqdm(range(1, 644), desc="Loading PDFs"):
        file_name = f"{i}.pdf"
        pdf_path = os.path.join(source_path, file_name)
        if os.path.exists(pdf_path):
            text = read_pdf(pdf_path)
            corpus_dict[str(i)] = text
        else:
            logging.warning(f"文件 {file_name} 不存在。")
    logging.info("已完成載入所有 PDF 文件")
    return corpus_dict

# 生成文件嵌入
def generate_embeddings(corpus_dict, contexts):
    embeddings = {}
    embeddings_file = 'embeddings/embeddings.pkl'

    # 檢查是否已有快取的嵌入檔案
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            logging.info("已載入快取的嵌入檔案")
    else:
        logging.info("未找到嵌入快取檔案，將創建新的嵌入檔案")

    for key, text in tqdm(corpus_dict.items(), desc="Generating embeddings"):
        if key in embeddings:
            logging.info(f"已存在嵌入，跳過：{key}")
            continue  # 已有嵌入，跳過計算
        else:
            # 生成新的嵌入
            context = contexts.get(key, "")
            text_with_context = f"{context}\n\n{text}"
            embedding = get_embedding(text_with_context)
            if embedding is not None:
                embeddings[key] = embedding
                logging.info(f"已生成並添加嵌入：{key}")
            else:
                logging.error(f"無法生成嵌入：{key}")

    # 將所有嵌入保存到快取檔案
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
        logging.info("已保存所有嵌入到快取檔案")

    return embeddings

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

# 建立 Contextual BM25 檢索器
def create_bm25_retriever(nodes, insurance_contexts, similarity_top_k=3):
    # 打印參數類型
    print(f"函數中 nodes 的類型: {type(nodes)}")
    if nodes:
        print(f"nodes 中第一個元素的類型: {type(nodes[0])}")
    else:
        print("nodes 列表為空。")
    
    # 將文本和上下文合併
    for node in nodes:
        # 獲取節點對應的文檔ID
        doc_id = node.ref_doc_id
        # 獲取上下文
        context = insurance_contexts.get(doc_id, "")
        # 合併上下文和文本，並存儲在 node.text
        combined_text = f"{context}\n\n{node.text}"
        node.text = combined_text

    # 使用合併後的文本創建 BM25 檢索器
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        tokenizer=ckip_tokenize,
        language="chinese"
    )
    logging.info("已建立 BM25 檢索器")
    return bm25_retriever

# 建立 Contextual Embedding 檢索器
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

# 使用 OpenAI 重排序器
def rerank_with_openai(query, nodes, model="gpt-4"):
    scored_nodes = []
    for node in nodes:
        logging.info(f"正在處理節點：{node.ref_doc_id}")
        prompt = f"查詢：{query}\n\n文檔內容：{node.text}\n\n請根據查詢和文檔的相關性，從0到10進行評分。僅輸出一個數字，表示相關性得分。我希望你可以根據文檔的內容和查詢的相關性，給出一個合理的評分。"
        max_retries = 5
        for i in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                break
            except Exception as e:
                logging.error(f"重排序時出現錯誤：{e}")
                time.sleep(2 ** i)
        else:
            logging.error("無法完成重排序，達到最大重試次數")
            continue
        # 提取評分
        score_text = response.choices[0].message.content.strip()
        try:
            score = int(score_text)
            logging.info(f"文檔 {node.ref_doc_id} 的評分為 {score}")
        except ValueError:
            logging.warning(f"無法解析評分：{score_text}")
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
if __name__ == "__main__":
    answer_dict = {"answers": []}
    question_path = "dataset/preliminary/questions_example.json"
    source_path = "reference"
    output_path = "dataset/preliminary/insurance_pred_retrieve.json"

    # 載入問題數據集
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)
    logging.info("已載入問題數據集")

    # 載入 PDF 文檔並生成上下文和嵌入
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)
    insurance_contexts = generate_contexts_parallel(corpus_dict_insurance, max_workers=5)
    insurance_embeddings = generate_embeddings(corpus_dict_insurance, insurance_contexts)

    # 創建 Document 對象列表
    documents = []
    for key in corpus_dict_insurance.keys():
        text = corpus_dict_insurance[key]
        context = insurance_contexts.get(key, "")
        text_with_context = f"{context}\n\n{text}"
        doc = Document(text=text_with_context, doc_id=key)
        documents.append(doc)
    logging.info("已創建 Document 對象列表")

    # 解析節點
    parser = SentenceSplitter()
    all_nodes = parser.get_nodes_from_documents(documents)
    logging.info("已解析節點")

    # 為每個節點添加嵌入
    for node in all_nodes:
        embedding = insurance_embeddings.get(node.ref_doc_id)
        if embedding is not None:
            node.embedding = embedding
        else:
            logging.warning(f"嵌入缺失，跳過節點：{node.node_id}")
    logging.info("已為節點添加嵌入")


    # 初始化檢索器
    contextual_embedding_retriever = create_embedding_retriever(all_nodes)
    print(f"Type of nodes: {type(all_nodes)}")
    contextual_bm25_retriever = create_bm25_retriever(all_nodes, insurance_contexts)

    # 問題檢索並保存結果
    for q_dict in tqdm(qs_ref['questions'], desc="Processing questions"):
        if q_dict['category'] == 'insurance':
            query = q_dict['query']
            qid = q_dict['qid']
            sources = q_dict.get('source', [])
            doc_ids = set(map(str, sources))
            logging.info(f"處理問題 {qid}: {query}")
            
            # 篩選對應的節點
            nodes = [node for node in all_nodes if node.ref_doc_id in doc_ids]
            if not nodes:
                logging.warning(f"問題 {qid} 沒有可用的節點")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue  # 跳過此問題
            
            # 新增：提取當前問題的上下文
            current_insurance_contexts = {doc_id: insurance_contexts[doc_id] for doc_id in doc_ids if doc_id in insurance_contexts}

            # 創建檢索器
            embedding_retriever = create_embedding_retriever(nodes)
            bm25_retriever = create_bm25_retriever(nodes, current_insurance_contexts)
            
            # 檢索並重新排序
            best_match_nodes = retrieve_and_rerank(
                query, embedding_retriever, bm25_retriever, model="gpt-4"
            )
            
            # 獲取得分最高的文檔ID
            if best_match_nodes:
                top_node = best_match_nodes[0]
                retrieved = int(top_node.ref_doc_id)  # 將文檔ID轉換為整數
                logging.info(f"問題 {qid} 的最佳匹配文件為 {retrieved}")
            else:
                retrieved = None
                logging.warning(f"問題 {qid} 未找到匹配的文件")
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

    # 保存檢索結果
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    logging.info("已保存檢索結果到文件")