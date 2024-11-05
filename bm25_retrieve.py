import os
import json
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從 PDF 文件中提取文字的工具
from tqdm import tqdm
from rank_bm25 import BM25Okapi  # 使用 BM25 演算法進行文件檢索


# 指定固定的路徑
question_path = "dataset/preliminary/questions_example.json"  # 問題文件的路徑
source_path = "reference"  # 參考資料的根路徑
output_path = "dataset/preliminary/pred_retrieve.json"  # 預測結果的輸出路徑


# 載入參考資料，返回一個字典，key 為檔案名稱，value 為 PDF 檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個 PDF 文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個 PDF 文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的 PDF 文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉 PDF 文件

    return pdf_text  # 返回提取出的文本


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用 BM25 演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名


if __name__ == "__main__":
    answer_dict = {"answers": []}  # 初始化字典

    with open(question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    # 加載保險和財務類型的參考資料
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(source_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance)

    # 讀取 FAQ 的參考資料
    with open(os.path.join(source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    # 根據問題類型進行檢索並儲存結果
    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 檢索財務類問題
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            # 檢索保險類問題
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            # 檢索 FAQ 類問題
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Unknown category")  # 如果類型不明，拋出錯誤

    # 將答案字典保存為 JSON 文件
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非 ASCII 字符