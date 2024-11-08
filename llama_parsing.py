import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse

# 初始化解析器
parser = LlamaParse(
    api_key="llx-PS1kBBrBAqDC3cOm2JrJj6ZF1LGm3Mb9fOkQRqoWn49TlM5V",  # 請替換為您的 API 金鑰
    result_type="markdown",  # 設定輸出格式，可以是 "markdown" 或 "text"
    verbose=True,
    language="ch_tra",  # 設定語言為繁體中文
)

# 解析單個 PDF 文件
documents = parser.load_data("./reference/finance/19.pdf")

# # 查看解析結果
# for doc in documents:
#     print(doc)

file_name = "parsed_document.md"

# 將解析結果儲存為 .md 檔案
with open(file_name, 'w', encoding="utf-8") as file:
    for doc in documents:
        file.write(doc.text + "\n\n")  # 每個文件之間留空行