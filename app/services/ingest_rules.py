import os
import sys

# 将项目根目录加入路径，防止找不到 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface  import HuggingFaceEmbeddings


def ingest_pdfs():
    print("🚀 开始处理 PDF 文档...")

    # 1. 加载 data/pdfs 下所有 pdf
    loader = DirectoryLoader('./data/pdfs', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 页文档")

    # 2. 文本切片 (规则文档通常较长，切细一点)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每块 500 字符
        chunk_overlap=100  # 重叠 100 字符防止语义断裂
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️ 切分为 {len(splits)} 个片段")

    # 3. 向量化并存入 Chroma (rules_db)
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/m3e-base",
        model_kwargs={'device': 'cuda'},  # 有显卡---'cuda'，没有---'cpu'
        encode_kwargs={'normalize_embeddings': True}
    )

    # 指定 collection_name 为 rules
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_FILE_PATH"),
        collection_name="competition_rules"
    )
    print("✅ PDF 规则库构建完成！")


if __name__ == "__main__":
    ingest_pdfs()