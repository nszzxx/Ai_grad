import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.db import mysql_client
from app.schemas.compeitions import CompetitionInfo  # 假设你有这个 SQLModel
from langchain_chroma import Chroma
from langchain_huggingface  import HuggingFaceEmbeddings


def ingest_matches():
    print("🚀 开始同步竞赛简介数据...")

    db = mysql_client.get_session()
    # 1. 从 MySQL 拉取所有比赛
    # 注意：用原生 SQL 或者定义的 Model，这里演示用 SQL 逻辑
    # results = db.execute("SELECT id, title, description, category FROM competitions").fetchall()
    results = db.query(
        CompetitionInfo.id,
        CompetitionInfo.title,
        CompetitionInfo.description,
        CompetitionInfo.category
    ).all()

    texts = []
    metadatas = []
    ids = []

    for row in results:
        comp_id, title, desc, category = row
        # 构造用于匹配的语义文本
        content = f"比赛名称：{title}。类别：{category}。简介：{desc}"

        texts.append(content)
        metadatas.append({"mysql_id": comp_id, "title": title})
        ids.append(str(comp_id))

    # 2. 存入 Chroma (match_db)
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/m3e-base",
        model_kwargs={'device': 'cuda'},  # 有显卡---'cuda'，没有---'cpu'
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="competition_matches",
    )

    # 先删掉已存在的 id，再写入
    vector_db.delete(ids=ids)
    vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"✅ {len(texts)} 条竞赛数据已同步到 match_db！")


if __name__ == "__main__":
    ingest_matches()