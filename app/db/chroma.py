import os
from typing import Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.core.logger import get_logger

# 确保 HuggingFace 下载时显示进度条
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")


class ChromaClient:
    """Chroma 向量数据库客户端封装"""

    def __init__(self):
        self.logger = get_logger("chroma")
        self._client = None
        self._connected = False
        self._embedding_model = None
        self._persist_dir = os.path.join(os.path.dirname(__file__), os.getenv("CHROMA_FILE_PATH"))
        # 模型名称：HuggingFace 模型 ID，用于自动下载
        self._model_name = os.getenv("EMBEDDING_MODEL_NAME")
        # 模型缓存目录：指定下载位置，避免下载到 C 盘
        self._model_cache_dir = os.getenv("EMBEDDING_CACHE_DIR")
        self._model_device = os.getenv("EMBEDDING_DEVICE", "cpu")

    def init(self):
        """初始化 Chroma 连接"""
        self.logger.info("=" * 20 + "CHROMA" + "=" * 20)
        try:
            self.logger.info("正在初始化 Chroma 连接...")

            # 确保模型缓存目录存在
            os.makedirs(self._model_cache_dir, exist_ok=True)

            # 初始化 embedding 模型，自动下载到指定目录
            self.logger.info(f"加载 Embedding 模型: {self._model_name}")
            self.logger.info(f"模型缓存目录: {os.path.abspath(self._model_cache_dir)}")
            self.logger.info(f"使用设备: {self._model_device}")

            self._embedding_model = SentenceTransformer(
                model_name_or_path=self._model_name,
                cache_folder=self._model_cache_dir,
                device=self._model_device
            )
            self.logger.info("Embedding 模型加载成功!")

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取已有的集合
            collections = self._client.list_collections()
            self._connected = True

            self.logger.info("Chroma 初始化成功!")
            self.logger.info(f"持久化目录: {os.path.abspath(self._persist_dir)}")
            self.logger.info(f"已有集合数量: {len(collections)}")

            if collections:
                for col in collections:
                    count = self._client.get_collection(col.name).count()
                    self.logger.info(f"  - {col.name}: {count} 条记录")
            else:
                # 创建默认集合
                self.get_collection("default")
                self.logger.info("已创建默认集合: default")

        except Exception as e:
            self._connected = False
            self.logger.error(f"Chroma 初始化失败: {e}")
            raise
        self.logger.info("=" * 20 + "CHROMA" + "=" * 20)

    @property
    def client(self) -> chromadb.PersistentClient:
        """获取 Chroma 客户端"""
        if not self._connected:
            raise ConnectionError("Chroma 未连接")
        return self._client

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """使用本地模型生成向量"""
        embeddings = self._embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_collection(self, name: str = "default"):
        """获取或创建集合"""
        if not self._connected:
            raise ConnectionError("Chroma 未连接")
        return self._client.get_or_create_collection(name=name)

    def add_documents(self, texts: list[str], metadatas: list[dict] = None,
                      ids: list[str] = None, collection_name: str = "default"):
        """添加文档到向量数据库"""
        collection = self.get_collection(collection_name)
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        # 使用自定义模型生成向量
        embeddings = self._embed_texts(texts)
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        self.logger.debug(f"添加 {len(texts)} 条文档到集合 {collection_name}")
        return len(texts)

    def search_similar(self, query: str, n_results: int = 5, collection_name: str = "default"):
        """搜索相似文档"""
        collection = self.get_collection(collection_name)
        # 使用自定义模型生成查询向量
        query_embedding = self._embed_texts([query])
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)
        self.logger.debug(f"搜索相似文档: query={query[:50]}..., n_results={n_results}")
        return results

    def close(self):
        """关闭连接"""
        self._client = None
        self._connected = False
        self.logger.info("Chroma 连接已关闭")

    # ==================== 新增功能函数 ====================

    def exists(self, doc_id: str, collection_name: str = "default") -> bool:
        """检查文档是否已存在"""
        collection = self.get_collection(collection_name)
        result = collection.get(ids=[doc_id])
        return len(result["ids"]) > 0

    def exists_batch(self, doc_ids: list[str], collection_name: str = "default") -> dict[str, bool]:
        """批量检查文档是否存在，返回 {id: exists} 字典"""
        collection = self.get_collection(collection_name)
        result = collection.get(ids=doc_ids)
        existing_ids = set(result["ids"])
        return {doc_id: doc_id in existing_ids for doc_id in doc_ids}

    def get_document(self, doc_id: str, collection_name: str = "default") -> Optional[dict]:
        """获取单个文档"""
        collection = self.get_collection(collection_name)
        result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else None
            }
        return None

    def get_documents_batch(self, doc_ids: list[str], collection_name: str = "default") -> list[dict]:
        """批量获取文档"""
        collection = self.get_collection(collection_name)
        result = collection.get(ids=doc_ids, include=["documents", "metadatas"])
        documents = []
        for i, doc_id in enumerate(result["ids"]):
            documents.append({
                "id": doc_id,
                "document": result["documents"][i] if result["documents"] else None,
                "metadata": result["metadatas"][i] if result["metadatas"] else None
            })
        return documents

    def update_document(self, doc_id: str, text: str = None, metadata: dict = None,
                        collection_name: str = "default") -> bool:
        """更新单个文档（文本和/或元数据）"""
        collection = self.get_collection(collection_name)
        if not self.exists(doc_id, collection_name):
            self.logger.warning(f"文档 {doc_id} 不存在，无法更新")
            return False

        update_kwargs = {"ids": [doc_id]}
        if text is not None:
            update_kwargs["documents"] = [text]
            update_kwargs["embeddings"] = self._embed_texts([text])
        if metadata is not None:
            update_kwargs["metadatas"] = [metadata]

        collection.update(**update_kwargs)
        self.logger.debug(f"更新文档 {doc_id} 成功")
        return True

    def upsert_document(self, doc_id: str, text: str, metadata: dict = None,
                        collection_name: str = "default") -> str:
        """
        插入或更新单个文档（防重复插入）
        返回: 'inserted' 或 'updated'
        """
        collection = self.get_collection(collection_name)
        existed = self.exists(doc_id, collection_name)

        # 使用自定义模型生成向量
        embeddings = self._embed_texts([text])
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=embeddings,
            metadatas=[metadata] if metadata else None
        )

        action = "updated" if existed else "inserted"
        self.logger.debug(f"文档 {doc_id} {action}")
        return action

    def upsert_documents_batch(self, ids: list[str], texts: list[str],
                               metadatas: list[dict] = None,
                               collection_name: str = "default") -> dict:
        """
        批量插入或更新文档（防重复插入）
        返回: {'inserted': int, 'updated': int}
        """
        collection = self.get_collection(collection_name)

        # 检查哪些已存在
        exists_map = self.exists_batch(ids, collection_name)
        inserted_count = sum(1 for exists in exists_map.values() if not exists)
        updated_count = sum(1 for exists in exists_map.values() if exists)

        # 使用自定义模型生成向量
        embeddings = self._embed_texts(texts)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self.logger.info(f"批量 upsert 完成: 新增 {inserted_count}, 更新 {updated_count}")
        return {"inserted": inserted_count, "updated": updated_count}

    def delete_document(self, doc_id: str, collection_name: str = "default") -> bool:
        """删除单个文档"""
        collection = self.get_collection(collection_name)
        if not self.exists(doc_id, collection_name):
            self.logger.warning(f"文档 {doc_id} 不存在，无需删除")
            return False

        collection.delete(ids=[doc_id])
        self.logger.debug(f"删除文档 {doc_id} 成功")
        return True

    def delete_documents_batch(self, doc_ids: list[str], collection_name: str = "default") -> int:
        """批量删除文档，返回实际删除的数量"""
        collection = self.get_collection(collection_name)
        exists_map = self.exists_batch(doc_ids, collection_name)
        existing_ids = [doc_id for doc_id, exists in exists_map.items() if exists]

        if existing_ids:
            collection.delete(ids=existing_ids)
            self.logger.debug(f"批量删除 {len(existing_ids)} 个文档")

        return len(existing_ids)

    def get_collection_count(self, collection_name: str = "default") -> int:
        """获取集合中的文档数量"""
        collection = self.get_collection(collection_name)
        return collection.count()

    def list_all_ids(self, collection_name: str = "default", limit: int = None) -> list[str]:
        """列出集合中所有文档ID"""
        collection = self.get_collection(collection_name)
        if limit:
            result = collection.get(limit=limit)
        else:
            result = collection.get()
        return result["ids"]

    def clear_collection(self, collection_name: str = "default") -> int:
        """清空集合中所有文档"""
        all_ids = self.list_all_ids(collection_name)
        if all_ids:
            collection = self.get_collection(collection_name)
            collection.delete(ids=all_ids)
            self.logger.info(f"清空集合 {collection_name}，删除 {len(all_ids)} 个文档")
        return len(all_ids)

    def delete_collection(self, collection_name: str) -> bool:
        """删除整个集合"""
        if not self._connected:
            raise ConnectionError("Chroma 未连接")
        try:
            self._client.delete_collection(collection_name)
            self.logger.info(f"删除集合 {collection_name} 成功")
            return True
        except Exception as e:
            self.logger.error(f"删除集合 {collection_name} 失败: {e}")
            return False

    def list_collections(self) -> list[str]:
        """列出所有集合名称"""
        if not self._connected:
            raise ConnectionError("Chroma 未连接")
        collections = self._client.list_collections()
        return [col.name for col in collections]


# 全局单例
chroma_client = ChromaClient()
