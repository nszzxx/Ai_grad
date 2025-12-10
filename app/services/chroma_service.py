"""
Chroma 向量数据库服务层
聚合业务逻辑：竞赛数据同步、向量搜索等功能
"""
from typing import Optional
from app.db import chroma_client, mysql_client
from app.core.logger import get_logger
from app.models.sql_models import Competition

logger = get_logger("chroma_service")


class ChromaService:
    """Chroma 向量数据库业务服务"""

    # 默认集合名称
    COLLECTION_COMPETITIONS = "competition_matches"

    def __init__(self):
        self._chroma = chroma_client
        self._mysql = mysql_client

    # ==================== 竞赛数据同步 ====================

    def sync_competitions_from_mysql(self) -> dict:
        """
        从 MySQL 同步竞赛数据到 Chroma（全量同步，使用 upsert 防重复）
        返回: {'total': int, 'inserted': int, 'updated': int}
        """
        logger.info("开始同步竞赛数据...")

        db = self._mysql.get_session()
        try:
            results = db.query(
                Competition.id,
                Competition.title,
                Competition.description,
                Competition.category
            ).all()

            if not results:
                logger.warning("MySQL 中没有竞赛数据")
                return {"total": 0, "inserted": 0, "updated": 0}

            texts = []
            metadatas = []
            ids = []

            for row in results:
                comp_id, title, desc, category = row
                content = f"比赛名称：{title}。类别：{category}。简介：{desc}"
                texts.append(content)
                metadatas.append({"mysql_id": comp_id, "title": title, "category": category})
                ids.append(str(comp_id))

            # 使用 upsert 批量同步（防重复）
            result = self._chroma.upsert_documents_batch(
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                collection_name=self.COLLECTION_COMPETITIONS
            )

            logger.info(f"竞赛数据同步完成: 总数 {len(texts)}, 新增 {result['inserted']}, 更新 {result['updated']}")
            return {
                "total": len(texts),
                "inserted": result["inserted"],
                "updated": result["updated"]
            }
        finally:
            db.close()

    def sync_single_competition(self, competition_id: int) -> dict:
        """
        同步单个竞赛数据到 Chroma
        返回: {'action': 'inserted'|'updated', 'id': str}
        """
        db = self._mysql.get_session()
        try:
            row = db.query(
                Competition.id,
                Competition.title,
                Competition.description,
                Competition.category
            ).filter(Competition.id == competition_id).first()

            if not row:
                raise ValueError(f"竞赛 ID {competition_id} 不存在")

            comp_id, title, desc, category = row
            content = f"比赛名称：{title}。类别：{category}。简介：{desc}"
            metadata = {"mysql_id": comp_id, "title": title, "category": category}

            action = self._chroma.upsert_document(
                doc_id=str(comp_id),
                text=content,
                metadata=metadata,
                collection_name=self.COLLECTION_COMPETITIONS
            )

            logger.info(f"竞赛 {comp_id} 同步完成: {action}")
            return {"action": action, "id": str(comp_id)}
        finally:
            db.close()

    def delete_competition(self, competition_id: int) -> bool:
        """
        从 Chroma 中删除单个竞赛数据
        """
        result = self._chroma.delete_document(
            doc_id=str(competition_id),
            collection_name=self.COLLECTION_COMPETITIONS
        )
        if result:
            logger.info(f"竞赛 {competition_id} 已从向量库删除")
        return result

    def delete_competitions_batch(self, competition_ids: list[int]) -> int:
        """
        批量删除竞赛数据
        """
        doc_ids = [str(cid) for cid in competition_ids]
        count = self._chroma.delete_documents_batch(
            doc_ids=doc_ids,
            collection_name=self.COLLECTION_COMPETITIONS
        )
        logger.info(f"批量删除竞赛数据: {count} 条")
        return count

    # ==================== 向量搜索 ====================

    def search_competitions(self, query: str, n_results: int = 5) -> list[dict]:
        """
        搜索相似竞赛
        返回: [{'id': str, 'document': str, 'metadata': dict, 'distance': float}, ...]
        """
        results = self._chroma.search_similar(
            query=query,
            n_results=n_results,
            collection_name=self.COLLECTION_COMPETITIONS
        )

        competitions = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                competitions.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })

        logger.debug(f"搜索竞赛: query='{query[:30]}...', 返回 {len(competitions)} 条结果")
        return competitions

    # ==================== 文档管理（通用） ====================

    def add_document(self, doc_id: str, text: str, metadata: dict = None,
                     collection_name: str = "default") -> str:
        """
        添加单个文档（使用 upsert 防重复）
        返回: 'inserted' 或 'updated'
        """
        return self._chroma.upsert_document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            collection_name=collection_name
        )

    def add_documents_batch(self, ids: list[str], texts: list[str],
                            metadatas: list[dict] = None,
                            collection_name: str = "default") -> dict:
        """
        批量添加文档（使用 upsert 防重复）
        返回: {'inserted': int, 'updated': int}
        """
        return self._chroma.upsert_documents_batch(
            ids=ids,
            texts=texts,
            metadatas=metadatas,
            collection_name=collection_name
        )

    def get_document(self, doc_id: str, collection_name: str = "default") -> Optional[dict]:
        """获取单个文档"""
        return self._chroma.get_document(doc_id, collection_name)

    def get_documents_batch(self, doc_ids: list[str], collection_name: str = "default") -> list[dict]:
        """批量获取文档"""
        return self._chroma.get_documents_batch(doc_ids, collection_name)

    def update_document(self, doc_id: str, text: str = None, metadata: dict = None,
                        collection_name: str = "default") -> bool:
        """更新单个文档"""
        return self._chroma.update_document(doc_id, text, metadata, collection_name)

    def delete_document(self, doc_id: str, collection_name: str = "default") -> bool:
        """删除单个文档"""
        return self._chroma.delete_document(doc_id, collection_name)

    def delete_documents_batch(self, doc_ids: list[str], collection_name: str = "default") -> int:
        """批量删除文档"""
        return self._chroma.delete_documents_batch(doc_ids, collection_name)

    def search_similar(self, query: str, n_results: int = 5,
                       collection_name: str = "default") -> list[dict]:
        """
        通用向量搜索
        """
        results = self._chroma.search_similar(
            query=query,
            n_results=n_results,
            collection_name=collection_name
        )

        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                documents.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })

        return documents

    def exists(self, doc_id: str, collection_name: str = "default") -> bool:
        """检查文档是否存在"""
        return self._chroma.exists(doc_id, collection_name)

    # ==================== 集合管理 ====================

    def get_collection_count(self, collection_name: str = "default") -> int:
        """获取集合中的文档数量"""
        return self._chroma.get_collection_count(collection_name)

    def get_competitions_count(self) -> int:
        """获取竞赛集合中的文档数量"""
        return self._chroma.get_collection_count(self.COLLECTION_COMPETITIONS)

    def list_all_ids(self, collection_name: str = "default", limit: int = None) -> list[str]:
        """列出集合中所有文档ID"""
        return self._chroma.list_all_ids(collection_name, limit)

    def clear_collection(self, collection_name: str = "default") -> int:
        """清空集合"""
        return self._chroma.clear_collection(collection_name)

    def clear_competitions(self) -> int:
        """清空竞赛集合"""
        return self._chroma.clear_collection(self.COLLECTION_COMPETITIONS)

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return self._chroma.delete_collection(collection_name)

    def list_collections(self) -> list[str]:
        """列出所有集合"""
        return self._chroma.list_collections()

    def get_collection_info(self, collection_name: str = "default") -> dict:
        """获取集合信息"""
        count = self._chroma.get_collection_count(collection_name)
        sample_ids = self._chroma.list_all_ids(collection_name, limit=5)
        return {
            "name": collection_name,
            "count": count,
            "sample_ids": sample_ids
        }

    def get_all_collections_info(self) -> list[dict]:
        """获取所有集合的信息"""
        collections = self._chroma.list_collections()
        return [self.get_collection_info(name) for name in collections]


# 全局单例
chroma_service = ChromaService()
