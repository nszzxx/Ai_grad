"""
Chroma 向量数据库服务层
聚合业务逻辑：竞赛数据同步、向量搜索等功能
"""
import hashlib
from typing import Optional, List, Tuple
from app.db import chroma_client, mysql_client
from app.db.chroma import TempVectorStore
from app.core.logger import get_logger
from app.models.sql_models import Competition
from app.utils.document_parser import document_parser
from app.utils.smart_splitter import SmartSectionSplitter

logger = get_logger("chroma_service")


class ChromaService:
    """Chroma 向量数据库业务服务"""

    # 集合名称
    COLLECTION_COMPETITIONS = "competition_matches"
    COLLECTION_RULES = "rules_documents"  # 竞赛规则文档
    COLLECTION_SCORE_RULES = "score_rule_documents"  # 评分细则文档

    # 智能切分配置
    MAX_CHUNK_SIZE = 500
    MIN_CHUNK_SIZE = 10

    def __init__(self):
        self._chroma = chroma_client
        self._mysql = mysql_client
        # 初始化智能章节切分器（基于正则表达式识别章节标题）
        self._text_splitter = SmartSectionSplitter(
            max_chunk_size=self.MAX_CHUNK_SIZE,
            min_chunk_size=self.MIN_CHUNK_SIZE,
            enable_fallback=True  # 如果没有章节结构，使用固定长度切分
        )

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
                Competition.category,
                Competition.track,
                Competition.tags,
                Competition.difficulty,
                Competition.rules_json
            ).all()

            if not results:
                logger.warning("MySQL 中没有竞赛数据")
                return {"total": 0, "inserted": 0, "updated": 0}

            texts = []
            metadatas = []
            ids = []

            for row in results:
                comp_id, title, desc, category, track, tags, difficulty, rules_json = row
                content = f"比赛名称：{title}。类别：{category}。赛道：{track}。简介：{desc}"
                texts.append(content)
                metadatas.append({
                    "mysql_id": comp_id,
                    "title": title,
                    "category": category,
                    "track": track,
                    "tags": tags,          # 比如从 MySQL 的标签字段解析成列表
                    "difficulty": difficulty,         # 初级/中级/高级
                    "rules_json": rules_json          # JSON 化的规则信息
                })
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
                    "metadata": {"mysql_id":results["metadatas"][0][i]["mysql_id"],
                                 "title":results["metadatas"][0][i]["title"],
                                 "difficulty":results["metadatas"][0][i]["difficulty"]}
                                if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })

        logger.info(f"搜索竞赛: query='{query[:30]}...', 返回 {len(competitions)} 条结果")
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

    def check_competitions_sync_status(self) -> dict:
        """
        检查竞赛数据同步状态，对比 MySQL 和向量库的差异
        返回: {
            'mysql_count': int,           # MySQL 中的竞赛数量
            'vector_count': int,          # 向量库中的竞赛数量
            'synced_count': int,          # 已同步的竞赛数量
            'missing_in_vector': list,    # MySQL 有但向量库没有的竞赛 ID
            'extra_in_vector': list,      # 向量库有但 MySQL 没有的竞赛 ID
            'is_synced': bool,            # 是否完全同步
            'missing_competitions': list  # 缺失竞赛的详细信息（ID 和标题）
        }
        """
        db = self._mysql.get_session()
        try:
            # 获取 MySQL 中所有竞赛 ID 和标题
            mysql_results = db.query(Competition.id, Competition.title).all()
            mysql_ids = {str(row[0]) for row in mysql_results}

            # 获取向量库中所有竞赛 ID
            vector_ids = set(self._chroma.list_all_ids(self.COLLECTION_COMPETITIONS))

            # 计算差异
            synced_ids = mysql_ids & vector_ids
            missing_in_vector = mysql_ids - vector_ids
            extra_in_vector = vector_ids - mysql_ids



            is_synced = len(missing_in_vector) == 0 and len(extra_in_vector) == 0
            logger.info(f"同步状态检查: MySQL {len(mysql_ids)} 条, 向量库 {len(vector_ids)} 条, "
                       f"已同步 {len(synced_ids)} 条, 缺失 {len(missing_in_vector)} 条")

            return {
                "mysql_count": len(mysql_ids),
                "vector_count": len(vector_ids),
                "synced_count": len(synced_ids),
                "missing_in_vector": sorted([int(x) for x in missing_in_vector]),
                "extra_in_vector": sorted([int(x) for x in extra_in_vector]),
                "is_synced": is_synced
            }
        finally:
            db.close()

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

    # ==================== 规则文档管理 ====================

    @staticmethod
    def _generate_doc_id(file_path: str) -> str:
        """
        根据文件路径生成唯一的文档ID
        使用规范化路径的 MD5 哈希值
        """
        normalized_path = file_path.replace('\\', '/').lower()
        return hashlib.md5(normalized_path.encode('utf-8')).hexdigest()

    def _delete_document_chunks(self, parent_doc_id: str) -> int:
        """
        删除指定父文档的所有chunks

        Args:
            parent_doc_id: 父文档ID

        Returns:
            删除的chunk数量
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)

            # 查询该父文档的所有chunks
            result = collection.get(
                where={"parent_doc_id": parent_doc_id},
                include=[]  # 只需要ID
            )

            chunk_ids = result["ids"]

            if chunk_ids:
                collection.delete(ids=chunk_ids)
                logger.debug(f"删除父文档 {parent_doc_id} 的 {len(chunk_ids)} 个chunks")
                return len(chunk_ids)
            else:
                logger.debug(f"父文档 {parent_doc_id} 没有chunks")
                return 0

        except Exception as e:
            logger.error(f"删除父文档chunks失败 {parent_doc_id}: {e}")
            return 0

    def add_rule_document(
        self,
        file_path: str,
        competition_id: int,
        custom_metadata: dict = None
    ) -> dict:
        """
        添加规则文档并自动关联到竞赛（支持文档切分）

        Args:
            file_path: 文件路径（本地路径或HTTP URL）
            competition_id: 竞赛ID
            custom_metadata: 额外的自定义元数据

        Returns:
            {
                'success': bool,
                'action': 'inserted' | 'updated',
                'doc_id': str,
                'file_path': str,
                'chunks_count': int,  # 切分后的chunk数量
                'error': str (仅在失败时)
            }
        """
        try:
            logger.info(f"开始解析规则文档: {file_path}")

            # 1. 解析文档
            text, metadata = document_parser.parse_document(file_path)

            # 2. 切分文档为多个chunks
            chunks = self._text_splitter.split_text(text)
            total_chunks = len(chunks)
            logger.info(f"文档已切分为 {total_chunks} 个chunks")

            # 3. 生成父文档ID
            parent_doc_id = self._generate_doc_id(file_path)

            # 4. 检查是否已存在（通过查询第一个chunk判断）
            first_chunk_id = f"{parent_doc_id}_chunk_0"
            existed = self._chroma.exists(first_chunk_id, self.COLLECTION_RULES)
            action = "updated" if existed else "inserted"

            # 5. 如果是更新，先删除所有旧的chunks
            if existed:
                logger.info(f"检测到文档已存在，删除旧的chunks")
                self._delete_document_chunks(parent_doc_id)

            # 6. 准备批量插入的数据
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk_text in enumerate(chunks):
                # 生成chunk ID
                chunk_id = f"{parent_doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)

                # 构建chunk元数据
                chunk_metadata = {
                    'competition_id': competition_id,
                    'parent_doc_id': parent_doc_id,
                    'chunk_index': i,
                    'total_chunks': total_chunks,
                    **metadata  # 包含原始文档元数据
                }

                # 合并自定义元数据
                if custom_metadata:
                    chunk_metadata.update(custom_metadata)

                chunk_metadatas.append(chunk_metadata)

            # 7. 批量插入所有chunks
            self._chroma.upsert_documents_batch(
                ids=chunk_ids,
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                collection_name=self.COLLECTION_RULES
            )

            logger.info(f"规则文档添加成功: {file_path} ({action}, {total_chunks} chunks)")
            return {
                'success': True,
                'action': action,
                'doc_id': parent_doc_id,
                'file_path': file_path,
                'chunks_count': total_chunks
            }

        except Exception as e:
            logger.error(f"添加规则文档失败 {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }

    def add_rule_documents_batch(
        self,
        file_paths: List[str],
        competition_id: int,
        custom_metadata: dict = None
    ) -> dict:
        """
        批量添加规则文档并关联到竞赛

        Args:
            file_paths: 文件路径列表
            competition_id: 竞赛ID
            custom_metadata: 额外的自定义元数据

        Returns:
            {
                'total': int,
                'success': int,
                'inserted': int,
                'updated': int,
                'failed': int,
                'results': [...]
            }
        """
        logger.info(f"开始批量导入 {len(file_paths)} 个规则文档")

        results = []
        inserted_count = 0
        updated_count = 0
        failed_count = 0

        for file_path in file_paths:
            result = self.add_rule_document(
                file_path=file_path,
                competition_id=competition_id,
                custom_metadata=custom_metadata
            )
            results.append(result)

            if result['success']:
                if result['action'] == 'inserted':
                    inserted_count += 1
                elif result['action'] == 'updated':
                    updated_count += 1
            else:
                failed_count += 1

        summary = {
            'total': len(file_paths),
            'success': inserted_count + updated_count,
            'inserted': inserted_count,
            'updated': updated_count,
            'failed': failed_count,
            'results': results
        }

        logger.info(
            f"批量导入完成: 总数 {summary['total']}, "
            f"成功 {summary['success']} (新增 {inserted_count}, 更新 {updated_count}), "
            f"失败 {failed_count}"
        )

        return summary

    def add_rule_documents_from_directory(
        self,
        directory_path: str,
        competition_id: int,
        recursive: bool = True,
        custom_metadata: dict = None
    ) -> dict:
        """
        从目录批量添加规则文档并关联到竞赛

        Args:
            directory_path: 目录路径
            competition_id: 竞赛ID
            recursive: 是否递归扫描
            custom_metadata: 额外的自定义元数据

        Returns:
            {
                'total': int,
                'success': int,
                'inserted': int,
                'updated': int,
                'failed': int,
                'results': [...]
            }
        """
        logger.info(f"开始从目录导入规则文档: {directory_path}")

        try:
            # 扫描目录获取所有支持的文档文件
            file_paths = document_parser.scan_directory(directory_path, recursive=recursive)

            if not file_paths:
                logger.warning(f"目录中没有找到支持的文档文件: {directory_path}")
                return {
                    'total': 0,
                    'success': 0,
                    'inserted': 0,
                    'updated': 0,
                    'failed': 0,
                    'results': []
                }

            # 批量处理文档
            return self.add_rule_documents_batch(
                file_paths=file_paths,
                competition_id=competition_id,
                custom_metadata=custom_metadata
            )

        except Exception as e:
            logger.error(f"从目录导入规则文档失败 {directory_path}: {e}")
            raise

    def delete_rule_document_by_path(
        self,
        file_path: str
    ) -> dict:
        """
        根据文件路径删除规则文档的所有chunks

        Args:
            file_path: 文件路径

        Returns:
            {
                'success': bool,
                'deleted': bool,
                'chunks_deleted': int,  # 删除的chunk数量
                'doc_id': str,
                'file_path': str
            }
        """
        try:
            parent_doc_id = self._generate_doc_id(file_path)
            chunks_deleted = self._delete_document_chunks(parent_doc_id)

            if chunks_deleted > 0:
                logger.info(f"规则文档删除成功: {file_path} ({chunks_deleted} chunks)")
                deleted = True
            else:
                logger.warning(f"规则文档不存在: {file_path}")
                deleted = False

            return {
                'success': True,
                'deleted': deleted,
                'chunks_deleted': chunks_deleted,
                'doc_id': parent_doc_id,
                'file_path': file_path
            }

        except Exception as e:
            logger.error(f"删除规则文档失败 {file_path}: {e}")
            return {
                'success': False,
                'deleted': False,
                'chunks_deleted': 0,
                'error': str(e),
                'file_path': file_path
            }

    def delete_rule_documents_by_paths(
        self,
        file_paths: List[str]
    ) -> dict:
        """
        根据文件路径列表批量删除规则文档

        Args:
            file_paths: 文件路径列表

        Returns:
            {
                'total': int,
                'deleted': int,
                'not_found': int,
                'results': [...]
            }
        """
        logger.info(f"开始批量删除 {len(file_paths)} 个规则文档")

        results = []
        deleted_count = 0
        not_found_count = 0

        for file_path in file_paths:
            result = self.delete_rule_document_by_path(file_path)
            results.append(result)

            if result['success'] and result['deleted']:
                deleted_count += 1
            elif result['success'] and not result['deleted']:
                not_found_count += 1

        summary = {
            'total': len(file_paths),
            'deleted': deleted_count,
            'not_found': not_found_count,
            'results': results
        }

        logger.info(
            f"批量删除完成: 总数 {summary['total']}, "
            f"删除 {deleted_count}, 未找到 {not_found_count}"
        )

        return summary

    def search_competition_rules(
        self,
        query: str,
        competition_id: int = None,
        subject_keyword: str = None,
        n_results: int = 5
    ) -> list[dict]:
        """
        搜索竞赛规则文档（支持多条件过滤）

        Args:
            query: 搜索查询
            competition_id: 竞赛ID（可选）
            subject_keyword: 科目关键词，用于过滤文件名（可选，检索后过滤）
            n_results: 返回结果数量

        Returns:
            [{'id': str, 'document': str, 'metadata': dict, 'distance': float}, ...]
        """
        collection = self._chroma.get_collection(self.COLLECTION_RULES)
        query_embedding = self._chroma._embed_texts([query])

        # 如果需要科目过滤，扩大召回数量以确保过滤后有足够结果
        recall_count = n_results * 3 if subject_keyword else n_results

        # 构建查询参数
        query_params = {
            "query_embeddings": query_embedding,
            "n_results": recall_count
        }

        # Chroma where 过滤只支持精确匹配，科目关键词需要后处理
        if competition_id is not None:
            query_params["where"] = {"competition_id": competition_id}

        results = collection.query(**query_params)

        # 格式化返回结果
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc_item = {
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None
                }

                # 科目关键词后处理过滤（检查文件名是否包含关键词）
                if subject_keyword:
                    filename = doc_item["metadata"].get("filename", "") if doc_item["metadata"] else ""
                    if subject_keyword.lower() not in filename.lower():
                        continue

                documents.append(doc_item)

                # 达到目标数量后停止
                if len(documents) >= n_results:
                    break

        logger.info(
            f"搜索规则文档: query='{query[:30]}...', "
            f"competition_id={competition_id}, subject={subject_keyword}, 返回 {len(documents)} 条结果"
        )
        return documents

    def route_competition(self, query: str) -> Optional[Tuple[int, float]]:
        """
        竞赛路由：根据查询词匹配竞赛

        Args:
            query: 竞赛指代词（如：蓝桥杯、ICPC等）

        Returns:
            (competition_id, distance) 或 None
        """
        results = self.search_competitions(query, n_results=1)
        if not results:
            logger.info(f"竞赛路由: 未找到匹配竞赛 query='{query}'")
            return None

        top_result = results[0]
        competition_id = top_result["metadata"].get("mysql_id")
        distance = top_result.get("distance", 1.0)

        logger.info(f"竞赛路由: query='{query}' -> id={competition_id}, distance={distance:.4f}")
        return (competition_id, distance)

    def delete_competition_rules(
        self,
        competition_id: int
    ) -> dict:
        """
        删除某个竞赛的所有规则文档

        Args:
            competition_id: 竞赛ID

        Returns:
            {'deleted': int, 'competition_id': int}
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)

            # 获取该竞赛的所有文档ID
            result = collection.get(
                where={"competition_id": competition_id},
                include=[]  # 只需要ID
            )

            doc_ids = result["ids"]

            if doc_ids:
                collection.delete(ids=doc_ids)
                logger.info(f"删除竞赛 {competition_id} 的 {len(doc_ids)} 条规则文档")
            else:
                logger.info(f"竞赛 {competition_id} 没有规则文档")

            return {
                'deleted': len(doc_ids),
                'competition_id': competition_id
            }

        except Exception as e:
            logger.error(f"删除竞赛 {competition_id} 的规则文档失败: {e}")
            raise

    def get_competition_rules_count(self, competition_id: int) -> int:
        """获取某个竞赛的规则文档数量"""
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)
            result = collection.get(
                where={"competition_id": competition_id},
                include=[]
            )
            return len(result["ids"])
        except Exception as e:
            logger.error(f"获取竞赛 {competition_id} 规则文档数量失败: {e}")
            return 0

    def get_rules_collection_count(self) -> int:
        """获取规则文档集合的总数量"""
        return self._chroma.get_collection_count(self.COLLECTION_RULES)

    # ==================== 评分细则文档管理 ====================

    def add_score_rule_document(
        self,
        file_path: str,
        competition_id: int,
        custom_metadata: dict = None
    ) -> dict:
        """
        添加评分细则文档（自动切分）

        Args:
            file_path: 文件路径
            competition_id: 竞赛ID
            custom_metadata: 自定义元数据

        Returns:
            {'success': bool, 'doc_id': str, 'chunks_count': int, 'error': str}
        """
        try:
            logger.info(f"解析评分细则: {file_path}")
            text, metadata = document_parser.parse_document(file_path)
            chunks = self._text_splitter.split_text(text)

            parent_doc_id = self._generate_doc_id(file_path)

            # 删除旧的 chunks
            self._delete_chunks_by_parent(parent_doc_id, self.COLLECTION_SCORE_RULES)

            # 准备批量数据
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk_text in enumerate(chunks):
                chunk_ids.append(f"{parent_doc_id}_chunk_{i}")
                chunk_texts.append(chunk_text)
                chunk_meta = {
                    'competition_id': competition_id,
                    'parent_doc_id': parent_doc_id,
                    'chunk_index': i,
                    **metadata
                }
                if custom_metadata:
                    chunk_meta.update(custom_metadata)
                chunk_metadatas.append(chunk_meta)

            self._chroma.upsert_documents_batch(
                ids=chunk_ids,
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                collection_name=self.COLLECTION_SCORE_RULES
            )

            logger.info(f"评分细则添加成功: {len(chunks)} chunks")
            return {'success': True, 'doc_id': parent_doc_id, 'chunks_count': len(chunks)}

        except Exception as e:
            logger.error(f"添加评分细则失败: {e}")
            return {'success': False, 'error': str(e)}

    def search_score_rules(
        self,
        query: str,
        competition_id: int,
        n_results: int = 5
    ) -> List[dict]:
        """
        搜索评分细则

        Args:
            query: 查询文本
            competition_id: 竞赛ID
            n_results: 返回数量

        Returns:
            [{'id': str, 'document': str, 'metadata': dict, 'distance': float}]
        """
        collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)
        query_embedding = self._chroma.embed_texts([query])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"competition_id": competition_id}
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

        logger.debug(f"评分细则搜索: competition_id={competition_id}, 返回 {len(documents)} 条")
        return documents

    def get_all_score_rules(self, competition_id: int) -> List[str]:
        """获取竞赛的所有评分细则文本"""
        try:
            collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)
            result = collection.get(
                where={"competition_id": competition_id},
                include=["documents"]
            )
            return result["documents"] if result["documents"] else []
        except Exception as e:
            logger.error(f"获取评分细则失败: {e}")
            return []

    def delete_score_rules(self, competition_id: int) -> int:
        """删除竞赛的所有评分细则"""
        try:
            collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)
            result = collection.get(where={"competition_id": competition_id}, include=[])
            if result["ids"]:
                collection.delete(ids=result["ids"])
                logger.info(f"删除竞赛 {competition_id} 的评分细则: {len(result['ids'])} 条")
                return len(result["ids"])
            return 0
        except Exception as e:
            logger.error(f"删除评分细则失败: {e}")
            return 0

    def _delete_chunks_by_parent(self, parent_doc_id: str, collection_name: str) -> int:
        """删除指定父文档的所有chunks（通用方法）"""
        try:
            collection = self._chroma.get_collection(collection_name)
            result = collection.get(
                where={"parent_doc_id": parent_doc_id},
                include=[]
            )
            if result["ids"]:
                collection.delete(ids=result["ids"])
                return len(result["ids"])
            return 0
        except Exception as e:
            logger.error(f"删除chunks失败: {e}")
            return 0

    # ==================== 后台管理接口（文档列表与统计） ====================

    def get_competition_rule_documents(self, competition_id: int) -> List[dict]:
        """
        获取竞赛的规则文档列表（按源文件分组）

        Args:
            competition_id: 竞赛ID

        Returns:
            [
                {
                    'parent_doc_id': str,
                    'filename': str,
                    'file_type': str,
                    'file_size': int,
                    'total_chunks': int,
                    'original_path': str,
                    'source': str
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)

            # 查询该竞赛的所有chunks
            result = collection.get(
                where={"competition_id": competition_id},
                include=["metadatas"]
            )

            if not result["ids"]:
                return []

            # 按 parent_doc_id 分组，提取每个源文件的信息
            documents_map = {}
            for i, doc_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                parent_doc_id = metadata.get("parent_doc_id", "")

                if parent_doc_id and parent_doc_id not in documents_map:
                    documents_map[parent_doc_id] = {
                        "parent_doc_id": parent_doc_id,
                        "filename": metadata.get("filename", ""),
                        "file_type": metadata.get("file_type", ""),
                        "file_size": metadata.get("file_size", 0),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "original_path": metadata.get("original_path", ""),
                        "source": metadata.get("source", "")
                    }

            documents = list(documents_map.values())
            logger.info(f"获取竞赛 {competition_id} 的规则文档列表: {len(documents)} 个文件")
            return documents

        except Exception as e:
            logger.error(f"获取竞赛 {competition_id} 规则文档列表失败: {e}")
            return []

    def get_rules_statistics(self) -> List[dict]:
        """
        获取所有竞赛的规则文档统计

        Returns:
            [
                {
                    'competition_id': int,
                    'document_count': int,  # 源文件数量
                    'chunk_count': int,     # chunk总数
                    'documents': [          # 文档列表
                        {'filename': str, 'total_chunks': int},
                        ...
                    ]
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)

            # 获取所有数据
            result = collection.get(include=["metadatas"])

            if not result["ids"]:
                return []

            # 按 competition_id 分组统计
            stats_map = {}
            for i, doc_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                comp_id = metadata.get("competition_id")
                parent_doc_id = metadata.get("parent_doc_id", "")
                filename = metadata.get("filename", "")
                total_chunks = metadata.get("total_chunks", 0)

                if comp_id is None:
                    continue

                if comp_id not in stats_map:
                    stats_map[comp_id] = {
                        "competition_id": comp_id,
                        "document_count": 0,
                        "chunk_count": 0,
                        "documents": {},  # 使用dict去重
                    }

                stats_map[comp_id]["chunk_count"] += 1

                # 按 parent_doc_id 去重统计文档数
                if parent_doc_id and parent_doc_id not in stats_map[comp_id]["documents"]:
                    stats_map[comp_id]["documents"][parent_doc_id] = {
                        "filename": filename,
                        "total_chunks": total_chunks
                    }
                    stats_map[comp_id]["document_count"] += 1

            # 转换为列表格式
            statistics = []
            for comp_id, stat in stats_map.items():
                statistics.append({
                    "competition_id": stat["competition_id"],
                    "document_count": stat["document_count"],
                    "chunk_count": stat["chunk_count"],
                    "documents": list(stat["documents"].values())
                })

            # 按 competition_id 排序
            statistics.sort(key=lambda x: x["competition_id"])
            logger.info(f"获取规则文档统计: {len(statistics)} 个竞赛")
            return statistics

        except Exception as e:
            logger.error(f"获取规则文档统计失败: {e}")
            return []

    def get_competition_score_rule_documents(self, competition_id: int) -> List[dict]:
        """
        获取竞赛的评分细则文档列表（按源文件分组）

        Args:
            competition_id: 竞赛ID

        Returns:
            [
                {
                    'parent_doc_id': str,
                    'filename': str,
                    'file_type': str,
                    'file_size': int,
                    'total_chunks': int,
                    'original_path': str,
                    'source': str
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)

            # 查询该竞赛的所有chunks
            result = collection.get(
                where={"competition_id": competition_id},
                include=["metadatas"]
            )

            if not result["ids"]:
                return []

            # 按 parent_doc_id 分组
            documents_map = {}
            for i, doc_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                parent_doc_id = metadata.get("parent_doc_id", "")

                if parent_doc_id and parent_doc_id not in documents_map:
                    documents_map[parent_doc_id] = {
                        "parent_doc_id": parent_doc_id,
                        "filename": metadata.get("filename", ""),
                        "file_type": metadata.get("file_type", ""),
                        "file_size": metadata.get("file_size", 0),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "original_path": metadata.get("original_path", ""),
                        "source": metadata.get("source", "")
                    }

            documents = list(documents_map.values())
            logger.info(f"获取竞赛 {competition_id} 的评分细则文档列表: {len(documents)} 个文件")
            return documents

        except Exception as e:
            logger.error(f"获取竞赛 {competition_id} 评分细则文档列表失败: {e}")
            return []

    def get_rule_document_chunks(self, parent_doc_id: str) -> List[dict]:
        """
        获取指定规则文档的所有chunks

        Args:
            parent_doc_id: 父文档ID

        Returns:
            [
                {
                    'chunk_id': str,
                    'chunk_index': int,
                    'content': str,
                    'metadata': dict
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_RULES)

            result = collection.get(
                where={"parent_doc_id": parent_doc_id},
                include=["documents", "metadatas"]
            )

            if not result["ids"]:
                return []

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", i),
                    "content": result["documents"][i] if result["documents"] else "",
                    "metadata": metadata
                })

            # 按 chunk_index 排序
            chunks.sort(key=lambda x: x["chunk_index"])
            logger.info(f"获取文档 {parent_doc_id} 的chunks: {len(chunks)} 个")
            return chunks

        except Exception as e:
            logger.error(f"获取文档 {parent_doc_id} chunks失败: {e}")
            return []

    def get_score_rule_document_chunks(self, parent_doc_id: str) -> List[dict]:
        """
        获取指定评分细则文档的所有chunks

        Args:
            parent_doc_id: 父文档ID

        Returns:
            [
                {
                    'chunk_id': str,
                    'chunk_index': int,
                    'content': str,
                    'metadata': dict
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)

            result = collection.get(
                where={"parent_doc_id": parent_doc_id},
                include=["documents", "metadatas"]
            )

            if not result["ids"]:
                return []

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", i),
                    "content": result["documents"][i] if result["documents"] else "",
                    "metadata": metadata
                })

            chunks.sort(key=lambda x: x["chunk_index"])
            logger.info(f"获取评分文档 {parent_doc_id} 的chunks: {len(chunks)} 个")
            return chunks

        except Exception as e:
            logger.error(f"获取评分文档 {parent_doc_id} chunks失败: {e}")
            return []

    def get_score_rules_statistics(self) -> List[dict]:
        """
        获取所有竞赛的评分细则统计

        Returns:
            [
                {
                    'competition_id': int,
                    'document_count': int,
                    'chunk_count': int,
                    'documents': [
                        {'filename': str, 'total_chunks': int},
                        ...
                    ]
                },
                ...
            ]
        """
        try:
            collection = self._chroma.get_collection(self.COLLECTION_SCORE_RULES)

            result = collection.get(include=["metadatas"])

            if not result["ids"]:
                return []

            # 按 competition_id 分组统计
            stats_map = {}
            for i, doc_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                comp_id = metadata.get("competition_id")
                parent_doc_id = metadata.get("parent_doc_id", "")
                filename = metadata.get("filename", "")
                total_chunks = metadata.get("total_chunks", 0)

                if comp_id is None:
                    continue

                if comp_id not in stats_map:
                    stats_map[comp_id] = {
                        "competition_id": comp_id,
                        "document_count": 0,
                        "chunk_count": 0,
                        "documents": {},
                    }

                stats_map[comp_id]["chunk_count"] += 1

                if parent_doc_id and parent_doc_id not in stats_map[comp_id]["documents"]:
                    stats_map[comp_id]["documents"][parent_doc_id] = {
                        "filename": filename,
                        "total_chunks": total_chunks
                    }
                    stats_map[comp_id]["document_count"] += 1

            statistics = []
            for comp_id, stat in stats_map.items():
                statistics.append({
                    "competition_id": stat["competition_id"],
                    "document_count": stat["document_count"],
                    "chunk_count": stat["chunk_count"],
                    "documents": list(stat["documents"].values())
                })

            statistics.sort(key=lambda x: x["competition_id"])
            logger.info(f"获取评分细则统计: {len(statistics)} 个竞赛")
            return statistics

        except Exception as e:
            logger.error(f"获取评分细则统计失败: {e}")
            return []

    # ==================== 临时向量存储 ====================

    def create_temp_store(self) -> TempVectorStore:
        """
        创建临时向量存储（内存模式，用完即销毁）

        Returns:
            TempVectorStore 实例
        """
        return TempVectorStore(self._chroma)


# 全局单例
chroma_service = ChromaService()
