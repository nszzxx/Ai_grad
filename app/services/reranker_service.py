"""
Reranker 服务
使用 Cross-Encoder 模型对检索结果进行重排序
"""
import json
import os
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download
from app.core.logger import get_logger

logger = get_logger("reranker_service")


class RerankerService:
    """
    重排序服务
    使用 BGE-Reranker-Base 对向量检索结果进行精排
    """

    def __init__(self):
        self._model = None
        self._config = None
        # 配置将在 init() 中从数据库加载
        self._model_name = None
        self._model_cache_dir = None
        self._model_device = None
        self._initialized = False

    def init(self, db_session):
        """初始化 Reranker 模型 (服务启动时调用)"""
        from app.models.sql_models import GlobalConfig

        if self._initialized:
            logger.info("Reranker 已经初始化，跳过")
            return

        logger.info("=" * 20 + "RERANKER" + "=" * 20)
        try:
            logger.info("正在初始化 Reranker 模型...")

            # 从数据库加载配置
            config = db_session.query(GlobalConfig).filter(
                GlobalConfig.config_key == "CHROMA_CONFIG"
            ).first()

            if not config:
                raise ValueError("数据库中没有找到 CHROMA_CONFIG 配置!")

            self._config = json.loads(config.config_value)
            logger.info(f"加载 Reranker 配置: model={self._config.get('RERANKER_MODEL_NAME')}")

            # 解析配置参数
            self._model_name = self._config.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
            self._model_cache_dir = self._config.get("RERANKER_CACHE_DIR", "./app/models/bge-reranker-base")
            self._model_device = self._config.get("EMBEDDING_DEVICE", "cpu")

            logger.info(f"模型名称: {self._model_name}")
            logger.info(f"模型缓存目录: {os.path.abspath(self._model_cache_dir)}")
            logger.info(f"使用设备: {self._model_device}")

            # 确保缓存目录存在
            os.makedirs(self._model_cache_dir, exist_ok=True)

            # 确定模型路径
            # 如果配置的是本地路径（已存在的目录），直接使用
            # 否则使用 huggingface_hub 管理缓存
            if os.path.isdir(self._model_name):
                # 配置的是本地路径，直接使用
                local_model_path = self._model_name
                logger.info(f"使用本地模型路径: {local_model_path}")
            else:
                # 配置的是 HuggingFace 模型 ID，使用缓存机制
                logger.info("正在检查本地缓存...")
                try:
                    # 优先使用本地缓存（不联网）
                    local_model_path = snapshot_download(
                        repo_id=self._model_name,
                        cache_dir=self._model_cache_dir,
                        local_files_only=True  # 只使用本地缓存
                    )
                    logger.info(f"从本地缓存加载: {local_model_path}")
                except Exception:
                    # 本地没有，需要下载
                    logger.info("本地缓存不存在，正在下载模型...")
                    local_model_path = snapshot_download(
                        repo_id=self._model_name,
                        cache_dir=self._model_cache_dir,
                        local_files_only=False
                    )
                    logger.info(f"模型下载完成: {local_model_path}")

            # 从本地路径加载模型
            self._model = CrossEncoder(
                model_name_or_path=local_model_path,
                device=self._model_device,
                max_length=512  # 最大序列长度
            )

            self._initialized = True
            logger.info("Reranker 模型加载成功!")

        except Exception as e:
            self._initialized = False
            logger.error(f"Reranker 模型加载失败: {e}")
            logger.error(f"详细错误信息: {str(e)}")
            import traceback
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            raise

        logger.info("=" * 20 + "RERANKER" + "=" * 20)

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    @property
    def config(self) -> dict:
        """获取当前配置"""
        return self._config

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果

        Returns:
            [(文档索引, 分数), ...] 按分数降序排列
        """
        if not self._initialized:
            raise RuntimeError("Reranker 未初始化，请先调用 init() 方法")

        if not documents:
            return []

        try:
            # 构建 query-document pairs
            pairs = [[query, doc] for doc in documents]

            # 使用模型打分
            scores = self._model.predict(pairs)

            # 将索引和分数配对，并按分数降序排序
            ranked_results = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )

            # 返回 top_k 个结果
            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            # 如果失败，返回原始顺序的前 top_k 个
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

    def rerank_with_metadata(
        self,
        query: str,
        results: List[dict],
        top_k: int = 3
    ) -> List[dict]:
        """
        对带有元数据的检索结果进行重排序

        Args:
            query: 查询文本
            results: 检索结果列表，每个元素包含 'document' 和 'metadata'
            top_k: 返回前 k 个结果

        Returns:
            重排序后的结果列表（包含原始的 metadata）
        """
        if not results:
            logger.warning("输入结果为空，返回空列表")
            return []

        logger.info(f"开始重排序: query='{query[:30]}...', 输入结果数={len(results)}, top_k={top_k}")

        # 提取文档文本
        documents = [result.get('document', '') for result in results]
        logger.info(f"提取了 {len(documents)} 个文档文本")

        # 进行重排序
        ranked_indices = self.rerank(query, documents, top_k)
        logger.info(f"Rerank 返回了 {len(ranked_indices)} 个结果")

        # 构建重排序后的结果
        reranked_results = []
        for idx, score in ranked_indices:
            result = results[idx].copy()
            result['rerank_score'] = float(score)  # 添加重排序分数
            reranked_results.append(result)
            logger.debug(f"  结果 {len(reranked_results)}: idx={idx}, score={score:.4f}")

        logger.info(f"重排序完成: 输入 {len(results)} 条，输出 {len(reranked_results)} 条")

        return reranked_results


# 全局单例
reranker_service = RerankerService()
