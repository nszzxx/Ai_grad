from app.core.logger import get_logger
from app.services.llm_factory import llm_client
from app.services.chroma_service import chroma_service
from app.services.user_service import user_service
from app.services.ai_service import ai_service
from app.services.reranker_service import reranker_service
from app.db.chroma import chroma_client

logger = get_logger("services")


def init_services(db_session):
    """初始化所有服务 (服务启动时调用)"""
    # 初始化 LLM 客户端
    llm_client.init(db_session)

    # 初始化用户画像服务（设置 LLM）
    user_service.set_llm(llm_client.client)

    # 初始化 AI 服务（设置 LLM）
    ai_service.set_llm(llm_client.client)

    # 初始化 Chroma 向量数据库（从数据库读取配置）
    chroma_client.init(db_session)

    # 初始化 Reranker 模型（用于检索结果重排序，从数据库读取配置）
    try:
        logger.info("正在初始化 Reranker 服务...")
        reranker_service.init(db_session)
    except Exception as e:
        logger.warning(f"Reranker 初始化失败: {e}")
        logger.warning("将使用降级方案（不进行重排序）")


__all__ = [
    "llm_client",
    "chroma_service",
    "user_service",
    "ai_service",
    "reranker_service",
    "init_services",
]
