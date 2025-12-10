from app.core.logger import get_logger
from app.services.llm_factory import llm_client
from app.services.chroma_service import chroma_service

logger = get_logger("services")


def init_services(db_session):
    """初始化所有服务 (服务启动时调用)"""
    # 初始化 LLM 客户端
    llm_client.init(db_session)


__all__ = [
    "llm_client",
    "chroma_service",
    "init_services",
]
