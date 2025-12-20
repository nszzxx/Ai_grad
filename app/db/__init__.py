from app.core.logger import get_logger
from app.db.mysql import mysql_client
from app.db.redis import redis_client
from app.db.chroma import chroma_client

logger = get_logger("db")


def init_databases():
    """统一初始化所有数据库连接 (不依赖配置的部分)"""
    logger.info("="*30+"初始化数据库"+"="*30)

    # 1. 初始化 MySQL (必须)
    mysql_client.init()

    # 2. 初始化 Redis (可选，失败不影响服务启动)
    redis_client.init()

    # 注意: Chroma 需要从数据库读取配置，在 init_services() 中初始化

    logger.info(logger.info("="*30+"数据库初始化完成"+"="*30))


def close_databases():
    """关闭所有数据库连接"""
    logger.info(logger.info("="*25+"关闭数据库"+"="*25))

    mysql_client.close()
    redis_client.close()
    chroma_client.close()

    logger.info(logger.info("="*25+"数据库连接已关闭"+"="*25))


# 导出单例供其他模块使用
__all__ = [
    "mysql_client",
    "redis_client",
    "chroma_client",
    "init_databases",
    "close_databases",
]


def session():
    return None