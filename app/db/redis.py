import os
import redis
from app.core.logger import get_logger


class RedisClient:
    """Redis 数据库客户端封装"""

    def __init__(self):
        self.logger = get_logger("redis")
        self._pool = None
        self._client = None
        self._connected = False

    def init(self):
        """初始化 Redis 连接"""
        self.logger.info("=" * 20 + "REDIS" + "=" * 20)
        try:
            self.logger.info("正在初始化 Redis 连接...")
            host = os.getenv("REDIS_HOST")
            port = os.getenv("REDIS_PORT")
            password = os.getenv("REDIS_PASSWORD")
            db = os.getenv("REDIS_DB")

            self._pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True
            )

            self._client = redis.Redis(connection_pool=self._pool)

            # 测试连接
            self._client.ping()
            self._connected = True
            self.logger.info(f"Redis 连接成功!")

        except Exception as e:
            self._connected = False
            self.logger.warning(f"Redis 连接失败: {e} (服务将继续运行，但缓存功能不可用)")
        self.logger.info("=" * 20 + "REDIS" + "=" * 20)

    @property
    def client(self) -> redis.Redis:
        """获取 Redis 客户端"""
        if not self._connected:
            raise ConnectionError("Redis 未连接")
        return self._client

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def get(self, key: str) -> str:
        """获取值"""
        if not self._connected:
            return None
        return self._client.get(key)

    def set(self, key: str, value: str, ex: int = None):
        """设置值"""
        if not self._connected:
            return False
        return self._client.set(key, value, ex=ex)

    def delete(self, key: str):
        """删除键"""
        if not self._connected:
            return False
        return self._client.delete(key)

    def close(self):
        """关闭连接"""
        if self._pool:
            self._pool.disconnect()
            self._connected = False
            self.logger.info("Redis 连接已关闭")

# 全局单例
redis_client = RedisClient()
