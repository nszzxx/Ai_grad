import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from app.core.logger import get_logger


class MySQLClient:
    """MySQL 数据库客户端封装"""

    def __init__(self):
        self.logger = get_logger("mysql")
        self._engine = None
        self._session_local = None
        self._connected = False

    def init(self):
        """初始化 MySQL 连接"""
        self.logger.info("=" * 20 + "MYSQL" + "=" * 20)
        try:
            self.logger.info("正在初始化 MySQL 连接...")
            url = os.getenv("MYSQL_URL")
            if not url:
                raise ValueError("MYSQL_URL 环境变量未设置")

            self._engine = create_engine(url, pool_pre_ping=True)
            self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

            # 测试连接并输出信息
            with self._engine.connect() as conn:
                self._connected = True
                self.logger.info("MySQL 连接成功!")
                # 获取所有表名
                inspector = inspect(self._engine)
                tables = inspector.get_table_names()

                # 查询 global_config 表数据
                if "global_config" in tables:
                    self.logger.info("global_config 表数据:")
                    result = conn.execute(text("SELECT * FROM global_config"))
                    rows = result.fetchall()
                    if rows:
                        col_names = result.keys()
                        for row in rows:
                            if row.id == 2:
                                row_dict = dict(zip(col_names, row))
                                self.logger.info(f"  - {row_dict}")
                    else:
                        self.logger.info("  (表中暂无数据)")
                else:
                    self.logger.warning("global_config 表不存在")

        except Exception as e:
            self._connected = False
            self.logger.error(f"MySQL 连接失败: {e}")
            raise
        self.logger.info("=" * 20 + "MYSQL" + "=" * 20)

    @property
    def engine(self):
        """获取 SQLAlchemy Engine"""
        return self._engine

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self._connected:
            raise ConnectionError("MySQL 未连接")
        return self._session_local()

    def get_db(self):
        """FastAPI 依赖注入用的生成器"""
        db = self._session_local()
        try:
            yield db
        finally:
            db.close()

    def close(self):
        """关闭连接"""
        if self._engine:
            self._engine.dispose()
            self._connected = False
            self.logger.info("MySQL 连接已关闭")


# 全局单例
mysql_client = MySQLClient()
