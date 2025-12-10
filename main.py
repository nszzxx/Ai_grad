from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from app.api.aiRouters import router as ai_router
from app.api.chromaRouters import router as chroma_router
from app.db import init_databases, close_databases, mysql_client
from app.services import init_services
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动和关闭时的生命周期管理"""
    # 启动时执行
    logger.info("服务启动中..")

    # 1. 初始化所有数据库
    init_databases()

    # 2. 初始化所有服务 (依赖数据库)
    db_session = mysql_client.get_session()
    try:
        init_services(db_session)
    finally:
        db_session.close()

    logger.info("服务启动完成")

    yield

    # 关闭时执行
    logger.info("服务关闭中...")
    close_databases()


app = FastAPI(
    title="AI Competition Server",
    description="竞赛系统 AI 服务",
    version="1.0.0",
    lifespan=lifespan
)

# 注册路由
app.include_router(ai_router, prefix="/api", tags=["AI"])
app.include_router(chroma_router, prefix="/api/chroma", tags=["Chroma向量库"])


if __name__ == "__main__":
    import uvicorn
    import logging

    # 统一 uvicorn 日志输出到 stdout (避免红色)
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.error").handlers = []

    uvicorn.run(
        "main:app",
        host="Localhost",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "default",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        }
    )
