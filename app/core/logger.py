import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# 日志目录
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class LevelFilter(logging.Filter):
    """只通过指定等级范围内的日志记录"""
    def __init__(self, min_level: int, max_level: int = None):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level if max_level is not None else min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level


def get_logger(name: str = "ai_server") -> logging.Logger:
    """获取日志记录器"""
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 阻止日志向上传播到 root logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    #
    # # 文件输出 (带轮转，最大 10MB，保留 5 个备份)
    # file_handler = RotatingFileHandler(
    #     os.path.join(LOG_DIR, "app.log"),
    #     maxBytes=10 * 1024 * 1024,
    #     backupCount=5,
    #     encoding="utf-8"
    # )
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # 文件输出 (带轮转，最大 10MB，保留 5 个备份)
    # debug.log - 仅 DEBUG
    debug_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "debug.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(LevelFilter(logging.DEBUG, logging.DEBUG))
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # info.log - 仅 INFO
    info_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "info.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(LevelFilter(logging.INFO, logging.INFO))
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # warning.log - WARNING 及以上
    warning_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "warning.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(LevelFilter(logging.WARNING, logging.CRITICAL))
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)

    # error.log - ERROR 及以上
    error_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "error.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(LevelFilter(logging.ERROR, logging.CRITICAL))
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


# 默认 logger 实例
logger = get_logger()
