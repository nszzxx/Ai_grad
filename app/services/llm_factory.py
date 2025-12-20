import json
from langchain_openai import ChatOpenAI
from app.core.logger import get_logger


class LLMClient:
    """LLM 客户端封装"""

    def __init__(self):
        self.logger = get_logger("llm_client")
        self._client = None
        self._config = None
        self._initialized = False

    def init(self, db_session):
        """初始化 LLM 客户端 (服务启动时调用)"""
        from app.models.sql_models import GlobalConfig

        self.logger.info("=" * 25 + "LLM INIT" + "=" * 25)
        try:
            self.logger.info("正在初始化 LLM 客户端...")

            # 从数据库加载配置
            config = db_session.query(GlobalConfig).filter(
                GlobalConfig.config_key == "AI_SERVICE_CONFIG"
            ).first()

            if not config:
                raise ValueError("数据库中没有找到 AI_SERVICE_CONFIG 配置!")

            self._config = json.loads(config.config_value)
            self.logger.info(f"加载 AI 配置: service_name={self._config.get('service_name')}, model={self._config.get('model')}")

            # 检查是否启用
            if not self._config.get("service_enabled", False):
                self.logger.warning("AI 服务未启用，跳过初始化")
                return

            # 解析配置参数
            api_key = self._config.get("api_key")
            base_url = self._config.get("base_url")
            model = self._config.get("model")
            max_tokens = int(self._config.get("max_tokens", 4000))
            temperature = float(self._config.get("temperature", 0.7))
            timeout = int(self._config.get("time_out", 100000)) / 1000

            # 创建 LLM 客户端
            self._client = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )

            self._initialized = True
            self.logger.info(f"LLM 客户端初始化成功! 使用模型: {model}")

        except Exception as e:
            self._initialized = False
            self.logger.error(f"LLM 客户端初始化失败: {e}")
            raise
        finally:
            self.logger.info("=" * 25 + "LLM INIT" + "=" * 25)

    @property
    def client(self) -> ChatOpenAI:
        """获取 LLM 客户端"""
        if not self._initialized:
            raise RuntimeError("LLM 客户端未初始化")
        return self._client

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    @property
    def config(self) -> dict:
        """获取当前配置"""
        return self._config

    def reset(self):
        """重置 LLM 客户端"""
        self._client = None
        self._config = None
        self._initialized = False
        self.logger.info("LLM 客户端已重置")


# 全局单例
llm_client = LLMClient()
