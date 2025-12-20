"""
意图识别服务
使用 LLM 解析用户输入，提取竞赛、科目、问题等结构化信息
"""
import json
import re
from dataclasses import dataclass
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.logger import get_logger

logger = get_logger("intent_service")


@dataclass
class RuleQueryIntent:
    """规则查询意图"""
    competition_query: Optional[str] = None  # 竞赛指代词
    subject_keyword: Optional[str] = None    # 科目关键词
    rule_query: Optional[str] = None         # 核心问题
    parse_success: bool = False              # 解析是否成功


class IntentService:
    """意图识别服务"""

    def __init__(self, llm=None):
        self.llm = llm
        self._prompts = None

    def set_llm(self, llm):
        """设置 LLM 客户端"""
        self.llm = llm

    def _load_prompts(self) -> dict:
        """加载提示词配置"""
        if self._prompts is None:
            import os
            prompts_path = os.path.join(
                os.path.dirname(__file__), "..", "prompts", "prompts.json"
            )
            with open(prompts_path, "r", encoding="utf-8") as f:
                self._prompts = json.load(f)
        return self._prompts

    def _extract_json(self, text: str) -> Optional[dict]:
        """从 LLM 响应中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        return None

    async def parse_rule_query(self, user_message: str) -> RuleQueryIntent:
        """
        解析规则查询意图

        Args:
            user_message: 用户输入

        Returns:
            RuleQueryIntent: 解析结果
        """
        if not self.llm:
            logger.error("LLM 未初始化")
            return RuleQueryIntent(parse_success=False)

        try:
            prompts = self._load_prompts()
            system_prompt = prompts.get("intent", {}).get("rule_query", "")

            if not system_prompt:
                logger.error("意图识别 prompt 未配置")
                return RuleQueryIntent(parse_success=False)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]

            response = await self.llm.ainvoke(messages)
            result = self._extract_json(response.content)

            if not result:
                logger.warning(f"意图解析失败，无法提取 JSON: {response.content[:100]}")
                return RuleQueryIntent(parse_success=False)

            intent = RuleQueryIntent(
                competition_query=result.get("competition_query"),
                subject_keyword=result.get("subject_keyword"),
                rule_query=result.get("rule_query"),
                parse_success=True
            )

            logger.info(f"意图解析成功: competition={intent.competition_query}, "
                       f"subject={intent.subject_keyword}, query={intent.rule_query}")
            return intent

        except Exception as e:
            logger.error(f"意图解析异常: {e}")
            return RuleQueryIntent(parse_success=False)


# 全局单例
intent_service = IntentService()
