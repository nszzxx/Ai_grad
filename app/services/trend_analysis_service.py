"""
趋势分析服务
联网搜索竞赛趋势、热点聚焦
"""
import json
import re
from typing import List, Optional
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.logger import get_logger

logger = get_logger("trend_analysis_service")


class TrendAnalysisService:
    """趋势分析服务 - 基于 DuckDuckGo 搜索"""

    def __init__(self, llm=None):
        self.llm = llm
        self._search_tool = None
        self._prompts = None

    def set_llm(self, llm):
        """设置 LLM 客户端"""
        self.llm = llm

    @property
    def search_tool(self):
        """懒加载搜索工具"""
        if self._search_tool is None:
            self._search_tool = DuckDuckGoSearchResults(
                num_results=8,
                output_format="list"
            )
        return self._search_tool

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

    def _get_prompt(self, category: str, key: str) -> str:
        """获取指定提示词"""
        prompts = self._load_prompts()
        return prompts.get(category, {}).get(key, "")

    def _search(self, query: str) -> List[dict]:
        """执行 DuckDuckGo 搜索"""
        try:
            logger.info(f"【搜索】query: {query}")
            results = self.search_tool.invoke(query)
            logger.info(f"【搜索】返回 {len(results)} 条结果")
            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def _extract_json(self, content: str) -> dict:
        """从 LLM 响应中提取 JSON"""
        content = content.strip()
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group())
            raise

    async def analyze(self, competition_name: Optional[str] = None) -> dict:
        """
        趋势分析主入口

        Args:
            competition_name: 竞赛名称（可选，不传则分析通用趋势）

        Returns:
            {
                "success": bool,
                "data": {
                    "trends": [...],
                    "hot_competitions": [...],
                    "summary": str
                },
                "error_msg": str
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        logger.info(f"【趋势分析】开始，competition_name={competition_name}")

        # 1. 构建搜索查询
        if competition_name:
            queries = [
                f"{competition_name} 2024 2025 趋势 变化",
                f"{competition_name} 竞赛 热点 最新"
            ]
        else:
            queries = [
                "2024 2025 大学生竞赛趋势 热门比赛",
                "2024 2025 编程竞赛 算法竞赛 趋势"
            ]

        # 2. 执行搜索
        all_results = []
        for query in queries:
            results = self._search(query)
            all_results.extend(results)

        if not all_results:
            return {
                "success": False,
                "data": None,
                "error_msg": "搜索无结果，请稍后重试"
            }

        # 3. 去重并格式化搜索结果
        seen_links = set()
        unique_results = []
        for item in all_results:
            link = item.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                unique_results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": link
                })

        search_context = "\n".join([
            f"- {r['title']}: {r['snippet']}"
            for r in unique_results[:10]
        ])

        # 4. LLM 分析总结
        system_prompt = self._get_prompt("trend_analysis", "analyze")
        if not system_prompt:
            system_prompt = self._default_prompt()

        user_input = f"【搜索结果】\n{search_context}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        try:
            logger.info("【趋势分析】调用 LLM 分析...")
            result = await self.llm.ainvoke(messages)
            analysis = self._extract_json(result.content)
            logger.info("【趋势分析】完成")

            return {
                "success": True,
                "data": self._normalize_result(analysis)
            }

        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": str(e)
            }

    def _default_prompt(self) -> str:
        """默认趋势分析 prompt"""
        return """你是竞赛趋势分析专家。根据搜索结果分析近两年大学生竞赛趋势。

返回严格JSON格式：
{"trends":[{"name":"趋势名称","description":"简述(30字内)"}],"hot_competitions":[{"name":"竞赛名称","heat":"热度(高/中)","reason":"推荐理由(20字内)"}],"summary":"总结(50字内)"}

要求：
- trends: 3-5个主要趋势
- hot_competitions: 3-5个热门竞赛
- 内容基于搜索结果，客观准确

只返回JSON，无其他内容。"""

    def _normalize_result(self, data: dict) -> dict:
        """规范化分析结果"""
        trends = data.get("trends", [])[:5]
        hot_competitions = data.get("hot_competitions", [])[:5]
        summary = data.get("summary", "")[:200]

        return {
            "trends": trends,
            "hot_competitions": hot_competitions,
            "summary": summary
        }


# 全局单例
trend_analysis_service = TrendAnalysisService()
