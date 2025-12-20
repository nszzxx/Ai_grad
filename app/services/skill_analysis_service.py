"""
技能分析服务
能力雷达图 + 差异化诊断
"""
import json
import re
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger import get_logger

logger = get_logger("skill_analysis_service")


class SkillAnalysisService:
    """技能分析服务"""

    # 默认雷达图维度
    DEFAULT_DIMENSIONS = ["算法基础", "后端工程", "前端交互", "文档撰写", "竞赛经验"]

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

    def _get_prompt(self, category: str, key: str) -> str:
        """获取指定提示词"""
        prompts = self._load_prompts()
        return prompts.get(category, {}).get(key, "")

    def _default_analysis_prompt(self) -> str:
        """默认技能分析 prompt"""
        return """你是一位资深的竞赛指导专家。请根据用户的技能列表和画像信息，进行能力诊断分析。

必须返回严格的JSON格式，包含以下字段：
{
    "radar_chart": [
        {"name": "算法基础", "score": 0-100},
        {"name": "后端工程", "score": 0-100},
        {"name": "前端交互", "score": 0-100},
        {"name": "文档撰写", "score": 0-100},
        {"name": "竞赛经验", "score": 0-100}
    ],
    "core_strength": "核心优势描述（30字内）",
    "weakness_diagnosis": "劣势诊断（50字内，结合目标竞赛分析短板）",
    "learning_suggestions": ["建议1", "建议2", "建议3"]
}

评分标准：
- 算法基础：数据结构、算法设计、复杂度分析能力
- 后端工程：服务端开发、数据库、架构设计能力
- 前端交互：UI开发、用户体验、可视化能力
- 文档撰写：技术文档、项目报告、答辩材料能力
- 竞赛经验：参赛经历、获奖情况、实战经验

只返回JSON，不要任何其他内容。"""

    def _extract_json(self, content: str) -> dict:
        """从 LLM 响应中提取 JSON"""
        content = content.strip()

        # 处理 markdown 代码块
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1)

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 {} 包裹的内容
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group())
            raise

    async def analyze(
        self,
        user_profile: dict,
        skills: List[str],
        target_competition: Optional[str] = None
    ) -> dict:
        """
        技能分析主入口

        Args:
            user_profile: 用户画像 {analysis, summary, intent_keywords}
            skills: 用户技能列表
            target_competition: 目标竞赛（可选）

        Returns:
            {
                "success": bool,
                "data": {
                    "radar_chart": [...],
                    "core_strength": str,
                    "weakness_diagnosis": str,
                    "learning_suggestions": [...]
                },
                "error_msg": str (可选)
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        logger.info(f"【技能分析】开始分析，skills={skills}, target={target_competition}")

        # 构建输入
        profile_text = user_profile.get("analysis", "") or user_profile.get("summary", "")
        skills_text = "、".join(skills) if skills else "暂无"
        target_text = target_competition if target_competition else "未指定"

        user_input = f"""【用户技能】{skills_text}
【用户画像】{profile_text}
【目标竞赛】{target_text}"""

        # 获取 prompt
        system_prompt = self._get_prompt("skill_analysis", "analyze")
        if not system_prompt:
            system_prompt = self._default_analysis_prompt()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_input)
        ])

        try:
            logger.info("【技能分析】调用 LLM...")
            chain = prompt | self.llm
            result = await chain.ainvoke({})

            # 解析 JSON
            analysis_data = self._extract_json(result.content)
            logger.info("【技能分析】完成")

            # 验证并规范化数据
            return {
                "success": True,
                "data": self._normalize_result(analysis_data)
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": "分析结果解析失败"
            }
        except Exception as e:
            logger.error(f"技能分析失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": str(e)
            }

    def _normalize_result(self, data: dict) -> dict:
        """规范化分析结果"""
        # 规范化雷达图
        radar_chart = []
        raw_radar = data.get("radar_chart", [])

        for item in raw_radar:
            name = item.get("name", "未知")
            score = item.get("score", 50)
            # 确保分数在 0-100 范围内
            score = max(0, min(100, int(score)))
            radar_chart.append({"name": name, "score": score})

        # 如果雷达图数据不完整，补充默认维度
        existing_names = {item["name"] for item in radar_chart}
        for dim in self.DEFAULT_DIMENSIONS:
            if dim not in existing_names:
                radar_chart.append({"name": dim, "score": 50})

        # 规范化其他字段
        return {
            "radar_chart": radar_chart[:5],  # 只保留前5个维度
            "core_strength": data.get("core_strength", "")[:100],
            "weakness_diagnosis": data.get("weakness_diagnosis", "")[:150],
            "learning_suggestions": data.get("learning_suggestions", [])[:5]
        }


# 全局单例
skill_analysis_service = SkillAnalysisService()
