"""
学习路径服务
结构化思维链 (CoT) 生成阶段性学习计划
"""
import json
import re
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.logger import get_logger
from app.services.chroma_service import chroma_service

logger = get_logger("learning_path_service")


class LearningPathService:
    """学习路径服务 - 基于 CoT 思维链"""

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

    def _calculate_phases(self, competition_date: datetime) -> dict:
        """计算学习阶段时间"""
        now = datetime.now()
        # 目标：比赛前两周
        end_date = competition_date - timedelta(days=14)

        if end_date <= now:
            return {"error": "距离比赛时间不足两周，无法制定学习计划"}

        total_days = (end_date - now).days

        return {
            "start_date": now.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "competition_date": competition_date.strftime("%Y-%m-%d"),
            "total_days": total_days,
            "total_weeks": total_days // 7
        }

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

    def _get_competition_info(self, competition_id: int) -> Optional[dict]:
        """
        从向量库获取竞赛信息

        Returns:
            {
                "id": str,
                "title": str,
                "category": str,
                "track": str,
                "tags": str,
                "difficulty": str,
                "description": str
            } 或 None
        """
        try:
            doc = chroma_service.get_document(
                doc_id=str(competition_id),
                collection_name=chroma_service.COLLECTION_COMPETITIONS
            )
            if not doc:
                logger.warning(f"竞赛 {competition_id} 在向量库中不存在")
                return None

            metadata = doc.get("metadata", {})
            return {
                "id": competition_id,
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "track": metadata.get("track", ""),
                "tags": metadata.get("tags", ""),
                "difficulty": metadata.get("difficulty", ""),
                "description": doc.get("document", "")
            }
        except Exception as e:
            logger.error(f"获取竞赛信息失败: {e}")
            return None

    async def generate(
        self,
        user_profile: dict,
        competition_id: int,
        competition_date: str
    ) -> dict:
        """
        生成学习路径

        Args:
            user_profile: 用户画像 {analysis, summary, intent_keywords}
            competition_id: 目标竞赛ID
            competition_date: 比赛开始时间 (YYYY-MM-DD)

        Returns:
            {
                "success": bool,
                "data": {
                    "time_info": {...},
                    "competition_name": str,
                    "phases": [...]
                },
                "error_msg": str
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        # 1. 从向量库获取竞赛信息
        competition_info = self._get_competition_info(competition_id)
        if not competition_info:
            return {
                "success": False,
                "data": None,
                "error_msg": f"竞赛ID {competition_id} 不存在，请检查竞赛是否已同步到向量库"
            }

        competition_name = competition_info.get("title", f"竞赛{competition_id}")
        logger.info(f"【学习路径】获取到竞赛信息: {competition_name}")

        # 2. 解析比赛时间
        try:
            comp_date = datetime.strptime(competition_date, "%Y-%m-%d")
        except ValueError:
            return {
                "success": False,
                "data": None,
                "error_msg": "日期格式错误，请使用 YYYY-MM-DD"
            }

        # 3. 计算时间阶段
        time_info = self._calculate_phases(comp_date)
        if "error" in time_info:
            return {
                "success": False,
                "data": None,
                "error_msg": time_info["error"]
            }

        logger.info(f"【学习路径】竞赛={competition_name}, 总天数={time_info['total_days']}")

        # 4. 构建 CoT 提示（包含竞赛详情）
        profile_text = user_profile.get("analysis", "") or user_profile.get("summary", "")

        # 格式化竞赛信息
        competition_context = f"""
        竞赛名称：{competition_name}| 
        类别：{competition_info.get('category', '未知')}|
        难度：{competition_info.get('difficulty', '未知')}|
        """

        user_input = f"""【目标竞赛信息】
{competition_context}

【当前日期】{time_info['start_date']}
【截止日期】{time_info['end_date']}（比赛前两周）
【可用时间】{time_info['total_days']}天（约{time_info['total_weeks']}周）
【用户画像】{profile_text}"""

        system_prompt = self._get_prompt("learning_path", "generate")
        if not system_prompt:
            system_prompt = self._default_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        try:
            logger.info(f"调用--{messages}")
            logger.info("【学习路径】调用 LLM 生成计划...")
            result = await self.llm.ainvoke(messages)
            path_data = self._extract_json(result.content)
            logger.info("【学习路径】生成完成")

            return {
                "success": True,
                "data": {
                    "time_info": time_info,
                    "competition_name": competition_name,
                    "current_level": path_data.get("current_level", ""),
                    "target_level": path_data.get("target_level", ""),
                    "phases": path_data.get("phases", [])
                }
            }

        except Exception as e:
            logger.error(f"学习路径生成失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": str(e)
            }

    def _default_prompt(self) -> str:
        """默认学习路径 prompt"""
        return """你是竞赛备赛规划专家。根据用户画像和时间安排，制定阶段性学习计划。

思考步骤（CoT）：
1. 分析用户当前水平和短板
2. 明确目标竞赛所需能力
3. 根据可用时间划分学习阶段
4. 每阶段设置具体目标和任务

返回严格JSON格式：
{"current_level":"当前水平评估(20字内)","target_level":"目标水平(20字内)","phases":[{"phase":1,"name":"阶段名称","duration":"时长如:第1-2周","goals":["目标1","目标2"],"tasks":["任务1","任务2","任务3"]}]}

要求：
- phases: 2-4个阶段，根据总时间合理划分
- 每阶段goals: 2-3个目标
- 每阶段tasks: 3-5个具体任务
- 任务要具体可执行

只返回JSON，无其他内容。"""


# 全局单例
learning_path_service = LearningPathService()
