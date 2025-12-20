"""
赛事智能推荐服务
加权混合检索 + LLM 匹配度分析
"""
import json
from typing import List, Optional
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from app.services.chroma_service import chroma_service
from app.core.logger import get_logger

logger = get_logger("recommendation_service")


@dataclass
class MatchScore:
    """单个竞赛的匹配度评分"""
    competition_id: int
    title: str
    category: str
    track: str
    difficulty: str
    tech_match: int       # 技术栈匹配度 0-100
    difficulty_match: int # 难度匹配度 0-100
    intent_match: int     # 意图匹配度 0-100
    comment: str          # 专业点评


class RecommendationService:
    """赛事智能推荐服务"""

    # 权重配置
    INTENT_WEIGHT = 0.7   # 意图关键词权重
    SKILL_WEIGHT = 0.3    # 技能权重

    # 检索配置
    RECALL_COUNT = 10     # 每路召回数量
    TOP_K = 5             # 最终返回数量

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

    # ==================== 加权混合检索 ====================

    def _weighted_recall(
        self,
        intent_keywords: List[str],
        skills: List[str]
    ) -> List[dict]:
        """
        加权混合检索
        - Query A: intent_keywords (70%)
        - Query B: skills (30%)
        """
        score_map = {}  # {competition_id: {"score": float, "data": dict}}

        # Query A: 意图关键词检索
        if intent_keywords:
            intent_query = " ".join(intent_keywords)
            logger.info(f"【意图检索】query: {intent_query}")
            intent_results = chroma_service.search_competitions(
                query=intent_query,
                n_results=self.RECALL_COUNT
            )
            for rank, item in enumerate(intent_results):
                comp_id = item["metadata"].get("mysql_id")
                if comp_id is None:
                    continue
                # 排名越高分数越高
                rank_score = (self.RECALL_COUNT - rank) / self.RECALL_COUNT
                weighted_score = rank_score * self.INTENT_WEIGHT

                if comp_id not in score_map:
                    score_map[comp_id] = {"score": 0, "data": item}
                score_map[comp_id]["score"] += weighted_score

        # Query B: 技能检索
        if skills:
            skill_query = " ".join(skills)
            logger.info(f"【技能检索】query: {skill_query}")
            skill_results = chroma_service.search_competitions(
                query=skill_query,
                n_results=self.RECALL_COUNT
            )
            for rank, item in enumerate(skill_results):
                comp_id = item["metadata"].get("mysql_id")
                if comp_id is None:
                    continue
                rank_score = (self.RECALL_COUNT - rank) / self.RECALL_COUNT
                weighted_score = rank_score * self.SKILL_WEIGHT

                if comp_id not in score_map:
                    score_map[comp_id] = {"score": 0, "data": item}
                score_map[comp_id]["score"] += weighted_score

        # 按加权分数排序
        sorted_items = sorted(
            score_map.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        # 返回 Top K
        results = []
        for comp_id, item in sorted_items[:self.TOP_K]:
            data = item["data"]
            results.append({
                "competition_id": comp_id,
                "title": data["metadata"].get("title", ""),
                "category": data["metadata"].get("category", ""),
                "track": data["metadata"].get("track", ""),
                "difficulty": data["metadata"].get("difficulty", ""),
                "tags": data["metadata"].get("tags", "")
            })

        logger.info(f"【加权检索】召回 {len(results)} 个竞赛")
        return results

    # ==================== LLM 匹配度分析 ====================

    async def _analyze_match_scores(
        self,
        user_profile: dict,
        competitions: List[dict]
    ) -> List[dict]:
        """
        LLM 分析匹配度
        输入: 用户画像 + 竞赛列表
        输出: 每个竞赛的匹配度评分和点评
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        if not competitions:
            return []

        # 构建竞赛列表文本（精简）
        comp_list_text = []
        for i, comp in enumerate(competitions, 1):
            comp_list_text.append(
                f"{i}. {comp['title']}｜{comp['category']}｜{comp['track']}｜难度:{comp['difficulty']}"
            )
        competitions_text = "\n".join(comp_list_text)

        # 构建用户画像文本（精简）
        profile_text = (
            f"分析:{user_profile.get('analysis', '')}\n"
            f"意图:{','.join(user_profile.get('intent_keywords', []))}"
        )

        # 获取 prompt
        system_prompt = self._get_prompt("recommendation", "score_analysis")
        if not system_prompt:
            system_prompt = self._default_score_prompt()

        # 构建输入
        user_input = f"【用户画像】\n{profile_text}\n\n【候选竞赛】\n{competitions_text}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_input)
        ])

        logger.info("【LLM分析】开始匹配度评分...")
        chain = prompt | self.llm
        result = await chain.ainvoke({})

        # 解析 JSON 结果
        try:
            scores = self._parse_llm_scores(result.content, competitions)
            logger.info(f"【LLM分析】完成，解析 {len(scores)} 条评分")
            return scores
        except Exception as e:
            logger.error(f"解析 LLM 评分失败: {e}")
            # 返回默认评分
            return self._default_scores(competitions)

    def _default_score_prompt(self) -> str:
        """默认评分 prompt"""
        return """你是竞赛推荐专家。根据用户画像分析每个竞赛的匹配度。
                返回JSON数组，每个竞赛包含：
                - id: 序号(1开始)
                - tech: 技术匹配度(0-100)
                - diff: 难度匹配度(0-100)
                - intent: 意图匹配度(0-100)
                - comment: 点评(15字内)
                评分标准：
                - tech: 用户技能与竞赛技术栈的匹配程度
                - diff: 竞赛难度是否适合用户当前水平(太难或太简单都扣分)
                - intent: 用户意图/目标与竞赛的契合度
                只返回JSON数组，无其他内容。示例：
                [{"id":1,"tech":85,"diff":70,"intent":90,"comment":"技术匹配，适合提升"}]"""

    def _parse_llm_scores(self, content: str, competitions: List[dict]) -> List[dict]:
        """解析 LLM 返回的评分 JSON"""
        # 提取 JSON 部分
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        scores_data = json.loads(content)

        results = []
        for score in scores_data:
            idx = score.get("id", 0) - 1
            if 0 <= idx < len(competitions):
                comp = competitions[idx]
                results.append({
                    "competition_id": comp["competition_id"],
                    "title": comp["title"],
                    "category": comp["category"],
                    "track": comp["track"],
                    "difficulty": comp["difficulty"],
                    "scores": {
                        "tech_match": min(100, max(0, score.get("tech", 60))),
                        "difficulty_match": min(100, max(0, score.get("diff", 60))),
                        "intent_match": min(100, max(0, score.get("intent", 60)))
                    },
                    "comment": score.get("comment", "")[:30]
                })

        return results

    def _default_scores(self, competitions: List[dict]) -> List[dict]:
        """返回默认评分（LLM 解析失败时使用）"""
        return [
            {
                "competition_id": comp["competition_id"],
                "title": comp["title"],
                "category": comp["category"],
                "track": comp["track"],
                "difficulty": comp["difficulty"],
                "scores": {
                    "tech_match": 60,
                    "difficulty_match": 60,
                    "intent_match": 60
                },
                "comment": "推荐参加"
            }
            for comp in competitions
        ]

    # ==================== 主入口 ====================

    async def recommend(
        self,
        user_profile: dict,
        skills: List[str] = None,
        top_k: int = 5
    ) -> dict:
        """
        智能推荐竞赛

        Args:
            user_profile: 用户画像 {analysis, summary, intent_keywords}
            skills: 用户技能列表（可选，用于技能检索）
            top_k: 返回数量

        Returns:
            {
                "success": bool,
                "recommendations": [
                    {
                        "competition_id": int,
                        "title": str,
                        "category": str,
                        "track": str,
                        "difficulty": str,
                        "scores": {
                            "tech_match": int,
                            "difficulty_match": int,
                            "intent_match": int
                        },
                        "comment": str
                    }
                ],
                "total": int
            }
        """
        self.TOP_K = top_k
        intent_keywords = user_profile.get("intent_keywords", [])

        # 1. 加权混合检索
        logger.info(f"【推荐服务】开始推荐，intent={intent_keywords}, skills={skills}")
        competitions = self._weighted_recall(intent_keywords, skills or [])

        if not competitions:
            logger.warning("【推荐服务】未召回任何竞赛")
            return {
                "success": False,
                "recommendations": [],
                "total": 0,
                "error_msg": "未找到匹配的竞赛"
            }

        # 2. LLM 匹配度分析
        recommendations = await self._analyze_match_scores(user_profile, competitions)

        # 3. 按综合分排序（可选）
        for rec in recommendations:
            scores = rec["scores"]
            rec["overall_score"] = round(
                scores["tech_match"] * 0.4 +
                scores["difficulty_match"] * 0.3 +
                scores["intent_match"] * 0.3
            )

        recommendations.sort(key=lambda x: x["overall_score"], reverse=True)

        logger.info(f"【推荐服务】推荐完成，返回 {len(recommendations)} 个竞赛")
        return {
            "success": True,
            "recommendations": recommendations,
            "total": len(recommendations)
        }


# 全局单例
recommendation_service = RecommendationService()
