"""
项目规划服务
从想法生成项目书大纲、已有项目诊断
"""
import json
import re
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.logger import get_logger
from app.services.chroma_service import chroma_service
from app.utils.document_parser import document_parser
from app.utils.project_splitter import project_splitter

logger = get_logger("project_planning_service")


class ProjectPlanningService:
    """项目规划服务"""

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

    async def generate_outline(
        self,
        idea: str,
        user_profile: Optional[dict] = None,
        competition_name: Optional[str] = None
    ) -> dict:
        """
        从想法生成项目书大纲

        Args:
            idea: 项目想法/创意描述
            user_profile: 用户画像（可选，辅助生成更匹配的方案）
            competition_name: 目标竞赛名称（如：互联网+创新创业大赛、挑战杯）

        Returns:
            {
                "success": bool,
                "data": {
                    "title": str,
                    "slogan": str,
                    "background": {...},
                    "pain_points": [...],
                    "solution": {...},
                    "business_model": {...},
                    "team_requirements": [...],
                    "risks_and_challenges": [...]
                },
                "error_msg": str
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        logger.info(f"【项目规划】开始生成大纲: idea={idea[:50]}...")

        # 构建用户输入
        user_input_parts = [f"【项目想法】\n{idea}"]

        if competition_name:
            user_input_parts.append(f"\n【目标竞赛】{competition_name}")

        if user_profile:
            profile_text = user_profile.get("analysis", "") or user_profile.get("summary", "")
            if profile_text:
                user_input_parts.append(f"\n【用户背景】{profile_text}")

        user_input = "\n".join(user_input_parts)

        # 获取 prompt
        system_prompt = self._get_prompt("project_planning", "generate_outline")
        if not system_prompt:
            system_prompt = self._default_outline_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        try:
            logger.info("【项目规划】调用 LLM 生成大纲...")
            result = await self.llm.ainvoke(messages)
            outline_data = self._extract_json(result.content)
            logger.info(f"【项目规划】生成完成: title={outline_data.get('title', '')}")

            # 规范化数据结构
            normalized_data = self._normalize_outline(outline_data)

            return {
                "success": True,
                "data": normalized_data
            }

        except json.JSONDecodeError as e:
            logger.error(f"项目大纲 JSON 解析失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": "AI 返回格式异常，请重试"
            }
        except Exception as e:
            logger.error(f"项目大纲生成失败: {e}")
            return {
                "success": False,
                "data": None,
                "error_msg": str(e)
            }

    def _normalize_outline(self, data: dict) -> dict:
        """规范化项目大纲数据"""
        # 确保所有必需字段存在
        return {
            "title": data.get("title", "未命名项目"),
            "slogan": data.get("slogan", ""),
            "background": {
                "industry_background": data.get("background", {}).get("industry_background", ""),
                "policy_support": data.get("background", {}).get("policy_support", ""),
                "market_opportunity": data.get("background", {}).get("market_opportunity", "")
            },
            "pain_points": [
                {
                    "point": p.get("point", ""),
                    "target_group": p.get("target_group", ""),
                    "severity": p.get("severity", "中")
                }
                for p in data.get("pain_points", [])
            ],
            "solution": {
                "core_idea": data.get("solution", {}).get("core_idea", ""),
                "key_features": data.get("solution", {}).get("key_features", []),
                "tech_stack": data.get("solution", {}).get("tech_stack", []),
                "innovation_points": data.get("solution", {}).get("innovation_points", [])
            },
            "business_model": {
                "target_customers": data.get("business_model", {}).get("target_customers", []),
                "value_proposition": data.get("business_model", {}).get("value_proposition", ""),
                "revenue_streams": data.get("business_model", {}).get("revenue_streams", []),
                "competitive_advantage": data.get("business_model", {}).get("competitive_advantage", "")
            },
            "team_requirements": data.get("team_requirements", []),
            "risks_and_challenges": data.get("risks_and_challenges", [])
        }

    def _default_outline_prompt(self) -> str:
        """默认项目大纲生成 prompt"""
        return """
        你是资深创业导师和竞赛项目策划专家。根据用户的项目想法，生成完整的项目书大纲。
        返回严格JSON格式：
        {"title":"项目名称","slogan":"一句话介绍","background":{"industry_background":"行业背景","policy_support":"政策支持","market_opportunity":"市场机遇"},"pain_points":[{"point":"痛点","target_group":"群体","severity":"高/中/低"}],"solution":{"core_idea":"核心理念","key_features":["功能1","功能2"],"tech_stack":["技术1"],"innovation_points":["创新点1"]},"business_model":{"target_customers":["客户1"],"value_proposition":"价值主张","revenue_streams":["盈利方式"],"competitive_advantage":"竞争优势"},"team_requirements":["角色1"],"risks_and_challenges":["风险1"]}
        只返回JSON，无其他内容。"""

    # ==================== 项目诊断 ====================

    async def diagnose(
        self,
        file_path: str,
        competition_id: int,
        competition_name: str = ""
    ) -> dict:
        """
        项目书诊断 - RAG 模拟评审

        流程:
        1. 解析项目书文档
        2. 创建临时向量库存储项目书
        3. 获取评分细则
        4. LLM 原子化规则为考察点
        5. 针对每个考察点检索证据
        6. LLM 综合评审

        Args:
            file_path: 项目书路径
            competition_id: 竞赛ID
            competition_name: 竞赛名称

        Returns:
            {
                "success": bool,
                "data": {
                    "competition_name": str,
                    "total_score": int,
                    "evaluation_points": [...],
                    "strengths": [...],
                    "weaknesses": [...],
                    "suggestions": [...]
                },
                "error_msg": str
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        logger.info(f"【项目诊断】开始: file={file_path}, competition_id={competition_id}")
        temp_store = None

        try:
            # 1. 解析项目书
            logger.info("【项目诊断】解析项目书...")
            doc_text, doc_meta = document_parser.parse_document(file_path)
            if not doc_text or len(doc_text.strip()) < 100:
                return {"success": False, "data": None, "error_msg": "项目书内容过少或解析失败"}

            # 2. 切分并存入临时向量库
            logger.info("【项目诊断】创建临时向量库...")
            chunks = project_splitter.split_to_texts(doc_text)
            if not chunks:
                chunks = [doc_text[:2000]]  # fallback

            temp_store = chroma_service.create_temp_store()
            temp_store.add_documents(chunks)
            logger.info(f"【项目诊断】项目书已切分为 {len(chunks)} 个片段")

            # 3. 获取评分细则
            logger.info("【项目诊断】获取评分细则...")
            score_rules = chroma_service.get_all_score_rules(competition_id)
            if not score_rules:
                return {"success": False, "data": None, "error_msg": "该竞赛暂无评分细则，请先上传"}

            rules_text = "\n".join(score_rules[:5])  # 限制长度

            # 4. LLM 原子化规则为考察点
            logger.info("【项目诊断】提取考察点...")
            eval_points = await self._extract_evaluation_points(rules_text)
            if not eval_points:
                eval_points = self._default_evaluation_points()

            # 5. 针对每个考察点检索证据
            logger.info(f"【项目诊断】检索证据，共 {len(eval_points)} 个考察点...")
            point_evidences = []
            for point in eval_points:
                query = point.get("query", point.get("name", ""))
                evidences = temp_store.search(query, n_results=3)
                evidence_texts = [e["document"][:200] for e in evidences if e.get("document")]
                point_evidences.append({
                    "name": point.get("name", ""),
                    "weight": point.get("weight", 20),
                    "evidences": evidence_texts
                })

            # 6. LLM 综合评审
            logger.info("【项目诊断】LLM 综合评审...")
            result = await self._evaluate_project(rules_text, point_evidences)

            result["competition_name"] = competition_name
            logger.info(f"【项目诊断】完成: total_score={result.get('total_score', 0)}")

            return {"success": True, "data": result}

        except Exception as e:
            logger.error(f"项目诊断失败: {e}")
            return {"success": False, "data": None, "error_msg": str(e)}

        finally:
            # 清理临时向量库
            if temp_store:
                temp_store.clear()

    async def _extract_evaluation_points(self, rules_text: str) -> List[dict]:
        """LLM 提取考察点"""
        system_prompt = self._get_prompt("project_diagnostic", "extract_points")
        if not system_prompt:
            return self._default_evaluation_points()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"【评分细则】\n{rules_text[:3000]}")
        ]

        try:
            result = await self.llm.ainvoke(messages)
            points = self._extract_json_array(result.content)
            return points if points else self._default_evaluation_points()
        except Exception as e:
            logger.warning(f"提取考察点失败: {e}")
            return self._default_evaluation_points()

    def _extract_json_array(self, content: str) -> list:
        """提取 JSON 数组"""
        content = content.strip()
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1)
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else []
        except:
            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                return json.loads(match.group())
            return []

    def _default_evaluation_points(self) -> List[dict]:
        """默认考察点"""
        return [
            {"name": "创新性", "weight": 25, "query": "创新点 技术创新 模式创新"},
            {"name": "可行性", "weight": 25, "query": "技术方案 实现路径 可行性分析"},
            {"name": "商业价值", "weight": 25, "query": "商业模式 盈利模式 市场规模"},
            {"name": "团队能力", "weight": 25, "query": "团队介绍 成员背景 分工"}
        ]

    async def _evaluate_project(self, rules_text: str, point_evidences: List[dict]) -> dict:
        """LLM 综合评审"""
        system_prompt = self._get_prompt("project_diagnostic", "evaluate")
        if not system_prompt:
            system_prompt = self._default_evaluate_prompt()

        # 构建评审输入（控制长度）
        evidence_summary = []
        for pe in point_evidences:
            ev_text = " | ".join(pe["evidences"][:2]) if pe["evidences"] else "无相关证据"
            evidence_summary.append(f"【{pe['name']}(权重{pe['weight']}%)】{ev_text[:300]}")

        user_input = f"""【评分细则摘要】
{rules_text[:1500]}

【项目书证据】
{"".join(evidence_summary)}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        try:
            result = await self.llm.ainvoke(messages)
            data = self._extract_json(result.content)
            return self._normalize_diagnostic_result(data)
        except Exception as e:
            logger.error(f"评审失败: {e}")
            return self._empty_diagnostic_result()

    def _normalize_diagnostic_result(self, data: dict) -> dict:
        """规范化诊断结果"""
        return {
            "total_score": data.get("total_score", 0),
            "evaluation_points": [
                {
                    "point_name": ep.get("point_name", ""),
                    "evidence": ep.get("evidence", []),
                    "score": ep.get("score", 0),
                    "comment": ep.get("comment", "")
                }
                for ep in data.get("evaluation_points", [])
            ],
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "suggestions": data.get("suggestions", [])
        }

    def _empty_diagnostic_result(self) -> dict:
        """空诊断结果"""
        return {
            "total_score": 0,
            "evaluation_points": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": ["诊断失败，请重试"]
        }

    def _default_evaluate_prompt(self) -> str:
        """默认评审 prompt"""
        return """你是竞赛评审专家。根据评分规则和项目书证据评审。
返回JSON：{"evaluation_points":[{"point_name":"考察点","evidence":["证据"],"score":0-100,"comment":"意见"}],"total_score":综合分,"strengths":["亮点"],"weaknesses":["不足"],"suggestions":["建议"]}
只返回JSON。"""


# 全局单例
project_planning_service = ProjectPlanningService()
