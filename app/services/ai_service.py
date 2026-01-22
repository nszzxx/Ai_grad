"""
AI 服务
支持用户画像上下文 + 历史对话记忆 + 竞赛规则查询 + 竞赛相关 AI 功能
"""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.db import redis_client
from app.schemas.chat import ChatPair, RuleReference
from app.core.logger import get_logger
from app.services.user_service import user_service
from app.services.chroma_service import chroma_service
from app.services.reranker_service import reranker_service
from app.services.intent_service import intent_service

logger = get_logger("ai_service")


@dataclass
class RuleSearchResult:
    """规则检索结果"""
    context: Optional[str] = None      # 格式化的上下文
    success: bool = False              # 是否成功找到规则
    error_msg: Optional[str] = None    # 错误信息
    competition_id: Optional[int] = None
    competition_name: Optional[str] = None
    raw_results: Optional[List[dict]] = None  # 原始检索结果，用于生成参考片段


class AIService:
    """AI 服务 - 聊天、竞赛推荐等"""

    # 配置
    MAX_HISTORY_ROUNDS = 10  # 最大历史轮数
    CACHE_PREFIX = "ai:chat:"
    CACHE_TTL_HISTORY = 4800  # 历史记录缓存时间（秒），80分钟

    # 检索阈值配置
    COMPETITION_DISTANCE_THRESHOLD = 0.7  # 竞赛路由相似度阈值
    RERANK_SCORE_THRESHOLD = -2.0         # Rerank 最低分数阈值
    RECALL_COUNT = 20                      # 粗召回数量
    RERANK_TOP_K = 3                        # 重排序后保留数量

    def __init__(self, llm=None):
        self.llm = llm
        self._prompts = None

    def set_llm(self, llm):
        """设置 LLM 客户端，同时设置意图识别服务"""
        self.llm = llm
        intent_service.set_llm(llm)

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

    # ==================== 历史记录缓存 ====================

    def _history_cache_key(self, user_id: int, group_id: str) -> str:
        """生成历史记录缓存 key（基于 user_id + group_id）"""
        return f"{self.CACHE_PREFIX}{user_id}:{group_id}:history"

    def get_cached_history(self, user_id: int, group_id: str) -> List[ChatPair]:
        """从 Redis 获取缓存的历史记录（对话对格式）"""
        try:
            key = self._history_cache_key(user_id, group_id)
            cached = redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return [ChatPair(**pair) for pair in data]
        except Exception as e:
            logger.error(f"获取历史记录缓存失败: {e}")
        return []

    def save_history_to_cache(self, user_id: int, group_id: str, history: List[ChatPair]):
        """保存历史记录到 Redis（对话对格式）"""
        try:
            key = self._history_cache_key(user_id, group_id)
            # 只保留最近的 N 轮对话
            recent_history = history[-self.MAX_HISTORY_ROUNDS:]
            data = [pair.model_dump() for pair in recent_history]
            redis_client.set(key, json.dumps(data, ensure_ascii=False), ex=self.CACHE_TTL_HISTORY)
        except Exception as e:
            logger.error(f"保存历史记录缓存失败: {e}")

    def clear_history_cache(self, user_id: int, group_id: str):
        """清除指定对话组的历史记录缓存"""
        try:
            key = self._history_cache_key(user_id, group_id)
            redis_client.delete(key)
            logger.info(f"用户 {user_id} 对话组 {group_id} 历史记录已清除")
        except Exception as e:
            logger.error(f"清除历史记录失败: {e}")

    # ==================== 规则检索流水线 ====================

    async def _search_rules_pipeline(self, user_message: str) -> RuleSearchResult:
        """
        规则检索流水线：LLM意图识别 -> 竞赛路由 -> 规则检索 -> 重排序

        Args:
            user_message: 用户原始输入

        Returns:
            RuleSearchResult: 检索结果
        """
        # ==================== 阶段一：LLM 意图识别 ====================
        logger.info(f"【意图识别】开始解析用户输入: {user_message[:50]}...")

        intent = await intent_service.parse_rule_query(user_message)

        if not intent.parse_success:
            logger.warning("意图识别失败，无法解析用户问题")
            return RuleSearchResult(
                success=False,
                error_msg="无法理解您的问题，请尝试更明确地描述您想查询的竞赛和规则内容"
            )

        logger.info(f"【意图识别】解析成功: competition={intent.competition_query}, "
                   f"subject={intent.subject_keyword}, query={intent.rule_query}")

        # ==================== 阶段二：竞赛路由 ====================
        competition_id = None
        competition_name = None

        if intent.competition_query:
            logger.info(f"【竞赛路由】搜索竞赛: {intent.competition_query}")
            route_result = chroma_service.route_competition(intent.competition_query)

            if route_result is None:
                logger.warning(f"竞赛路由: 未找到匹配竞赛")
                return RuleSearchResult(
                    success=False,
                    error_msg=f"未找到{intent.competition_query}相关的竞赛，请确认竞赛名称"
                )

            competition_id, distance = route_result

            # 检查相似度阈值
            if distance > self.COMPETITION_DISTANCE_THRESHOLD:
                logger.warning(f"竞赛路由: 相似度过低 distance={distance:.4f} > {self.COMPETITION_DISTANCE_THRESHOLD}")
                return RuleSearchResult(
                    success=False,
                    error_msg=f"未找到{intent.competition_query}相关的竞赛，请确认竞赛名称"
                )

            logger.info(f"【竞赛路由】匹配成功: id={competition_id}, distance={distance:.4f}")

        # ==================== 阶段三：规则检索 ====================
        rule_query = intent.rule_query or user_message
        logger.info(f"【规则检索】query='{rule_query[:30]}...', competition_id={competition_id}, subject={intent.subject_keyword}")

        raw_results = chroma_service.search_competition_rules(
            query=rule_query,
            competition_id=competition_id,
            subject_keyword=intent.subject_keyword,
            n_results=self.RECALL_COUNT
        )

        if not raw_results:
            logger.info("规则检索: 未找到任何结果")
            return RuleSearchResult(
                success=False,
                error_msg="未找到相关规则信息",
                competition_id=competition_id
            )

        logger.info(f"【规则检索】召回 {len(raw_results)} 条候选结果")

        # ==================== 阶段四：重排序 ====================
        final_results = raw_results

        if reranker_service.is_initialized:
            logger.info(f"【重排序】使用 BGE-Reranker 对 {len(raw_results)} 条结果精排...")

            try:
                final_results = reranker_service.rerank_with_metadata(
                    query=rule_query,
                    results=raw_results,
                    top_k=self.RERANK_TOP_K
                )

                # 检查最高分阈值
                if final_results:
                    top_score = final_results[0].get('rerank_score', 0)
                    if top_score < self.RERANK_SCORE_THRESHOLD:
                        logger.warning(f"重排序: 最高分过低 score={top_score:.4f} < {self.RERANK_SCORE_THRESHOLD}")
                        return RuleSearchResult(
                            success=False,
                            error_msg="未找到与您问题高度相关的规则信息",
                            competition_id=competition_id
                        )

                logger.info(f"【重排序】完成，返回 Top {len(final_results)} 结果")

            except Exception as e:
                logger.error(f"重排序失败: {e}，使用原始结果")
                final_results = raw_results[:self.RERANK_TOP_K]
        else:
            logger.warning("Reranker 未初始化，跳过重排序")
            final_results = raw_results[:self.RERANK_TOP_K]

        # ==================== 阶段五：格式化上下文 ====================
        context = self._format_rules_context(final_results)

        return RuleSearchResult(
            context=context,
            success=True,
            competition_id=competition_id,
            competition_name=competition_name,
            raw_results=final_results  # 保存原始检索结果
        )

    def _format_rules_context(self, results: list) -> str:
        """格式化规则结果为上下文字符串（精简元数据）"""
        parts = ["以下是相关的竞赛规则信息：\n"]

        for i, result in enumerate(results, 1):
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', '未知文档')

            # 只保留必要信息：文件名和内容
            parts.append(f"【规则片段 {i}】来源：{filename}")
            parts.append(f"{doc}\n")

        parts.append("请根据以上规则信息回答用户的问题。如果规则信息不足以回答，请如实告知。")
        return "\n".join(parts)

    def _build_rule_references(self, raw_results: List[dict]) -> List[RuleReference]:
        """将原始检索结果转换为 RuleReference 列表"""
        references = []
        for result in raw_results:
            metadata = result.get('metadata', {})
            references.append(RuleReference(
                filename=metadata.get('filename', '未知文档'),
                chunk_index=metadata.get('chunk_index', 0),
                total_chunks=metadata.get('total_chunks', 1),
                rerank_score=result.get('rerank_score')
            ))
        return references

    # ==================== 消息构建 ====================

    def _build_messages(
        self,
        user_message: str,
        history: List[ChatPair],
        profile_context: Optional[dict] = None,
        rules_context: Optional[str] = None
    ) -> List:
        """构建 LangChain 消息列表（支持 ChatPair 格式 + 规则上下文）"""
        messages = []

        # 1. 系统消息（包含用户画像上下文）
        if profile_context:
            # 构建完整的用户上下文字符串
            context_parts = []
            if profile_context.get("analysis"):
                context_parts.append(f"用户画像分析：{profile_context['analysis']}")
            if profile_context.get("summary"):
                context_parts.append(f"用户摘要：{profile_context['summary']}")
            if profile_context.get("intent_keywords"):
                keywords = "、".join(profile_context["intent_keywords"])
                context_parts.append(f"用户意图关键词：{keywords}")

            user_context = "\n".join(context_parts)
            system_template = self._get_prompt("chat", "with_context")
            system_content = system_template.format(user_context=user_context)
        else:
            system_content = self._get_prompt("chat", "general")

        # 添加基础要求
        requirements = self._get_prompt("system", "requirements")
        if requirements:
            system_content += f"\n{requirements}"

        # 添加规则上下文（如果有）
        if rules_context:
            system_content += f"\n\n{rules_context}"

        messages.append(SystemMessage(content=system_content))

        # 2. 历史对话（限制轮数，使用 ChatPair 格式）
        recent_history = history[-self.MAX_HISTORY_ROUNDS:]
        for pair in recent_history:
            messages.append(HumanMessage(content=pair.user))
            messages.append(AIMessage(content=pair.assistant))

        # 3. 当前用户消息
        messages.append(HumanMessage(content=user_message))

        return messages

    # ==================== 核心聊天方法 ====================

    async def chat(
        self,
        user_id: int,
        group_id: str,
        message: str,
        history: Optional[List[ChatPair]] = None,
        enable_rules: bool = False,
        stream: bool = False
    ) -> dict:
        """
        带画像、历史记录和规则查询的聊天

        Args:
            user_id: 用户ID，用于获取画像和历史记录
            group_id: 对话组ID，用于区分不同对话
            message: 用户消息
            history: 历史对话（可选，如果不传则从缓存获取）
            enable_rules: 是否启用规则检索
            stream: 是否使用流式输出

        Returns:
            {
                "reply": str,           # AI 回复
                "history": List[dict],  # 更新后的历史记录（ChatPair 格式）
                "profile_used": str,    # 是否使用了画像
                "rules_used": bool,     # 是否使用了规则检索
                "error_msg": str        # 错误信息（可选）
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        # 1. 获取用户画像上下文（规则模式下跳过）
        profile_context = None

        if not enable_rules:
            try:
                profile_context = await user_service.get_user_context_by_id(user_id)
            except Exception as e:
                logger.error(f"获取用户 {user_id} 画像上下文异常: {e}")
                return {
                    "reply": "",
                    "history": [],
                    "profile_used": "error",
                    "rules_used": False,
                    "error_msg": f"获取用户画像失败: {e}"
                }

            if not profile_context:
                return {
                    "reply": "",
                    "history": [],
                    "profile_used": "error",
                    "rules_used": False,
                    "error_msg": "用户画像不存在，请先绘制用户画像"
                }

            logger.info(f"用户 {user_id} 画像上下文已加载")
        else:
            logger.info("规则检索模式，跳过画像加载")

        # 2. 获取历史记录
        if history is not None and len(history) > 0:
            cached_history = self.get_cached_history(user_id, group_id)
            if cached_history:
                history = cached_history
                logger.info(f"使用缓存历史记录: len={len(history)}")
            else:
                self.save_history_to_cache(user_id, group_id, history)
        else:
            history = self.get_cached_history(user_id, group_id)

        history = history or []

        # 3. 规则检索（使用新流水线）
        rules_context = None
        rules_used = False
        search_error = None
        raw_results = None  # 保存原始检索结果

        if enable_rules:
            logger.info(f"【规则检索流水线】开始处理: {message[:50]}...")

            search_result = await self._search_rules_pipeline(message)

            if search_result.success:
                rules_context = search_result.context
                rules_used = True
                raw_results = search_result.raw_results  # 保存原始结果
                logger.info("规则检索成功")
            else:
                search_error = search_result.error_msg
                logger.warning(f"规则检索失败: {search_error}")

        # 4. 构建消息
        if enable_rules:
            messages = self._build_messages(message, [], None, rules_context)
        else:
            messages = self._build_messages(message, history, profile_context, None)

        # 5. 调用 LLM（支持流式/非流式）
        if stream:
            return await self._chat_stream(
                user_id, group_id, message, messages, history,
                enable_rules, rules_used, search_error, raw_results
            )
        else:
            return await self._chat_sync(
                user_id, group_id, message, messages, history,
                enable_rules, rules_used, search_error, raw_results
            )

    async def _chat_sync(
        self,
        user_id: int,
        group_id: str,
        message: str,
        messages: List,
        history: List[ChatPair],
        enable_rules: bool,
        rules_used: bool,
        search_error: Optional[str],
        raw_results: Optional[List[dict]] = None
    ) -> dict:
        """同步聊天（非流式）"""
        response = await self.llm.ainvoke(messages)
        reply = response.content

        # 更新历史记录
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_pair = ChatPair(user=message, assistant=reply, created_at=created_at)
        history.append(new_pair)
        self.save_history_to_cache(user_id, group_id, history)

        result = {
            "reply": reply,
            "history": [pair.model_dump() for pair in history[-self.MAX_HISTORY_ROUNDS:]],
            "profile_used": "false" if enable_rules else "true",
            "rules_used": rules_used
        }

        # 如果规则检索成功，添加参考规则片段
        if rules_used and raw_results:
            result["rule_references"] = [ref.model_dump() for ref in self._build_rule_references(raw_results)]

        # 如果规则检索失败，将错误信息附加到回复
        if enable_rules and search_error and not rules_used:
            result["error_msg"] = search_error

        return result

    async def _chat_stream(
        self,
        user_id: int,
        group_id: str,
        message: str,
        messages: List,
        history: List[ChatPair],
        enable_rules: bool,
        rules_used: bool,
        search_error: Optional[str],
        raw_results: Optional[List[dict]] = None
    ):
        """
        流式聊天生成器

        Yields:
            dict: {"type": "chunk", "content": str} 或 {"type": "done", ...}
        """
        reply_chunks = []

        # 如果规则检索失败，先返回错误提示
        if enable_rules and search_error and not rules_used:
            yield {"type": "error", "content": search_error}

        try:
            async for chunk in self.llm.astream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    reply_chunks.append(content)
                    yield {"type": "chunk", "content": content}

            # 完成后拼接完整回复
            reply = "".join(reply_chunks)

            # 更新历史记录
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_pair = ChatPair(user=message, assistant=reply, created_at=created_at)
            history.append(new_pair)
            self.save_history_to_cache(user_id, group_id, history)

            # 构建完成信号
            done_data = {
                "type": "done",
                "reply": reply,
                "history": [pair.model_dump() for pair in history[-self.MAX_HISTORY_ROUNDS:]],
                "profile_used": "false" if enable_rules else "true",
                "rules_used": rules_used
            }

            # 如果规则检索成功，添加参考规则片段
            if rules_used and raw_results:
                done_data["rule_references"] = [ref.model_dump() for ref in self._build_rule_references(raw_results)]

            yield done_data

        except Exception as e:
            logger.error(f"流式输出异常: {e}")
            yield {"type": "error", "content": f"生成回复时出错: {str(e)}"}

    # ==================== 竞赛 AI 功能（预留） ====================

    async def get_competition_advice(
        self,
        user_id: int,
        competition_id: int
    ) -> dict:
        """
        获取竞赛建议

        Args:
            user_id: 用户ID，用于获取画像
            competition_id: 竞赛ID，用于获取竞赛信息

        Returns:
            {
                "success": bool,
                "advice": str,       # 竞赛建议文本
                "competition_name": str,  # 竞赛名称
                "error_msg": str     # 错误信息（可选）
            }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        logger.info(f"【竞赛建议】user_id={user_id}, competition_id={competition_id}")

        # 1. 获取用户画像
        try:
            user_profile = await user_service.get_user_context_by_id(user_id)
        except Exception as e:
            logger.error(f"获取用户 {user_id} 画像失败: {e}")
            return {
                "success": False,
                "advice": "",
                "competition_name": "",
                "error_msg": f"获取用户画像失败: {e}"
            }

        if not user_profile:
            return {
                "success": False,
                "advice": "",
                "competition_name": "",
                "error_msg": "用户画像不存在，请先绘制用户画像"
            }

        # 2. 获取竞赛信息（从向量库）
        comp_doc = chroma_service.get_document(
            doc_id=str(competition_id),
            collection_name=chroma_service.COLLECTION_COMPETITIONS
        )

        if not comp_doc:
            return {
                "success": False,
                "advice": "",
                "competition_name": "",
                "error_msg": f"未找到竞赛 ID={competition_id} 的信息"
            }

        competition_info = comp_doc.get("document", "")
        metadata = comp_doc.get("metadata", {})
        competition_name = metadata.get("title", "未知竞赛")

        # 3. 构建提示词
        system_prompt = self._get_prompt("competition_advice", "generate")

        # 构建用户画像文本
        profile_parts = []
        if user_profile.get("analysis"):
            profile_parts.append(f"用户画像分析：{user_profile['analysis']}")
        if user_profile.get("summary"):
            profile_parts.append(f"用户摘要：{user_profile['summary']}")
        if user_profile.get("intent_keywords"):
            keywords = "、".join(user_profile["intent_keywords"])
            profile_parts.append(f"用户兴趣方向：{keywords}")
        user_profile_text = "\n".join(profile_parts)

        # 构建竞赛信息文本
        competition_text = f"竞赛名称：{competition_name}\n"
        competition_text += f"竞赛类别：{metadata.get('category', '未知')}\n"
        competition_text += f"难度：{metadata.get('difficulty', '未知')}\n"

        user_message = f"【用户信息】\n{user_profile_text}\n\n【目标竞赛】\n{competition_text}\n\n请为该用户提供针对此竞赛的参赛建议。"

        # 5. 调用 LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            logger.info("【竞赛建议】调用 LLM 生成建议...")
            response = await self.llm.ainvoke(messages)
            advice = response.content

            logger.info(f"【竞赛建议】生成成功，字数：{len(advice)}")

            return {
                "success": True,
                "advice": advice,
                "competition_name": competition_name,
                "error_msg": None
            }

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {
                "success": False,
                "advice": "",
                "competition_name": competition_name,
                "error_msg": f"生成建议失败: {e}"
            }



# 全局单例
ai_service = AIService()
