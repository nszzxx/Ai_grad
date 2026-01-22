"""
用户画像服务
负责用户画像构建、缓存管理、意图分析等
"""
import asyncio
import json
import hashlib
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from app.db import redis_client
from app.schemas.profile import UserProfileDTO
from app.core.logger import get_logger

logger = get_logger("user_service")


class UserService:
    """用户画像服务"""

    # 缓存配置
    CACHE_PREFIX = "ai:user:"
    CACHE_TTL_PROFILE = 5400        # 完整画像缓存 90 分钟
    CACHE_TTL_INTENT = 5400         # 意图分析缓存 90 分钟
    CACHE_TTL_SUMMARY = 5400        # 摘要缓存 90 分钟

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

    # ==================== 缓存管理 ====================

    def _cache_key(self, user_id: int, suffix: str) -> str:
        """生成缓存 key"""
        return f"{self.CACHE_PREFIX}{user_id}:{suffix}"

    def _profile_hash(self, profile: UserProfileDTO) -> str:
        """计算画像数据的 hash，用于判断是否需要更新缓存"""
        data = {
            "major": profile.major,
            "skills": sorted(profile.skills),
            "honors": sorted(profile.honors),
            "experience": sorted(profile.experience),
            "logs": [{"action": log.action, "content": log.content} for log in profile.recent_logs[:10]]
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def _get_cache(self, key: str) -> Optional[str]:
        """从 Redis 获取缓存（统一返回 str）"""
        try:
            value = redis_client.get(key)
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", errors="ignore")
            return str(value)
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None

    def _set_cache(self, key: str, value: str, ttl: int = None):
        """设置 Redis 缓存"""
        try:
            redis_client.set(key, value, ex=ttl or self.CACHE_TTL_PROFILE)
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")

    def _delete_cache(self, key: str):
        """删除缓存"""
        try:
            redis_client.delete(key)
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")

    def invalidate_user_cache(self, user_id: int):
        """清除用户所有缓存"""
        suffixes = ["profile", "intent", "summary", "hash"]
        for suffix in suffixes:
            self._delete_cache(self._cache_key(user_id, suffix))
        logger.info(f"用户 {user_id} 缓存已清除")

    # ==================== 画像文本构建 ====================

    def _build_profile_text(self, profile: UserProfileDTO) -> str:
        """将用户画像转为结构化文本"""
        template = self._get_prompt("templates", "profile_input")
        if not template:
            template = "用户专业：{major}。技能：{skills}。荣誉：{honors}。参赛/项目经历：{experience}。近期行为：{recent_logs}。"

        # 处理近期行为日志
        logs_text = []
        for log in profile.recent_logs[:10]:
            action_map = {"search": "搜索", "chat": "咨询", "browse": "浏览"}
            action_cn = action_map.get(log.action, log.action)
            logs_text.append(f"{action_cn}:{log.content}")

        return template.format(
            major=profile.major,
            skills="、".join(profile.skills) if profile.skills else "暂无",
            honors="、".join(profile.honors) if profile.honors else "暂无",
            experience="；".join(profile.experience) if profile.experience else "暂无",
            recent_logs="；".join(logs_text) if logs_text else "暂无"
        )

    # ==================== 核心方法：画像分析 ====================

    async def analyze_user_profile(self, profile: UserProfileDTO, force_refresh: bool = False) -> dict:
        """
        完整的用户画像分析
        返回: {
            "user_id": int,
            "analysis": str,      # 完整分析
            "summary": str,       # 一句话摘要
            "intent_keywords": list,  # 意图关键词
            "cached": bool        # 是否来自缓存
        }
        """
        if not self.llm:
            raise ValueError("LLM 客户端未初始化")

        user_id = profile.user_id
        current_hash = self._profile_hash(profile)

        # 检查缓存是否有效（hash 一致表示用户数据未变化）
        if not force_refresh:
            cached_hash = self._get_cache(self._cache_key(user_id, "hash"))
            if cached_hash == current_hash:
                cached_analysis = self._get_cache(self._cache_key(user_id, "profile"))
                cached_summary = self._get_cache(self._cache_key(user_id, "summary"))
                cached_intent = self._get_cache(self._cache_key(user_id, "intent"))

                if cached_analysis and cached_summary:
                    logger.debug(f"用户 {user_id} 画像命中缓存")
                    return {
                        "user_id": user_id,
                        "analysis": cached_analysis,
                        "summary": cached_summary,
                        "intent_keywords": json.loads(cached_intent) if cached_intent else [],
                        "cached": True
                    }

        logger.info(f"用户 {user_id} 开始画像分析...")
        profile_text = self._build_profile_text(profile)

        # 并行执行三个分析任务
        analysis_result, summary_result, intent_result = await asyncio.gather(
            self._run_analysis(profile_text),
            self._run_summary(profile_text),
            self._run_intent_extract(profile_text),
        )

        # 更新缓存
        self._set_cache(self._cache_key(user_id, "hash"), current_hash, self.CACHE_TTL_PROFILE)
        self._set_cache(self._cache_key(user_id, "profile"), analysis_result, self.CACHE_TTL_PROFILE)
        self._set_cache(self._cache_key(user_id, "summary"), summary_result, self.CACHE_TTL_SUMMARY)
        self._set_cache(self._cache_key(user_id, "intent"), json.dumps(intent_result), self.CACHE_TTL_INTENT)

        logger.info(f"用户 {user_id} 画像分析完成")
        return {
            "user_id": user_id,
            "analysis": analysis_result,
            "summary": summary_result,
            "intent_keywords": intent_result,
            "cached": False
        }

    async def _run_analysis(self, profile_text: str) -> str:
        """执行完整画像分析"""
        logger.info("画像分析中...")
        system_prompt = self._get_prompt("user_profile", "analysis")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", profile_text)
        ])
        chain = prompt | self.llm
        logger.info(f"请求地址={getattr(self.llm, 'base_url', None) or getattr(self.llm, 'endpoint', None) or getattr(self.llm, 'api_base', None)}，请求内容={profile_text}")
        result = await chain.ainvoke({})
        logger.info("画像分析完成")
        return result.content

    async def _run_summary(self, profile_text: str) -> str:
        """执行摘要生成"""
        logger.info("摘要生成中...")
        system_prompt = self._get_prompt("user_profile", "summary")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", profile_text)
        ])
        chain = prompt | self.llm
        result = await chain.ainvoke({})
        logger.info("摘要生成完成")
        return result.content

    async def _run_intent_extract(self, profile_text: str) -> list:
        """提取意图关键词"""
        logger.info("意图关键词提取中...")
        system_prompt = self._get_prompt("user_profile", "intent_extract")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", profile_text)
        ])
        chain = prompt | self.llm
        result = await chain.ainvoke({})
        logger.info("意图关键词提取完成")
        # 解析关键词（假设返回格式为逗号分隔或换行分隔）
        keywords = [k.strip() for k in result.content.replace("\n", "、").split("、") if k.strip()]
        return keywords[:5]

    # ==================== 快速方法：仅获取意图上下文 ====================

    async def get_user_intent_context(self, profile: UserProfileDTO) -> str:
        """
        快速获取用户意图上下文（用于聊天增强）
        优先返回缓存，无缓存则生成简化版本
        """
        user_id = profile.user_id
        cache_key = self._cache_key(user_id, "profile")

        # 尝试从缓存获取
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # 缓存未命中，执行快速分析
        logger.info(f"用户 {user_id} 快速画像分析...")
        profile_text = self._build_profile_text(profile)
        analysis = await self._run_analysis(profile_text)

        # 缓存结果
        self._set_cache(cache_key, analysis, self.CACHE_TTL_INTENT)
        return analysis

    # ==================== 辅助方法 ====================

    async def get_user_summary(self, profile: UserProfileDTO) -> str:
        """获取用户一句话摘要"""
        user_id = profile.user_id
        cache_key = self._cache_key(user_id, "summary")

        cached = self._get_cache(cache_key)
        if cached:
            return cached

        profile_text = self._build_profile_text(profile)
        summary = await self._run_summary(profile_text)
        self._set_cache(cache_key, summary, self.CACHE_TTL_SUMMARY)
        return summary

    async def get_intent_keywords(self, profile: UserProfileDTO) -> list:
        """获取用户意图关键词"""
        user_id = profile.user_id
        cache_key = self._cache_key(user_id, "intent")

        cached = self._get_cache(cache_key)
        if cached:
            return json.loads(cached)

        profile_text = self._build_profile_text(profile)
        keywords = await self._run_intent_extract(profile_text)
        self._set_cache(cache_key, json.dumps(keywords), self.CACHE_TTL_INTENT)
        return keywords

    def get_cache_status(self, user_id: int) -> dict:
        """获取用户缓存状态"""
        suffixes = ["profile", "intent", "summary", "hash"]
        status = {}
        for suffix in suffixes:
            key = self._cache_key(user_id, suffix)
            value = self._get_cache(key)
            logger.info(f"缓存检查 - Key: {key}, Exists: {value}")
            status[suffix] = {
                "exists": value is not None,
                "preview": (value[:50] + "...") if value and len(value) > 50 else (value or "")
            }
        return status

    # ==================== 通过 ID 获取用户上下文 ====================

    async def get_user_context_by_id(self, user_id: int) -> Optional[dict]:
        """
        通过用户 ID 获取完整的用户画像上下文

        优先从缓存获取，如果缓存不存在则返回 None，
        调用方可根据返回值判断是否需要重新构建用户画像

        Args:
            user_id: 用户 ID

        Returns:
            dict: 完整用户画像（如果缓存存在）
                {
                    "analysis": str,        # 完整分析
                    "summary": str,         # 一句话摘要
                    "intent_keywords": list # 意图关键词
                }
            None: 缓存不存在，需要重新绘制用户画像
        """
        # 获取三个缓存字段
        cached_analysis = self._get_cache(self._cache_key(user_id, "profile"))
        cached_summary = self._get_cache(self._cache_key(user_id, "summary"))
        cached_intent = self._get_cache(self._cache_key(user_id, "intent"))

        # 只要 analysis 存在就认为画像有效
        if cached_analysis:
            logger.info(f"用户 {user_id} 画像上下文命中缓存")
            return {
                "analysis": cached_analysis,
                "summary": cached_summary or "",
                "intent_keywords": json.loads(cached_intent) if cached_intent else []
            }

        # 缓存不存在，返回 None 提示调用方需要重新构建画像
        logger.info(f"用户 {user_id} 画像缓存不存在，需要重新绘制用户画像")
        return None


# 全局单例
user_service = UserService()
