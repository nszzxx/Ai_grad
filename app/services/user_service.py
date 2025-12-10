import json
from langchain_core.prompts import ChatPromptTemplate
from app.db import redis_client
from app.schemas.profile import UserProfileDTO


class UserService:
    def __init__(self, llm):
        self.llm = llm
        self.redis = redis_client()
        self.CACHE_TTL = 3600  # 缓存 1 小时

    def _build_prompt(self, profile: UserProfileDTO) -> str:
        """将 JSON 画像转为自然语言"""
        # 简化版 Prompt，结合了静态和动态日志
        return f"""
            用户专业：{profile.major}。
            技能：{', '.join(profile.skills)}。
            荣誉：{', '.join(profile.honors)}。
            参赛/项目经历：{'; '.join(profile.experience)}。
            近期行为：{[log.content for log in profile.recent_logs]}。
        """

    async def get_user_intent_context(self, profile: UserProfileDTO) -> str:
        """
        【全局核心方法】获取用户意图上下文
        逻辑：查 Redis -> 没有则 LLM 分析 -> 存 Redis -> 返回
        """
        user_id = profile.user_id
        cache_key = f"ai:user_context:{user_id}"

        # 1. 查缓存
        cached_data = self.redis.get(cache_key)
        if cached_data:
            return cached_data

        # 2. 缓存未命中，调用 LLM 分析
        print(f"⚡ 用户 {user_id} 缓存失效，正在进行 AI 画像分析...")

        profile_text = self._build_prompt(profile)
        system_tmpl = "你是一个竞赛顾问。根据用户资料，总结他当前的参赛意图、技术短板和推荐方向。100字以内。"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_tmpl),
            ("user", profile_text)
        ])

        chain = prompt | self.llm
        analysis_result = await chain.ainvoke({})
        intent_text = analysis_result.content

        # 3. 存缓存
        self.redis.set(cache_key, intent_text, ex=self.CACHE_TTL)

        return intent_text