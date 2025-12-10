from pydantic import BaseModel, Field
from typing import List, Optional


# --- 子模型：用户行为日志 ---
class UserBehaviorLog(BaseModel):
    action: str  # search, chat, browser
    content: str  # 搜索关键词 / 聊天摘要 / 浏览的比赛名
    duration: Optional[int] = 0  # 停留时长(秒)


# --- 核心模型：用户全量画像 ---
class UserProfileDTO(BaseModel):
    user_id: int  # 用于 Redis Key

    # 1. 静态画像 (Static)
    major: str  # 专业
    skills: List[str]  # 技能列表
    honors: List[str]  # 荣誉列表
    experience: List[str]  # 参赛/项目经历文本

    # 2. 动态画像 (Dynamic)
    recent_logs: List[UserBehaviorLog]  # 最近 10-20 条有价值日志


# --- 请求模型：推荐接口 ---
class RecommendRequest(BaseModel):
    profile: UserProfileDTO
    top_k: int = 6  # 返回多少个比赛


# --- 响应模型：推荐结果 ---
class RecommendItem(BaseModel):
    competition_id: int  # MySQL ID
    title: str  # 比赛标题
    match_score: float  # 匹配分数 (0-100)
    reason: str  # 推荐理由 (简短)


class RecommendResponse(BaseModel):
    user_intent: str  # 返回给前端展示：AI 分析出的用户意图
    items: List[RecommendItem]  # 推荐列表