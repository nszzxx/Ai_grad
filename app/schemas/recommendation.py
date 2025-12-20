from pydantic import BaseModel, Field
from typing import List, Optional


# ==================== 智能推荐相关 ====================
class RecommendRequest(BaseModel):
    """智能推荐请求"""
    user_id: int = Field(..., description="用户ID")
    skills: List[str] = Field(default=[], description="用户技能列表")
    top_k: int = Field(default=5, ge=1, le=10, description="返回推荐数量")


class MatchScores(BaseModel):
    """匹配度评分"""
    tech_match: int = Field(..., ge=0, le=100, description="技术栈匹配度")
    difficulty_match: int = Field(..., ge=0, le=100, description="难度匹配度")
    intent_match: int = Field(..., ge=0, le=100, description="意图匹配度")


class RecommendItem(BaseModel):
    """单个推荐竞赛"""
    competition_id: int = Field(..., description="竞赛ID")
    scores: MatchScores = Field(..., description="匹配度评分")
    overall_score: int = Field(default=0, description="综合评分")
    comment: str = Field(default="", description="推荐点评")


class RecommendResponse(BaseModel):
    """智能推荐响应"""
    success: bool = Field(default=True)
    recommendations: List[RecommendItem] = Field(default=[], description="推荐列表")
    total: int = Field(default=0, description="推荐数量")
    error_msg: Optional[str] = Field(None, description="错误信息")