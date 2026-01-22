"""
技能分析相关数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# ==================== 技能分析 ====================

class SkillAnalysisRequest(BaseModel):
    """技能分析请求"""
    user_id: int = Field(..., description="用户ID")
    skills: List[str] = Field(..., description="用户技能列表")
    target_competition: Optional[str] = Field(None, description="目标竞赛（可选，用于针对性诊断）")

class RadarItem(BaseModel):
    """雷达图单项"""
    name: str = Field(..., description="维度名称")
    score: int = Field(..., ge=0, le=100, description="得分 0-100")


class SkillAnalysisResult(BaseModel):
    """技能分析结果"""
    radar_chart: List[RadarItem] = Field(..., description="雷达图数据")
    core_strength: str = Field(..., description="核心优势")
    weakness_diagnosis: str = Field(..., description="劣势诊断")
    learning_suggestions: List[str] = Field(default=[], description="学习建议")


class SkillAnalysisResponse(BaseModel):
    """技能分析响应"""
    success: bool = Field(default=True)
    data: Optional[SkillAnalysisResult] = Field(None, description="分析结果")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 趋势分析 ====================

class TrendAnalysisRequest(BaseModel):
    """趋势分析请求"""
    competition_name: Optional[str] = Field(None, description="竞赛名称（可选，不传则分析通用趋势）")


class TrendItem(BaseModel):
    """趋势项"""
    name: str = Field(..., description="趋势名称")
    description: str = Field(..., description="趋势描述")


class HotCompetition(BaseModel):
    """热门竞赛"""
    name: str = Field(..., description="竞赛名称")
    heat: str = Field(..., description="热度（高/中）")
    reason: str = Field(..., description="推荐理由")


class TrendAnalysisResult(BaseModel):
    """趋势分析结果"""
    trends: List[TrendItem] = Field(default=[], description="趋势列表")
    hot_competitions: List[HotCompetition] = Field(default=[], description="热门竞赛")
    summary: str = Field(default="", description="总结")


class TrendAnalysisResponse(BaseModel):
    """趋势分析响应"""
    success: bool = Field(default=True)
    data: Optional[TrendAnalysisResult] = Field(None, description="分析结果")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 学习路径 ====================

class LearningPathRequest(BaseModel):
    """学习路径请求"""
    user_id: int = Field(..., description="用户ID")
    competition_id: int = Field(..., description="目标竞赛ID")
    competition_date: str = Field(..., description="比赛开始时间 (YYYY-MM-DD)")


class TimeInfo(BaseModel):
    """时间信息"""
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="截止日期（比赛前两周）")
    competition_date: str = Field(..., description="比赛日期")
    total_days: int = Field(..., description="总天数")
    total_weeks: int = Field(..., description="总周数")


class LearningPhase(BaseModel):
    """学习阶段"""
    phase: int = Field(..., description="阶段序号")
    name: str = Field(..., description="阶段名称")
    duration: str = Field(..., description="时长描述")
    goals: List[str] = Field(default=[], description="阶段目标")
    tasks: List[str] = Field(default=[], description="具体任务")


class LearningPathResult(BaseModel):
    """学习路径结果"""
    time_info: TimeInfo = Field(..., description="时间信息")
    competition_name: str = Field(default="", description="竞赛名称")
    current_level: str = Field(default="", description="当前水平")
    target_level: str = Field(default="", description="目标水平")
    phases: List[LearningPhase] = Field(default=[], description="学习阶段")


class LearningPathResponse(BaseModel):
    """学习路径响应"""
    success: bool = Field(default=True)
    data: Optional[LearningPathResult] = Field(None, description="学习路径")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 项目规划 ====================

class ProjectIdeaRequest(BaseModel):
    """项目想法请求 - 从0生成项目书大纲"""
    user_id: Optional[int] = Field(None, description="用户ID（可选，用于获取画像辅助生成）")
    idea: str = Field(..., description="项目想法/创意描述")
    competition_name: Optional[str] = Field(None, description="目标竞赛名称（如：互联网+创新创业大赛、挑战杯）")


class BackgroundSection(BaseModel):
    """背景分析"""
    industry_background: str = Field(..., description="行业背景")
    policy_support: str = Field(default="", description="政策支持")
    market_opportunity: str = Field(default="", description="市场机遇")


class PainPointItem(BaseModel):
    """痛点项"""
    point: str = Field(..., description="痛点描述")
    target_group: str = Field(default="", description="受影响群体")
    severity: str = Field(default="中", description="严重程度（高/中/低）")


class SolutionSection(BaseModel):
    """解决方案"""
    core_idea: str = Field(..., description="核心理念")
    key_features: List[str] = Field(default=[], description="核心功能点")
    tech_stack: List[str] = Field(default=[], description="技术栈")
    innovation_points: List[str] = Field(default=[], description="创新点")


class BusinessModel(BaseModel):
    """商业模式"""
    target_customers: List[str] = Field(default=[], description="目标客户群")
    value_proposition: str = Field(default="", description="价值主张")
    revenue_streams: List[str] = Field(default=[], description="盈利模式")
    competitive_advantage: str = Field(default="", description="竞争优势")


class ProjectOutline(BaseModel):
    """项目书大纲"""
    title: str = Field(..., description="项目名称")
    slogan: str = Field(default="", description="项目口号/一句话介绍")
    background: BackgroundSection = Field(..., description="背景分析")
    pain_points: List[PainPointItem] = Field(default=[], description="痛点分析")
    solution: SolutionSection = Field(..., description="解决方案")
    business_model: BusinessModel = Field(..., description="商业模式")
    team_requirements: List[str] = Field(default=[], description="团队配置建议")
    risks_and_challenges: List[str] = Field(default=[], description="风险与挑战")


class ProjectOutlineResponse(BaseModel):
    """项目大纲响应"""
    success: bool = Field(default=True)
    data: Optional[ProjectOutline] = Field(None, description="项目大纲")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 项目诊断 ====================

class ProjectDiagnosticRequest(BaseModel):
    """项目诊断请求"""
    file_path: str = Field(..., description="项目书文件路径（PDF/Word）")
    competition_id: int = Field(..., description="目标竞赛ID")


class EvaluationPoint(BaseModel):
    """评审考察点"""
    name: str = Field(..., description="考察点名称")
    weight: float = Field(default=0, description="权重占比")
    description: str = Field(default="", description="评分标准")


class PointEvidence(BaseModel):
    """考察点证据"""
    point_name: str = Field(..., description="考察点名称")
    evidence: List[str] = Field(default=[], description="项目书中的相关证据")
    score: float = Field(default=0, ge=0, le=100, description="得分 0-100")
    comment: str = Field(default="", description="评审意见")


class DiagnosticResult(BaseModel):
    """诊断结果"""
    competition_name: str = Field(default="", description="竞赛名称")
    total_score: float = Field(default=0, ge=0, le=100, description="综合得分")
    evaluation_points: List[PointEvidence] = Field(default=[], description="各考察点评审")
    strengths: List[str] = Field(default=[], description="项目亮点")
    weaknesses: List[str] = Field(default=[], description="待改进项")
    suggestions: List[str] = Field(default=[], description="优化建议")


class ProjectDiagnosticResponse(BaseModel):
    """项目诊断响应"""
    success: bool = Field(default=True)
    data: Optional[DiagnosticResult] = Field(None, description="诊断结果")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 竞赛建议 ====================

class CompetitionAdviceRequest(BaseModel):
    """竞赛建议请求"""
    user_id: int = Field(..., description="用户ID")
    competition_id: int = Field(..., description="竞赛ID")


class CompetitionAdviceResponse(BaseModel):
    """竞赛建议响应"""
    success: bool = Field(default=True)
    advice: str = Field(default="", description="竞赛建议文本")
    competition_name: str = Field(default="", description="竞赛名称")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ==================== 匹配度计算 ====================

class MatchItem(BaseModel):
    """待匹配的单个项目（团队需求或用户技能）"""
    id: int = Field(..., description="项目ID（团队ID或用户ID）")
    description: str = Field(..., description="描述文本（团队需求描述或用户技能描述）")


class MatchRequest(BaseModel):
    """
    匹配度计算请求

    场景1 - 用户看团队列表:
      source_description = 用户技能描述（如"擅长Java，熟悉数据结构"）
      targets = 团队需求描述列表

    场景2 - 队长看申请人列表:
      source_description = 团队需求描述（如"需要机器学习，算法选手"）
      targets = 申请人技能描述列表
    """
    source_description: str = Field(..., description="源向量描述（用户技能或团队需求）")
    targets: List[MatchItem] = Field(..., description="目标项列表")


class MatchResult(BaseModel):
    """单个匹配结果"""
    id: int = Field(..., description="目标项ID")
    score: float = Field(..., ge=0, le=100, description="匹配度分数 0-100")
    similarity: float = Field(..., ge=-1, le=1, description="原始余弦相似度 -1到1")


class MatchResponse(BaseModel):
    """匹配度计算响应"""
    success: bool = Field(default=True)
    results: List[MatchResult] = Field(default=[], description="匹配结果列表（按分数降序）")
    error_msg: Optional[str] = Field(None, description="错误信息")
