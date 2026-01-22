"""
AI 竞赛相关 API 路由
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from app.services import ai_service
from app.services import llm_client
from app.services.recommendation_service import recommendation_service
from app.services.skill_analysis_service import skill_analysis_service
from app.services.trend_analysis_service import trend_analysis_service
from app.services.learning_path_service import learning_path_service
from app.services.project_planning_service import project_planning_service
from app.services.chroma_service import chroma_service
from app.services.user_service import user_service
from app.db import chroma_client
from app.core.logger import get_logger
from app.schemas.chat import ChatRequest, RuleReference
from app.schemas.recommendation import RecommendRequest, RecommendResponse
from app.schemas.analysis import (
    SkillAnalysisRequest, SkillAnalysisResponse,
    TrendAnalysisRequest, TrendAnalysisResponse,
    LearningPathRequest, LearningPathResponse,
    ProjectIdeaRequest, ProjectOutlineResponse,
    ProjectDiagnosticRequest, ProjectDiagnosticResponse,
    CompetitionAdviceRequest, CompetitionAdviceResponse,
    MatchRequest, MatchResponse, MatchResult
)

logger = get_logger("ai_api")

router = APIRouter()

# ==================== 响应模型 ====================

class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool = True
    reply: str = ""
    history: Optional[List[dict]] = None
    profile_used: str = "false"  # "true" | "false" | "error"
    rules_used: bool = False  # 是否使用了规则检索
    rule_references: Optional[List[RuleReference]] = None  # 参考规则片段
    error_msg: Optional[str] = None


# ==================== 聊天接口 ====================

@router.post("/chat", summary="AI 聊天（支持画像+历史记录+流式输出）")
async def chat_with_ai(req: ChatRequest):
    """
    AI 聊天接口

    - 通过 user_id 自动获取用户画像上下文，AI 会根据用户背景给出个性化回答
    - 通过 group_id 区分不同对话组，每个对话组有独立的历史记录
    - 支持对话历史：传入 history 或自动从缓存获取，支持 5-10 轮对话记忆
    - enable_rules: 启用规则检索模式（LLM意图识别 -> 竞赛路由 -> 规则检索 -> 重排序）
    - stream: 启用流式输出（SSE格式）

    流式输出格式:
    - {"type": "chunk", "content": "..."} - 内容块
    - {"type": "done", ...} - 完成信号，包含完整回复和历史记录
    - {"type": "error", "content": "..."} - 错误信息
    """
    log = logger.getChild("chat")
    log.info("=" * 25 + "CHAT REQUEST" + "=" * 25)

    try:
        msg_preview = req.message[:20] + "..." if len(req.message) > 20 else req.message
        log.info(f"收到聊天请求: user_id={req.user_id}, group_id={req.group_id}, "
                f"stream={req.stream}, enable_rules={req.enable_rules}, message={msg_preview}")

        # 确保 ai_service 已设置 LLM
        if not ai_service.llm:
            ai_service.set_llm(llm_client.client)

        # 流式输出
        if req.stream:
            async def stream_generator():
                try:
                    async for chunk in await ai_service.chat(
                        user_id=req.user_id,
                        group_id=req.group_id,
                        message=req.message,
                        history=req.history if req.history else None,
                        enable_rules=req.enable_rules,
                        stream=True
                    ):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    log.error(f"流式输出异常: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # 非流式输出
        result = await ai_service.chat(
            user_id=req.user_id,
            group_id=req.group_id,
            message=req.message,
            history=req.history if req.history else None,
            enable_rules=req.enable_rules,
            stream=False
        )

        # 处理画像获取失败的情况
        if result.get("profile_used") == "error":
            return ChatResponse(
                success=False,
                reply="",
                history=[],
                profile_used="error",
                rules_used=False,
                error_msg=result.get("error_msg")
            )

        log.info(f"聊天完成: profile_used={result['profile_used']}, "
                f"rules_used={result.get('rules_used', False)}, history_len={len(result['history'])}")

        return ChatResponse(
            reply=result["reply"],
            history=result["history"],
            profile_used=result["profile_used"],
            rules_used=result.get("rules_used", False),
            rule_references=result.get("rule_references"),
            error_msg=result.get("error_msg")
        )

    except ValueError as e:
        log.error(f"参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"LLM 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
    finally:
        log.info("=" * 25 + "CHAT REQUEST END" + "=" * 25)


@router.delete("/chat/history/{user_id}/{group_id}", summary="清除指定对话组的历史记录")
async def clear_chat_history(user_id: int, group_id: str):
    """清除指定用户和对话组的历史记录缓存"""
    try:
        ai_service.clear_history_cache(user_id, group_id)
        return {
            "success": True,
            "message": f"用户 {user_id} 对话组 {group_id} 历史记录已清除"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{user_id}/{group_id}", summary="获取指定对话组的历史记录")
async def get_chat_history(user_id: int, group_id: str):
    """获取指定用户和对话组的历史记录缓存"""
    try:
        history = ai_service.get_cached_history(user_id, group_id)
        return {
            "success": True,
            "user_id": user_id,
            "group_id": group_id,
            "history": [pair.model_dump() for pair in history],
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 竞赛 AI 接口 ====================

@router.post("/recommend", response_model=RecommendResponse, summary="智能推荐竞赛")
async def recommend_competitions(req: RecommendRequest):
    """
    智能推荐竞赛

    基于用户画像进行加权混合检索：
    - 意图关键词检索（权重70%）
    - 技能检索（权重30%）

    返回每个竞赛的匹配度评分（雷达图数据）：
    - tech_match: 技术栈匹配度 (0-100)
    - difficulty_match: 难度匹配度 (0-100)
    - intent_match: 意图匹配度 (0-100)
    """
    log = logger.getChild("recommend")
    log.info("=" * 25 + "RECOMMEND REQUEST" + "=" * 25)

    try:
        log.info(f"推荐请求: user_id={req.user_id}, skills={req.skills}, top_k={req.top_k}")

        # 1. 获取用户画像
        user_profile = await user_service.get_user_context_by_id(req.user_id)
        if not user_profile:
            log.warning(f"用户 {req.user_id} 画像不存在")
            return RecommendResponse(
                success=False,
                recommendations=[],
                total=0,
                error_msg="用户画像不存在，请先绘制用户画像"
            )

        # 2. 设置 LLM
        if not recommendation_service.llm:
            recommendation_service.set_llm(llm_client.client)

        # 3. 执行推荐
        result = await recommendation_service.recommend(
            user_profile=user_profile,
            skills=req.skills,
            level=req.level,
            top_k=req.top_k
        )

        log.info(f"推荐完成: success={result['success']}, total={result['total']}")

        return RecommendResponse(
            success=result["success"],
            recommendations=result["recommendations"],
            total=result["total"],
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"推荐失败: {e}")
        raise HTTPException(status_code=500, detail=f"推荐服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "RECOMMEND REQUEST END" + "=" * 25)


@router.post("/skill-analysis", response_model=SkillAnalysisResponse, summary="技能分析")
async def analyze_skills(req: SkillAnalysisRequest):
    """
    技能分析接口

    基于用户技能列表和画像进行能力诊断：
    - 返回雷达图数据（5个维度：算法基础、后端工程、前端交互、文档撰写、竞赛经验）
    - 核心优势分析
    - 劣势诊断（结合目标竞赛）
    - 学习建议
    """
    log = logger.getChild("skill_analysis")
    log.info("=" * 25 + "SKILL ANALYSIS REQUEST" + "=" * 25)

    try:
        log.info(f"技能分析请求: user_id={req.user_id}, skills={req.skills}, target={req.target_competition}")

        # 1. 获取用户画像
        user_profile = await user_service.get_user_context_by_id(req.user_id)
        if not user_profile:
            log.warning(f"用户 {req.user_id} 画像不存在")
            return SkillAnalysisResponse(
                success=False,
                data=None,
                error_msg="用户画像不存在，请先绘制用户画像"
            )

        # 2. 设置 LLM
        if not skill_analysis_service.llm:
            skill_analysis_service.set_llm(llm_client.client)

        # 3. 执行分析
        result = await skill_analysis_service.analyze(
            user_profile=user_profile,
            skills=req.skills,
            target_competition=req.target_competition
        )

        log.info(f"技能分析完成: success={result['success']}")

        return SkillAnalysisResponse(
            success=result["success"],
            data=result.get("data"),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"技能分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"技能分析服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "SKILL ANALYSIS REQUEST END" + "=" * 25)


@router.post("/trend-analysis", response_model=TrendAnalysisResponse, summary="趋势分析")
async def analyze_trends(req: TrendAnalysisRequest):
    """
    趋势分析接口

    基于 DuckDuckGo 搜索分析近两年竞赛趋势：
    - 传入竞赛名称：分析该竞赛的趋势变化
    - 不传：分析通用大学生竞赛趋势

    返回：
    - trends: 主要趋势列表
    - hot_competitions: 热门竞赛
    - summary: 总结
    """
    log = logger.getChild("trend_analysis")
    log.info("=" * 25 + "TREND ANALYSIS REQUEST" + "=" * 25)

    try:
        log.info(f"趋势分析请求: competition_name={req.competition_name}")

        # 设置 LLM
        if not trend_analysis_service.llm:
            trend_analysis_service.set_llm(llm_client.client)

        # 执行分析
        result = await trend_analysis_service.analyze(
            competition_name=req.competition_name
        )

        log.info(f"趋势分析完成: success={result['success']}")

        return TrendAnalysisResponse(
            success=result["success"],
            data=result.get("data"),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"趋势分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"趋势分析服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "TREND ANALYSIS REQUEST END" + "=" * 25)


@router.post("/learning-path", response_model=LearningPathResponse, summary="学习路径规划")
async def generate_learning_path(req: LearningPathRequest):
    """
    学习路径规划接口

    基于用户画像和目标竞赛，使用 CoT 思维链生成阶段性学习计划：
    - 通过 competition_id 从向量库获取竞赛详细信息
    - 计算从当前时间到比赛前两周的可用时间
    - 分析用户当前水平和目标差距
    - 划分学习阶段，每阶段设置目标和具体任务

    返回时间轴数据，供前端渲染。
    """
    log = logger.getChild("learning_path")
    log.info("=" * 25 + "LEARNING PATH REQUEST" + "=" * 25)

    try:
        log.info(f"学习路径请求: user_id={req.user_id}, competition_id={req.competition_id}, date={req.competition_date}")

        # 1. 获取用户画像
        user_profile = await user_service.get_user_context_by_id(req.user_id)
        if not user_profile:
            log.warning(f"用户 {req.user_id} 画像不存在")
            return LearningPathResponse(
                success=False,
                data=None,
                error_msg="用户画像不存在，请先绘制用户画像"
            )

        # 2. 设置 LLM
        if not learning_path_service.llm:
            learning_path_service.set_llm(llm_client.client)

        # 3. 生成学习路径（service 内部会从向量库获取竞赛信息）
        result = await learning_path_service.generate(
            user_profile=user_profile,
            competition_id=req.competition_id,
            competition_date=req.competition_date
        )

        log.info(f"学习路径生成完成: success={result['success']}")

        return LearningPathResponse(
            success=result["success"],
            data=result.get("data"),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"学习路径生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"学习路径服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "LEARNING PATH REQUEST END" + "=" * 25)


# ==================== 项目规划接口 ====================

@router.post("/project/generate-outline", response_model=ProjectOutlineResponse, summary="生成项目书大纲")
async def generate_project_outline(req: ProjectIdeaRequest):
    """
    从想法生成项目书大纲

    输入一个项目想法/创意，AI 生成完整的项目书大纲，包含：
    - 项目名称和口号
    - 背景分析（行业背景、政策支持、市场机遇）
    - 痛点分析（核心痛点、目标群体、严重程度）
    - 解决方案（核心理念、功能点、技术栈、创新点）
    - 商业模式（目标客户、价值主张、盈利模式、竞争优势）
    - 团队配置建议
    - 风险与挑战

    可选参数：
    - user_id: 传入用户ID可获取画像，辅助生成更匹配的方案
    - competition_name: 目标竞赛名称（如：互联网+创新创业大赛、挑战杯）
    """
    log = logger.getChild("project_outline")
    log.info("=" * 25 + "PROJECT OUTLINE REQUEST" + "=" * 25)

    try:
        idea_preview = req.idea[:30] + "..." if len(req.idea) > 30 else req.idea
        log.info(f"项目大纲请求: user_id={req.user_id}, competition_name={req.competition_name}, idea={idea_preview}")

        # 1. 获取用户画像（可选）
        user_profile = None
        if req.user_id:
            user_profile = await user_service.get_user_context_by_id(req.user_id)
            if user_profile:
                log.info(f"已获取用户 {req.user_id} 画像")

        # 2. 设置 LLM
        if not project_planning_service.llm:
            project_planning_service.set_llm(llm_client.client)

        # 3. 生成项目大纲
        result = await project_planning_service.generate_outline(
            idea=req.idea,
            user_profile=user_profile,
            competition_name=req.competition_name
        )

        log.info(f"项目大纲生成完成: success={result['success']}, title={result.get('data', {}).get('title', '')}")

        return ProjectOutlineResponse(
            success=result["success"],
            data=result.get("data"),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"项目大纲生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"项目规划服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "PROJECT OUTLINE REQUEST END" + "=" * 25)


@router.post("/project/diagnose", response_model=ProjectDiagnosticResponse, summary="项目书诊断")
async def diagnose_project(req: ProjectDiagnosticRequest):
    """
    项目书诊断 - RAG 模拟评审

    上传项目书（PDF/Word），AI 结合评分细则进行模拟评审：
    1. 解析项目书文档
    2. 从向量库检索该竞赛的评分细则
    3. LLM 提取核心考察点（创新性、可行性、商业价值等）
    4. 针对每个考察点检索项目书证据
    5. LLM 综合评审打分

    返回：
    - 综合得分
    - 各考察点评分及证据
    - 项目亮点
    - 待改进项
    - 优化建议

    注意：需先上传评分细则到 score_rule_documents 集合
    """
    log = logger.getChild("project_diagnose")
    log.info("=" * 25 + "PROJECT DIAGNOSE REQUEST" + "=" * 25)

    try:
        log.info(f"项目诊断请求: file_path={req.file_path}, competition_id={req.competition_id}")

        # 1. 获取竞赛名称
        competition_name = ""
        comp_doc = chroma_service.get_document(
            doc_id=str(req.competition_id),
            collection_name=chroma_service.COLLECTION_COMPETITIONS
        )
        if comp_doc and comp_doc.get("metadata"):
            competition_name = comp_doc["metadata"].get("title", "")

        # 2. 设置 LLM
        if not project_planning_service.llm:
            project_planning_service.set_llm(llm_client.client)

        # 3. 执行诊断
        result = await project_planning_service.diagnose(
            file_path=req.file_path,
            competition_id=req.competition_id,
            competition_name=competition_name
        )

        log.info(f"项目诊断完成: success={result['success']}, score={result.get('data', {}).get('total_score', 0)}")

        return ProjectDiagnosticResponse(
            success=result["success"],
            data=result.get("data"),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"项目诊断失败: {e}")
        raise HTTPException(status_code=500, detail=f"项目诊断服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "PROJECT DIAGNOSE REQUEST END" + "=" * 25)


# ==================== 竞赛建议接口 ====================

@router.post("/competition-advice", response_model=CompetitionAdviceResponse, summary="获取竞赛建议")
async def get_competition_advice(req: CompetitionAdviceRequest):
    """
    获取竞赛建议接口

    根据用户画像和竞赛信息，生成针对性的参赛建议：
    - 通过 user_id 获取用户画像
    - 通过 competition_id 获取竞赛信息和相关规则
    - 综合分析生成专业、可行、有特色的建议

    返回：
    - advice: 纯文字建议（500-1200字）
    - competition_name: 竞赛名称

    建议内容包含：
    - 竞赛概述与定位分析
    - 用户优势与匹配度分析
    - 备赛策略与重点方向
    - 团队组建建议（如适用）
    - 常见问题与避坑指南
    - 提升建议与资源推荐
    """
    log = logger.getChild("competition_advice")
    log.info("=" * 25 + "COMPETITION ADVICE REQUEST" + "=" * 25)

    try:
        log.info(f"竞赛建议请求: user_id={req.user_id}, competition_id={req.competition_id}")

        # 确保 ai_service 已设置 LLM
        if not ai_service.llm:
            ai_service.set_llm(llm_client.client)

        # 调用服务获取建议
        result = await ai_service.get_competition_advice(
            user_id=req.user_id,
            competition_id=req.competition_id
        )

        log.info(f"竞赛建议完成: success={result['success']}, "
                f"competition={result.get('competition_name', '')}, "
                f"advice_len={len(result.get('advice', ''))}")

        return CompetitionAdviceResponse(
            success=result["success"],
            advice=result.get("advice", ""),
            competition_name=result.get("competition_name", ""),
            error_msg=result.get("error_msg")
        )

    except Exception as e:
        log.error(f"获取竞赛建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"竞赛建议服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "COMPETITION ADVICE REQUEST END" + "=" * 25)


# ==================== 匹配度计算接口 ====================

@router.post("/match", response_model=MatchResponse, summary="计算匹配度")
async def calculate_match_score(req: MatchRequest):
    """
    计算匹配度（余弦相似度）

    核心原理：使用 Embedding 模型将文本向量化，计算余弦相似度

    场景1 - 用户看团队列表:
      - source_description: 用户的技能描述（如 "擅长Java，熟悉数据结构"）
      - targets: 各团队的需求描述列表

    场景2 - 队长看申请人列表:
      - source_description: 团队的需求描述（如 "需要机器学习，算法选手"）
      - targets: 申请人的技能描述列表

    返回：
    - results: 匹配结果列表，按分数降序排列
      - id: 目标项ID
      - score: 匹配度分数 0-100
      - similarity: 原始余弦相似度 -1到1
    """
    log = logger.getChild("match")
    log.info("=" * 25 + "MATCH REQUEST" + "=" * 25)

    try:
        log.info(f"匹配度计算请求: source_len={len(req.source_description)}, targets_count={len(req.targets)}")

        if not req.targets:
            return MatchResponse(
                success=True,
                results=[],
                error_msg=None
            )

        # 提取目标描述文本列表
        target_texts = [item.description for item in req.targets]
        target_ids = [item.id for item in req.targets]

        # 计算余弦相似度
        similarities = chroma_client.compute_cosine_similarity(
            source_text=req.source_description,
            target_texts=target_texts
        )

        # 构建结果列表
        # 分数映射说明：
        # 语义模型特性：无关文本的相似度通常也在 0.6-0.8 区间（基线噪声）
        # 真正相关的文本相似度应该在 0.8+ 以上
        # 设置基线阈值 baseline=0.75，低于此值认为不匹配（得分0）
        # 高匹配阈值 high=0.95，高于此值认为完全匹配（得分100）
        BASELINE = 0.7  # 基线阈值（低于此值得分为0）
        HIGH = 0.92      # 高匹配阈值（高于此值得分为100）

        results = []
        for i, (target_id, similarity) in enumerate(zip(target_ids, similarities)):
            # 将余弦相似度映射到匹配度分数 [0, 100]
            if similarity <= BASELINE:
                score = 0.0
            elif similarity >= HIGH:
                score = 100.0
            else:
                score = (similarity - BASELINE) / (HIGH - BASELINE) * 100
            score = round(score, 2)

            results.append(MatchResult(
                id=target_id,
                score=score,
                similarity=round(similarity, 4)
            ))

        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)

        log.info(f"匹配度计算完成: results_count={len(results)}, "
                f"top_score={results[0].score if results else 'N/A'}")

        return MatchResponse(
            success=True,
            results=results,
            error_msg=None
        )

    except Exception as e:
        log.error(f"匹配度计算失败: {e}")
        raise HTTPException(status_code=500, detail=f"匹配度计算服务异常: {str(e)}")
    finally:
        log.info("=" * 25 + "MATCH REQUEST END" + "=" * 25)
