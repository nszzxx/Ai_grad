"""
用户画像 API 路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services import llm_client
from app.services.user_service import user_service
from app.core.logger import get_logger
from app.schemas.profile import UserProfileDTO

logger = get_logger("user")

router = APIRouter()


# ==================== 请求模型 ====================

class ProfileAnalysisRequest(BaseModel):
    """用户画像分析请求"""
    profile: UserProfileDTO
    force_refresh: bool = Field(False, description="是否强制刷新缓存")


class ProfileRequest(BaseModel):
    """通用用户画像请求"""
    profile: UserProfileDTO


# ==================== 用户画像接口 ====================

@router.post("/analyze", summary="分析用户画像")
async def analyze_user_profile(req: ProfileAnalysisRequest):
    """
    完整的用户画像分析
    返回：分析结果、一句话摘要、意图关键词
    """
    log = logger.getChild("analyze")
    try:
        log.info(f"用户 {req.profile.user_id} 请求画像分析, force_refresh={req.force_refresh}")

        # 确保 user_service 已设置 LLM
        if not user_service.llm:
            user_service.set_llm(llm_client.client)

        result = await user_service.analyze_user_profile(req.profile, req.force_refresh)

        log.info(f"用户 {req.profile.user_id} 画像分析完成, cached={result['cached']}")
        return {
            "success": True,
            "data": result
        }

    except ValueError as e:
        log.error(f"参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"画像分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"画像分析失败: {str(e)}")

@router.get("/cache/{user_id}", summary="获取用户缓存状态")
async def get_user_cache_status(user_id: int):
    """获取用户画像缓存状态"""
    try:
        status = user_service.get_cache_status(user_id)
        return {
            "success": True,
            "user_id": user_id,
            "cache_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{user_id}", summary="清除用户缓存")
async def invalidate_user_cache(user_id: int):
    """清除用户画像缓存"""
    try:
        user_service.invalidate_user_cache(user_id)
        return {
            "success": True,
            "message": f"用户 {user_id} 缓存已清除"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
