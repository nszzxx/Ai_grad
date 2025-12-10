from fastapi import APIRouter, HTTPException
from app.services import llm_client
from app.core.logger import get_logger
from app.schemas.chat import ChatRequest

logger = get_logger("api")

router = APIRouter()


@router.post("/chat")
async def chat_with_ai(req: ChatRequest):
    log = logger.getChild("chat")
    log.info("=" * 25 + "CHAT REQUEST" + "=" * 25)
    try:
        log.info("收到聊天请求: %s", req.message[:10] + "..." if len(req.message) > 10 else req.message)

        # 直接使用已初始化的 LLM 客户端
        response = await llm_client.client.ainvoke(req.message)

        log.info("聊天请求处理成功: %s", response.content[:10] + "..." if len(response.content) > 10 else response.content)
        return {"reply": response.content}

    except ValueError as e:
        log.error(f"参数错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error(f"LLM 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
    finally:
        log.info("=" * 25 + "CHAT REQUEST" + "=" * 25)
