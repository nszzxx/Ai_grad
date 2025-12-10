from pydantic import BaseModel
from typing import List, Optional
from app.schemas.profile import UserProfileDTO

class ChatMessage(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    profile: Optional[UserProfileDTO] = None
    # 新增：历史对话上下文
    history: List[ChatMessage] = []