from pydantic import BaseModel, Field
from typing import List, Optional


class ChatPair(BaseModel):
    """对话对：一个用户消息 + 一个助手回复"""
    user: str = Field(..., description="用户消息")
    assistant: str = Field(..., description="助手回复")
    created_at: Optional[str] = Field(None, description="对话创建时间")


class RuleReference(BaseModel):
    """规则参考片段：用于展示规则来源"""
    filename: str = Field(..., description="文档名称")
    chunk_index: int = Field(..., description="片段索引（从0开始）")
    total_chunks: int = Field(..., description="该文档的总片段数")
    rerank_score: Optional[float] = Field(None, description="重排序分数（越高越相关）")


class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: int = Field(..., description="用户ID，用于获取画像和历史记录")
    group_id: str = Field(..., description="群组ID，用于区分不同对话组")
    message: str = Field(..., description="用户消息")
    history: Optional[List[ChatPair]] = Field(None, description="历史对话（可选，不传则从缓存获取）")
    enable_rules: bool = Field(False, description="是否启用规则检索（默认关闭）")
    stream: bool = Field(False, description="是否启用流式输出（默认关闭）")


