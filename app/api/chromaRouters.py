"""
Chroma 向量数据库 API 路由
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from app.services.chroma_service import chroma_service
from app.core.logger import get_logger

logger = get_logger("chroma_api")
router = APIRouter()


# ==================== 请求/响应模型 ====================

class DocumentRequest(BaseModel):
    """单个文档请求"""
    doc_id: str = Field(..., description="文档ID")
    text: str = Field(..., description="文档内容")
    metadata: Optional[dict] = Field(None, description="元数据")
    collection_name: str = Field("default", description="集合名称")


class DocumentUpdateRequest(BaseModel):
    """文档更新请求"""
    doc_id: str = Field(..., description="文档ID")
    text: Optional[str] = Field(None, description="文档内容")
    metadata: Optional[dict] = Field(None, description="元数据")
    collection_name: str = Field("default", description="集合名称")


class BatchDocumentRequest(BaseModel):
    """批量文档请求"""
    ids: list[str] = Field(..., description="文档ID列表")
    texts: list[str] = Field(..., description="文档内容列表")
    metadatas: Optional[list[dict]] = Field(None, description="元数据列表")
    collection_name: str = Field("default", description="集合名称")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    n_results: int = Field(5, ge=1, le=100, description="返回结果数量")
    collection_name: str = Field("default", description="集合名称")


class CompetitionSearchRequest(BaseModel):
    """竞赛搜索请求"""
    query: str = Field(..., description="搜索查询")
    n_results: int = Field(5, ge=1, le=100, description="返回结果数量")


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""
    doc_ids: list[str] = Field(..., description="文档ID列表")
    collection_name: str = Field("default", description="集合名称")


class BatchCompetitionDeleteRequest(BaseModel):
    """批量删除竞赛请求"""
    competition_ids: list[int] = Field(..., description="竞赛ID列表")


# ==================== 竞赛数据接口 ====================

@router.post("/competitions/sync", summary="同步所有竞赛数据")
async def sync_competitions():
    """从 MySQL 同步所有竞赛数据到 Chroma 向量库"""
    try:
        result = chroma_service.sync_competitions_from_mysql()
        return {
            "success": True,
            "message": f"同步完成: 总数 {result['total']}, 新增 {result['inserted']}, 更新 {result['updated']}",
            "data": result
        }
    except Exception as e:
        logger.error(f"同步竞赛数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitions/sync/{competition_id}", summary="同步单个竞赛数据")
async def sync_single_competition(competition_id: int):
    """同步单个竞赛数据到 Chroma 向量库"""
    try:
        result = chroma_service.sync_single_competition(competition_id)
        return {
            "success": True,
            "message": f"竞赛 {competition_id} 同步成功 ({result['action']})",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"同步竞赛 {competition_id} 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitions/search", summary="搜索相似竞赛")
async def search_competitions(req: CompetitionSearchRequest):
    """根据查询内容搜索相似竞赛"""
    try:
        results = chroma_service.search_competitions(req.query, req.n_results)
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        logger.error(f"搜索竞赛失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/competitions/{competition_id}", summary="删除单个竞赛向量数据")
async def delete_competition(competition_id: int):
    """从向量库中删除单个竞赛数据"""
    try:
        result = chroma_service.delete_competition(competition_id)
        return {
            "success": result,
            "message": f"竞赛 {competition_id} 已删除" if result else f"竞赛 {competition_id} 不存在"
        }
    except Exception as e:
        logger.error(f"删除竞赛 {competition_id} 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/competitions/batch", summary="批量删除竞赛向量数据")
async def delete_competitions_batch(req: BatchCompetitionDeleteRequest):
    """批量删除竞赛数据"""
    try:
        count = chroma_service.delete_competitions_batch(req.competition_ids)
        return {
            "success": True,
            "message": f"成功删除 {count} 条竞赛数据",
            "deleted_count": count
        }
    except Exception as e:
        logger.error(f"批量删除竞赛失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitions/count", summary="获取竞赛向量数据数量")
async def get_competitions_count():
    """获取竞赛集合中的向量数据数量"""
    try:
        count = chroma_service.get_competitions_count()
        return {"success": True, "count": count}
    except Exception as e:
        logger.error(f"获取竞赛数量失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/competitions/clear", summary="清空竞赛向量数据")
async def clear_competitions():
    """清空竞赛集合中的所有数据"""
    try:
        count = chroma_service.clear_competitions()
        return {
            "success": True,
            "message": f"已清空 {count} 条竞赛数据",
            "deleted_count": count
        }
    except Exception as e:
        logger.error(f"清空竞赛数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 通用文档接口 ====================

@router.post("/documents", summary="添加单个文档")
async def add_document(req: DocumentRequest):
    """添加单个文档到向量库（使用 upsert 防重复）"""
    try:
        action = chroma_service.add_document(
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata,
            collection_name=req.collection_name
        )
        return {
            "success": True,
            "action": action,
            "message": f"文档 {req.doc_id} {action}"
        }
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/batch", summary="批量添加文档")
async def add_documents_batch(req: BatchDocumentRequest):
    """批量添加文档（使用 upsert 防重复）"""
    try:
        if len(req.ids) != len(req.texts):
            raise HTTPException(status_code=400, detail="ids 和 texts 长度必须相同")

        result = chroma_service.add_documents_batch(
            ids=req.ids,
            texts=req.texts,
            metadatas=req.metadatas,
            collection_name=req.collection_name
        )
        return {
            "success": True,
            "message": f"新增 {result['inserted']}, 更新 {result['updated']}",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量添加文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}", summary="获取单个文档")
async def get_document(doc_id: str, collection_name: str = Query("default")):
    """获取单个文档"""
    try:
        doc = chroma_service.get_document(doc_id, collection_name)
        if doc:
            return {"success": True, "data": doc}
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents", summary="更新单个文档")
async def update_document(req: DocumentUpdateRequest):
    """更新单个文档"""
    try:
        if req.text is None and req.metadata is None:
            raise HTTPException(status_code=400, detail="text 和 metadata 不能同时为空")

        result = chroma_service.update_document(
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata,
            collection_name=req.collection_name
        )
        if result:
            return {"success": True, "message": f"文档 {req.doc_id} 更新成功"}
        raise HTTPException(status_code=404, detail=f"文档 {req.doc_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", summary="删除单个文档")
async def delete_document(doc_id: str, collection_name: str = Query("default")):
    """删除单个文档"""
    try:
        result = chroma_service.delete_document(doc_id, collection_name)
        return {
            "success": result,
            "message": f"文档 {doc_id} 已删除" if result else f"文档 {doc_id} 不存在"
        }
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/batch", summary="批量删除文档")
async def delete_documents_batch(req: BatchDeleteRequest):
    """批量删除文档"""
    try:
        count = chroma_service.delete_documents_batch(req.doc_ids, req.collection_name)
        return {
            "success": True,
            "message": f"成功删除 {count} 个文档",
            "deleted_count": count
        }
    except Exception as e:
        logger.error(f"批量删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/search", summary="向量搜索")
async def search_documents(req: SearchRequest):
    """向量相似度搜索"""
    try:
        results = chroma_service.search_similar(
            query=req.query,
            n_results=req.n_results,
            collection_name=req.collection_name
        )
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        logger.error(f"搜索文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/exists/{doc_id}", summary="检查文档是否存在")
async def check_document_exists(doc_id: str, collection_name: str = Query("default")):
    """检查文档是否存在"""
    try:
        exists = chroma_service.exists(doc_id, collection_name)
        return {"success": True, "exists": exists}
    except Exception as e:
        logger.error(f"检查文档存在失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 集合管理接口 ====================

@router.get("/collections", summary="列出所有集合")
async def list_collections():
    """列出所有集合"""
    try:
        collections = chroma_service.list_collections()
        return {"success": True, "collections": collections}
    except Exception as e:
        logger.error(f"列出集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/info", summary="获取所有集合信息")
async def get_all_collections_info():
    """获取所有集合的详细信息"""
    try:
        info = chroma_service.get_all_collections_info()
        return {"success": True, "data": info}
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}", summary="获取集合信息")
async def get_collection_info(collection_name: str):
    """获取指定集合的详细信息"""
    try:
        info = chroma_service.get_collection_info(collection_name)
        return {"success": True, "data": info}
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/count", summary="获取集合文档数量")
async def get_collection_count(collection_name: str):
    """获取指定集合的文档数量"""
    try:
        count = chroma_service.get_collection_count(collection_name)
        return {"success": True, "count": count}
    except Exception as e:
        logger.error(f"获取集合数量失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/ids", summary="列出集合所有文档ID")
async def list_collection_ids(collection_name: str, limit: int = Query(None, ge=1, le=1000)):
    """列出指定集合的所有文档ID"""
    try:
        ids = chroma_service.list_all_ids(collection_name, limit)
        return {"success": True, "count": len(ids), "ids": ids}
    except Exception as e:
        logger.error(f"列出文档ID失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}/clear", summary="清空集合")
async def clear_collection(collection_name: str):
    """清空指定集合的所有文档"""
    try:
        count = chroma_service.clear_collection(collection_name)
        return {
            "success": True,
            "message": f"已清空集合 {collection_name}，删除 {count} 个文档",
            "deleted_count": count
        }
    except Exception as e:
        logger.error(f"清空集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}", summary="删除集合")
async def delete_collection(collection_name: str):
    """删除整个集合"""
    try:
        result = chroma_service.delete_collection(collection_name)
        return {
            "success": result,
            "message": f"集合 {collection_name} 已删除" if result else f"集合 {collection_name} 删除失败"
        }
    except Exception as e:
        logger.error(f"删除集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
