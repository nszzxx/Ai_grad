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


class FileDocumentRequest(BaseModel):
    """文件文档请求"""
    file_path: str = Field(..., description="文件路径（本地路径或 HTTP URL）")
    collection_name: str = Field("default", description="集合名称")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class DirectoryDocumentRequest(BaseModel):
    """目录文档请求"""
    directory_path: str = Field(..., description="目录路径（仅支持本地路径）")
    collection_name: str = Field("default", description="集合名称")
    recursive: bool = Field(True, description="是否递归扫描子目录")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class FileListDocumentRequest(BaseModel):
    """文件列表文档请求"""
    file_paths: list[str] = Field(..., description="文件路径列表")
    collection_name: str = Field("default", description="集合名称")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class RuleDocumentRequest(BaseModel):
    """规则文档单文件请求"""
    file_path: str = Field(..., description="文件路径（本地路径或 HTTP URL）")
    competition_id: int = Field(..., description="竞赛ID")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class RuleBatchRequest(BaseModel):
    """规则文档批量文件请求"""
    file_paths: list[str] = Field(..., description="文件路径列表")
    competition_id: int = Field(..., description="竞赛ID")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class RuleDirectoryRequest(BaseModel):
    """规则文档目录扫描请求"""
    directory_path: str = Field(..., description="目录路径")
    competition_id: int = Field(..., description="竞赛ID")
    recursive: bool = Field(True, description="是否递归扫描子目录")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class RuleSearchRequest(BaseModel):
    """规则文档搜索请求"""
    query: str = Field(..., description="搜索查询")
    competition_id: Optional[int] = Field(None, description="竞赛ID（可选，不传则搜索所有竞赛规则）")
    n_results: int = Field(5, ge=1, le=100, description="返回结果数量")


class RuleDeleteByPathRequest(BaseModel):
    """规则文档单文件删除请求"""
    file_path: str = Field(..., description="文件路径")
    competition_id: int = Field(..., description="竞赛ID")


class RuleBatchDeleteByPathRequest(BaseModel):
    """规则文档批量文件删除请求"""
    file_paths: list[str] = Field(..., description="文件路径列表")
    competition_id: int = Field(..., description="竞赛ID")


class ScoreRuleRequest(BaseModel):
    """评分细则上传请求"""
    file_path: str = Field(..., description="文件路径（本地路径或 HTTP URL）")
    competition_id: int = Field(..., description="竞赛ID")
    metadata: Optional[dict] = Field(None, description="自定义元数据")


class ScoreRuleSearchRequest(BaseModel):
    """评分细则搜索请求"""
    query: str = Field(..., description="搜索查询")
    competition_id: int = Field(..., description="竞赛ID")
    n_results: int = Field(5, ge=1, le=20, description="返回结果数量")


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


# ==================== 规则文档接口 ====================

@router.post("/rules/file", summary="添加单个规则文档")
async def add_rule_document(req: RuleDocumentRequest):
    """
    从文件路径添加单个规则文档
    - 支持本地路径（如 E:/docs/rules.pdf）和 HTTP URL
    - 支持格式：pdf, docx, doc
    - 自动关联到指定竞赛
    - 自动处理重复文档（更新）
    """
    try:
        result = chroma_service.add_rule_document(
            file_path=req.file_path,
            competition_id=req.competition_id,
            custom_metadata=req.metadata
        )

        if result['success']:
            return {
                "success": True,
                "message": f"规则文档 {result['action']}: {req.file_path}",
                "data": result
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '添加失败'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/files-batch", summary="批量添加规则文档")
async def add_rule_documents_batch(req: RuleBatchRequest):
    """
    从文件路径列表批量添加规则文档
    - 支持本地路径和 HTTP URL 混合
    - 自动关联到指定竞赛
    - 自动处理重复文档（更新）
    """
    try:
        if not req.file_paths:
            raise HTTPException(status_code=400, detail="file_paths 不能为空")

        result = chroma_service.add_rule_documents_batch(
            file_paths=req.file_paths,
            competition_id=req.competition_id,
            custom_metadata=req.metadata
        )

        return {
            "success": True,
            "message": f"批量导入完成: 总数 {result['total']}, 成功 {result['success']} (新增 {result['inserted']}, 更新 {result['updated']}), 失败 {result['failed']}",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量添加规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/directory", summary="从目录批量添加规则文档")
async def add_rule_documents_from_directory(req: RuleDirectoryRequest):
    """
    从目录批量添加规则文档
    - 自动遍历目录下的 pdf, docx, doc 文件
    - 支持递归扫描子目录
    - 自动关联到指定竞赛
    - 自动处理重复文档（更新）
    """
    try:
        result = chroma_service.add_rule_documents_from_directory(
            directory_path=req.directory_path,
            competition_id=req.competition_id,
            recursive=req.recursive,
            custom_metadata=req.metadata
        )

        return {
            "success": True,
            "message": f"批量导入完成: 总数 {result['total']}, 成功 {result['success']} (新增 {result['inserted']}, 更新 {result['updated']}), 失败 {result['failed']}",
            "data": result
        }

    except Exception as e:
        logger.error(f"从目录添加规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/search", summary="搜索竞赛规则")
async def search_competition_rules(req: RuleSearchRequest):
    """
    搜索竞赛规则文档（向量语义搜索）
    - 可搜索特定竞赛的规则（传入 competition_id）
    - 可搜索所有竞赛的规则（不传 competition_id）
    """
    try:
        results = chroma_service.search_competition_rules(
            query=req.query,
            competition_id=req.competition_id,
            n_results=req.n_results
        )
        return {
            "success": True,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        logger.error(f"搜索规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/file", summary="根据文件路径删除单个规则文档")
async def delete_rule_document_by_path(req: RuleDeleteByPathRequest):
    """
    根据文件路径删除单个规则文档
    - 使用与添加时相同的路径即可删除
    """
    try:
        result = chroma_service.delete_rule_document_by_path(req.file_path)

        if result['success']:
            message = f"规则文档已删除: {req.file_path}" if result['deleted'] else f"规则文档不存在: {req.file_path}"
            return {
                "success": True,
                "message": message,
                "data": result
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '删除失败'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/files-batch", summary="根据文件路径列表批量删除规则文档")
async def delete_rule_documents_by_paths(req: RuleBatchDeleteByPathRequest):
    """根据文件路径列表批量删除规则文档"""
    try:
        if not req.file_paths:
            raise HTTPException(status_code=400, detail="file_paths 不能为空")

        result = chroma_service.delete_rule_documents_by_paths(req.file_paths)

        return {
            "success": True,
            "message": f"批量删除完成: 总数 {result['total']}, 删除 {result['deleted']}, 未找到 {result['not_found']}",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除规则文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/competition/{competition_id}", summary="删除竞赛的所有规则文档")
async def delete_competition_rules(competition_id: int):
    """删除指定竞赛的所有规则文档"""
    try:
        result = chroma_service.delete_competition_rules(competition_id)
        return {
            "success": True,
            "message": f"竞赛 {competition_id} 的 {result['deleted']} 条规则文档已删除",
            "data": result
        }

    except Exception as e:
        logger.error(f"删除竞赛规则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules/competition/{competition_id}/count", summary="获取竞赛规则文档数量")
async def get_competition_rules_count(competition_id: int):
    """获取指定竞赛的规则文档数量"""
    try:
        count = chroma_service.get_competition_rules_count(competition_id)
        return {
            "success": True,
            "competition_id": competition_id,
            "count": count
        }

    except Exception as e:
        logger.error(f"获取规则文档数量失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules/count", summary="获取所有规则文档数量")
async def get_rules_total_count():
    """获取规则文档集合的总数量"""
    try:
        count = chroma_service.get_rules_collection_count()
        return {"success": True, "count": count}

    except Exception as e:
        logger.error(f"获取规则文档总数失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 评分细则接口 ====================

@router.post("/score-rules/file", summary="上传评分细则")
async def add_score_rule(req: ScoreRuleRequest):
    """
    上传评分细则文档

    - 支持 PDF/Word 格式
    - 自动切分并向量化
    - 关联到指定竞赛
    - 用于项目诊断时的评审依据
    """
    try:
        result = chroma_service.add_score_rule_document(
            file_path=req.file_path,
            competition_id=req.competition_id,
            custom_metadata=req.metadata
        )

        if result['success']:
            return {
                "success": True,
                "message": f"评分细则上传成功: {result['chunks_count']} 个片段",
                "data": result
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '上传失败'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传评分细则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score-rules/search", summary="搜索评分细则")
async def search_score_rules(req: ScoreRuleSearchRequest):
    """
    搜索指定竞赛的评分细则

    - 基于向量语义搜索
    - 返回与查询最相关的评分标准
    """
    try:
        results = chroma_service.search_score_rules(
            query=req.query,
            competition_id=req.competition_id,
            n_results=req.n_results
        )
        return {
            "success": True,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        logger.error(f"搜索评分细则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score-rules/competition/{competition_id}", summary="获取竞赛的所有评分细则")
async def get_competition_score_rules(competition_id: int):
    """获取指定竞赛的所有评分细则文本"""
    try:
        rules = chroma_service.get_all_score_rules(competition_id)
        return {
            "success": True,
            "competition_id": competition_id,
            "count": len(rules),
            "data": rules
        }

    except Exception as e:
        logger.error(f"获取评分细则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/score-rules/competition/{competition_id}", summary="删除竞赛的评分细则")
async def delete_competition_score_rules(competition_id: int):
    """删除指定竞赛的所有评分细则"""
    try:
        count = chroma_service.delete_score_rules(competition_id)
        return {
            "success": True,
            "message": f"竞赛 {competition_id} 的 {count} 条评分细则已删除",
            "deleted_count": count
        }

    except Exception as e:
        logger.error(f"删除评分细则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 通用文档接口（保留旧版兼容） ====================

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
