from fastapi import APIRouter, HTTPException
import os
from backend.document_store import DocumentStore
from backend.utils import logger

router = APIRouter()

@router.delete("/knowledge/{file_id}")
async def delete_knowledge_file(file_id: str):
    try:
        # 获取文件信息
        file_info = get_file_info(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = file_info['path']
        
        # 1. 从文件系统删除文件
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # 2. 从向量数据库删除文档
        doc_store = DocumentStore()
        doc_store.delete_document(file_path)
        
        # 3. 从数据库删除文件记录
        delete_file_record(file_id)
        
        return {"message": "File deleted successfully"}
    except Exception as e:
        logger.error(f"删除文件时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 添加一个同步接口
@router.post("/knowledge/sync")
async def sync_knowledge_base():
    try:
        doc_store = DocumentStore()
        doc_store.sync_with_filesystem(UPLOAD_DIR)
        return {"message": "Knowledge base synchronized successfully"}
    except Exception as e:
        logger.error(f"同步知识库时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/reset")
async def reset_knowledge_base():
    """重置整个知识库"""
    try:
        doc_store = DocumentStore()
        
        # 1. 清空向量数据库
        success = doc_store.reset_database()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset vector database")
            
        # 2. 清空上传目录
        upload_dir = "./uploads"
        if os.path.exists(upload_dir):
            try:
                # 删除目录中的所有文件
                for filename in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                logger.info(f"已清空上传目录: {upload_dir}")
            except Exception as e:
                logger.error(f"清空上传目录时出错: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 3. 重新创建上传目录
        os.makedirs(upload_dir, exist_ok=True)
        
        return {"message": "Knowledge base reset successfully"}
    except Exception as e:
        logger.error(f"重置知识库时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/clear")
async def clear_knowledge_base():
    """清空知识库中的所有文档"""
    try:
        doc_store = DocumentStore()
        success = doc_store.clear_collection()
        if success:
            return {"message": "Knowledge base cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
    except Exception as e:
        logger.error(f"清空知识库时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 