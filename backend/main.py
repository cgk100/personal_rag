from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
import uuid
import shutil
from datetime import datetime
from fastapi.responses import StreamingResponse
import traceback
import json
import glob  # 用于文件模式匹配

from llm_client import LLMClient
# 导入文件处理相关的类
from readfile import FileProcessor, DocumentProcessor, TextEmbedding, DocumentStore

# 配置更详细的日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别以获取更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("knowledge-base")

app = FastAPI(title="本地知识库 API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型请求模式
class ConnectionTestRequest(BaseModel):
    baseUrl: str
    model: str
    apiKey: str

# 聊天请求模式
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str
    apiKey: str
    baseUrl: str
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    use_knowledge_base: bool = False

# 知识库文件存储目录
UPLOAD_DIR = "./uploads"
DB_DIR = "./chroma_db"  # 向量数据库目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 知识库文件模型
class KnowledgeFile(BaseModel):
    id: str
    name: str
    type: str
    size: int
    path: str
    created_at: str

# 内存中存储知识库文件信息（实际应用中应使用数据库）
knowledge_files = []

# 启动时加载已有文件
def load_existing_files():
    """启动时加载已有的文件信息"""
    try:
        logger.info("开始加载已有知识库文件")
        if not os.path.exists(UPLOAD_DIR):
            logger.warning(f"上传目录不存在: {UPLOAD_DIR}")
            return
            
        # 记录上传目录内容
        files_in_dir = os.listdir(UPLOAD_DIR)
        logger.info(f"上传目录中的文件: {files_in_dir}")
        
        # 检查向量数据库目录
        if os.path.exists(DB_DIR):
            db_files = glob.glob(f"{DB_DIR}/**/*", recursive=True)
            logger.info(f"向量数据库目录中的文件: {db_files}")
        else:
            logger.warning(f"向量数据库目录不存在: {DB_DIR}")
            
        # 遍历上传目录
        for filename in files_in_dir:
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                logger.debug(f"处理文件: {filename}")
                
                # 尝试从文件名中提取ID
                try:
                    parts = filename.split('_')
                    if len(parts) < 2:
                        logger.warning(f"文件名格式不正确: {filename}")
                        continue
                        
                    file_id = parts[0]
                    # 验证是否为有效的UUID
                    try:
                        uuid.UUID(file_id)
                    except ValueError:
                        logger.warning(f"文件ID不是有效的UUID: {file_id}")
                        continue
                    
                    # 获取原始文件名
                    original_filename = '_'.join(parts[1:])
                    logger.debug(f"原始文件名: {original_filename}")
                    
                    # 获取文件扩展名
                    file_extension = original_filename.split('.')[-1] if '.' in original_filename else ''
                    
                    # 获取文件大小
                    file_size = os.path.getsize(file_path)
                    
                    # 创建文件记录
                    file_record = {
                        "id": file_id,
                        "name": original_filename,
                        "type": file_extension,
                        "size": file_size,
                        "path": file_path,
                        "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        "processed": True,
                        "status": "success",
                        "message": "文件已加载",
                        "document_count": 0  # 稍后会尝试获取实际文档数量
                    }
                    
                    # 尝试从向量数据库获取文档数量
                    try:
                        from readfile import DocumentStore
                        doc_store = DocumentStore(persist_directory=DB_DIR)
                        
                        # 查询与此文件ID相关的文档
                        doc_count = doc_store.count_documents_by_file_id(file_id)
                        logger.info(f"文件 {file_id} 在向量数据库中有 {doc_count} 个文档")
                        
                        file_record["document_count"] = doc_count
                        if doc_count > 0:
                            file_record["status"] = "success"
                            file_record["message"] = f"文件已处理，生成了 {doc_count} 个文档"
                        else:
                            file_record["status"] = "warning"
                            file_record["message"] = "文件已上传但未找到相关文档"
                    except Exception as e:
                        logger.error(f"获取文档数量失败: {str(e)}")
                        logger.debug(f"错误详情: {traceback.format_exc()}")
                    
                    # 添加到文件列表
                    knowledge_files.append(file_record)
                    logger.info(f"已加载文件: {original_filename}")
                    
                except Exception as e:
                    logger.error(f"处理文件 {filename} 时出错: {str(e)}")
                    logger.debug(f"错误详情: {traceback.format_exc()}")
                    continue
                    
        logger.info(f"共加载了 {len(knowledge_files)} 个文件")
        logger.debug(f"知识库文件列表: {json.dumps(knowledge_files, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"加载已有文件失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")

# 在应用启动时加载文件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    logger.info("应用启动，开始初始化...")
    load_existing_files()
    logger.info("应用初始化完成")

@app.post("/api/test-connection")
async def test_connection(request: ConnectionTestRequest):
    """测试与LLM API的连接"""
    try:
        # 创建LLM客户端
        client = LLMClient(
            base_url=request.baseUrl,
            api_key=request.apiKey,
            model=request.model
        )
        
        # 测试连接
        result = client.test_connection()
        
        # 确保结果包含必要的字段
        if "success" not in result:
            result["success"] = True
            
        if result["success"] and "message" not in result:
            result["message"] = "成功连接到API"
            
        return result
    except Exception as e:
        logger.error(f"连接测试失败: {str(e)}")
        return {"success": False, "error": str(e), "message": f"连接失败: {str(e)}"}

@app.post("/api/chat")
async def chat(request: dict = Body(...)):
    """聊天接口"""
    try:
        query = request.get("query", "")
        logger.info(f"收到聊天请求: {query}")
        
        # 记录原始查询
        logger.info(f"原始查询: {query}")
        
        # 从知识库中检索相关文档
        from readfile import DocumentStore, TextEmbedding
        
        # 初始化嵌入模型和文档存储
        embedding_model = TextEmbedding(
            model_name="shibing624/text2vec-base-chinese",
            cache_dir="./model_cache"
        )
        doc_store = DocumentStore()
        
        # 生成查询向量
        query_embedding = embedding_model.embed_query(query)
        
        # 增加日志，记录查询向量
        logger.info(f"查询向量维度: {len(query_embedding)}")
        
        # 增加相似度阈值和检索数量参数
        similarity_threshold = 0.6  # 提高相似度阈值
        top_k = 5  # 增加检索数量
        
        # 检索相关文档
        results = doc_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # 记录检索结果
        logger.info(f"检索到 {len(results)} 个相关文档")
        for i, result in enumerate(results):
            logger.info(f"文档 {i+1}: 相似度={result['score']:.4f}, 来源={result['metadata'].get('source', '未知')}")
            logger.info(f"文档内容片段: {result['text'][:100]}...")
        
        # 如果没有找到相关文档，记录警告
        if not results:
            logger.warning(f"未找到与查询 '{query}' 相关的文档")
            
        # 构建上下文
        context = ""
        sources = []
        
        if results:
            # 构建上下文
            context = "根据知识库中的信息:\n\n"
            for i, result in enumerate(results):
                context += f"{i+1}. {result['text']}\n\n"
                
                # 添加来源信息
                source = {
                    "filename": result["metadata"].get("file_name", "未知文档"),
                    "content": result["text"][:200] + "..."
                }
                sources.append(source)
        
        # 构建提示
        system_prompt = """你是一个基于本地知识库的AI助手。请根据提供的知识库内容回答用户问题。
如果知识库中包含相关信息，请基于这些信息提供准确、简洁的回答。
如果知识库中没有相关信息，请明确告知用户"知识库中没有相关信息"，不要编造答案。
回答应当客观、准确，并引用知识库中的相关内容。
直接以Markdown格式输出内容，不要说明你在使用Markdown格式或者其他元描述。
不要在回答中包含"采用Markdown格式呈现"或类似的说明。"""

        user_prompt = f"问题: {query}\n\n{context}"
        
        # 记录完整提示
        logger.info(f"系统提示: {system_prompt}")
        logger.info(f"用户提示: {user_prompt}")
        
        # 调用LLM生成回答
        from llm import get_llm_response
        
        response = await get_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        # 如果没有找到相关文档，在回答中明确说明
        if not results:
            response = "知识库中没有找到与孟子相关的文档。请确保知识库中包含孟子的相关资料，或者尝试上传孟子的相关文档后再提问。"
        
        logger.info(f"生成回答: {response[:100]}...")
        
        # 返回结果
        return {
            "response": response,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """上传文件到知识库"""
    try:
        logger.info(f"开始处理文件上传: {file.filename}")
        
        # 获取原始文件名和扩展名
        original_filename = file.filename
        file_extension = original_filename.split('.')[-1] if '.' in original_filename else ''
        
        # 生成唯一ID作为文件名前缀，但保留原始文件名
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{original_filename}"
        
        # 确保文件名安全
        safe_filename = safe_filename.replace(' ', '_')
        
        # 构建文件路径
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        logger.info(f"保存文件到: {file_path}")
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        # 创建文件记录
        file_record = {
            "id": file_id,
            "name": original_filename,  # 使用原始文件名
            "type": file_extension,
            "size": file_size,
            "path": file_path,
            "created_at": datetime.now().isoformat(),
            "processed": False,
            "status": "processing",
            "message": "文件正在处理中",
            "document_count": 0
        }
        
        # 添加到内存中的文件列表
        knowledge_files.append(file_record)
        
        logger.info(f"文件已保存: {file_record}")
        
        # 异步处理文件内容
        if background_tasks:
            background_tasks.add_task(process_file_content, file_path, file_id, file_record)
            logger.info(f"已添加后台任务处理文件: {file_id}")
        else:
            logger.warning("BackgroundTasks 不可用，将不会处理文件内容")
        
        # 返回成功响应
        return {
            "success": True,
            "file": file_record
        }
        
    except Exception as e:
        logger.error(f"上传文件失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")

async def process_file_content(file_path: str, file_id: str, file_record: dict):
    """处理文件内容并添加到知识库"""
    try:
        logger.info(f"开始处理文件内容: {file_path}")
        
        # 初始化文件处理器
        from readfile import FileProcessor, DocumentProcessor, TextEmbedding, DocumentStore
        
        # 创建处理组件
        file_processor = FileProcessor()
        doc_processor = DocumentProcessor(
            chunk_size=1024,
            chunk_overlap=200
        )
        embedding_model = TextEmbedding(
            model_name="shibing624/text2vec-base-chinese",
            cache_dir="./model_cache"
        )
        doc_store = DocumentStore()
        
        # 1. 处理文件
        logger.info(f"处理文件: {file_path}")
        documents = file_processor.process_file(file_path)
        
        if not documents:
            logger.warning(f"文件处理未生成任何文档: {file_path}")
            file_record["processed"] = True
            file_record["status"] = "warning"
            file_record["message"] = "文件处理未生成任何文档"
            return
        
        logger.info(f"文件处理成功，生成了 {len(documents)} 个文档")
        
        # 2. 文档切割
        logger.info("开始文档切割")
        chunks = doc_processor.process_documents(documents)
        logger.info(f"文档分块成功，生成了 {len(chunks)} 个块")
        
        # 3. 生成向量
        logger.info("开始生成向量")
        embeddings = embedding_model.encode([chunk[0] for chunk in chunks])
        logger.info(f"向量生成成功，共 {len(embeddings)} 个向量")
        
        # 4. 准备元数据
        logger.info("准备元数据")
        metadatas = []
        for i, _ in enumerate(chunks):
            metadatas.append({
                "source": file_path,
                "file_id": file_id,
                "file_name": file_record["name"],
                "chunk_id": i
            })
        
        # 5. 存储到向量数据库
        logger.info("存储到向量数据库")
        doc_store.add_documents(
            texts=[chunk[0] for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"{file_id}_{i}" for i in range(len(chunks))]
        )
        
        # 更新文件记录
        file_record["processed"] = True
        file_record["status"] = "success"
        file_record["document_count"] = len(chunks)
        file_record["message"] = f"文件处理成功，生成了 {len(chunks)} 个块"
        
        logger.info(f"文件 {file_path} 处理完成")
        
    except Exception as e:
        logger.error(f"处理文件内容失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        
        # 更新文件记录
        file_record["processed"] = True
        file_record["status"] = "error"
        file_record["message"] = f"处理失败: {str(e)}"

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """获取知识库文件列表"""
    try:
        logger.info("获取知识库文件列表")
        
        # 检查知识库文件列表是否为空
        if not knowledge_files:
            logger.warning("知识库文件列表为空")
            
            # 尝试重新加载文件
            logger.info("尝试重新加载文件...")
            load_existing_files()
            
            if not knowledge_files:
                logger.warning("重新加载后知识库文件列表仍为空")
                return []
            else:
                logger.info(f"重新加载成功，找到 {len(knowledge_files)} 个文件")
        else:
            logger.info(f"知识库文件列表包含 {len(knowledge_files)} 个文件")
            
        # 记录返回的数据结构
        logger.debug(f"返回的文件列表: {json.dumps(knowledge_files, ensure_ascii=False)[:200]}...")
            
        # 确保返回的是列表类型
        if not isinstance(knowledge_files, list):
            logger.error(f"knowledge_files 不是列表类型: {type(knowledge_files)}")
            return []
            
        # 返回内存中存储的文件列表
        return knowledge_files
    except Exception as e:
        logger.error(f"获取知识库文件列表失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回空数组而不是抛出异常
        return []

@app.delete("/api/knowledge/{file_id}")
async def delete_file(file_id: str):
    """从知识库中删除文件"""
    try:
        logger.info(f"准备删除文件: {file_id}")
        
        # 查找文件记录
        file_record = None
        for file in knowledge_files:
            if file["id"] == file_id:
                file_record = file
                break
        
        if not file_record:
            logger.warning(f"文件不存在: {file_id}")
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 1. 从向量数据库中删除文档
        try:
            doc_store = DocumentStore()
            file_path = file_record["path"]
            logger.info(f"从向量数据库中删除文档: {file_path}")
            
            # 调用删除方法
            delete_result = doc_store.delete_document(file_path)
            if delete_result:
                logger.info(f"成功从向量数据库中删除文档: {file_path}")
            else:
                logger.warning(f"从向量数据库中删除文档失败: {file_path}")
                
            # 验证删除结果
            doc_count = doc_store.count_documents_by_file_id(file_id)
            logger.info(f"删除后文档数量检查: {doc_count}")
            
        except Exception as e:
            logger.error(f"从向量数据库删除文档时出错: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 继续执行文件删除，即使向量数据库删除失败
        
        # 2. 删除物理文件
        file_path = file_record["path"]
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除物理文件: {file_path}")
        else:
            logger.warning(f"物理文件不存在: {file_path}")
        
        # 3. 从列表中移除文件记录
        knowledge_files.remove(file_record)
        logger.info(f"已从知识库中移除文件记录: {file_id}")
        
        # 4. 打印最终状态
        try:
            remaining_files = [f["id"] for f in knowledge_files]
            logger.info(f"剩余文件: {remaining_files}")
            
            # 检查向量数据库状态
            stats = doc_store.get_collection_stats()
            logger.info(f"向量数据库统计: {json.dumps(stats, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"获取状态信息时出错: {str(e)}")
        
        return {
            "success": True,
            "message": "文件已成功删除",
            "file_id": file_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@app.post("/api/chat-search")
async def chat_search(request: ChatRequest):
    """使用RAG搜索回答问题"""
    try:
        # 创建LLM客户端
        client = LLMClient(
            base_url=request.baseUrl,
            api_key=request.apiKey,
            model=request.model
        )
        
        # 获取最后一条用户消息作为查询
        user_messages = [msg for msg in request.messages if msg["role"] == "user"]
        if not user_messages:
            return {"success": False, "error": "没有找到用户消息"}
            
        query = user_messages[-1]["content"]
        
        # 使用流式响应
        if request.stream:
            # 直接返回生成器
            return StreamingResponse(
                client.rag_search_completion(
                    query=query,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                media_type="text/plain"  # 使用纯文本格式
            )
        else:
            # 非流式响应，收集所有内容
            content = ""
            for chunk in client.rag_search_completion(
                query=query,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                content += chunk
                
            return {
                "success": True,
                "content": content
            }
            
    except Exception as e:
        logger.error(f"聊天搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """流式聊天API，支持普通聊天和知识库搜索"""
    try:
        # 记录请求信息
        logger.info(f"收到流式聊天请求: model={request.model}, stream=True")
        logger.info(f"请求参数: temperature={request.temperature}, max_tokens={request.max_tokens}")
        logger.debug(f"消息数量: {len(request.messages)}")
        
        # 创建LLM客户端
        client = LLMClient(
            base_url=request.baseUrl,
            api_key=request.apiKey,
            model=request.model
        )
        
        # 检查是否使用知识库
        use_knowledge_base = request.use_knowledge_base if hasattr(request, 'use_knowledge_base') else False
        
        # 获取最后一条用户消息作为查询
        user_messages = [msg for msg in request.messages if msg["role"] == "user"]
        if not user_messages:
            return {"success": False, "error": "没有找到用户消息"}
            
        query = user_messages[-1]["content"]
        logger.info(f"用户查询: {query}")
        
        # 根据是否使用知识库选择不同的流式响应方法
        if use_knowledge_base:
            logger.info("使用知识库流式响应")
            return StreamingResponse(
                client.stream_chat_completion_with_knowledge(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                media_type="text/plain"
            )
        else:
            logger.info("使用普通流式响应")
            return StreamingResponse(
                client.stream_chat_completion(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                media_type="text/plain"
            )
    except Exception as e:
        logger.error(f"流式聊天请求失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 