from typing import Dict, Any, Optional, List, Generator
import time
import logging
import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage
import json

# 向量搜索相关导入
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMClient:
    """
    大语言模型客户端类，用于处理与不同LLM API的连接和交互
    """
    
    def __init__(self, base_url: str, api_key: str, model: str):
        """
        初始化LLM客户端
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 模型名称
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM客户端初始化完成，模型: {model}, 基础URL: {base_url}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试与API的连接
        
        Returns:
            包含测试结果的字典
        """
        start_time = time.time()
        
        try:
            # 发送一个简单的请求来测试连接
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                max_tokens=10
            )
            
            # 计算响应时间
            response_time = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "model": self.model,
                "response_time": response_time,
                "message": "成功连接到API"
            }
        except Exception as e:
            logger.error(f"API连接测试失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"连接失败: {str(e)}"
            }
    
    def chat_completion(self, 
                        messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 2000,
                        stream: bool = False) -> ChatCompletion:
        """
        创建聊天完成请求
        
        Args:
            messages: 消息列表，包含角色和内容
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            stream: 是否使用流式响应
            
        Returns:
            聊天完成响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"聊天完成请求失败: {str(e)}")
            raise
    
    def stream_chat_completion(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7, 
                              max_tokens: int = 2000) -> Generator[str, None, None]:
        """
        创建流式聊天完成请求
        
        Args:
            messages: 消息列表，包含角色和内容
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            
        Returns:
            流式聊天完成响应的生成器
        """
        try:
            # 记录请求信息
            logger.info(f"开始流式聊天请求，消息数量: {len(messages)}")
            
            # 记录最后几条消息以便调试
            if messages:
                last_messages = messages[-min(3, len(messages)):]
                for i, msg in enumerate(last_messages):
                    logger.info(f"最近消息 {i+1}: 角色={msg.get('role', 'unknown')}, 内容={msg.get('content', '')[:100]}...")
            
            # 确保消息格式正确
            formatted_messages = []
            for msg in messages:
                if 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    logger.warning(f"跳过格式不正确的消息: {msg}")
            
            if not formatted_messages:
                logger.error("没有有效的消息可发送")
                yield "错误: 没有有效的消息"
                return
            
            # 创建流式响应
            logger.info(f"向模型 {self.model} 发送请求，温度={temperature}, 最大令牌数={max_tokens}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # 流式输出内容
            content_buffer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_buffer += content
                    yield content
            
            logger.info(f"流式响应完成，总生成内容长度: {len(content_buffer)}")
            
        except Exception as e:
            error_msg = f"流式聊天完成请求失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"处理请求时出错: {str(e)}"
            
    def stream_chat_completion_with_knowledge(self, 
                                             messages: List[Dict[str, str]], 
                                             temperature: float = 0.7, 
                                             max_tokens: int = 2000) -> Generator[str, None, None]:
        """
        创建带有知识库检索的流式聊天完成请求
        
        Args:
            messages: 消息列表，包含角色和内容
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            
        Returns:
            流式聊天完成响应的生成器，包含知识库来源信息
        """
        try:
            # 记录请求日志
            logger.info(f"开始知识库检索流式请求，消息数量: {len(messages)}")
            
            # 获取最后一条用户消息
            user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            logger.info(f"用户查询: {user_query}")
            
            # 初始化文本嵌入模型
            embedding_model = TextEmbedding()
            
            # 初始化文档存储
            doc_store = DocumentStore()
            
            # 生成查询向量
            query_embedding = embedding_model.encode([user_query])[0]
            
            # 搜索相关文档
            logger.info("搜索相关文档...")
            search_results = doc_store.search(query_embedding, n_results=5)
            
            # 提取文档内容和元数据
            docs = search_results.get('documents', [[]])[0]
            metadatas = search_results.get('metadatas', [[]])[0]
            
            # 准备源文档信息
            sources_data = []
            context = ""
            
            if docs and len(docs) > 0:
                logger.info(f"找到 {len(docs)} 个相关文档")
                
                # 构建上下文和源信息
                for i, (doc, meta) in enumerate(zip(docs, metadatas)):
                    # 添加文档到上下文
                    source_path = meta.get('source', '未知')
                    context += f"\n文档 {i+1} (来源: {source_path}):\n{doc}\n"
                    
                    # 提取文件名
                    if source_path and source_path != '未知':
                        # 从路径中提取文件名
                        filename = os.path.basename(source_path)
                        
                        # 如果文件名格式为 uuid_原始文件名，则提取原始文件名
                        if '_' in filename and len(filename.split('_')[0]) >= 32:
                            # UUID通常至少32个字符
                            original_filename = '_'.join(filename.split('_')[1:])
                        else:
                            original_filename = filename
                        
                        # 添加到源数据
                        sources_data.append({
                            "filename": original_filename,
                            "title": f"文档 {i+1}",
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            "source_path": source_path  # 添加完整路径以便调试
                        })
                    else:
                        # 处理未知来源
                        sources_data.append({
                            "filename": "未知文件",
                            "title": f"文档 {i+1}",
                            "content": doc[:200] + "..." if len(doc) > 200 else doc
                        })
                
                # 构建系统提示
                system_prompt = f"""你是一个专业的助手。请基于以下参考文档回答问题。
参考文档：
{context}

用户问题：{user_query}
请给出详细的答案，确保回答专业、准确和客观。
直接以Markdown格式输出内容，不要说明你在使用Markdown格式或者其他元描述。
"""
                
                # 创建增强的消息列表
                augmented_messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # 添加用户最后一条消息
                augmented_messages.append({"role": "user", "content": user_query})
                
                logger.info("发送增强消息到LLM")
            else:
                logger.info("没有找到相关文档，使用原始消息")
                augmented_messages = messages
            
            # 创建流式响应
            response = self.client.chat.completions.create(
                model=self.model,
                messages=augmented_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # 流式输出内容
            content_buffer = ""
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_buffer += content
                    yield content
            
            # 在内容后附加知识库来源信息
            if sources_data:
                # 记录源数据以便调试
                logger.info(f"源数据详情: {json.dumps(sources_data, ensure_ascii=False)}")
                
                # 格式化知识库来源
                sources_json = json.dumps(sources_data, ensure_ascii=False)
                logger.info(f"输出知识库来源信息: {sources_json}")
                yield f"\n\nSOURCES:{sources_json}"
            
            logger.info("流式响应完成")
            
        except Exception as e:
            logger.error(f"流式聊天完成请求失败: {str(e)}", exc_info=True)
            yield f"处理请求时出错: {str(e)}"
            
    def rag_search_completion(self, 
                             query: str, 
                             temperature: float = 0.7, 
                             max_tokens: int = 2000) -> Generator[str, None, None]:
        """
        执行RAG搜索并生成回答
        
        Args:
            query: 用户查询
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            流式回答生成器
        """
        try:
            # 初始化向量模型
            logger.info("加载向量模型")
            model_path = "./model_cache/text2vec_model"
            
            # 检查模型是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型未找到: {model_path}")
                yield "错误: 向量模型未找到，请先下载模型。"
                return
                
            # 加载模型
            model = SentenceTransformer(model_path, device='cpu', local_files_only=True)
            
            # 初始化向量数据库
            logger.info("初始化向量数据库")
            chroma_db_client = chromadb.PersistentClient('./chroma_db')
            
            # 获取集合
            try:
                collection = chroma_db_client.get_collection(name="documents")
            except Exception as e:
                logger.error(f"获取集合失败: {str(e)}")
                yield "错误: 向量数据库集合不存在，请先上传文档。"
                return
                
            # 检查集合中的文档数量
            doc_count = collection.count()
            if doc_count == 0:
                logger.warning("集合中没有文档")
                yield "错误: 知识库中没有文档，请先上传文档。"
                return
                
            logger.info(f"找到 {doc_count} 个文档")
            
            # 生成查询向量
            logger.info(f"处理查询: {query}")
            query_embedding = model.encode(query)
            
            # 搜索相关文档
            result = collection.query(query_embedding, n_results=5)
            
            # 提取文档和元数据
            documents = result['documents'][0]
            metadatas = result['metadatas'][0]
            
            # 去除重复文档
            seen = set()
            unique_docs = []
            unique_metas = []
            
            for doc, meta in zip(documents, metadatas):
                if doc not in seen:
                    seen.add(doc)
                    unique_docs.append(doc)
                    unique_metas.append(meta)
            
            logger.info(f"找到 {len(unique_docs)} 个相关文档")
            
            # 构造提示词
            context = "\n\n".join(
                f"文档 {i+1} (来源: {meta.get('source', '未知')}):\n{doc}"
                for i, (doc, meta) in enumerate(zip(unique_docs, unique_metas))
            )
            
            # 提取文件名作为来源
            sources = []
            source_details = []
            
            for i, meta in enumerate(unique_metas):
                source_path = meta.get('source', '')
                if source_path:
                    # 从路径中提取文件名
                    filename = os.path.basename(source_path)
                    
                    # 如果文件名格式为 uuid_原始文件名，则提取原始文件名
                    if '_' in filename and len(filename.split('_')[0]) >= 32:
                        original_filename = '_'.join(filename.split('_')[1:])
                    else:
                        original_filename = filename
                    
                    if original_filename not in sources:
                        sources.append(original_filename)
                        
                    # 添加详细信息用于调试
                    source_details.append({
                        "index": i+1,
                        "original_path": source_path,
                        "extracted_filename": original_filename
                    })
            
            # 记录源文件详情以便调试
            logger.info(f"源文件详情: {json.dumps(source_details, ensure_ascii=False)}")
            
            # 添加来源信息到提示词
            source_info = "参考文件: " + ", ".join(sources)
            
            prompt = f"""你是一个专业的助手。请基于以下参考文档回答问题。
参考文档：
{context}

{source_info}

用户问题：{query}
请给出详细的答案，使用 markdown 格式，确保回答专业、准确和客观，并在回答中标注信息来源。
"""
            logger.info("发送请求到LLM模型")
            
            # 创建流式响应
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个知识渊博的助手，提供准确且有条理的回答。"},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 首先输出来源信息
            sources_str = ", ".join(sources)
            logger.info(f"使用的来源文件: {sources_str}")
            yield f"(来源: {sources_str})\n\n"
            
            # 返回流式响应
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"RAG搜索失败: {str(e)}")
            yield f"搜索过程中发生错误: {str(e)}"


class DocumentProcessor:
    """文档处理类，用于处理和分割文档"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_documents(self, documents: List[Any]) -> List[tuple]:
        """处理文档并返回文本块"""
        # 简单实现，实际应用中应该使用更复杂的分块策略
        chunks = []
        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            # 简单分块
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append((chunk, {'source': doc.metadata.get('file_path', 'unknown') if hasattr(doc, 'metadata') else 'unknown'}))
        return chunks


class TextEmbedding:
    """文本嵌入类，用于生成文本向量"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self.model_name = model_name
        self.model_path = "./model_cache/text2vec_model"
        
        # 检查模型是否存在
        if not os.path.exists(self.model_path):
            logger.error(f"模型未找到: {self.model_path}")
            raise FileNotFoundError(f"模型未找到: {self.model_path}")
            
        # 加载模型
        self.model = SentenceTransformer(self.model_path, device='cpu', local_files_only=True)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        return self.model.encode(texts)


class DocumentStore:
    """文档存储类，用于存储文档向量"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient('./chroma_db')
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
            
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """添加文档到向量数据库"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
            
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """搜索相似文档"""
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        ) 