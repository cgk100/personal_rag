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
            流式聊天完成响应的生成器
        """
        try:
            # 获取最后一条用户消息
            user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            logger.info(f"\n=== 开始新的查询 ===")
            logger.info(f"用户查询: {user_query}")
            
            # 初始化文档存储并打印当前状态
            doc_store = DocumentStore()
            doc_store._print_collection_info()
            
            # 生成查询向量
            embedding_model = TextEmbedding()
            query_embedding = embedding_model.encode([user_query])[0]
            
            # 执行向量搜索
            logger.info("\n执行向量搜索...")
            search_results = doc_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=10,
                where={"is_deleted": {"$ne": "1"}},
                include=['documents', 'metadatas', 'distances']
            )
            
            # 检查是否有搜索结果
            if not search_results['documents'] or len(search_results['documents'][0]) == 0:
                logger.warning("没有找到匹配的文档")
                yield "未找到相关内容，请尝试调整搜索关键词。"
                return
            
            # 记录搜索结果数量
            result_count = len(search_results['documents'][0])
            logger.info(f"找到 {result_count} 个相关文档")
            
            # 处理搜索结果
            seen = set()
            unique_docs = []
            unique_metas = []
            sources_data = []
            context = ""
            
            # 处理向量搜索结果
            for doc, meta, distance in zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            ):
                source_path = meta.get('source', '')
                if source_path and source_path not in seen:
                    seen.add(source_path)
                    unique_docs.append(doc)
                    unique_metas.append(meta)
                    
                    # 构建上下文
                    context += f"\n文档 {len(unique_docs)} (来源: {source_path}):\n{doc}\n"
                    
                    # 准备源数据
                    sources_data.append({
                        "filename": os.path.basename(source_path),
                        "content": doc[:500] + "..." if len(doc) > 500 else doc,
                        "source_path": source_path
                    })
            
            if not unique_docs:
                logger.warning("没有找到有效的唯一文档")
                yield "未找到相关内容，请尝试调整搜索关键词。"
                return
                
            # 构造提示词
            prompt = f"""基于以下参考文档回答问题。如果无法从参考文档中找到相关信息，请明确说明。

参考文档：
{context}

用户问题：{user_query}

请给出详细的答案，使用markdown格式。确保回答专业、准确和客观。
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
            sources_str = ", ".join([os.path.basename(source_path) for source_path in seen])
            logger.info(f"使用的来源文件: {sources_str}")
            yield f"(来源: {sources_str})\n\n"
            
            # 返回流式响应
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
            yield f"处理查询时发生错误: {str(e)}"

            
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
        chunks = []
        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            # 获取文档元数据
            metadata = {
                'source': doc.metadata.get('file_path', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
                'is_deleted': False,  # 添加删除标记
                'create_time': int(time.time()),
                'file_type': doc.metadata.get('file_type', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
                'file_size': doc.metadata.get('file_size', 0) if hasattr(doc, 'metadata') else 0,
                'last_modified': doc.metadata.get('last_modified', int(time.time())) if hasattr(doc, 'metadata') else int(time.time())
            }
            
            # 分块处理
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append((chunk, metadata))
                    
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
        self.db_path = './chroma_db'
        self.client = chromadb.PersistentClient(self.db_path)
        
        try:
            # 尝试获取已存在的集合
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"成功获取已存在的集合: {collection_name}")
        except Exception as e:
            logger.info(f"集合不存在，创建新集合: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 设置向量空间
            )
        
        self._print_collection_info()

    def _print_collection_info(self):
        """打印集合详细信息，包括活跃和已删除文档的统计"""
        try:
            results = self.collection.get()
            total_count = len(results['ids']) if results.get('ids') else 0
            
            # 统计活跃和已删除的文档
            active_count = 0
            deleted_count = 0
            
            if results and 'metadatas' in results:
                for meta in results['metadatas']:
                    if meta.get('is_deleted', False):
                        deleted_count += 1
                    else:
                        active_count += 1
            
            logger.info(f"\n集合统计信息:")
            logger.info(f"- 总文档数: {total_count}")
            logger.info(f"- 活跃文档: {active_count}")
            logger.info(f"- 已删除文档: {deleted_count}")
            
            # 打印部分文档示例
            if results and 'metadatas' in results and results['metadatas']:
                logger.info("\n文档示例:")
                for i, (doc, meta) in enumerate(zip(results['documents'][:3], results['metadatas'][:3])):
                    status = "已删除" if meta.get('is_deleted', False) else "活跃"
                    logger.info(f"\n文档 {i+1}:")
                    logger.info(f"- 状态: {status}")
                    logger.info(f"- 来源: {meta.get('source', '未知')}")
                    logger.info(f"- 内容预览: {doc[:100]}...")
                
        except Exception as e:
            logger.error(f"获取集合信息时出错: {str(e)}")

    def delete_document(self, source_path: str) -> bool:
        """通过标记方式逻辑删除文档"""
        try:
            logger.info(f"开始标记删除文档: {source_path}")
            
            # 1. 先检查文档是否存在
            try:
                results = self.collection.get(
                    where={"source": source_path}
                )
                result_info = {
                    'ids_count': len(results['ids']) if results.get('ids') else 0,
                    'documents_count': len(results['documents']) if results.get('documents') else 0,
                    'metadatas': results.get('metadatas', [])
                }
                logger.info(f"查询结果: {json.dumps(result_info, ensure_ascii=False)}")
                
                if not results or len(results['ids']) == 0:
                    logger.warning(f"未找到要删除的文档: {source_path}")
                    return False
                    
                # 获取现有文档的信息
                doc_ids = results['ids']
                documents = results['documents']
                metadatas = results['metadatas']
                embeddings = results.get('embeddings', [])
                
                logger.info(f"找到 {len(doc_ids)} 个文档需要标记删除")
                
            except Exception as e:
                logger.error(f"查询文档时出错: {str(e)}", exc_info=True)
                return False
            
            # 2. 更新元数据，添加删除标记
            try:
                # 为每个文档添加删除标记
                new_metadatas = []
                for meta in metadatas:
                    new_meta = meta.copy()  # 复制原有元数据
                    new_meta['is_deleted'] = 1  # 添加删除标记
                    new_meta['delete_time'] = int(time.time())  # 添加删除时间
                    new_metadatas.append(new_meta)
                logger.info("is_deleted 更新完成")
                # 先删除原有文档
                self.collection.delete(ids=doc_ids)
                
                # 重新添加带有删除标记的文档
                self.collection.add(
                    ids=doc_ids,
                    documents=documents,
                    metadatas=new_metadatas,
                    embeddings=embeddings if embeddings else None
                )
                
                logger.info(f"成功标记删除 {len(doc_ids)} 个文档")
                
            except Exception as e:
                logger.error(f"标记删除文档时出错: {str(e)}", exc_info=True)
                return False
            
            # 3. 验证更新结果
            try:
                verify_results = self.collection.get(
                    where={
                        "source": source_path,
                        "is_deleted": True
                    }
                )
                
                if len(verify_results['ids']) != len(doc_ids):
                    logger.error(f"标记删除验证失败: 预期 {len(doc_ids)} 个文档，实际标记 {len(verify_results['ids'])} 个")
                    return False
                    
                logger.info("标记删除验证成功")
                
            except Exception as e:
                logger.error(f"验证标记删除结果时出错: {str(e)}", exc_info=True)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"标记删除过程中出错: {str(e)}", exc_info=True)
            return False

    def query_documents(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """查询文档时排除已删除的文档"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # 多查询一些，因为有些可能被标记删除
                where={"is_deleted": {"$ne": True}},  # 排除已删除的文档
                include=['documents', 'metadatas', 'distances']
            )
            
            # 只返回未删除的前 n_results 个结果
            return {
                'documents': results['documents'][0][:n_results],
                'metadatas': results['metadatas'][0][:n_results],
                'distances': results['distances'][0][:n_results]
            }
            
        except Exception as e:
            logger.error(f"查询文档时出错: {str(e)}")
            return {'documents': [], 'metadatas': [], 'distances': []}

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息，区分活跃和已删除文档"""
        try:
            # 获取所有文档
            all_results = self.collection.get()
            
            # 统计活跃和已删除的文档
            active_sources = set()
            deleted_sources = set()
            
            if all_results and 'metadatas' in all_results:
                for metadata in all_results['metadatas']:
                    if metadata and 'source' in metadata:
                        if metadata.get('is_deleted', False):
                            deleted_sources.add(metadata['source'])
                        else:
                            active_sources.add(metadata['source'])
            
            return {
                "total_documents": len(all_results['ids']) if 'ids' in all_results else 0,
                "active_documents": len(active_sources),
                "deleted_documents": len(deleted_sources),
                "active_sources": list(active_sources),
                "deleted_sources": list(deleted_sources)
            }
        except Exception as e:
            logger.error(f"获取集合统计信息时出错: {str(e)}")
            return {
                "total_documents": 0,
                "active_documents": 0,
                "deleted_documents": 0,
                "active_sources": [],
                "deleted_sources": []
            }

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """添加文档到向量数据库"""
        try:
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
            
            current_time = int(time.time())
            
            # 确保每个文档的元数据都包含必需字段
            enhanced_metadatas = []
            for meta in metadatas:
                enhanced_meta = {
                    # 基础字段
                    "source": meta.get("source", "unknown"),
                    "is_deleted": False,  # 明确设置删除标记
                    "create_time": current_time,
                    "delete_time": None,  # 未删除时为None
                    "file_type": meta.get("file_type", "unknown"),
                    "file_size": meta.get("file_size", 0),
                    "last_modified": meta.get("last_modified", current_time),
                }
                enhanced_metadatas.append(enhanced_meta)
            
            logger.info(f"添加文档，元数据示例:")
            logger.info(json.dumps(enhanced_metadatas[0], indent=2, ensure_ascii=False))
            
            # 添加文档到数据库
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=enhanced_metadatas,
                ids=ids
            )
            
            logger.info(f"成功添加 {len(texts)} 个文档")
            self._print_collection_info()
            
        except Exception as e:
            logger.error(f"添加文档时出错: {str(e)}")
            raise

    def verify_document_exists(self, source_path: str) -> bool:
        """验证文档是否存在于向量数据库中"""
        try:
            results = self.collection.get(
                where={"source": source_path}
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.error(f"验证文档存在时出错: {str(e)}")
            return False

    def sync_with_filesystem(self, knowledge_dir: str):
        """与文件系统同步，删除不存在的文档"""
        try:
            logger.info(f"开始同步知识库，目录: {knowledge_dir}")
            
            # 获取所有文档
            results = self.collection.get()
            if not results or 'metadatas' not in results:
                logger.info("知识库为空")
                return
                
            total_docs = len(results['metadatas'])
            logger.info(f"当前知识库文档总数: {total_docs}")

            # 获取同步后的统计信息
            stats = self.get_collection_stats()
            logger.info(f"知识库状态: {json.dumps(stats, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            logger.error(f"同步文件系统时出错: {str(e)}")
            raise

    def process_file(self, file_path: str):
        """处理单个文件"""
        try:
            # 获取向量数据库统计信息
            stats = self.get_collection_stats()
            logger.info(f"当前向量数据库状态: {stats}")
            
            # 如果文件已存在于向量数据库中，先删除旧数据
            if file_path in stats['sources']:
                logger.info(f"文件 {file_path} 已存在于向量数据库中，将更新数据")
            
            # 继续处理文件...
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            raise

    def reset_database(self):
        """完全重置向量数据库"""
        try:
            logger.info("开始重置向量数据库...")
            
            # 1. 删除集合
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"成功删除集合: {self.collection_name}")
            except Exception as e:
                logger.warning(f"删除集合时出错: {str(e)}")
            
            # 2. 清理持久化存储
            import shutil
            try:
                # 删除整个数据库目录
                shutil.rmtree(self.db_path)
                logger.info(f"成功删除数据库目录: {self.db_path}")
            except Exception as e:
                logger.warning(f"删除数据库目录时出错: {str(e)}")
            
            # 3. 重新初始化客户端和集合
            self.client = chromadb.PersistentClient(self.db_path)
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info("成功重新初始化客户端和集合")
            
            return True
        except Exception as e:
            logger.error(f"重置数据库时出错: {str(e)}")
            return False

    def clear_collection(self):
        """清空当前集合中的所有文档"""
        try:
            logger.info(f"开始清空集合: {self.collection_name}")
            
            # 获取所有文档数量
            count_before = self.collection.count()
            logger.info(f"当前文档数量: {count_before}")
            
            # 删除所有文档
            self.collection.delete()
            
            # 验证清空结果
            count_after = self.collection.count()
            logger.info(f"清空后文档数量: {count_after}")
            
            return True
        except Exception as e:
            logger.error(f"清空集合时出错: {str(e)}")
            return False 