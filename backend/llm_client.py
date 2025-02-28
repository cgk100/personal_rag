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
            流式聊天完成响应的生成器，包含知识库来源信息
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
            
            # 执行向量搜索，增加返回结果数量
            logger.info("\n执行向量搜索...")
            search_results = doc_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=10,  # 增加返回结果数量
                include=['documents', 'metadatas', 'distances']
            )
            
            # 记录搜索结果
            logger.info("\n搜索结果:")
            if len(search_results['documents'][0]) == 0:
                logger.warning("没有找到任何匹配的文档")
            
            for i, (doc, meta, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            )):
                logger.info(f"\n结果 {i+1}:")
                logger.info(f"  来源: {meta.get('source', '未知')}")
                logger.info(f"  相似度距离: {distance}")
                logger.info(f"  内容预览: {doc[:100]}...")
            
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
            
            # 如果没有找到文档，尝试关键词搜索
            if not unique_docs and "管理办法" in user_query:
                logger.info("\n尝试关键词搜索...")
                all_docs = doc_store.collection.get()
                for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                    source_path = meta.get('source', '')
                    if '管理办法' in source_path and source_path not in seen:
                        seen.add(source_path)
                        unique_docs.append(doc)
                        unique_metas.append(meta)
                        logger.info(f"通过关键词找到文档: {source_path}")
                        
                        context += f"\n文档 {len(unique_docs)} (来源: {source_path}):\n{doc}\n"
                        sources_data.append({
                            "filename": os.path.basename(source_path),
                            "content": doc[:500] + "..." if len(doc) > 500 else doc,
                            "source_path": source_path
                        })

            if not unique_docs:
                logger.warning("\n没有找到相关文档")
                yield "抱歉，我没有找到相关的参考信息。"
                return

            logger.info(f"\n最终使用的文档数量: {len(unique_docs)}")
            for i, meta in enumerate(unique_metas):
                logger.info(f"文档 {i+1}: {meta.get('source', '未知')}")
            
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
        self.db_path = './chroma_db'
        self.client = chromadb.PersistentClient(self.db_path)
        
        try:
            # 尝试获取已存在的集合
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"成功获取已存在的集合: {collection_name}")
            self._print_collection_info()
        except Exception as e:
            logger.info(f"集合不存在，创建新集合: {collection_name}")
            self.collection = self.client.create_collection(name=collection_name)

    def _print_collection_info(self):
        """打印集合详细信息，用于调试"""
        try:
            count = self.collection.count()
            results = self.collection.get()
            logger.info(f"集合中的文档数量: {count}")
            if results and 'metadatas' in results:
                logger.info("文档列表:")
                for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                    logger.info(f"{i+1}. 源文件: {meta.get('source', '未知')}")
                    logger.info(f"   内容预览: {doc[:100]}...")
        except Exception as e:
            logger.error(f"获取集合信息时出错: {str(e)}")

    def delete_document(self, source_path: str) -> bool:
        """删除指定源文件的所有文档"""
        try:
            logger.info(f"开始删除文档: {source_path}")
            
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
                
                # 如果找到文档，记录其IDs
                doc_ids = results.get('ids', [])
                logger.info(f"找到的文档IDs: {doc_ids}")
                
            except Exception as e:
                logger.error(f"查询文档时出错: {str(e)}", exc_info=True)
                return False
            
            if not results or len(results['ids']) == 0:
                logger.warning(f"未找到要删除的文档: {source_path}")
                return False
                
            # 2. 记录删除前的文档数
            try:
                before_count = self.collection.count()
                logger.info(f"删除前文档总数: {before_count}")
            except Exception as e:
                logger.error(f"获取文档数量时出错: {str(e)}", exc_info=True)
                return False
            
            # 3. 尝试两种删除方式
            try:
                # 方式1: 使用where条件删除
                self.collection.delete(
                    where={"source": source_path}
                )
                logger.info("方式1: where条件删除完成")
                
                # 方式2: 使用IDs删除
                if doc_ids:
                    self.collection.delete(
                        ids=doc_ids
                    )
                    logger.info("方式2: IDs删除完成")
                    
                # 方式3: 强制重新加载集合
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info("方式3: 重新加载集合完成")
                except Exception as e:
                    logger.error(f"重新加载集合失败: {str(e)}", exc_info=True)
                
            except Exception as e:
                logger.error(f"执行删除操作时出错: {str(e)}", exc_info=True)
                return False
            
            # 4. 验证删除结果
            try:
                # 强制等待一小段时间，确保删除操作完成
                import time
                time.sleep(0.5)
                
                # 检查文档是否还存在
                verify_results = self.collection.get(
                    where={"source": source_path}
                )
                after_count = self.collection.count()
                
                logger.info(f"删除后状态:")
                logger.info(f"- 总文档数: {after_count} (删除前: {before_count})")
                logger.info(f"- 目标文档是否还存在: {len(verify_results['ids']) > 0}")
                
                if len(verify_results['ids']) > 0:
                    logger.error(f"文档删除失败，仍能查询到文档: {source_path}")
                    logger.error(f"残留文档信息: {json.dumps(verify_results, ensure_ascii=False)}")
                    
                    # 如果仍然存在，尝试最后的重置方法
                    try:
                        logger.info("尝试重置集合...")
                        self.client.delete_collection(self.collection_name)
                        self.collection = self.client.create_collection(name=self.collection_name)
                        logger.info("集合重置完成")
                    except Exception as e:
                        logger.error(f"重置集合失败: {str(e)}", exc_info=True)
                    
                    return False
                    
                if after_count >= before_count:
                    logger.error(f"文档总数未减少: 删除前={before_count}, 删除后={after_count}")
                    return False
                    
            except Exception as e:
                logger.error(f"验证删除结果时出错: {str(e)}", exc_info=True)
                return False
                
            logger.info(f"文档删除成功: {source_path}")
            logger.info(f"删除前文档数: {before_count}, 删除后文档数: {after_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"删除文档过程中出错: {str(e)}", exc_info=True)
            return False

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """添加文档到向量数据库"""
        try:
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
            
            logger.info("开始添加新文档...")
            logger.info(f"文档数量: {len(texts)}")
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                logger.info(f"文档 {i+1}:")
                logger.info(f"  源文件: {metadata.get('source', '未知')}")
                logger.info(f"  内容预览: {text[:100]}...")
            
            # 添加新文档
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("文档添加完成")
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

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            results = self.collection.get()
            unique_sources = set()
            
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        source_path = metadata['source']
                        unique_sources.add(source_path)
            
            return {
                "total_documents": count,
                "unique_sources": len(unique_sources),
                "sources": list(unique_sources)
            }
        except Exception as e:
            logger.error(f"获取集合统计信息时出错: {str(e)}")
            return {
                "total_documents": 0,
                "unique_sources": 0,
                "sources": []
            }

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