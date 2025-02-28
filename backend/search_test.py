from sentence_transformers import SentenceTransformer
import logging
import chromadb
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import os

from openai import OpenAI  # 使用 OpenAI 兼容客户端调用 DeepSeek

# 禁用 huggingface 的在线检查
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免警告

# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化向量模型，使用本地缓存的模型
logger.info("Loading sentence transformer model from local cache")
model_path = "./model_cache/text2vec_model"  # 使用下载脚本保存的模型路径

# 检查模型是否存在
if not os.path.exists(model_path):
    logger.error(f"Model not found at {model_path}. Please run download_model.py first.")
    exit(1)

# 加载本地模型
model = SentenceTransformer(model_path, device='cpu', local_files_only=True)
logger.info("Model loaded successfully from local cache")

# 初始化向量数据库
logger.info("Init chroma db")
chroma_db_client = chromadb.PersistentClient('./chroma_db')

# get collection
collection = chroma_db_client.get_collection(name="documents")

# 集合结果数量判断
doc_count = collection.count()
if doc_count == 0:
    logger.warning("No documents in collection, please add documents first")
    exit(1)

logger.info(f"Found {doc_count} documents in collection")
logger.info(f"Total collections: {chroma_db_client.count_collections()}")

# 初始化翻译模型 - 同样使用本地缓存
logger.info("Loading translation models from local cache")
zh2en_model_name = 'Helsinki-NLP/opus-mt-zh-en'
en2zh_model_name = 'Helsinki-NLP/opus-mt-en-zh'

# 检查翻译模型是否存在
zh2en_cache_dir = './model_cache/zh2en_model'
en2zh_cache_dir = './model_cache/en2zh_model'

# 如果翻译模型不存在，使用简单的替代方案
if not (os.path.exists(zh2en_cache_dir) and os.path.exists(en2zh_cache_dir)):
    logger.warning("Translation models not found locally. Using simple translation fallback.")
    
    def translate_zh_to_en(text):
        logger.info("Using fallback translation (no change)")
        return text
        
    def translate_en_to_zh(text):
        logger.info("Using fallback translation (no change)")
        return text
else:
    # 使用本地缓存的翻译模型
    zh2en_tokenizer = MarianTokenizer.from_pretrained(zh2en_cache_dir, local_files_only=True)
    zh2en_model = MarianMTModel.from_pretrained(zh2en_cache_dir, local_files_only=True)

    en2zh_tokenizer = MarianTokenizer.from_pretrained(en2zh_cache_dir, local_files_only=True)
    en2zh_model = MarianMTModel.from_pretrained(en2zh_cache_dir, local_files_only=True)

    def split_text(text, max_length=400):
        """
        将长文本分割成短文本
        """
        if not isinstance(text, str):
            return []
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def translate_en_to_zh(text):
        if not text or not isinstance(text, str):
            return ""
        
        try:
            chunks = split_text(text)
            translated_chunks = []
            
            for chunk in chunks:
                inputs = en2zh_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = en2zh_model.generate(**inputs)
                translated_chunks.append(en2zh_tokenizer.decode(outputs[0], skip_special_tokens=True))
            
            return " ".join(translated_chunks)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def translate_zh_to_en(text):
        if not text or not isinstance(text, str):
            return ""
        
        try:
            chunks = split_text(text)
            translated_chunks = []
            
            for chunk in chunks:
                inputs = zh2en_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = zh2en_model.generate(**inputs)
                translated_chunks.append(zh2en_tokenizer.decode(outputs[0], skip_special_tokens=True))
            
            return " ".join(translated_chunks)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

def remove_duplicates(documents, metadatas):
    """
    去除重复的文档
    """
    seen = set()
    unique_docs = []
    unique_metas = []
    
    for doc, meta in zip(documents, metadatas):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_metas.append(meta)
    
    return unique_docs, unique_metas

def list_all_documents():
    """列出所有文档"""
    try:
        # 获取集合中的所有文档
        results = collection.get()
        
        if not results or 'metadatas' not in results:
            logger.warning("No documents found in collection")
            return
        
        logger.info("=== 所有文档列表 ===")
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            source = metadata.get('source', '未知')
            logger.info(f"{i+1}. {source}")
            # 打印文档内容的前100个字符
            logger.info(f"   内容预览: {doc[:100]}...")
        
        return results
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")

def search_documents(query: str, use_keyword_search: bool = True):
    """改进的文档搜索"""
    logger.info(f"搜索查询: {query}")
    
    # 1. 向量搜索
    query_embedding = model.encode(query)
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=10,  # 增加返回结果数量
        include=['documents', 'metadatas', 'distances']
    )
    
    # 2. 如果启用关键词搜索
    if use_keyword_search:
        # 获取所有文档
        all_docs = collection.get()
        keyword_results = {'documents': [], 'metadatas': [], 'distances': []}
        
        # 简单的关键词匹配
        for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
            source = metadata.get('source', '').lower()
            if '管理办法' in source or query.lower() in source.lower():
                keyword_results['documents'].append(doc)
                keyword_results['metadatas'].append(metadata)
                keyword_results['distances'].append(0)  # 关键词匹配赋予最高优先级
    
        # 合并结果
        all_docs = vector_results['documents'][0] + keyword_results['documents']
        all_metas = vector_results['metadatas'][0] + keyword_results['metadatas']
    else:
        all_docs = vector_results['documents'][0]
        all_metas = vector_results['metadatas'][0]
    
    # 去重
    seen = set()
    unique_docs = []
    unique_metas = []
    
    for doc, meta in zip(all_docs, all_metas):
        source = meta.get('source', '')
        if source not in seen:
            seen.add(source)
            unique_docs.append(doc)
            unique_metas.append(meta)
    
    logger.info(f"找到 {len(unique_docs)} 个相关文档")
    for i, meta in enumerate(unique_metas):
        logger.info(f"{i+1}. {meta.get('source', '未知')}")
    
    return unique_docs, unique_metas

if __name__ == "__main__":
    # 首先列出所有文档
    logger.info("=== 列出所有文档 ===")
    list_all_documents()
    
    # 测试搜索
    logger.info("\n=== 测试搜索 ===")
    query = "管理办法"
    docs, metas = search_documents(query)

    # 搜索内容生成向量
    query = """ 
    根据 甘肃金融控股集团有限公司 采购管理办法，参考如下配置的电脑：
    现货M4pro芯片【14+20】64G内存1TB硬   优券后¥16987|优惠前 ￥16990 。
    对具体采购方案进行分析，提出瑕疵和修改建议
    """

    logger.info(f"Processing query: {query}")

    # 尝试翻译查询
    try:
        query_en = translate_zh_to_en(query)
        logger.info(f"Translated query: {query_en}")
    except Exception as e:
        logger.error(f"Error translating query: {str(e)}")
        query_en = query
        logger.info("Using original query as fallback")

    # 生成中英文查询向量
    query_zh_embedding = model.encode(query)
    query_en_embedding = model.encode(query_en)

    # 分别搜索中英文结果
    result_zh = collection.query(query_zh_embedding, n_results=5)
    result_en = collection.query(query_en_embedding, n_results=5)

    # 合并结果
    all_documents = []
    all_metadatas = []

    for doc, meta in zip(result_zh['documents'][0], result_zh['metadatas']):
        all_documents.append(doc)
        all_metadatas.append(meta[0])

    for doc, meta in zip(result_en['documents'][0], result_en['metadatas']):
        try:
            translated_doc = translate_en_to_zh(doc)
            all_documents.append(translated_doc)
            all_metadatas.append(meta[0])
        except Exception as e:
            logger.error(f"Error translating document: {str(e)}")
            all_documents.append(doc)
            all_metadatas.append(meta[0])

    # 去重
    all_documents, all_metadatas = remove_duplicates(all_documents, all_metadatas)
    logger.info(f"找到 {len(all_documents)} 个相关文档")

    # 构造提示词
    context = "\n\n".join(
        f"文档 {i+1} (来源: {meta.get('source', '未知')}):\n{doc}"
        for i, (doc, meta) in enumerate(zip(all_documents, all_metadatas))
    )

    prompt = f"""你是一个专业的助手。请基于以下参考文档回答问题。
    参考文档：
    {context}
    用户问题：{query}
    请给出详细的答案，使用 markdown 格式，确保回答专业、准确和客观，并标注信息来源。
    """
    logger.info("Sending request to DeepSeek model")

    # 初始化 DeepSeek 客户端
    DEEPSEEK_API_KEY = "sk-copsfwzizttlrfjwmbkrrdkfiihrgxptyquyadckcfswxqfs"
    DEEPSEEK_API_BASE = "https://api.siliconflow.cn/v1"

    client = OpenAI(
        base_url=DEEPSEEK_API_BASE,
        api_key=DEEPSEEK_API_KEY,
    )

    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-V3',
        messages=[
            {"role": "system", "content": "你是一个知识渊博的助手，提供准确且有条理的回答。"},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        max_tokens=512
    )

    try:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end='')
    except Exception as e:
        logger.error(f"Error during response streaming: {str(e)}")