from sentence_transformers import SentenceTransformer
import logging
import chromadb
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

from openai import OpenAI  # 使用 OpenAI 兼容客户端调用 DeepSeek


# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化向量模型，mps是mac下的gpu
logger.info("Loading sentence transformer model")
model = SentenceTransformer('shibing624/text2vec-base-chinese', './model_cache', device='mps')

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

# 初始化翻译模型
logger.info("Loading translation models")
zh2en_model_name = 'Helsinki-NLP/opus-mt-zh-en'
en2zh_model_name = 'Helsinki-NLP/opus-mt-en-zh'

zh2en_tokenizer = MarianTokenizer.from_pretrained(zh2en_model_name, cache_dir='./model_cache')
zh2en_model = MarianMTModel.from_pretrained(zh2en_model_name, cache_dir='./model_cache')

en2zh_tokenizer = MarianTokenizer.from_pretrained(en2zh_model_name, cache_dir='./model_cache')
en2zh_model = MarianMTModel.from_pretrained(en2zh_model_name, cache_dir='./model_cache')

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

# 搜索内容生成向量
query = "佛教中的八万四千法门指的是什么？"
logger.info(f"Processing query: {query}")

query_en = translate_zh_to_en(query)
logger.info(f"Translated query: {query_en}")

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
    translated_doc = translate_en_to_zh(doc)
    all_documents.append(translated_doc)
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

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V3',
    messages=[
        {"role": "system", "content": "你是一个知识渊博的助手，提供准确且有条理的回答。"},
        {"role": "user", "content": prompt}
    ],
    stream=True,
    temperature=0.7,
    max_tokens=2048
)

try:
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='')
except Exception as e:
    logger.error(f"Error during response streaming: {str(e)}")