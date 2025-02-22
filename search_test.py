from sentence_transformers import SentenceTransformer
import logging
import chromadb
from openai import OpenAI  # 使用 OpenAI 兼容客户端调用 DeepSeek


# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 初始化向量模型，mps是mac下的gpu
logging.info("Loading sentence transformer model")
model = SentenceTransformer('shibing624/text2vec-base-chinese', './model_cache', device='mps')

# 初始化向量数据库
logging.info("Init chroma db")
chroma_db_client=chromadb.PersistentClient('./chroma_db')

# get collection
collection=chroma_db_client.get_collection(name="documents")

# 集合结果数量判断
doc_count=collection.count()
# 集合结果数量判断
if doc_count==0:
    logging.warning("No documents in collection, please add documents first")
    exit(1)

logging.info("Init chroma db done")

print(chroma_db_client.count_collections())

# 搜索内容生成向量
query="介绍一下观世音菩萨"
query_embedding = model.encode(query)

# 搜索结果
result=collection.query(query_embedding, n_results=10)
logging.info("匹配文档数量：%d",len(result['documents'][0]))


# 检查匹配结果
if not result['documents']:
    logging.warning("No documents found")
    exit(1)

# 构造提示词
context = "\n\n".join(
    f"文档 {i+1} (来源: {meta[0].get('source', '未知')}):\n{doc[0]}"
    for i, (doc, meta) in enumerate(zip(result['documents'], result['metadatas']))
)

prompt = f"""你是一个专业的助手。请基于以下参考文档回答问题。
参考文档：
{context}
用户问题：{query}
请给出详细的答案，使用 markdown 格式，确保回答专业、准确和客观，并标注信息来源。
"""
logging.info("Sending request to DeepSeek model")


# 初始化 DeepSeek 客户端（假设使用 SiliconFlow 的 DeepSeek API）
DEEPSEEK_API_KEY = "sk-copsfwzizttlrfjwmbkrrdkfiihrgxptyquyadckcfswxqfs"  # 替换为你的 DeepSeek API 密钥
DEEPSEEK_API_BASE = "https://api.siliconflow.cn/v1"  # 假设使用 SiliconFlow 的 DeepSeek 端点

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V2.5',
    messages=[
                {"role": "system", "content": "你是一个知识渊博的助手，提供准确且有条理的回答。"},
                {"role": "user", "content": prompt}
            ],
    stream=True,
    temperature=0.7,
    max_tokens=2048
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='')