import os
import torch
from sentence_transformers import SentenceTransformer
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name="shibing624/text2vec-base-chinese", cache_dir="./model_cache"):
    """
    下载模型到本地缓存
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
    """
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"开始下载模型 {model_name} 到 {cache_dir}")
    
    try:
        # 下载模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )
        
        # 保存模型到本地
        save_path = os.path.join(cache_dir, "text2vec_model")
        logger.info(f"保存模型到 {save_path}")
        model.save(save_path)
        
        logger.info("模型下载并保存成功!")
        
        # 验证模型是否可以加载
        logger.info("验证模型...")
        loaded_model = SentenceTransformer(save_path)
        test_text = "这是一个测试句子"
        embedding = loaded_model.encode(test_text)
        logger.info(f"测试成功! 生成的向量维度: {embedding.shape}")
        
        return True
    except Exception as e:
        logger.error(f"下载模型时出错: {str(e)}")
        return False

if __name__ == "__main__":
    download_model() 