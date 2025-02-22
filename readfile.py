import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import traceback
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate,Document
from llama_index.readers.file.docs import PDFReader,DocxReader
from llama_index.readers.file.html.base import HTMLTagReader
from llama_index.readers.file.epub import EpubReader
from llama_index.readers.file.unstructured import UnstructuredReader
# 切割
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.node_parser.text.code import CodeSplitter
from llama_index.core import Document
from typing import List
from markdown import markdown
from opencc import OpenCC
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from chromadb.config import Settings
import re
import jieba
from itertools import chain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import jieba.analyse

class FileProcessor:
    def __init__(self, log_dir: str = "logs"):
        # 设置日志
        self._setup_logging(log_dir)
        
        # 支持的文件类型
        self.supported_formats = {
            '.pdf': self._read_pdf,
            '.doc': self._read_doc,
            '.docx': self._read_docx,
            '.txt': self._read_txt,
            '.html': self._read_html,
            '.htm': self._read_html,
            '.epub': self._read_epub,
            '.csv': self._read_csv,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
            '.md': self._read_md,
        }
        
        # 初始化计数器
        self.stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 初始化各种reader
        self._init_readers()
        
    def _setup_logging(self, log_dir: str):
        """设置日志"""
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件名（包含时间戳）
        log_file = os.path.join(
            log_dir, 
            f'file_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _init_readers(self):
        """初始化各种文档读取器"""
        try:
            self.pdf_reader = PDFReader()
            self.docx_reader = DocxReader()
            self.html_reader = HTMLTagReader()
            self.epub_reader = EpubReader()
            self.unstructured_reader = UnstructuredReader()
            
            self.logger.info("成功初始化所有文档读取器")
            
        except Exception as e:
            self.logger.error(f"初始化读取器时出错: {str(e)}")
            raise

    def _convert_to_simplified(self, text: str) -> str:
        """将繁体中文转换为简体中文"""
        cc = OpenCC('t2s')  # 繁体到简体
        return cc.convert(text)
            
    def _read_pdf(self, file_path: str) -> List[Document]:
        """读取PDF文件"""
        try:
            self.logger.info(f"开始读取PDF文件: {file_path}")
            documents = self.pdf_reader.load_data(file_path)
            self.logger.info(f"成功读取PDF文件，共 {len(documents)} 页")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取PDF文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_doc(self, file_path: str) -> List[Document]:
        """读取DOC文件"""
        try:
            self.logger.info(f"开始读取DOC文件: {file_path}")
            documents = self.unstructured_reader.load_data(file_path)
            self.logger.info(f"成功读取DOC文件")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取DOC文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_docx(self, file_path: str) -> List[Document]:
        """读取DOCX文件"""
        try:
            self.logger.info(f"开始读取DOCX文件: {file_path}")
            documents = self.docx_reader.load_data(file_path)
            self.logger.info(f"成功读取DOCX文件")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取DOCX文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_txt(self, file_path: str) -> List[Document]:
        """读取TXT文件"""
        try:
            self.logger.info(f"开始读取TXT文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # 转换为简体中文    
            text = self._convert_to_simplified(text)
            document = Document(text=text, metadata={'file_path': file_path})
            self.logger.info(f"成功读取TXT文件")
            return [document]
            
        except UnicodeDecodeError:
            self.logger.warning(f"UTF-8解码失败，尝试其他编码: {file_path}")
            # 尝试其他编码
            encodings = ['gbk', 'gb2312', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    # 转换为简体中文    
                    text = self._convert_to_simplified(text)
                    document = Document(text=text, metadata={'file_path': file_path})
                    self.logger.info(f"使用 {encoding} 编码成功读取TXT文件")
                    return [document]
                except UnicodeDecodeError:
                    continue
                    
            self.logger.error(f"所有编码都无法正确读取TXT文件: {file_path}")
            raise
            
        except Exception as e:
            self.logger.error(f"读取TXT文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_html(self, file_path: str) -> List[Document]:
        """读取HTML文件"""
        try:
            self.logger.info(f"开始读取HTML文件: {file_path}")
            documents = self.html_reader.load_data(file_path)
            self.logger.info(f"成功读取HTML文件")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取HTML文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_epub(self, file_path: str) -> List[Document]:
        """读取EPUB文件"""
        try:
            self.logger.info(f"开始读取EPUB文件: {file_path}")
            documents = self.epub_reader.load_data(file_path)
            self.logger.info(f"成功读取EPUB文件")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取EPUB文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_csv(self, file_path: str) -> List[Document]:
        """读取CSV文件"""
        try:
            self.logger.info(f"开始读取CSV文件: {file_path}")
            
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError(f"无法使用任何编码读取CSV文件: {file_path}")
                
            # 转换为文本格式
            text = df.to_string(index=False)
            document = Document(
                text=text,
                metadata={
                    'file_path': file_path,
                    'file_type': 'csv',
                    'columns': list(df.columns)
                }
            )
            
            self.logger.info(f"成功读取CSV文件，共 {len(df)} 行")
            return [document]
            
        except Exception as e:
            self.logger.error(f"读取CSV文件 {file_path} 时出错: {str(e)}")
            raise
            
    def _read_excel(self, file_path: str) -> List[Document]:
        """读取Excel文件"""
        try:
            self.logger.info(f"开始读取Excel文件: {file_path}")
            
            # 读取所有sheet
            df_dict = pd.read_excel(file_path, sheet_name=None)
            documents = []
            
            for sheet_name, df in df_dict.items():
                text = df.to_string(index=False)
                document = Document(
                    text=text,
                    metadata={
                        'file_path': file_path,
                        'file_type': 'excel',
                        'sheet_name': sheet_name,
                        'columns': list(df.columns)
                    }
                )
                documents.append(document)
                
            self.logger.info(f"成功读取Excel文件，共 {len(df_dict)} 个工作表")
            return documents
            
        except Exception as e:
            self.logger.error(f"读取Excel文件 {file_path} 时出错: {str(e)}")
            raise

    def _read_md(self, file_path: str) -> List[Document]:
        """读取Markdown文件"""
        try:
            self.logger.info(f"开始读取Markdown文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # 将Markdown转换为HTML
            html = markdown(text)
            document = Document(text=html, metadata={'file_path': file_path})
            self.logger.info(f"成功读取Markdown文件")
            return [document]
            
        except UnicodeDecodeError:
            self.logger.warning(f"UTF-8解码失败，尝试其他编码: {file_path}")
            # 尝试其他编码
            encodings = ['gbk', 'gb2312', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    html = markdown(text)
                    document = Document(text=html, metadata={'file_path': file_path})
                except UnicodeDecodeError:
                    continue
                    
            self.logger.error(f"所有编码都无法正确读取Markdown文件: {file_path}")
            raise
            
        except Exception as e:
            self.logger.error(f"读取Markdown文件 {file_path} 时出错: {str(e)}")
            raise
            
    def process_file(self, file_path: str) -> Optional[List[Document]]:
        """处理单个文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.logger.error(f"文件不存在: {file_path}")
                self.stats['failed'] += 1
                return None
                
            # 获取文件扩展名
            ext = Path(file_path).suffix.lower()
            
            # 检查文件类型是否支持
            if ext not in self.supported_formats:
                self.logger.warning(f"不支持的文件类型: {ext}")
                self.stats['skipped'] += 1
                return None
                
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                self.logger.warning(f"文件过大 ({file_size/1024/1024:.2f}MB): {file_path}")
                
            # 读取文件
            documents = self.supported_formats[ext](file_path)
            self.stats['success'] += 1
            return documents
            
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self.stats['failed'] += 1
            return None
            
    def process_directory(self, directory_path: str) -> Dict[str, List[Document]]:
        """处理整个目录"""
        self.logger.info(f"开始处理目录: {directory_path}")
        
        # 重置计数器
        self.stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        results = {}
        total_files = sum(1 for f in Path(directory_path).rglob('*') 
                         if f.suffix.lower() in self.supported_formats)
        
        # 使用tqdm显示进度
        with tqdm(total=total_files, desc="处理文件") as pbar:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        documents = self.process_file(file_path)
                        if documents:
                            results[file_path] = documents
                            
                    except Exception as e:
                        self.logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
                        continue
                        
                    finally:
                        pbar.update(1)
                        
        # 打印统计信息
        self.logger.info(f"处理完成。成功: {self.stats['success']}, "
                        f"失败: {self.stats['failed']}, "
                        f"跳过: {self.stats['skipped']}")
                        
        return results

    def print_documents(documents: Dict[str, List[Document]]):
        """打印文档内容"""
        for file_path, docs in documents.items():
            print("\n" + "="*80)
            print(f"文件路径: {file_path}")
            print("="*80)
            
            for i, doc in enumerate(docs, 1):
                print(f"\n文档 {i}/{len(docs)}:")
                print("-"*40)
                
                # 打印元数据
                if doc.metadata:
                    print("元数据:")
                    for key, value in doc.metadata.items():
                        print(f"  {key}: {value}")
                    print("-"*40)
                    
                # 打印内容预览
                content = doc.text
                preview_length = 500
                if len(content) > preview_length:
                    # print(f"内容预览 (前 {preview_length} 字符):")
                    print(content[:preview_length] + "...")
                else:
                    print("完整内容:")
                    print(content)
                    
                print("\n")


class TextPreprocessor:
    """文本预处理类"""
    def __init__(self):
        self.cc = OpenCC('t2s')  # 繁体转简体
        # 标点符号和特殊字符的正则表达式
        self.punctuation_pattern = re.compile(
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s]'  # 保留中文、英文、数字和空白字符
        )
        # 多余空白字符的正则表达式
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 初始化 jieba 分析器
        jieba.initialize()
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 繁体转简体
        text = self.cc.convert(text)
        
        # 去除特殊字符，但保留基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\s]', '', text)
        
        # 统一空白字符
        text = self.whitespace_pattern.sub(' ', text)
        
        # 去除行首行尾空白
        text = text.strip()
        
        return text
        
    def split_paragraphs(self, text: str) -> List[str]:
        """智能分段"""
        # 基于换行和标点符号分段
        paragraphs = re.split(r'[\n\r]+|(?<=[。！？])', text)
        # 过滤空段落并清理
        return [p.strip() for p in paragraphs if p.strip()]
        
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        try:
            # 使用 jieba TF-IDF 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10)
            return keywords
        except Exception as e:
            print(f"Warning: Failed to extract keywords: {str(e)}")
            # 如果关键词提取失败，返回空列表
            return []

class SmartChunker:
    """智能分块策略"""
    def __init__(self, 
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.text_processor = TextPreprocessor()
        
    def create_chunks(self, text: str) -> List[Tuple[str, Dict]]:
        """创建智能文本块"""
        # 预处理文本
        clean_text = self.text_processor.clean_text(text)
        paragraphs = self.text_processor.split_paragraphs(clean_text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # 如果当前段落本身就超过了chunk_size
            if para_length > self.chunk_size:
                # 如果当前chunk不为空，先保存它
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk_with_metadata(chunk_text))
                    current_chunk = []
                    current_length = 0
                
                # 处理大段落
                sentences = re.split(r'(?<=[。！？])', para)
                temp_chunk = []
                temp_length = 0
                
                for sent in sentences:
                    if temp_length + len(sent) > self.chunk_size:
                        if temp_chunk:
                            chunk_text = "".join(temp_chunk)
                            chunks.append(self._create_chunk_with_metadata(chunk_text))
                        temp_chunk = [sent]
                        temp_length = len(sent)
                    else:
                        temp_chunk.append(sent)
                        temp_length += len(sent)
                
                if temp_chunk:
                    chunk_text = "".join(temp_chunk)
                    chunks.append(self._create_chunk_with_metadata(chunk_text))
                
            # 正常段落处理
            elif current_length + para_length > self.chunk_size:
                # 保存当前chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk_with_metadata(chunk_text))
                
                # 开始新的chunk，并包含重叠内容
                if len(current_chunk) > 1:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, para]
                    current_length = len(overlap_text) + para_length
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # 处理最后一个chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk_with_metadata(chunk_text))
        
        return chunks
    
    def _create_chunk_with_metadata(self, text: str) -> Tuple[str, Dict]:
        """为文本块创建增强的元数据"""
        # 提取关键词
        keywords = self.text_processor.extract_keywords(text)
        
        # 计算文本统计信息
        char_count = len(text)
        word_count = len(jieba.lcut(text))
        
        # 创建元数据
        metadata = {
            'keywords': keywords,
            'char_count': char_count,
            'word_count': word_count,
            'language': self._detect_language(text),
            'text_type': self._detect_text_type(text),
            'timestamp': datetime.now().isoformat()
        }
        
        return text, metadata
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / len(text)
        english_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        
        if chinese_ratio > 0.5:
            return 'chinese'
        elif english_ratio > 0.5:
            return 'english'
        return 'mixed'
    
    def _detect_text_type(self, text: str) -> str:
        """检测文本类型"""
        code_patterns = [r'def\s+', r'class\s+', r'import\s+', r'function\s+']
        if any(re.search(pattern, text) for pattern in code_patterns):
            return 'code'
            
        if len(re.findall(r'[\d.]+', text)) / len(text) > 0.3:
            return 'numeric'
            
        return 'general'

class SemanticProcessor:
    """语义处理器"""
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self.model = SentenceTransformer(model_name)
        
    def compute_semantic_similarity(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """计算文本块之间的语义相似度并优化"""
        # 检查是否有文本块
        if not chunks:
            print("Warning: No chunks to process")
            return []
            
        texts = [chunk[0] for chunk in chunks]
        
        try:
            # 编码文本
            embeddings = self.model.encode(texts)
            
            # 确保embeddings是2D数组
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # 计算相似度矩阵
            similarities = cosine_similarity(embeddings)
            
            # 为每个chunk添加语义相关性信息
            enhanced_chunks = []
            for i, (text, metadata) in enumerate(chunks):
                # 找出最相关的其他块
                # 如果只有一个块，就跳过相关性计算
                if len(chunks) > 1:
                    related_indices = np.argsort(similarities[i])[-4:-1]  # 取前3个最相关的（除去自身）
                    related_chunks = [
                        {
                            'index': idx,
                            'similarity': float(similarities[i][idx]),
                            'preview': texts[idx][:100] + '...'  # 预览文本
                        }
                        for idx in related_indices
                    ]
                else:
                    related_chunks = []
                
                # 更新元数据
                metadata['semantic_relations'] = related_chunks
                metadata['semantic_score'] = float(np.mean(similarities[i])) if len(chunks) > 1 else 1.0
                
                enhanced_chunks.append((text, metadata))
                
            return enhanced_chunks
            
        except Exception as e:
            print(f"Warning: Error in semantic processing: {str(e)}")
            # 如果处理失败，返回原始chunks
            return chunks

class ContextManager:
    """上下文管理器"""
    def __init__(self, max_context_chunks: int = 3):
        self.max_context_chunks = max_context_chunks
        
    def add_context(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """为每个文本块添加上下文信息"""
        enhanced_chunks = []
        
        for i, (text, metadata) in enumerate(chunks):
            # 获取前后文
            start_idx = max(0, i - self.max_context_chunks)
            end_idx = min(len(chunks), i + self.max_context_chunks + 1)
            
            # 提取上下文
            context = {
                'previous': [
                    {
                        'text': chunks[j][0][:200] + '...',
                        'distance': i - j
                    }
                    for j in range(start_idx, i)
                ],
                'next': [
                    {
                        'text': chunks[j][0][:200] + '...',
                        'distance': j - i
                    }
                    for j in range(i + 1, end_idx)
                ]
            }
            
            # 更新元数据
            metadata['context'] = context
            enhanced_chunks.append((text, metadata))
            
        return enhanced_chunks

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunker = SmartChunker(chunk_size, chunk_overlap)
        self.semantic_processor = SemanticProcessor()
        self.context_manager = ContextManager()
        
    def process_documents(self, documents: List[Document]) -> List[Tuple[str, Dict]]:
        """处理文档列表"""
        all_chunks = []
        
        for doc in documents:
            # 1. 智能分块
            chunks = self.chunker.create_chunks(doc.text)
            
            # 2. 添加语义相关性
            chunks = self.semantic_processor.compute_semantic_similarity(chunks)
            
            # 3. 添加上下文信息
            chunks = self.context_manager.add_context(chunks)
            
            # 4. 添加文档源信息
            for text, metadata in chunks:
                metadata['source_document'] = doc.metadata
                all_chunks.append((text, metadata))
        
        return all_chunks

class TextEmbedding:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """初始化文本向量化模型
        
        Args:
            model_name: 模型名称,默认使用text2vec-base-chinese(约400MB)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量"""
        return self.model.encode(texts).tolist()

class DocumentStore:
    def __init__(self, collection_name: str = "documents"):
        """初始化文档存储"""
        import os
        import logging
        
        # 获取当前工作目录和完整的存储路径
        current_dir = os.getcwd()
        self.db_path = os.path.abspath("./chroma_db")
        
        # 打印详细的路径信息
        print(f"Current working directory: {current_dir}")
        print(f"ChromaDB will be stored at: {self.db_path}")
        
        try:
            # 确保目录存在
            os.makedirs(self.db_path, exist_ok=True)
            print(f"Created/Verified ChromaDB directory at: {self.db_path}")
            
            # 使用 PersistentClient
            self.client = chromadb.PersistentClient(
                path=self.db_path
            )
            
            # 创建或获取集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Document chunks collection"}
            )
            print("Successfully initialized ChromaDB client and collection")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            raise
            
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: List[List[float]], 
                     metadatas: List[dict] = None,
                     ids: List[str] = None):
        """添加文档到存储"""
        try:
            if ids is None:
                ids = [str(i) for i in range(len(texts))]
                
            print(f"Adding {len(texts)} documents to ChromaDB...")
            
            # 检查输入数据
            print(f"Texts count: {len(texts)}")
            print(f"Embeddings count: {len(embeddings)}")
            print(f"Metadata count: {len(metadatas) if metadatas else 0}")
            print(f"IDs count: {len(ids)}")
            
            # 添加文档
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas if metadatas else [{}] * len(texts),
                ids=ids
            )
            print(f"Successfully added documents to ChromaDB")
            
            # 验证数据是否已保存
            count = self.collection.count()
            print(f"Total documents in collection: {count}")
            
            # 检查存储目录内容
            if os.path.exists(self.db_path):
                print(f"ChromaDB directory contents:")
                for root, dirs, files in os.walk(self.db_path):
                    level = root.replace(self.db_path, '').count(os.sep)
                    indent = ' ' * 4 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f"{subindent}{f}")
            
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            raise

def main():
    """主函数：完整的文档处理流程"""
    # 初始化组件
    file_processor = FileProcessor()
    doc_processor = DocumentProcessor(
        chunk_size=1024,
        chunk_overlap=200
    )
    embedding_model = TextEmbedding()
    doc_store = DocumentStore()
    
    directory_path = "./documents"
    
    # 确保目录存在且包含文件
    if not os.path.exists(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return
        
    files = os.listdir(directory_path)
    if not files:
        logging.error(f"No files found in directory: {directory_path}")
        return
        
    logging.info(f"Found {len(files)} files in {directory_path}")
    
    try:
        # 1. 读取并处理文件
        logging.info("=== 第1阶段: 读取并处理文件 ===")
        documents = file_processor.process_directory(directory_path)
        logging.info(f"成功处理文件数: {len(documents)}")
        
        # 2. 合并文档
        logging.info("=== 第2阶段: 合并文档 ===")
        all_docs = []
        for file_path, docs in documents.items():
            logging.info(f"处理文件 {file_path} 中的 {len(docs)} 个文档")
            for doc in docs:
                # 确保每个文档都有file_path元数据
                if 'file_path' not in doc.metadata:
                    doc.metadata['file_path'] = file_path
            all_docs.extend(docs)
        logging.info(f"合并后总文档数: {len(all_docs)}")
        
        # 3. 文档切割
        logging.info("=== 第3阶段: 文档切割 ===")
        chunks = doc_processor.process_documents(all_docs)
        logging.info(f"生成文本块数: {len(chunks)}")
        
        # 4. 生成向量
        logging.info("=== 第4阶段: 生成向量 ===")
        embeddings = embedding_model.encode([chunk[0] for chunk in chunks])
        logging.info(f"生成向量数: {len(embeddings)}")
        
        # 5. 准备元数据
        logging.info("=== 第5阶段: 准备元数据 ===")
        metadatas = []
        chunk_per_doc = len(chunks) // len(all_docs)
        for i, doc in enumerate(all_docs):
            file_path = doc.metadata.get('file_path', 'unknown')
            # 为每个chunk创建对应的metadata
            for _ in range(chunk_per_doc):
                metadatas.append({
                    "source": file_path,
                    "doc_index": i,
                    "total_docs": len(all_docs)
                })
        
        # 如果chunks数量与metadata数量不匹配，进行调整
        if len(metadatas) < len(chunks):
            diff = len(chunks) - len(metadatas)
            last_metadata = metadatas[-1]
            metadatas.extend([last_metadata.copy() for _ in range(diff)])
        elif len(metadatas) > len(chunks):
            metadatas = metadatas[:len(chunks)]
            
        logging.info(f"准备元数据数: {len(metadatas)}")
        
        # 6. 存储到ChromaDB
        logging.info("=== 第6阶段: 存储到ChromaDB ===")
        doc_store.add_documents(
            texts=[chunk[0] for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        
        logging.info(f"\n处理完成! 成功存储了 {len(chunks)} 个文本块")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()