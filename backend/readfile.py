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
from torch import nn
import time
import json
import sqlite3

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
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.logger.error(f"DOC文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            self.logger.info(f"DOC文件大小: {file_size} 字节")
            
            # 检查文件权限
            self.logger.info(f"检查文件权限...")
            try:
                with open(file_path, 'rb') as f:
                    # 只读取前几个字节，检查文件是否可读
                    f.read(10)
                self.logger.info(f"文件权限正常，可以读取")
            except Exception as e:
                self.logger.error(f"文件权限检查失败: {str(e)}")
                raise
            
            # 尝试使用olefile直接读取DOC文件
            try:
                import olefile
                self.logger.info("尝试使用olefile读取DOC文件...")
                
                if olefile.isOleFile(file_path):
                    self.logger.info("文件是有效的OLE文件")
                    
                    try:
                        ole = olefile.OleFileIO(file_path)
                        self.logger.info(f"OLE文件已打开，可用流: {ole.listdir()}")
                        
                        # 尝试读取WordDocument流
                        if ole.exists('WordDocument'):
                            self.logger.info("找到WordDocument流")
                            
                            # 尝试提取文本
                            text_parts = []
                            
                            # 尝试读取主要文本流
                            try:
                                word_stream = ole.openstream('WordDocument')
                                word_data = word_stream.read()
                                self.logger.info(f"读取到WordDocument流，大小: {len(word_data)} 字节")
                                
                                # 尝试从二进制数据中提取ASCII文本
                                text = ''.join(chr(b) for b in word_data if 32 <= b <= 126 or b in [10, 13])
                                if len(text) > 100:  # 至少有100个字符
                                    text_parts.append(text)
                                    self.logger.info(f"从WordDocument流提取了 {len(text)} 个字符")
                            except Exception as e:
                                self.logger.warning(f"读取WordDocument流时出错: {str(e)}")
                            
                            # 尝试读取其他可能包含文本的流
                            for stream_name in ole.listdir():
                                if 'Text' in str(stream_name) or 'Contents' in str(stream_name):
                                    try:
                                        stream = ole.openstream(stream_name)
                                        data = stream.read()
                                        text = ''.join(chr(b) for b in data if 32 <= b <= 126 or b in [10, 13])
                                        if len(text) > 50:  # 至少有50个字符
                                            text_parts.append(text)
                                            self.logger.info(f"从流 {stream_name} 提取了 {len(text)} 个字符")
                                    except Exception as e:
                                        self.logger.warning(f"读取流 {stream_name} 时出错: {str(e)}")
                            
                            # 如果成功提取了文本
                            if text_parts:
                                combined_text = '\n\n'.join(text_parts)
                                self.logger.info(f"成功从DOC文件中提取了 {len(combined_text)} 个字符")
                                document = Document(text=combined_text, metadata={
                                    'file_path': file_path,
                                    'extraction_method': 'olefile'
                                })
                                ole.close()  # 确保关闭文件
                                return [document]
                        else:
                            self.logger.warning("未找到WordDocument流")
                        
                        ole.close()  # 确保关闭文件
                    except Exception as ole_error:
                        self.logger.warning(f"使用olefile处理时出错: {str(ole_error)}")
                else:
                    self.logger.warning("文件不是有效的OLE文件")
            except ImportError:
                self.logger.warning("olefile库未安装，跳过此方法")
            
            # 尝试使用python-docx2txt读取
            try:
                import docx2txt
                self.logger.info(f"使用docx2txt尝试读取...")
                try:
                    # 尝试将DOC文件转换为DOCX
                    self.logger.info("尝试将DOC转换为DOCX后再处理...")
                    
                    # 创建临时DOCX文件
                    import tempfile
                    import shutil
                    
                    temp_docx = tempfile.NamedTemporaryFile(suffix='.docx', delete=False).name
                    self.logger.info(f"创建临时DOCX文件: {temp_docx}")
                    
                    # 尝试复制并重命名为DOCX
                    shutil.copy2(file_path, temp_docx)
                    self.logger.info(f"已复制文件到临时DOCX")
                    
                    # 尝试使用docx2txt处理
                    try:
                        text = docx2txt.process(temp_docx)
                        self.logger.info(f"使用docx2txt成功读取临时DOCX，文本长度: {len(text)}")
                        document = Document(text=text, metadata={'file_path': file_path})
                        
                        # 删除临时文件
                        try:
                            os.unlink(temp_docx)
                        except Exception as del_error:
                            self.logger.warning(f"删除临时文件失败: {str(del_error)}")
                        
                        return [document]
                    except Exception as temp_error:
                        self.logger.warning(f"处理临时DOCX失败: {str(temp_error)}")
                        # 删除临时文件
                        try:
                            os.unlink(temp_docx)
                        except Exception as del_error:
                            self.logger.warning(f"删除临时文件失败: {str(del_error)}")
                    
                    # 直接尝试处理原始DOC文件
                    try:
                        text = docx2txt.process(file_path)
                        self.logger.info(f"使用docx2txt成功读取原始DOC，文本长度: {len(text)}")
                        document = Document(text=text, metadata={'file_path': file_path})
                        return [document]
                    except Exception as docx2txt_error:
                        self.logger.warning(f"docx2txt读取失败: {str(docx2txt_error)}")
                        # 继续尝试其他方法
                except Exception as docx2txt_error:
                    self.logger.warning(f"docx2txt读取失败: {str(docx2txt_error)}")
                    # 继续尝试其他方法
            except ImportError:
                self.logger.warning("docx2txt库未安装，跳过此方法")
            
            # 尝试使用textract
            try:
                import textract
                self.logger.info(f"使用textract尝试读取...")
                try:
                    text = textract.process(file_path).decode('utf-8', errors='ignore')
                    self.logger.info(f"使用textract成功读取，文本长度: {len(text)}")
                    document = Document(text=text, metadata={'file_path': file_path})
                    return [document]
                except Exception as textract_error:
                    self.logger.warning(f"textract读取失败: {str(textract_error)}")
                    # 继续尝试其他方法
            except ImportError:
                self.logger.warning("textract库未安装，跳过此方法")
            
            # 尝试使用antiword (如果安装了)
            try:
                import subprocess
                self.logger.info(f"尝试使用antiword读取...")
                try:
                    # 修复编码问题 - 使用二进制模式并手动处理编码
                    result = subprocess.run(
                        ['antiword', file_path], 
                        capture_output=True, 
                        text=False,  # 使用二进制模式
                        check=False  # 不要在失败时抛出异常
                    )
                    
                    # 检查命令是否成功执行
                    if result.returncode == 0 and result.stdout:
                        # 尝试多种编码解码输出
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                            try:
                                text = result.stdout.decode(encoding, errors='ignore')
                                if len(text.strip()) > 0:
                                    self.logger.info(f"使用antiword成功读取，文本长度: {len(text)}，编码: {encoding}")
                                    document = Document(text=text, metadata={
                                        'file_path': file_path,
                                        'extraction_method': 'antiword',
                                        'encoding': encoding
                                    })
                                    return [document]
                            except Exception as decode_error:
                                self.logger.warning(f"使用{encoding}解码antiword输出失败: {str(decode_error)}")
                    else:
                        self.logger.warning(f"antiword命令执行失败，返回码: {result.returncode}")
                        if result.stderr:
                            stderr_text = result.stderr.decode('utf-8', errors='ignore')
                            self.logger.warning(f"antiword错误输出: {stderr_text}")
                except Exception as antiword_error:
                    self.logger.warning(f"antiword读取失败: {str(antiword_error)}")
                    # 继续尝试其他方法
            except (ImportError, FileNotFoundError):
                self.logger.warning("antiword未安装或不可用，跳过此方法")
            
            # 最后尝试使用unstructured_reader，使用线程超时而不是信号
            self.logger.info(f"开始使用UnstructuredReader读取DOC文件...")
            self.logger.info(f"UnstructuredReader配置: {vars(self.unstructured_reader)}")
            
            # 使用线程和事件来实现超时
            import threading
            import time
            
            result = []
            exception = []
            
            def read_doc_with_timeout():
                try:
                    docs = self.unstructured_reader.load_data(file_path)
                    result.extend(docs)
                except Exception as e:
                    exception.append(e)
                    self.logger.error(f"UnstructuredReader读取失败: {str(e)}")
            
            # 创建线程
            thread = threading.Thread(target=read_doc_with_timeout)
            thread.daemon = True  # 设置为守护线程，这样主程序退出时它也会退出
            
            # 开始线程
            self.logger.info("启动UnstructuredReader读取线程...")
            thread.start()
            
            # 等待线程完成，最多等待60秒
            timeout = 60  # 60秒超时
            start_time = time.time()
            
            while thread.is_alive() and (time.time() - start_time) < timeout:
                time.sleep(0.5)  # 每0.5秒检查一次
                
            # 检查是否超时
            if thread.is_alive():
                self.logger.error(f"读取DOC文件超时 ({timeout}秒)")
                
                # 尝试使用二进制方法读取
                self.logger.info("尝试使用二进制方法读取文件内容...")
                try:
                    with open(file_path, 'rb') as f:
                        binary_content = f.read()
                    # 尝试提取可读文本
                    text_content = ''.join(chr(b) for b in binary_content if 32 <= b <= 126 or b in [10, 13])
                    self.logger.info(f"提取到的文本长度: {len(text_content)}")
                    document = Document(text=text_content, metadata={'file_path': file_path, 'extraction_method': 'binary'})
                    return [document]
                except Exception as binary_error:
                    self.logger.error(f"二进制读取失败: {str(binary_error)}")
                    raise Exception(f"读取DOC文件超时，且二进制读取失败: {str(binary_error)}")
            
            # 检查是否有异常
            if exception:
                raise exception[0]
            
            # 检查是否有结果
            if result:
                self.logger.info(f"成功读取DOC文件，文档数: {len(result)}")
                return result
            else:
                # 如果没有结果也没有异常，尝试二进制读取
                self.logger.warning("UnstructuredReader没有返回任何文档，尝试二进制读取...")
                with open(file_path, 'rb') as f:
                    binary_content = f.read()
                # 尝试提取可读文本
                text_content = ''.join(chr(b) for b in binary_content if 32 <= b <= 126 or b in [10, 13])
                self.logger.info(f"提取到的文本长度: {len(text_content)}")
                document = Document(text=text_content, metadata={'file_path': file_path, 'extraction_method': 'binary'})
                return [document]
            
        except Exception as e:
            self.logger.error(f"读取DOC文件 {file_path} 时出错: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 尝试使用最简单的方法提取一些文本，避免完全失败
            try:
                self.logger.info("尝试最后的应急方法提取文本...")
                with open(file_path, 'rb') as f:
                    content = f.read()
                # 尝试不同的编码
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                    try:
                        text = content.decode(encoding, errors='ignore')
                        # 只保留可打印字符
                        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\r', '\t'])
                        if len(text.strip()) > 100:  # 至少有100个有意义的字符
                            self.logger.info(f"使用{encoding}编码成功提取了{len(text)}个字符")
                            document = Document(text=text, metadata={
                                'file_path': file_path, 
                                'extraction_method': 'emergency',
                                'encoding': encoding
                            })
                            return [document]
                    except Exception:
                        continue
                
                # 如果所有尝试都失败，返回一个带有错误信息的文档
                error_text = f"无法读取文件内容。错误: {str(e)}"
                document = Document(text=error_text, metadata={
                    'file_path': file_path,
                    'extraction_failed': True,
                    'error': str(e)
                })
                return [document]
            
            except Exception as final_error:
                self.logger.error(f"所有读取方法都失败: {str(final_error)}")
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
            
    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件并返回文档列表"""
        try:
            self.logger.info(f"开始处理文件: {file_path}")
            
            # 获取文件扩展名
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 检查文件类型是否支持
            if file_ext not in self.supported_formats:
                self.logger.warning(f"不支持的文件类型: {file_ext}")
                self.stats['skipped'] += 1
                return []
            
            # 调用对应的读取方法
            read_method = self.supported_formats[file_ext]
            documents = read_method(file_path)
            
            # 更新统计信息
            self.stats['success'] += 1
            
            self.logger.info(f"成功处理文件: {file_path}, 生成 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.stats['failed'] += 1
            return []
            
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

    def chunk_documents(self, documents: List[Document]) -> List[Tuple[str, Dict]]:
        """
        将文档分割成较小的块
        
        Args:
            documents: 文档列表，每个文档是一个Document对象
            
        Returns:
            分块后的文档列表，每个元素是(文本块, 元数据)的元组
        """
        self.logger.info(f"开始分块文档，文档数量: {len(documents)}")
        chunks = []
        
        # 设置块大小和重叠大小
        chunk_size = 1000  # 每块大约1000个字符
        chunk_overlap = 200  # 块之间重叠200个字符
        
        for doc in documents:
            text = doc.text
            metadata = doc.metadata
            source = metadata.get('file_path', '未知来源')
            
            # 如果文本为空，跳过
            if not text:
                continue
            
            # 分块文本
            doc_chunks = self._split_text(text, chunk_size, chunk_overlap)
            
            # 为每个块创建元数据 - 扁平化元数据结构
            for i, chunk in enumerate(doc_chunks):
                # 创建扁平化的元数据字典，只包含基本类型
                flat_metadata = {
                    "source": source,
                    "chunk_index": i,
                    "chunk_count": len(doc_chunks)
                }
                
                # 添加原始元数据中的基本类型值
                if metadata:
                    for key, value in metadata.items():
                        # 只添加基本类型的值
                        if isinstance(value, (str, int, float, bool)):
                            flat_metadata[key] = value
                
                chunks.append((chunk, flat_metadata))
        
        self.logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        将文本分割成重叠的块
        
        Args:
            text: 要分割的文本
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小（字符数）
            
        Returns:
            文本块列表
        """
        # 如果文本长度小于块大小，直接返回
        if len(text) <= chunk_size:
            return [text]

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
        
        # 定义允许的字符模式
        allowed_chars = (
            r'['
            r'\u4e00-\u9fa5'  # 中文字符范围
            r'a-zA-Z0-9'      # 英文和数字
            r'，。！？、；：""''（）'  # 中文标点
            r'\s'             # 空白字符
            r']'
        )
        
        # 使用编译后的正则表达式
        pattern = re.compile(f'[^{allowed_chars}]')
        text = pattern.sub('', text)
        
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
        current_time = int(time.time())
        
        try:
            # 连接到 ChromaDB 的 SQLite 数据库
            db_path = os.path.join('./chroma_db', 'chroma.sqlite3')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for doc in documents:
                # 1. 智能分块
                chunks = self.chunker.create_chunks(doc.text)
                chunks = self.semantic_processor.compute_semantic_similarity(chunks)
                chunks = self.context_manager.add_context(chunks)
                
                # 4. 添加文档源信息和必要的元数据字段
                for text, metadata in chunks:
                    # 合并原有元数据和新的元数据
                    enhanced_metadata = {
                        # 基础元数据字段
                        'source': doc.metadata.get('file_path', 'unknown'),
                        'is_deleted': "0",  # 使用字符串"0"表示未删除
                        'create_time': str(current_time),
                        'file_type': doc.metadata.get('file_type', 'unknown'),
                        'file_size': str(doc.metadata.get('file_size', 0)),
                        'last_modified': str(current_time),
                    }
                    
                    # 添加其他元数据，确保所有值都是字符串类型
                    for k, v in metadata.items():
                        if isinstance(v, (bool, int, float, str)):
                            enhanced_metadata[k] = str(v)
                    
                    all_chunks.append((text, enhanced_metadata))
            
            # 验证元数据是否正确添加
            if all_chunks:
                # 打印示例元数据
                logging.info("\n添加的元数据示例:")
                logging.info(json.dumps(all_chunks[0][1], indent=2, ensure_ascii=False))
                
                # 查询数据库中的元数据
                cursor.execute("""
                    SELECT DISTINCT key 
                    FROM embedding_metadata 
                    ORDER BY key
                """)
                keys = cursor.fetchall()
                logging.info("\n数据库中的元数据键:")
                for key in keys:
                    logging.info(f"- {key[0]}")
            
            cursor.close()
            conn.close()
            
            return all_chunks
            
        except Exception as e:
            logging.error(f"处理文档时出错: {str(e)}")
            return all_chunks

    def _print_metadata_info(self):
        """打印元数据存储信息"""
        try:
            db_path = os.path.join('./chroma_db', 'chroma.sqlite3')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 1. 统计元数据信息
            cursor.execute("""
                SELECT COUNT(DISTINCT id) 
                FROM embedding_metadata 
                WHERE key = 'is_deleted' AND bool_value = 1
            """)
            deleted_count = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(DISTINCT id) FROM embedding_metadata")
            total_count = cursor.fetchone()[0] or 0
            
            logging.info(f"\n元数据统计:")
            logging.info(f"- 总记录数: {total_count}")
            logging.info(f"- 已删除记录: {deleted_count}")
            logging.info(f"- 活跃记录: {total_count - deleted_count}")
            
            # 2. 打印示例记录
            cursor.execute("""
                SELECT DISTINCT id, key, string_value, int_value, float_value, bool_value
                FROM embedding_metadata
                LIMIT 20
            """)
            results = cursor.fetchall()
            if results:
                logging.info("\n元数据记录示例:")
                current_id = None
                metadata = {}
                for id_, key, str_val, int_val, float_val, bool_val in results:
                    if current_id != id_:
                        if current_id is not None:
                            logging.info(f"\nID: {current_id}")
                            logging.info(f"Metadata: {metadata}")
                        current_id = id_
                        metadata = {}
                    value = str_val or int_val or float_val or bool_val
                    metadata[key] = value
                
                # 打印最后一条记录
                if current_id is not None:
                    logging.info(f"\nID: {current_id}")
                    logging.info(f"Metadata: {metadata}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logging.error(f"打印元数据信息时出错: {str(e)}")

# 禁用 huggingface 的在线检查
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免警告

# 设置日志级别为 ERROR，减少不必要的输出
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class TextEmbedding:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese", cache_dir: str = "./model_cache"):
        """初始化文本向量化模型
        
        Args:
            model_name: 模型名称，默认使用text2vec-base-chinese
            cache_dir: 模型缓存目录，默认为./model_cache
        """
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用保存的本地模型路径
        model_path = os.path.join(cache_dir, "text2vec_model")
        
        print(f"Looking for model at: {model_path}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # 检查目录是否存在
            if not os.path.exists(model_path):
                raise Exception(
                    f"Model directory not found: {model_path}. "
                    "Please run download_model.py first to download the model."
                )
            
            # 直接从本地路径加载模型，完全禁用在线检查
            print("Loading model from local path...")
            
            # 使用 SentenceTransformer 加载本地模型
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                local_files_only=True  # 只使用本地文件
            )
            print("Successfully loaded model from local cache")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to random vector generation")
            self.model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量"""
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            if self.model is not None:
                # 使用加载的模型
                return self.model.encode(texts).tolist()
            else:
                # 后备方案：生成随机向量
                print("Using random vectors as fallback")
                return np.random.rand(len(texts), 768).tolist()
                
        except Exception as e:
            print(f"Error during encoding: {str(e)}")
            # 返回随机向量作为后备方案
            return np.random.rand(len(texts), 768).tolist()

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
            
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """添加文档到向量数据库"""
        try:
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
            
            current_time = int(time.time())
            
            # 确保每个文档的元数据都包含必需字段
            enhanced_metadatas = []
            for i, meta in enumerate(metadatas):
                # 创建基础元数据，移除保留关键字
                enhanced_meta = {
                    "chunk_id": ids[i],          # 块ID
                    "file_id": meta.get("source", "unknown").split('/')[-1],  # 从源路径提取文件ID
                    "file_name": meta.get("source", "unknown").split('/')[-1],  # 从源路径提取文件名
                    "source": meta.get("source", "unknown"),  # 源路径
                    "is_deleted": "0",           # 删除标记，使用字符串"0"
                    "create_time": str(current_time),  # 创建时间
                    "file_type": meta.get("file_type", "unknown"),  # 文件类型
                    "file_size": str(meta.get("file_size", "0")),   # 文件大小
                    "last_modified": str(current_time)  # 最后修改时间
                }
                enhanced_metadatas.append(enhanced_meta)
            
            # 打印第一条元数据作为示例
            if enhanced_metadatas:
                logging.info("\n添加文档的元数据示例:")
                logging.info(json.dumps(enhanced_metadatas[0], indent=2, ensure_ascii=False))
            
            # 添加文档到数据库
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                metadatas=enhanced_metadatas,
                ids=ids
            )
            
            # 验证元数据是否正确添加
            db_path = os.path.join('./chroma_db', 'chroma.sqlite3')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 检查元数据字段
            cursor.execute("""
                SELECT DISTINCT key 
                FROM embedding_metadata 
                ORDER BY key
            """)
            keys = cursor.fetchall()
            logging.info("\n数据库中的元数据键:")
            for key in keys:
                logging.info(f"- {key[0]}")
            
            # 检查is_deleted字段的数据
            cursor.execute("""
                SELECT COUNT(*) 
                FROM embedding_metadata 
                WHERE key = 'is_deleted'
            """)
            count = cursor.fetchone()[0]
            logging.info(f"\nis_deleted字段统计:")
            logging.info(f"- 总记录数: {count}")
            
            cursor.close()
            conn.close()
            
            logging.info(f"成功添加 {len(texts)} 个文档")
            
        except Exception as e:
            logging.error(f"添加文档时出错: {str(e)}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            raise

    def _print_collection_info(self):
        """打印集合信息"""
        try:
            db_path = os.path.join('./chroma_db', 'chroma.sqlite3')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 检查元数据表中的字段
            cursor.execute("""
                SELECT DISTINCT key, COUNT(*) as count
                FROM embedding_metadata
                GROUP BY key
                ORDER BY key
            """)
            results = cursor.fetchall()
            
            logging.info("\n元数据字段统计:")
            for key, count in results:
                logging.info(f"- {key}: {count} 条记录")
            
            # 检查示例数据
            cursor.execute("""
                SELECT id, key, string_value
                FROM embedding_metadata
                WHERE key = 'is_deleted'
                LIMIT 5
            """)
            samples = cursor.fetchall()
            
            if samples:
                logging.info("\n示例记录:")
                for id_, key, value in samples:
                    logging.info(f"ID: {id_}, {key} = {value}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logging.error(f"获取集合信息时出错: {str(e)}")

    def delete_document(self, file_path: str) -> bool:
        """
        软删除文档（将is_deleted标记设为"1"）
        
        Args:
            file_path: 文件路径，用于标识要删除的文档
            
        Returns:
            bool: 删除是否成功
        """
        try:
            logging.info(f"准备软删除文档: {file_path}")
            
            # 查找与文件路径匹配的所有文档
            results = self.collection.get(
                where={"source": file_path}
            )
            
            if not results or not results['ids']:
                logging.warning(f"未找到与文件路径匹配的文档: {file_path}")
                return False
            
            # 获取要更新的文档ID
            doc_ids = results['ids']
            logging.info(f"找到 {len(doc_ids)} 个要标记删除的文档")
            
            logging.info("更新is_deleted")

            # 更新元数据，将is_deleted设置为"1"
            for i, doc_id in enumerate(doc_ids):
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[{"is_deleted": "1"}]
                )
            
            # 验证更新结果
            verify_results = self.collection.get(
                where={
                    "$and": [
                        {"source": file_path},
                        {"is_deleted": "1"}
                    ]
                }
            )
            
            success = len(verify_results['ids']) == len(doc_ids)
            if success:
                logging.info(f"文档软删除成功: {file_path}")
            else:
                logging.warning(f"部分文档可能未正确标记为删除状态")
            
            return success
                
        except Exception as e:
            logging.error(f"软删除文档时出错: {str(e)}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            return False
            
    def get_documents(self, include_deleted: bool = False) -> List[Dict]:
        """
        获取文档列表，默认不包含已删除的文档
        
        Args:
            include_deleted: 是否包含已删除的文档
            
        Returns:
            List[Dict]: 文档列表
        """
        try:
            where_clause = {} if include_deleted else {"is_deleted": "0"}
            results = self.collection.get(
                where=where_clause
            )
            return results
            
        except Exception as e:
            logging.error(f"获取文档列表时出错: {str(e)}")
            return []

    def count_documents_by_file_id(self, file_id: str) -> int:
        """
        统计特定文件ID的文档数量（不包括已删除的文档）
        
        Args:
            file_id: 文件ID
            
        Returns:
            int: 文档数量
        """
        try:
            # 构建查询条件：匹配file_id且未删除
            results = self.collection.get(
                where={
                    "$and": [
                        {"file_id": file_id},
                        {"is_deleted": "0"}
                    ]
                }
            )
            
            count = len(results['ids']) if results and 'ids' in results else 0
            logging.info(f"文件 {file_id} 的有效文档数量: {count}")
            return count
            
        except Exception as e:
            logging.error(f"统计文档数量时出错: {str(e)}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            return 0
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息，包括已删除和未删除的文档数量
        
        Returns:
            Dict: 包含集合统计信息的字典
        """
        try:
            # 获取所有文档
            all_results = self.collection.get()
            
            if not all_results or not all_results['metadatas']:
                return {
                    "total_documents": 0,
                    "active_documents": 0,
                    "deleted_documents": 0,
                    "sources": []
                }
            
            # 获取未删除的文档
            active_results = self.collection.get(
                where={"is_deleted": "0"}
            )
            
            # 获取已删除的文档
            deleted_results = self.collection.get(
                where={"is_deleted": "1"}
            )
            
            # 统计来源
            sources = set()
            for metadata in all_results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            return {
                "total_documents": len(all_results['ids']),
                "active_documents": len(active_results['ids']) if active_results else 0,
                "deleted_documents": len(deleted_results['ids']) if deleted_results else 0,
                "sources": list(sources)
            }
            
        except Exception as e:
            logging.error(f"获取集合统计信息时出错: {str(e)}")
            return {
                "error": str(e),
                "total_documents": 0,
                "active_documents": 0,
                "deleted_documents": 0,
                "sources": []
            }
            
    def verify_document_exists(self, file_path: str) -> bool:
        """
        验证文档是否存在且未被删除
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文档是否存在且未被删除
        """
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"source": file_path},
                        {"is_deleted": "0"}
                    ]
                }
            )
            exists = len(results['ids']) > 0 if results else False
            logging.info(f"文档 {file_path} 存在状态: {exists}")
            return exists
            
        except Exception as e:
            logging.error(f"验证文档存在时出错: {str(e)}")
            return False

def main():
    """主函数：完整的文档处理流程"""
    # 初始化组件
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
    
    directory_path = "./documents"
    
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
        
        # 3. 文档切割和元数据处理
        logging.info("=== 第3阶段: 文档切割和元数据处理 ===")
        chunks = doc_processor.process_documents(all_docs)
        logging.info(f"生成文本块数: {len(chunks)}")
        
        # 打印示例元数据
        if chunks:
            logging.info("\n示例文本块元数据:")
            logging.info(json.dumps(chunks[0][1], indent=2, ensure_ascii=False))
        
        # 4. 生成向量
        logging.info("=== 第4阶段: 生成向量 ===")
        embeddings = embedding_model.encode([chunk[0] for chunk in chunks])
        logging.info(f"生成向量数: {len(embeddings)}")
        
        # 5. 存储到ChromaDB
        logging.info("=== 第5阶段: 存储到ChromaDB ===")
        chunk_ids = [f"chunk_{i}_{int(time.time())}" for i in range(len(chunks))]
        
        # 确保所有元数据值都是字符串类型
        processed_metadatas = []
        for chunk in chunks:
            metadata = chunk[1]
            processed_metadata = {}
            for k, v in metadata.items():
                processed_metadata[k] = str(v)
            processed_metadatas.append(processed_metadata)
        
        # 添加文档到ChromaDB
        doc_store.add_documents(
            texts=[chunk[0] for chunk in chunks],
            embeddings=embeddings,
            metadatas=processed_metadatas,
            ids=chunk_ids
        )
        
        # 验证元数据是否正确添加
        logging.info("=== 验证元数据添加 ===")
        db_path = os.path.join('./chroma_db', 'chroma.sqlite3')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查is_deleted字段
        cursor.execute("""
            SELECT COUNT(DISTINCT id) 
            FROM embedding_metadata 
            WHERE key = 'is_deleted'
        """)
        count = cursor.fetchone()[0]
        logging.info(f"包含is_deleted字段的文档数: {count}")
        
        cursor.close()
        conn.close()
        
        # 6. 验证存储结果
        logging.info("=== 第6阶段: 验证存储结果 ===")
        collection_info = doc_store.get_collection_info()
        logging.info("\n存储结果统计:")
        logging.info(f"- 总文档数: {collection_info['total_documents']}")
        logging.info(f"- 活跃文档: {collection_info['active_documents']}")
        logging.info(f"- 已删除文档: {collection_info['deleted_documents']}")
        
        logging.info(f"\n处理完成! 成功存储了 {len(chunks)} 个文本块")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()