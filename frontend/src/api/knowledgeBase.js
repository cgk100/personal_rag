import axios from 'axios';
import API_CONFIG from './config';

// 使用config.js中定义的baseURL
const API_BASE_URL = API_CONFIG.baseURL || '';
console.log('使用config.js中的API基础URL:', API_BASE_URL);

// 创建axios实例
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_CONFIG.timeout || 30000, // 使用config中的超时设置或默认30秒
  headers: {
    'Content-Type': 'application/json'
  }
});

// 添加请求拦截器
api.interceptors.request.use(
  config => {
    console.log(`API请求: ${config.method.toUpperCase()} ${config.url}`, config);
    return config;
  },
  error => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

// 添加响应拦截器
api.interceptors.response.use(
  response => {
    console.log(`API响应: ${response.config.method.toUpperCase()} ${response.config.url}`, response.data);
    return response;
  },
  error => {
    console.error('API响应错误:', error.response || error);
    return Promise.reject(error);
  }
);

// 获取知识库文件列表
export const getKnowledgeFiles = async () => {
  try {
    // 添加时间戳防止缓存
    const timestamp = new Date().getTime();
    const response = await api.get(`${API_CONFIG.paths.getKnowledgeBase}?t=${timestamp}`);
    return response.data;
  } catch (error) {
    console.error('获取知识库文件失败:', error);
    throw error;
  }
};

// 上传文件到知识库
export const uploadFiles = async (files) => {
  try {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    const response = await api.post(API_CONFIG.paths.upload, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('上传文件失败:', error);
    throw error;
  }
};

// 删除知识库文件
export const deleteFile = async (fileId) => {
  try {
    const response = await api.delete(`${API_CONFIG.paths.delete}/${fileId}`);
    return response.data;
  } catch (error) {
    console.error('删除文件失败:', error);
    throw error;
  }
};

export default {
  getKnowledgeFiles,
  uploadFiles,
  deleteFile
};