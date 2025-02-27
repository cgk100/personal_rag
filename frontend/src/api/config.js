// API 配置文件
const API_CONFIG = {
  // API 基础 URL
  baseURL: 'http://localhost:8000',
  
  // API 超时时间（毫秒）
  timeout: 30000,
  
  // API 路径
  paths: {
    testConnection: '/api/test-connection',
    chat: '/api/chat',
    chatSearch: '/api/chat-search',
    upload: '/api/knowledge/upload',
    getKnowledgeBase: '/api/knowledge-base',
    stream: '/api/chat/stream',
    delete: '/api/knowledge'
  }
};

export default API_CONFIG; 