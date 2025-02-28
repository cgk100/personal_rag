<template>
  <el-container class="kb-container">
    <!-- 左侧导航栏 -->
    <SideNavigation 
      :default-active="activeMenu" 
      @menu-select="handleMenuSelect" 
    />
    
    <!-- 主内容区域 -->
    <el-main class="kb-main">
      <!-- 知识库管理页面 -->
      <KnowledgeManager v-if="activeMenu === 'knowledge'" />
      
      <!-- 对话历史页面 -->
      <div v-if="activeMenu === 'chat'" class="chat-page">
        <div class="chat-page-header">
          <h2>对话历史</h2>
          <el-button 
            type="primary" 
            plain 
            size="small" 
            :icon="Plus"
            @click="startNewChat"
          >
            新对话
          </el-button>
        </div>
        
        <div class="chat-container">
          <div class="chat-body">
            <ChatHistory :messages="messages" ref="chatHistoryRef" />
          </div>
          
          <div class="chat-footer">
            <ChatInput @send-message="handleSendMessage" @search-message="handleSearchMessage" />
          </div>
        </div>
      </div>
      
      <!-- 系统设置页面 -->
      <div v-if="activeMenu === 'settings'" class="settings-page">
        <div class="page-header">
          <h2 class="page-title">系统设置</h2>
        </div>
        
        <div class="settings-content">
          <!-- 设置选项卡 -->
          <el-tabs type="border-card" class="settings-tabs">
            <el-tab-pane label="模型设置">
              <ModelSettings />
            </el-tab-pane>
            <el-tab-pane label="应用设置">
              <div class="app-settings">
                <h3>应用设置</h3>
                <el-form label-position="top" size="small">
                  <el-form-item label="界面主题">
                    <el-radio-group v-model="theme">
                      <el-radio label="light">浅色</el-radio>
                      <el-radio label="dark">深色</el-radio>
                      <el-radio label="system">跟随系统</el-radio>
                    </el-radio-group>
                  </el-form-item>
                  
                  <el-form-item label="语言">
                    <el-select v-model="language" class="full-width">
                      <el-option label="简体中文" value="zh-CN" />
                      <el-option label="English" value="en-US" />
                    </el-select>
                  </el-form-item>
                  
                  <el-form-item>
                    <el-button type="primary">保存设置</el-button>
                  </el-form-item>
                </el-form>
              </div>
            </el-tab-pane>
            <el-tab-pane label="关于">
              <div class="about-section">
                <h3>关于本应用</h3>
                <p>本地知识库应用 v1.0.0</p>
                <p>基于 Vue.js 和 Element Plus 构建的本地知识库应用，支持文档管理和智能对话。</p>
              </div>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </el-main>
  </el-container>
</template>

<script>
import { ref, provide, onMounted } from 'vue'
import { Plus } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'
import API_CONFIG from '../api/config'

import SideNavigation from './SideNavigation.vue'
import ModelSettings from './ModelSettings.vue'
import ChatHistory from './ChatHistory.vue'
import ChatInput from './ChatInput.vue'
import KnowledgeManager from './KnowledgeManager.vue'

export default {
  name: 'KnowledgeBaseApp',
  components: {
    SideNavigation,
    ModelSettings,
    ChatHistory,
    ChatInput,
    KnowledgeManager
  },
  data() {
    return {
      activeMenu: 'knowledge', // 默认显示知识库管理
      theme: 'light',
      language: 'zh-CN'
    }
  },
  methods: {
    handleMenuSelect(key) {
      this.activeMenu = key;
    },
    startNewChat() {
      // 开始新对话的逻辑
      this.messages = [];
      this.$message.success('已开始新对话');
    },
    handleSendMessage(messageData) {
      console.log('handleSendMessage received:', messageData);
      
      // 检查是否有用户消息，如果有则添加
      if (messageData.userMessage) {
        this.addMessage(messageData.userMessage);
      }
      
      // 检查是否有助手消息
      if (messageData.assistantMessage) {
        // 如果是更新现有消息
        if (this.messages.length > 0 && 
            this.messages[this.messages.length - 1].role === 'assistant') {
          this.updateLastMessage(messageData.assistantMessage);
        } else {
          // 否则添加新消息
          this.addMessage(messageData.assistantMessage);
        }
      }
    },
    handleSearchMessage(messageData) {
      console.log('handleSearchMessage received:', messageData);
      
      // 检查是否有用户消息，如果有则添加
      if (messageData.userMessage) {
        this.addMessage(messageData.userMessage);
      }
      
      // 检查是否有助手消息
      if (messageData.assistantMessage) {
        // 如果是更新现有消息
        if (this.messages.length > 0 && 
            this.messages[this.messages.length - 1].role === 'assistant') {
          this.updateLastMessage(messageData.assistantMessage);
        } else {
          // 否则添加新消息
          this.addMessage(messageData.assistantMessage);
        }
      }
    }
  },
  setup() {
    // 聊天消息
    const messages = ref([]);
    const chatHistoryRef = ref(null);
    const knowledgeFiles = ref([]);
    const loadingKnowledgeFiles = ref(false);
    
    // 获取知识库文件列表
    const fetchKnowledgeFiles = async () => {
      try {
        console.log('KnowledgeBaseApp: 开始获取知识库文件列表');
        loadingKnowledgeFiles.value = true;
        
        // 添加时间戳防止缓存
        const timestamp = new Date().getTime();
        const response = await axios.get(`${API_CONFIG.baseURL}${API_CONFIG.paths.getKnowledgeBase}?t=${timestamp}`);
        
        console.log('KnowledgeBaseApp: 知识库API响应:', response);
        
        if (response.data) {
          if (Array.isArray(response.data)) {
            knowledgeFiles.value = response.data;
            console.log('KnowledgeBaseApp: 获取到知识库文件:', knowledgeFiles.value.length, '个');
            
            // 检查文件数据结构
            if (knowledgeFiles.value.length > 0) {
              console.log('KnowledgeBaseApp: 第一个文件数据:', JSON.stringify(knowledgeFiles.value[0]));
            }
          } else {
            console.warn('KnowledgeBaseApp: API返回的数据不是数组:', response.data);
            knowledgeFiles.value = [];
          }
        } else {
          console.warn('KnowledgeBaseApp: API返回了空数据');
          knowledgeFiles.value = [];
        }
      } catch (error) {
        console.error('KnowledgeBaseApp: 获取知识库文件失败:', error);
        ElMessage.error('获取知识库文件失败: ' + (error.response?.data?.detail || error.message));
        knowledgeFiles.value = [];
      } finally {
        loadingKnowledgeFiles.value = false;
      }
    };
    
    // 在组件挂载时获取知识库文件列表
    onMounted(() => {
      console.log('KnowledgeBaseApp: 组件已挂载，准备获取知识库文件列表');
      fetchKnowledgeFiles();
    });
    
    // 提供知识库文件列表给子组件
    provide('knowledgeFiles', knowledgeFiles);
    provide('loadingKnowledgeFiles', loadingKnowledgeFiles);
    provide('fetchKnowledgeFiles', fetchKnowledgeFiles);
    
    // 添加消息
    const addMessage = (message) => {
      console.log('Adding message:', message);
      
      // 防御性检查，确保消息是一个对象
      if (!message || typeof message !== 'object') {
        console.error('Invalid message object:', message);
        return;
      }
      
      // 确保消息对象有必要的属性
      const safeMessage = {
        role: message.role || 'user',
        content: message.content || '',
        time: message.time || new Date(),
        loading: !!message.loading,
        error: !!message.error,
        sources: Array.isArray(message.sources) ? message.sources : []
      };
      
      messages.value.push(safeMessage);
    };
    
    // 更新最后一条消息
    const updateLastMessage = (update) => {
      console.log('Updating last message with:', update);
      
      if (messages.value.length > 0) {
        const lastMessage = messages.value[messages.value.length - 1];
        
        // 防御性检查，确保更新是一个对象
        if (!update || typeof update !== 'object') {
          console.error('Invalid update object:', update);
          return;
        }
        
        // 更新消息属性
        Object.assign(lastMessage, update);
      } else {
        console.warn('No messages to update');
      }
    };
    
    // 模型设置
    const modelSettings = ref({
      baseUrl: 'https://api.openai.com/v1',
      apiKey: '',
      selectedModel: 'gpt-3.5-turbo',
      temperature: 0.7,
      maxLength: 2000
    });
    
    // 在组件创建时加载设置
    const loadSettings = () => {
      const savedSettings = localStorage.getItem('modelSettings');
      if (savedSettings) {
        try {
          modelSettings.value = JSON.parse(savedSettings);
        } catch (e) {
          console.error('Failed to parse saved settings:', e);
        }
      }
    };
    
    // 在组件创建时加载设置
    loadSettings();
    
    // 提供模型设置给子组件
    provide('getModelSettings', () => modelSettings.value);
    
    // 更新模型设置
    const updateModelSettings = (settings) => {
      modelSettings.value = { ...settings };
    };
    
    // 提供更新方法给子组件
    provide('updateModelSettings', updateModelSettings);
    
    return {
      messages,
      chatHistoryRef,
      addMessage,
      updateLastMessage,
      Plus,
      updateModelSettings,
      knowledgeFiles,
      loadingKnowledgeFiles,
      fetchKnowledgeFiles
    };
  }
};
</script>

<style scoped>
.kb-container {
  height: 100vh;
}

/* 主内容区域样式 */
.kb-main {
  padding: 0;
  background-color: var(--color-background);
  height: 100%;
  overflow: auto;
}

/* 对话页面样式 */
.chat-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--color-white);
}

.chat-page-header {
  padding: var(--spacing-medium);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-page-header h2 {
  margin: 0;
  font-size: var(--font-size-large);
  font-weight: 500;
  color: var(--color-text-primary);
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-medium);
}

.chat-footer {
  border-top: 1px solid var(--color-border);
  padding: var(--spacing-medium);
}

/* 设置页面样式 */
.settings-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--color-white);
}

.page-header {
  padding: var(--spacing-medium);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.page-title {
  margin: 0;
  font-size: var(--font-size-large);
  font-weight: 500;
  color: var(--color-text-primary);
}

.settings-content {
  flex: 1;
  padding: var(--spacing-medium);
  overflow: auto;
}

.settings-tabs {
  height: 100%;
}

.settings-tabs :deep(.el-tabs__content) {
  padding: var(--spacing-medium);
}

.app-settings h3, .about-section h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-medium);
  font-size: var(--font-size-medium);
  font-weight: 500;
  color: var(--color-text-primary);
}

.about-section p {
  margin-bottom: var(--spacing-medium);
  line-height: 1.6;
}

.full-width {
  width: 100%;
}
</style> 