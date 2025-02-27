<template>
  <div class="chat-input">
    <el-input
      v-model="message"
      type="textarea"
      :rows="2"
      placeholder="输入问题..."
      resize="none"
      @keydown.enter.prevent="handleEnter"
    />
    <div class="input-actions">
      <el-button 
        type="primary" 
        :icon="Send" 
        :disabled="!message.trim() || loading" 
        @click="sendMessage"
        :loading="loading"
      >
        发送
      </el-button>
      <el-tooltip content="使用知识库搜索" placement="top">
        <el-button 
          type="success" 
          :icon="Search" 
          :disabled="!message.trim() || loading" 
          @click="searchMessage"
          :loading="loading"
        >
          搜索
        </el-button>
      </el-tooltip>
    </div>
  </div>
</template>

<script setup>
import { ref, inject, onMounted } from 'vue'
import { Right as Send, Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import API_CONFIG from '../api/config.js'

// 定义事件
// eslint-disable-next-line no-undef
const emit = defineEmits(['send-message', 'search-message'])

// 状态变量
const message = ref('')
const loading = ref(false)

// 从父组件注入的方法
const getModelSettings = inject('getModelSettings')

// 在组件挂载时检查API设置
onMounted(() => {
  const settings = getModelSettings()
  if (!settings.apiKey) {
    ElMessage.warning('请在设置中配置API Key以启用聊天功能')
  }
})

// 发送普通消息 - 使用流式API
const sendMessage = async () => {
  if (!message.value.trim() || loading.value) return
  
  // 获取模型设置
  const settings = getModelSettings()
  if (!settings.apiKey) {
    ElMessage.error('请先在设置中配置API Key')
    return
  }
  
  // 创建用户消息
  const userMessage = {
    role: 'user',
    content: message.value,
    time: new Date()
  }
  
  // 创建助手消息（初始为空）
  const assistantMessage = {
    role: 'assistant',
    content: '',
    time: new Date(),
    loading: true
  }
  
  // 清空输入框
  const userInput = message.value
  message.value = ''
  loading.value = true
  
  try {
    // 准备消息历史
    const messages = [
      { role: 'system', content: '你是一个有用的助手。' },
      { role: 'user', content: userInput }
    ]
    
    // 发送事件给父组件（开始聊天）
    emit('send-message', {
      userMessage,
      assistantMessage
    })
    
    console.log('开始流式请求')
    
    // 使用fetch API处理流式响应
    const response = await fetch(`${API_CONFIG.baseURL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        messages: messages,
        model: settings.selectedModel,
        apiKey: settings.apiKey,
        baseUrl: settings.baseUrl,
        temperature: settings.temperature,
        max_tokens: settings.maxLength,
        use_knowledge_base: true  // 修改为 true，使用知识库
      })
    })
    
    if (!response.ok) {
      throw new Error(`请求失败: ${response.status} ${response.statusText}`)
    }
    
    // 获取响应的 ReadableStream
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    // 更新助手消息状态
    assistantMessage.loading = false
    
    let buffer = ''
    
    // 读取流 - 修复 ESLint 错误
    let done = false
    while (!done) {
      const result = await reader.read()
      done = result.done
      if (done) break
      
      // 解码并追加到消息内容
      const chunk = decoder.decode(result.value, { stream: true })
      buffer += chunk
      
      // 检查是否有特殊标记表示知识库来源
      if (buffer.includes('SOURCES:')) {
        const parts = buffer.split('SOURCES:')
        const content = parts[0].trim()
        
        try {
          // 解析知识库来源
          const sourcesJson = parts[1].trim()
          const sources = JSON.parse(sourcesJson)
          
          // 发送更新事件给父组件
          emit('send-message', {
            assistantMessage: {
              ...assistantMessage,
              content: content,
              loading: false,
              sources: sources
            }
          })
        } catch (e) {
          console.error('解析知识库来源失败:', e)
          // 如果解析失败，仍然更新内容
          emit('send-message', {
            assistantMessage: {
              ...assistantMessage,
              content: buffer,
              loading: false
            }
          })
        }
      } else {
        // 普通内容更新
        emit('send-message', {
          assistantMessage: {
            ...assistantMessage,
            content: buffer,
            loading: false
          }
        })
      }
    }
    
    console.log('流式响应完成')
    
  } catch (error) {
    console.error('请求失败:', error)
    
    // 发送错误事件给父组件
    emit('send-message', {
      assistantMessage: {
        ...assistantMessage,
        content: `请求失败: ${error.message}`,
        loading: false,
        error: true
      }
    })
  } finally {
    loading.value = false
  }
}

// 使用知识库搜索 - 使用流式API
const searchMessage = async () => {
  if (!message.value.trim() || loading.value) return
  
  // 获取模型设置
  const settings = getModelSettings()
  if (!settings.apiKey) {
    ElMessage.error('请先在设置中配置API Key')
    return
  }
  
  // 创建用户消息
  const userMessage = {
    role: 'user',
    content: message.value,
    time: new Date()
  }
  
  // 创建助手消息（初始为空）
  const assistantMessage = {
    role: 'assistant',
    content: '',
    time: new Date(),
    loading: true
  }
  
  // 清空输入框
  const userInput = message.value
  message.value = ''
  loading.value = true
  
  try {
    // 准备消息历史
    const messages = [
      { role: 'system', content: '你是一个有用的助手。' },
      { role: 'user', content: userInput }
    ]
    
    // 发送事件给父组件（开始搜索）
    emit('search-message', {
      userMessage,
      assistantMessage
    })
    
    console.log('开始知识库流式请求')
    
    // 使用fetch API处理流式响应
    const response = await fetch(`${API_CONFIG.baseURL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        messages: messages,
        model: settings.selectedModel,
        apiKey: settings.apiKey,
        baseUrl: settings.baseUrl,
        temperature: settings.temperature,
        max_tokens: settings.maxLength,
        use_knowledge_base: true // 使用知识库
      })
    })
    
    if (!response.ok) {
      throw new Error(`请求失败: ${response.status} ${response.statusText}`)
    }
    
    // 获取响应的 ReadableStream
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    // 更新助手消息状态
    assistantMessage.loading = false
    
    let buffer = ''
    
    // 读取流 - 修复 ESLint 错误
    let done = false
    while (!done) {
      const result = await reader.read()
      done = result.done
      if (done) break
      
      // 解码并追加到消息内容
      const chunk = decoder.decode(result.value, { stream: true })
      buffer += chunk
      
      // 检查是否有特殊标记表示知识库来源
      if (buffer.includes('SOURCES:')) {
        const parts = buffer.split('SOURCES:')
        const content = parts[0].trim()
        
        try {
          // 解析知识库来源
          const sourcesJson = parts[1].trim()
          const sources = JSON.parse(sourcesJson)
          
          // 发送更新事件给父组件
          emit('search-message', {
            assistantMessage: {
              ...assistantMessage,
              content: content,
              loading: false,
              sources: sources
            }
          })
        } catch (e) {
          console.error('解析知识库来源失败:', e)
          // 如果解析失败，仍然更新内容
          emit('search-message', {
            assistantMessage: {
              ...assistantMessage,
              content: buffer,
              loading: false
            }
          })
        }
      } else {
        // 普通内容更新
        emit('search-message', {
          assistantMessage: {
            ...assistantMessage,
            content: buffer,
            loading: false
          }
        })
      }
    }
    
    console.log('知识库流式响应完成')
    
  } catch (error) {
    console.error('搜索请求失败:', error)
    
    // 发送错误事件给父组件
    emit('search-message', {
      assistantMessage: {
        ...assistantMessage,
        content: `搜索失败: ${error.message}`,
        loading: false,
        error: true
      }
    })
  } finally {
    loading.value = false
  }
}

// 处理回车键
const handleEnter = (e) => {
  if (e.shiftKey) return // Shift+Enter 允许换行
  sendMessage()
}
</script>

<style scoped>
.chat-input {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-small);
  padding: var(--spacing-medium);
  border-top: 1px solid var(--color-border);
  background-color: var(--color-background);
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-small);
}
</style> 