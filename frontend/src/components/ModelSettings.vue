<template>
  <div class="model-settings">
    <el-form label-position="top" size="small" class="settings-form">
      <el-form-item label="API 地址">
        <el-input v-model="settings.baseUrl" placeholder="例如: https://api.openai.com/v1" />
      </el-form-item>
      
      <el-form-item label="API Key">
        <el-input 
          v-model="settings.apiKey" 
          placeholder="输入您的 API Key" 
          show-password
        />
      </el-form-item>
      
      <el-form-item label="模型">
        <el-select v-model="settings.selectedModel" class="full-width">
          <el-option label="GPT-3.5 Turbo" value="gpt-3.5-turbo" />
          <el-option label="GPT-4" value="gpt-4" />
          <el-option label="DeepSeek-V3" value="deepseek-ai/DeepSeek-V3" />
          <el-option label="Claude 3 Opus" value="claude-3-opus-20240229" />
          <el-option label="Claude 3 Sonnet" value="claude-3-sonnet-20240229" />
          <el-option label="Claude 3 Haiku" value="claude-3-haiku-20240307" />
        </el-select>
      </el-form-item>
      
      <el-form-item label="温度">
        <el-slider 
          v-model="settings.temperature" 
          :min="0" 
          :max="1" 
          :step="0.1" 
          show-stops
        />
        <div class="slider-description">
          <span>精确</span>
          <span>创造性</span>
        </div>
      </el-form-item>
      
      <el-form-item label="最大长度">
        <el-input-number 
          v-model="settings.maxLength" 
          :min="100" 
          :max="8000" 
          :step="100"
          class="full-width"
        />
      </el-form-item>
      
      <el-form-item>
        <el-button type="primary" @click="saveSettings">保存设置</el-button>
        <el-button @click="testConnection" :loading="testing">测试连接</el-button>
      </el-form-item>
    </el-form>
    
    <div v-if="testResult" class="test-result">
      <el-alert
        :title="testResult.success ? '连接成功' : '连接失败'"
        :type="testResult.success ? 'success' : 'error'"
        :description="testResult.message || (testResult.success ? '成功连接到API' : '无法连接到API')"
        show-icon
      />
      <div v-if="testResult.success" class="test-details">
        <p v-if="testResult.response_time">响应时间: {{ testResult.response_time }}ms</p>
        <p v-if="testResult.model">模型: {{ testResult.model }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, inject } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'
import API_CONFIG from '../api/config.js'

// 从父组件注入的方法
const updateModelSettings = inject('updateModelSettings', null)

// 状态变量
const settings = ref({
  baseUrl: 'https://api.openai.com/v1',
  apiKey: '',
  selectedModel: 'gpt-3.5-turbo',
  temperature: 0.7,
  maxLength: 2000
})
const testing = ref(false)
const testResult = ref(null)

// 在组件挂载时从本地存储加载设置
onMounted(() => {
  loadSettings()
})

// 保存设置到本地存储
const saveSettings = () => {
  localStorage.setItem('modelSettings', JSON.stringify(settings.value))
  
  // 如果父组件提供了更新方法，调用它
  if (updateModelSettings) {
    updateModelSettings(settings.value)
  }
  
  ElMessage.success('设置已保存')
}

// 从本地存储加载设置
const loadSettings = () => {
  const savedSettings = localStorage.getItem('modelSettings')
  if (savedSettings) {
    try {
      const parsed = JSON.parse(savedSettings)
      settings.value = { ...settings.value, ...parsed }
      
      // 如果父组件提供了更新方法，调用它
      if (updateModelSettings) {
        updateModelSettings(settings.value)
      }
    } catch (e) {
      console.error('加载设置失败:', e)
    }
  }
}

// 测试与API的连接
const testConnection = async () => {
  if (!settings.value.apiKey) {
    ElMessage.warning('请先输入API Key')
    return
  }
  
  testing.value = true
  testResult.value = null
  
  try {
    console.log('发送测试连接请求:', {
      baseUrl: settings.value.baseUrl,
      model: settings.value.selectedModel,
      apiKey: settings.value.apiKey
    })
    
    const response = await axios.post(
      `${API_CONFIG.baseURL}${API_CONFIG.paths.testConnection}`,
      {
        baseUrl: settings.value.baseUrl,
        model: settings.value.selectedModel,
        apiKey: settings.value.apiKey
      }
    )
    
    console.log('测试连接响应:', response.data)
    
    if (response.data.success) {
      ElMessage.success('连接测试成功')
      testResult.value = response.data
    } else {
      ElMessage.error(`连接测试失败: ${response.data.error || '未知错误'}`)
      testResult.value = {
        success: false,
        message: response.data.error || '未知错误'
      }
    }
  } catch (error) {
    console.error('测试连接失败:', error)
    ElMessage.error(`连接测试失败: ${error.response?.data?.error || error.message}`)
    testResult.value = {
      success: false,
      message: error.response?.data?.error || error.message
    }
  } finally {
    testing.value = false
  }
}
</script>

<style scoped>
.model-settings {
  max-width: 600px;
  margin: 0 auto;
}

.settings-form {
  margin-bottom: var(--spacing-large);
}

.full-width {
  width: 100%;
}

.slider-description {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-small);
  color: var(--color-text-secondary);
  font-size: var(--font-size-small);
}

.test-result {
  margin-top: var(--spacing-large);
}

.test-details {
  margin-top: var(--spacing-medium);
  padding: var(--spacing-medium);
  background-color: var(--color-background);
  border-radius: var(--border-radius-base);
}

.test-details p {
  margin: var(--spacing-small) 0;
}
</style> 