<template>
  <div class="chat-history" ref="chatContainer">
    <div v-if="messages.length === 0" class="empty-chat">
      <el-empty description="暂无聊天记录">
        <template #description>
          <p>在下方输入框中提问，开始聊天</p>
        </template>
      </el-empty>
    </div>
    
    <div v-else class="messages">
      <div 
        v-for="(message, index) in messages" 
        :key="index"
        :class="['message', message.role]"
      >
        <div class="message-avatar">
          <el-avatar 
            :icon="message.role === 'user' ? User : ChatRound"
            :size="36"
            :color="message.role === 'user' ? '#409EFF' : '#67C23A'"
          />
        </div>
        <div class="message-content">
          <div class="message-header">
            <span class="message-role">{{ message.role === 'user' ? '用户' : '助手' }}</span>
            <span class="message-time">{{ formatTime(message.time) }}</span>
          </div>
          <div class="message-body">
            <div v-if="message.loading" class="loading-indicator">
              <div class="thinking">
                思考中<span class="dot-one">.</span><span class="dot-two">.</span><span class="dot-three">.</span>
              </div>
            </div>
            <div v-else-if="message.error" class="error-message">
              {{ message.content }}
            </div>
            <div v-else class="markdown-content" v-html="renderMarkdown(message.content)"></div>
          </div>
          
          <!-- 知识库来源展示 -->
          <div v-if="message.sources && message.sources.length > 0" class="message-sources">
            <div class="sources-header">
              <el-icon><Document /></el-icon>
              <span>参考文档</span>
            </div>
            <ul class="sources-list">
              <li v-for="(source, idx) in message.sources" :key="idx" class="source-item">
                <el-tooltip :content="source.content" placement="top" :show-after="500">
                  <div class="source-name">
                    <el-icon><Document /></el-icon>
                    <span class="filename">{{ formatFilename(source.filename || '未知文档') }}</span>
                  </div>
                </el-tooltip>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'
import { User, ChatRound, Document } from '@element-plus/icons-vue'

// 使用全局变量
// 注意：CDN 加载的 marked 是一个对象，不是函数
const marked = window.marked
const DOMPurify = window.DOMPurify
const hljs = window.hljs

// 配置 marked
marked.setOptions({
  highlight: function(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  },
  langPrefix: 'hljs language-',
  gfm: true,        // 启用GitHub风格的Markdown
  breaks: true,     // 启用换行符转换为<br>
  pedantic: false,  // 不使用原始markdown.pl的bug
  sanitize: false,  // 不进行HTML过滤（我们使用DOMPurify）
  smartLists: true, // 使用更智能的列表行为
  smartypants: true // 使用更智能的标点符号
})

// 接收父组件传递的消息
// eslint-disable-next-line no-undef
const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  }
})

// 状态变量
const chatContainer = ref(null)

// 滚动到底部函数
const scrollToBottom = async () => {
  await nextTick()
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

// 监听消息变化，自动滚动到底部
watch(() => props.messages.length, () => {
  scrollToBottom()
}, { immediate: true })

// 监听消息内容变化，自动滚动到底部
watch(() => {
  return props.messages.map(m => m && m.content ? m.content : '')
}, () => {
  scrollToBottom()
}, { deep: true })

// 格式化时间
const formatTime = (date) => {
  if (!date) return ''
  const now = new Date()
  const messageDate = new Date(date)
  
  // 如果是今天，只显示时间
  if (now.toDateString() === messageDate.toDateString()) {
    return messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  
  // 否则显示日期和时间
  return messageDate.toLocaleString([], {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 渲染 Markdown
const renderMarkdown = (text) => {
  if (!text) return ''
  try {
    // 移除可能的元描述文本
    let cleanText = text;
    
    // 移除"采用Markdown格式呈现"等元描述
    const metaDescriptions = [
      /以下.*采用Markdown格式呈现[:：]?\s*/i,
      /下面.*采用Markdown格式呈现[:：]?\s*/i,
      /采用Markdown格式呈现[:：]?\s*/i,
      /以Markdown格式呈现[:：]?\s*/i,
      /使用Markdown格式[:：]?\s*/i
    ];
    
    for (const pattern of metaDescriptions) {
      cleanText = cleanText.replace(pattern, '');
    }
    
    console.log('清理后的文本:', cleanText.substring(0, 100) + '...');
    
    // 使用 marked.parse 解析 Markdown
    const html = marked.parse(cleanText);
    
    // 使用 DOMPurify 清理 HTML，防止 XSS 攻击
    return DOMPurify.sanitize(html);
  } catch (error) {
    console.error('Markdown 渲染失败:', error);
    return `<p class="error">Markdown 渲染失败: ${error.message}</p>`;
  }
}

// 格式化文件名，移除路径和UUID前缀
const formatFilename = (filename) => {
  // 移除路径
  let name = filename.split('/').pop();
  
  // 如果文件名包含UUID格式，则尝试提取原始文件名
  if (/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/.test(name)) {
    // 如果是完整路径，只保留文件名部分
    return name;
  }
  
  return name;
}

// 在组件挂载后滚动到底部
onMounted(() => {
  scrollToBottom()
})
</script>

<style scoped>
.chat-history {
  height: 100%;
  overflow-y: auto;
  padding: var(--spacing-medium);
  font-size: 14px;  /* 减小字体大小 */
  line-height: 1.6; /* 增加行间距 */
}

.empty-chat {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.messages {
  display: flex;
  flex-direction: column;
  gap: 24px; /* 增加消息之间的间距 */
}

.message {
  display: flex;
  gap: var(--spacing-medium);
  max-width: 100%;
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
}

.message-content {
  max-width: 85%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-small);
}

.message.user .message-content {
  align-items: flex-end;
}

.message-header {
  display: flex;
  gap: var(--spacing-small);
  font-size: 12px; /* 减小头部字体大小 */
  color: var(--color-text-secondary);
}

.message.user .message-header {
  flex-direction: row-reverse;
}

.message-body {
  padding: 12px 16px; /* 调整内边距 */
  border-radius: 12px; /* 增加圆角 */
  background-color: var(--color-white);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04); /* 减轻阴影 */
  font-size: 14px; /* 减小字体大小 */
  line-height: 1.6; /* 增加行间距 */
  width: 100%;
  box-sizing: border-box;
  min-width: 200px;
  max-width: 100%;
}

.message.user .message-body {
  background-color: var(--color-primary-light);
}

.error-message {
  color: var(--color-danger);
}

.loading-indicator {
  min-width: 200px;
}

.thinking {
  display: inline-block;
  font-style: italic;
  color: var(--color-text-secondary);
}

.dot-one, .dot-two, .dot-three {
  opacity: 0;
  animation: dotFade 1.5s infinite;
}

.dot-two {
  animation-delay: 0.5s;
}

.dot-three {
  animation-delay: 1s;
}

@keyframes dotFade {
  0% { opacity: 0; }
  50% { opacity: 1; }
  100% { opacity: 0; }
}

/* 改进的 Markdown 样式 */
.markdown-content {
  white-space: normal;
  word-break: break-word;
  line-height: 1.6; /* 增加行间距 */
  font-size: 14px; /* 减小字体大小 */
  color: var(--color-text-primary);
  width: 100%;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* 确保段落正确显示 */
:deep(.markdown-content p) {
  margin: 0 0 12px 0; /* 增加段落间距 */
  padding: 0;
  line-height: 1.6; /* 增加行间距 */
}

/* 确保最后一个段落没有底部边距 */
:deep(.markdown-content p:last-child) {
  margin-bottom: 0;
}

/* 统一所有文本元素的样式 */
:deep(.markdown-content p),
:deep(.markdown-content li),
:deep(.markdown-content blockquote),
:deep(.markdown-content td),
:deep(.markdown-content th) {
  font-size: 14px; /* 减小字体大小 */
  line-height: 1.6; /* 增加行间距 */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* 标题样式 */
:deep(.markdown-content h1),
:deep(.markdown-content h2),
:deep(.markdown-content h3),
:deep(.markdown-content h4),
:deep(.markdown-content h5),
:deep(.markdown-content h6) {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-weight: 600;
  line-height: 1.4; /* 增加标题行间距 */
  margin: 20px 0 12px 0; /* 调整标题间距 */
}

:deep(.markdown-content h1) { font-size: 20px; }
:deep(.markdown-content h2) { font-size: 18px; }
:deep(.markdown-content h3) { font-size: 16px; }
:deep(.markdown-content h4) { font-size: 15px; }
:deep(.markdown-content h5) { font-size: 14px; }
:deep(.markdown-content h6) { font-size: 14px; }

:deep(.markdown-content h1:first-child),
:deep(.markdown-content h2:first-child),
:deep(.markdown-content h3:first-child),
:deep(.markdown-content h4:first-child),
:deep(.markdown-content h5:first-child),
:deep(.markdown-content h6:first-child) {
  margin-top: 0;
}

/* 代码块样式优化 */
:deep(.markdown-content pre) {
  background-color: #f8f9fa;
  border-radius: 6px;
  padding: 12px;
  margin: 12px 0;
  overflow-x: auto;
}

:deep(.markdown-content code) {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 13px;
  background-color: #f8f9fa;
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.markdown-content pre code) {
  padding: 0;
  background-color: transparent;
}

/* 列表样式优化 */
:deep(.markdown-content ul),
:deep(.markdown-content ol) {
  padding-left: 20px;
  margin: 8px 0 12px 0;
}

:deep(.markdown-content li) {
  margin-bottom: 4px;
}

:deep(.markdown-content li:last-child) {
  margin-bottom: 0;
}

/* 引用块样式 */
:deep(.markdown-content blockquote) {
  border-left: 3px solid #e0e0e0;
  margin: 12px 0;
  padding: 8px 16px;
  color: #666;
  background-color: #f9f9f9;
}

/* 知识库来源样式 */
.message-sources {
  margin-top: 10px;
  font-size: 12px; /* 减小字体大小 */
  color: var(--color-text-secondary);
  background-color: var(--color-background-light);
  border-radius: 8px; /* 增加圆角 */
  padding: 8px 12px; /* 调整内边距 */
  border-left: 2px solid var(--color-primary);
  width: 100%;
  box-sizing: border-box;
}

.sources-header {
  font-weight: 600;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--color-primary);
}

.sources-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.source-item {
  display: flex;
  align-items: center;
  background-color: var(--color-white);
  border-radius: 4px;
  padding: 3px 6px; /* 减小内边距 */
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03); /* 减轻阴影 */
}

.source-name {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  color: var(--color-primary);
}

.filename {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style> 