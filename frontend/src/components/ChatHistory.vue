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
            <div class="sources-header" @click="toggleSources(message)" :class="{ 'is-expanded': message.showSources }">
              <span>找到了 {{ message.sources.length }} 篇知识库资料作为参考</span>
              <el-icon class="arrow-icon" :class="{ 'is-expanded': message.showSources }">
                <ArrowDown />
              </el-icon>
            </div>
            <transition name="expand">
              <div v-show="message.showSources" class="sources-list">
                <div 
                  v-for="(source, idx) in message.sources" 
                  :key="idx" 
                  class="source-item"
                >
                  <span class="source-number">{{ idx + 1 }}</span>
                  <el-popover
                    placement="top"
                    :width="400"
                    trigger="hover"
                    :show-after="500"
                    popper-class="source-popover"
                  >
                    <template #default>
                      <div class="source-preview">
                        <div class="preview-header">
                          <el-icon><Document /></el-icon>
                          <span>{{ source.filename }}</span>
                        </div>
                        <div class="preview-content">{{ source.content }}</div>
                      </div>
                    </template>
                    <template #reference>
                      <div class="source-name">
                        {{ source.filename }}
                      </div>
                    </template>
                  </el-popover>
                </div>
              </div>
            </transition>
          </div>
        </div>
      </div>
    </div>

    <!-- 添加一个返回底部的按钮 -->
    <div 
      v-show="userScrolling" 
      class="scroll-to-bottom"
      @click="scrollToBottomManually"
    >
      <el-button 
        type="primary" 
        circle
        size="small"
      >
        <el-icon><ArrowDown /></el-icon>
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { User, ChatRound, Document, ArrowDown } from '@element-plus/icons-vue'

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

// 添加一个变量来追踪是否用户正在查看历史消息
const userScrolling = ref(false)
const lastScrollTop = ref(0)

// 监听滚动事件
const handleScroll = () => {
  if (!chatContainer.value) return
  
  const { scrollTop, scrollHeight, clientHeight } = chatContainer.value
  const isAtBottom = scrollHeight - scrollTop - clientHeight < 50 // 50px 的缓冲区
  
  // 记录最后的滚动位置
  lastScrollTop.value = scrollTop
  
  // 如果用户向上滚动，标记为正在查看历史消息
  if (!isAtBottom) {
    userScrolling.value = true
  } else {
    userScrolling.value = false
  }
}

// 修改滚动到底部的函数
const scrollToBottom = async () => {
  await nextTick()
  if (!chatContainer.value) return
  
  // 只有在用户没有查看历史消息，或者本来就在底部时才自动滚动
  if (!userScrolling.value || 
      (chatContainer.value.scrollHeight - lastScrollTop.value - chatContainer.value.clientHeight < 150)) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

// 添加手动滚动到底部的函数
const scrollToBottomManually = () => {
  userScrolling.value = false
  scrollToBottom()
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

// 添加切换展开/折叠的方法
const toggleSources = (message) => {
  message.showSources = !message.showSources
}

// 在组件挂载时添加滚动事件监听
onMounted(() => {
  if (chatContainer.value) {
    chatContainer.value.addEventListener('scroll', handleScroll)
    scrollToBottom()
  }
})

// 在组件卸载时移除事件监听
onUnmounted(() => {
  if (chatContainer.value) {
    chatContainer.value.removeEventListener('scroll', handleScroll)
  }
})
</script>

<style scoped>
.chat-history {
  height: 100%;
  overflow-y: auto;
  scroll-behavior: smooth; /* 添加平滑滚动效果 */
  padding: var(--spacing-medium);
  position: relative; /* 为固定定位的滚动按钮提供参考 */
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
  margin-top: 12px;
  font-size: 14px;
  color: var(--color-text-regular);
  width: 100%;
  box-sizing: border-box;
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  color: var(--color-text-secondary);
  cursor: pointer;
  user-select: none;
  padding: 8px;
  border-radius: 4px;
  transition: all 0.3s ease;
}

.sources-header:hover {
  background-color: var(--color-background-light);
}

.arrow-icon {
  transition: transform 0.3s ease;
}

.arrow-icon.is-expanded {
  transform: rotate(180deg);
}

.sources-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 8px;
}

/* 展开动画 */
.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  transform: translateY(-10px);
  height: 0;
}

.source-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: var(--el-fill-color-light);
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.source-item:hover {
  background-color: var(--el-fill-color);
}

.source-number {
  color: var(--el-color-primary);
  font-weight: 500;
  min-width: 16px;
}

.source-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--el-text-color-regular);
}

/* Popover 样式 */
:deep(.source-popover) {
  padding: 0;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.source-preview {
  max-width: 400px;
}

.preview-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  font-weight: 500;
  color: var(--el-text-color-primary);
  background-color: var(--el-fill-color-lighter);
  border-radius: 8px 8px 0 0;
}

.preview-content {
  padding: 16px;
  line-height: 1.6;
  color: var(--el-text-color-regular);
  font-size: 14px;
  max-height: 300px;
  overflow-y: auto;
  white-space: pre-wrap;
  background-color: var(--el-bg-color);
  border-radius: 0 0 8px 8px;
}

.scroll-to-bottom {
  position: fixed;
  bottom: 100px;
  right: 30px;
  z-index: 99;
  cursor: pointer;
  transition: opacity 0.3s;
  opacity: 0.6;
}

.scroll-to-bottom:hover {
  opacity: 1;
}
</style> 