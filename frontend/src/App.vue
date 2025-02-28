<template>
  <div id="app">
    <KnowledgeBaseApp :default-active="'chat'" />
  </div>
</template>

<script>
import KnowledgeBaseApp from './components/KnowledgeBaseApp.vue'

export default {
  name: 'App',
  components: {
    KnowledgeBaseApp
  }
}
</script>

<style>
:root {
  /* 颜色变量 */
  --color-primary: #409EFF;
  --color-success: #67C23A;
  --color-warning: #E6A23C;
  --color-danger: #F56C6C;
  --color-info: #909399;
  
  /* 中性色 */
  --color-white: #FFFFFF;
  --color-background: #F5F7FA;
  --color-background-light: #F8F9FB;
  --color-text-primary: #303133;
  --color-text-regular: #606266;
  --color-text-secondary: #909399;
  --color-border: #DCDFE6;
  
  /* 主题色变体 */
  --color-primary-light: #ECF5FF;
  
  /* 间距 */
  --spacing-base: 8px;
  --spacing-small: 4px;
  --spacing-medium: 16px;
  --spacing-large: 24px;
  
  /* 字体大小 */
  --font-size-small: 12px;
  --font-size-base: 14px;
  --font-size-medium: 16px;
  --font-size-large: 18px;
  
  /* 圆角 */
  --border-radius-small: 2px;
  --border-radius-base: 4px;
  --border-radius-large: 8px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  height: 100%;
  overflow: hidden;
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;
  font-size: var(--font-size-base);
  color: var(--color-text-regular);
  background-color: var(--color-white);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  height: 100vh;
  overflow: hidden;
}

/* 自定义 Element Plus 样式 */
.el-button {
  font-weight: normal;
}

.el-button--primary {
  --el-button-bg-color: var(--color-primary);
  --el-button-border-color: var(--color-primary);
}

.el-input__inner {
  --el-input-border-color: var(--color-border);
}

.el-card {
  --el-card-border-color: transparent;
  --el-card-border-radius: var(--border-radius-base);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
}
</style>
