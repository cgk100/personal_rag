<template>
  <div class="sidebar-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapsed ? '64px' : '200px'" class="kb-sidebar">
      <div class="sidebar-header">
        <h2 class="app-title" v-if="!isCollapsed">Ai知识库</h2>
        <el-icon v-else class="collapsed-icon"><Document /></el-icon>
      </div>
      
      <el-menu 
        :default-active="activeMenu" 
        class="sidebar-menu"
        :collapse="isCollapsed"
        @select="handleMenuSelect"
      >
        <el-menu-item index="knowledge">
          <el-icon><Document /></el-icon>
          <template #title>知识库管理</template>
        </el-menu-item>

        <el-menu-item index="chat">
          <el-icon><ChatDotRound /></el-icon>
          <template #title>Ai对话</template>
        </el-menu-item>
        
        <el-menu-item index="settings">
          <el-icon><Setting /></el-icon>
          <template #title>系统设置</template>
        </el-menu-item>
      </el-menu>
      
      <div class="sidebar-footer">
        <div class="app-version" v-if="!isCollapsed">版本: 1.0.0</div>
        <el-button 
          type="text" 
          class="collapse-button" 
          @click="toggleCollapse"
          :title="isCollapsed ? '展开菜单' : '收起菜单'"
        >
          <el-icon>
            <component :is="isCollapsed ? 'ArrowRight' : 'ArrowLeft'" />
          </el-icon>
        </el-button>
      </div>
    </el-aside>
  </div>
</template>

<script>
import { ref } from 'vue';
import { ChatDotRound, Document, Setting, ArrowRight, ArrowLeft } from '@element-plus/icons-vue';

export default {
  name: 'SideNavigation',
  props: {
    defaultActive: {
      type: String,
      default: 'knowledge'
    }
  },
  setup(props, { emit }) {
    const activeMenu = ref(props.defaultActive);
    const isCollapsed = ref(false);
    
    const handleMenuSelect = (key) => {
      activeMenu.value = key;
      emit('menu-select', key);
    };
    
    const toggleCollapse = () => {
      isCollapsed.value = !isCollapsed.value;
      // 通知父组件侧边栏状态已更改
      emit('collapse-change', isCollapsed.value);
    };
    
    return {
      activeMenu,
      isCollapsed,
      handleMenuSelect,
      toggleCollapse,
      ChatDotRound,
      Document,
      Setting,
      ArrowRight,
      ArrowLeft
    };
  }
};
</script>

<style scoped>
/* 侧边栏容器 */
.sidebar-container {
  position: relative;
  height: 100%;
}

/* 左侧导航栏样式 */
.kb-sidebar {
  background-color: var(--color-white);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  height: 100%;
  transition: width 0.3s ease;
}

.sidebar-header {
  padding: var(--spacing-medium);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: center;
  align-items: center;
  height: 56px;
}

.app-title {
  font-size: var(--font-size-large);
  color: var(--color-text-primary);
  font-weight: 500;
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.collapsed-icon {
  font-size: 24px;
  color: var(--color-primary);
}

.sidebar-menu {
  flex: 1;
  border-right: none;
}

.sidebar-footer {
  padding: var(--spacing-medium);
  border-top: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  font-size: var(--font-size-small);
  text-align: center;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
}

.app-version {
  flex: 1;
}

.collapse-button {
  position: absolute;
  right: -12px;
  top: 50%;
  transform: translateY(-50%);
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background-color: var(--color-white);
  border: 1px solid var(--color-border);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  z-index: 10;
  padding: 0;
}

.collapse-button:hover {
  background-color: var(--color-primary-light);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .kb-sidebar {
    width: 64px !important;
  }
  
  .collapse-button {
    display: none;
  }
}
</style> 