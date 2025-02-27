<template>
  <div class="knowledge-manager">
    <div class="page-header">
      <h2 class="page-title">知识库管理</h2>
      <div class="header-actions">
        <el-input
          v-model="searchQuery"
          placeholder="搜索文件..."
          prefix-icon="Search"
          clearable
          style="width: 250px; margin-right: 10px;"
        />
        <el-button type="primary" :icon="Upload" @click="uploadDialogVisible = true">上传文件</el-button>
        <el-button :icon="Refresh" @click="fetchKnowledgeFiles">刷新</el-button>
      </div>
    </div>
    
    <div class="knowledge-content">
  
      
      <!-- API错误提示 -->
      <el-alert
        v-if="apiError"
        :title="'API错误: ' + apiError"
        type="error"
        :closable="true"
        show-icon
        style="margin-bottom: 15px;"
      />
      
      <!-- 加载中提示 -->
      <el-skeleton :loading="loadingKnowledgeFiles" animated :count="3" v-if="loadingKnowledgeFiles">
        <template #template>
          <div style="padding: 14px;">
            <el-skeleton-item variant="image" style="width: 100%; height: 100px;" />
            <div style="display: flex; align-items: center; margin-top: 16px;">
              <el-skeleton-item variant="circle" style="margin-right: 16px; width: 40px; height: 40px;" />
              <el-skeleton-item variant="p" style="width: 50%;" />
            </div>
            <div style="margin-top: 16px;">
              <el-skeleton-item variant="text" style="width: 30%;" />
              <el-skeleton-item variant="text" style="width: 50%; margin-top: 8px;" />
            </div>
          </div>
        </template>
      </el-skeleton>
      
      <!-- 文件类型筛选 -->
      <div class="filter-bar" v-if="!loadingKnowledgeFiles">
        <el-radio-group v-model="fileTypeFilter" size="small">
          <el-radio-button label="all">全部</el-radio-button>
          <el-radio-button label="pdf">PDF</el-radio-button>
          <el-radio-button label="doc">Word</el-radio-button>
          <el-radio-button label="txt">文本</el-radio-button>
          <el-radio-button label="other">其他</el-radio-button>
        </el-radio-group>
        
        <div class="view-options">
          <el-radio-group v-model="viewMode" size="small">
            <el-radio-button label="grid">
              <el-icon><Grid /></el-icon>
            </el-radio-button>
            <el-radio-button label="list">
              <el-icon><List /></el-icon>
            </el-radio-button>
          </el-radio-group>
        </div>
      </div>
      
      <!-- 空状态提示 -->
      <el-empty 
        v-if="!loadingKnowledgeFiles && filteredFiles.length === 0" 
        description="暂无文件" 
      >
        <el-button type="primary" @click="uploadDialogVisible = true">上传文件</el-button>
      </el-empty>
      
      <!-- 网格视图 -->
      <div v-if="!loadingKnowledgeFiles && filteredFiles.length > 0 && viewMode === 'grid'" class="file-grid">
        <el-card
          v-for="file in filteredFiles"
          :key="file.id"
          class="file-card"
          shadow="hover"
        >
          <div class="file-icon">
            <el-icon v-if="file.type.toLowerCase() === 'pdf'"><Document /></el-icon>
            <el-icon v-else-if="['doc', 'docx'].includes(file.type.toLowerCase())"><Document /></el-icon>
            <el-icon v-else-if="file.type.toLowerCase() === 'txt'"><Document /></el-icon>
            <el-icon v-else><Document /></el-icon>
          </div>
          
          <div class="file-info">
            <div class="file-name-container">
              <el-tooltip :content="file.name" placement="top" :show-after="500">
                <div class="file-name">{{ file.name }}</div>
              </el-tooltip>
            </div>
            
            <div class="file-meta">
              <span>{{ file.type.toUpperCase() }}</span>
              <span>{{ formatFileSize(file.size) }}</span>
            </div>
            
            <div class="file-date">
              {{ formatDate(file.created_at) }}
            </div>
            
            <div class="file-status">
              <el-tag v-if="file.status === 'success'" type="success" size="small">处理完成</el-tag>
              <el-tag v-else-if="file.status === 'processing'" type="warning" size="small">处理中</el-tag>
              <el-tag v-else-if="file.status === 'error'" type="danger" size="small">处理失败</el-tag>
              <el-tag v-else type="info" size="small">未知状态</el-tag>
            </div>
            
            <div class="file-actions">
              <el-button 
                type="danger" 
                :icon="Delete" 
                circle 
                size="small"
                @click="handleDelete(file)"
              />
            </div>
          </div>
        </el-card>
      </div>
      
      <!-- 列表视图 -->
      <el-table
        v-if="!loadingKnowledgeFiles && filteredFiles.length > 0 && viewMode === 'list'"
        :data="filteredFiles"
        style="width: 100%"
        border
      >
        <el-table-column label="文件名" min-width="300">
          <template #default="scope">
            <div class="file-name-row">
              <el-icon v-if="scope.row.type.toLowerCase() === 'pdf'"><Document /></el-icon>
              <el-icon v-else-if="['doc', 'docx'].includes(scope.row.type.toLowerCase())"><Document /></el-icon>
              <el-icon v-else-if="scope.row.type.toLowerCase() === 'txt'"><Document /></el-icon>
              <el-icon v-else><Document /></el-icon>
              
              <el-tooltip :content="scope.row.name" placement="top" :show-after="500">
                <span class="table-file-name">{{ scope.row.name }}</span>
              </el-tooltip>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column prop="type" label="类型" width="100">
          <template #default="scope">
            {{ scope.row.type.toUpperCase() }}
          </template>
        </el-table-column>
        
        <el-table-column label="大小" width="100">
          <template #default="scope">
            {{ formatFileSize(scope.row.size) }}
          </template>
        </el-table-column>
        
        <el-table-column label="上传时间" width="180">
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        
        <el-table-column label="状态" width="120">
          <template #default="scope">
            <el-tag v-if="scope.row.status === 'success'" type="success" size="small">处理完成</el-tag>
            <el-tag v-else-if="scope.row.status === 'processing'" type="warning" size="small">处理中</el-tag>
            <el-tag v-else-if="scope.row.status === 'error'" type="danger" size="small">处理失败</el-tag>
            <el-tag v-else type="info" size="small">未知状态</el-tag>
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="100" fixed="right">
          <template #default="scope">
            <el-button 
              type="danger" 
              :icon="Delete" 
              circle 
              size="small"
              @click="handleDelete(scope.row)"
            />
          </template>
        </el-table-column>
      </el-table>
    </div>
    
    <!-- 上传文件对话框 -->
    <el-dialog
      v-model="uploadDialogVisible"
      title="上传文件"
      width="500px"
    >
      <el-upload
        class="upload-demo"
        drag
        multiple
        :auto-upload="false"
        :limit="10"
        v-model:file-list="fileList"
        :accept="acceptedFileTypes"
        :on-change="handleFileChange"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽文件到此处或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持PDF、Word、Excel、TXT、Markdown、EPUB等文档格式
          </div>
        </template>
      </el-upload>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="uploadDialogVisible = false">取消</el-button>
          <el-button 
            type="primary" 
            @click="handleUpload" 
            :loading="uploadLoading"
          >
            上传
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, inject, onMounted, computed } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Delete, Upload, View, Search, Refresh, Grid, List } from '@element-plus/icons-vue';
import axios from 'axios';
import API_CONFIG from '../api/config';
import { Document } from '@element-plus/icons-vue';
import { UploadFilled } from '@element-plus/icons-vue';

export default {
  name: 'KnowledgeManager',
  setup() {
    // 从父组件注入知识库文件列表
    const knowledgeFiles = inject('knowledgeFiles', ref([]));
    const loadingKnowledgeFiles = inject('loadingKnowledgeFiles', ref(false));
    const fetchKnowledgeFiles = inject('fetchKnowledgeFiles', () => {
      console.log('KnowledgeManager: 使用默认的fetchKnowledgeFiles函数');
      return Promise.resolve([]);
    });
    
    const searchQuery = ref('');
    const fileTypeFilter = ref('all');
    const viewMode = ref('grid');
    const uploadDialogVisible = ref(false);
    const fileList = ref([]);
    const uploadLoading = ref(false);
    const apiError = ref(null);
    const showDebug = ref(false); // 控制是否显示调试信息
    
    // 定义接受的文件类型
    const acceptedFileTypes = '.pdf,.doc,.docx,.csv,.xlsx,.xls,.epub,.md,.txt';
    const allowedFileExtensions = ['pdf', 'doc', 'docx', 'csv', 'xlsx', 'xls', 'epub', 'md', 'txt'];
    
    // 添加调试日志
    console.log('KnowledgeManager组件初始化');
    
    // 过滤后的文件列表
    const filteredFiles = computed(() => {
      let files = knowledgeFiles.value || [];
      
      // 搜索过滤
      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase();
        files = files.filter(file => 
          file.name.toLowerCase().includes(query)
        );
      }
      
      // 类型过滤
      if (fileTypeFilter.value !== 'all') {
        files = files.filter(file => {
          const type = file.type.toLowerCase();
          
          if (fileTypeFilter.value === 'pdf') {
            return type === 'pdf';
          } else if (fileTypeFilter.value === 'doc') {
            return ['doc', 'docx'].includes(type);
          } else if (fileTypeFilter.value === 'txt') {
            return type === 'txt';
          } else if (fileTypeFilter.value === 'other') {
            return !['pdf', 'doc', 'docx', 'txt'].includes(type);
          }
          
          return true;
        });
      }
      
      return files;
    });
    
    // 组件挂载时获取文件列表
    onMounted(() => {
      console.log('KnowledgeManager组件已挂载');
      
      // 如果父组件没有提供文件列表，则自己获取
      if (!knowledgeFiles.value || knowledgeFiles.value.length === 0) {
        console.log('KnowledgeManager: 父组件未提供文件列表，自己获取');
        fetchKnowledgeFiles();
      } else {
        console.log('KnowledgeManager: 使用父组件提供的文件列表:', knowledgeFiles.value.length, '个文件');
      }
    });
    
    // 处理文件变更，验证文件类型
    const handleFileChange = (file, fileList) => {
      const extension = file.name.split('.').pop().toLowerCase();
      
      if (!allowedFileExtensions.includes(extension)) {
        ElMessage.error(`不支持的文件类型: ${extension}，请上传 ${allowedFileExtensions.join(', ')} 格式的文件`);
        
        // 从文件列表中移除不支持的文件
        const index = fileList.indexOf(file);
        if (index !== -1) {
          fileList.splice(index, 1);
        }
        
        return false;
      }
      
      return true;
    };
    
    // 上传文件
    const handleUpload = async () => {
      if (fileList.value.length === 0) {
        ElMessage.warning('请先选择要上传的文件');
        return;
      }
      
      // 验证文件类型
      const invalidFiles = fileList.value.filter(file => {
        const extension = file.name.split('.').pop().toLowerCase();
        return !allowedFileExtensions.includes(extension);
      });
      
      if (invalidFiles.length > 0) {
        const invalidFileNames = invalidFiles.map(f => f.name).join(', ');
        ElMessage.error(`不支持的文件类型: ${invalidFileNames}`);
        return;
      }
      
      try {
        uploadLoading.value = true;
        apiError.value = null;
        
        // 创建FormData对象
        const formData = new FormData();
        
        // 检查后端API是否期望单个文件或多个文件
        if (fileList.value.length === 1) {
          // 单个文件上传
          formData.append('file', fileList.value[0].raw);
        } else {
          // 多个文件上传 - 尝试两种可能的字段名
          fileList.value.forEach(file => {
            formData.append('files', file.raw);
          });
        }
        
        // 打印FormData内容以便调试
        console.log('FormData内容:');
        for (let [key, value] of formData.entries()) {
          console.log(`${key}: ${value instanceof File ? value.name : value}`);
        }
        
        console.log('开始上传文件:', fileList.value.length, '个文件');
        
        // 发送上传请求
        const response = await axios.post(
          `${API_CONFIG.baseURL}${API_CONFIG.paths.upload}`, 
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            // 添加上传进度处理
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              console.log(`上传进度: ${percentCompleted}%`);
            }
          }
        );
        
        console.log('上传响应:', response);
        
        if (response.status === 200 || response.status === 201) {
          // 关闭对话框
          uploadDialogVisible.value = false;
          
          // 清空文件列表
          fileList.value = [];
          
          // 显示上传成功消息，但提示用户文件正在处理中
          ElMessage({
            type: 'success',
            message: '文件已上传成功，正在后台处理中，请稍后刷新查看',
            duration: 5000
          });
          
          // 设置一个定时器，延迟刷新文件列表
          setTimeout(() => {
            fetchKnowledgeFiles();
          }, 3000); // 3秒后刷新一次
          
          // 再设置一个定时器，再次刷新文件列表
          setTimeout(() => {
            fetchKnowledgeFiles();
            ElMessage.info('文件处理可能需要一些时间，如未看到新文件，请手动刷新');
          }, 10000); // 10秒后再次刷新
        } else {
          ElMessage.error('上传文件失败: ' + (response.data?.detail || '未知错误'));
        }
      } catch (error) {
        console.error('上传文件失败:', error);
        
        let errorMessage = '上传文件失败';
        
        if (error.response) {
          console.error('错误响应:', error.response);
          
          if (error.response.data) {
            if (typeof error.response.data === 'object') {
              if (error.response.data.detail) {
                errorMessage += ': ' + error.response.data.detail;
              } else if (error.response.data.message) {
                errorMessage += ': ' + error.response.data.message;
              } else {
                // 打印完整的错误响应数据
                errorMessage += ': ' + JSON.stringify(error.response.data);
              }
            } else if (typeof error.response.data === 'string') {
              errorMessage += ': ' + error.response.data;
            }
          } else if (error.message) {
            errorMessage += ': ' + error.message;
          }
        }
        
        apiError.value = errorMessage;
        ElMessage.error(errorMessage);
      } finally {
        uploadLoading.value = false;
      }
    };
    
    // 删除文件
    const handleDelete = async (file) => {
      try {
        await ElMessageBox.confirm(
          `确定要删除文件 "${file.name}" 吗？`,
          '删除确认',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        );
        
        console.log('删除文件:', file);
        
        const response = await axios.delete(`${API_CONFIG.baseURL}${API_CONFIG.paths.delete}/${file.id}`);
        
        console.log('删除响应:', response);
        
        if (response.data && response.data.success) {
          ElMessage.success('文件删除成功');
          
          // 重新获取文件列表
          await fetchKnowledgeFiles();
        } else {
          ElMessage.error('文件删除失败: ' + (response.data?.message || '未知错误'));
        }
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除文件失败:', error);
          ElMessage.error('删除文件失败: ' + (error.response?.data?.detail || error.message));
        }
      }
    };
    
    // 格式化文件大小
    const formatFileSize = (size) => {
      if (!size && size !== 0) return '未知大小';
      
      if (size < 1024) {
        return size + ' B';
      } else if (size < 1024 * 1024) {
        return (size / 1024).toFixed(2) + ' KB';
      } else if (size < 1024 * 1024 * 1024) {
        return (size / (1024 * 1024)).toFixed(2) + ' MB';
      } else {
        return (size / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
      }
    };
    
    // 格式化日期
    const formatDate = (dateString) => {
      if (!dateString) return '未知时间';
      
      try {
        const date = new Date(dateString);
        
        // 检查日期是否有效
        if (isNaN(date.getTime())) {
          return '无效日期';
        }
        
        // 格式化为本地日期时间字符串
        return date.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        });
      } catch (error) {
        console.error('日期格式化错误:', error);
        return '日期错误';
      }
    };
    
    return {
      knowledgeFiles,
      loadingKnowledgeFiles,
      searchQuery,
      fileTypeFilter,
      viewMode,
      uploadDialogVisible,
      fileList,
      uploadLoading,
      apiError,
      filteredFiles,
      handleUpload,
      handleDelete,
      fetchKnowledgeFiles,
      Delete,
      Upload,
      View,
      Search,
      Refresh,
      Grid,
      List,
      formatFileSize,
      formatDate,
      Document,
      UploadFilled,
      acceptedFileTypes,
      handleFileChange,
      showDebug
    };
  }
};
</script>

<style scoped>
.knowledge-manager {
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

.header-actions {
  display: flex;
  gap: var(--spacing-small);
  align-items: center;
}

.knowledge-content {
  flex: 1;
  padding: var(--spacing-medium);
  overflow: auto;
}

.filter-bar {
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 网格视图样式 */
.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
}

.file-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.file-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 40px;
  color: #409EFF;
  margin-bottom: 10px;
}

.file-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.file-name-container {
  width: 100%;
}

.file-name {
  font-weight: 500;
  font-size: 16px;
  color: #303133;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}

.file-meta {
  display: flex;
  gap: 10px;
  font-size: 12px;
  color: #909399;
}

.file-date {
  font-size: 12px;
  color: #909399;
}

.file-status {
  margin-top: 5px;
}

.file-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
}

/* 列表视图样式 */
.file-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.table-file-name {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 400px;
  display: inline-block;
}

.el-upload {
  width: 100%;
}

.el-upload-dragger {
  width: 100%;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-small);
}
</style> 