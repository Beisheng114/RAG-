// 知识图谱提取页面 - 图谱抽取引擎脚本
// 用于执行船舶维修知识图谱抽取任务

(function() {
  'use strict';
  
  const API_BASE_URL = '/api';
  
  // DOM 元素引用（延迟初始化）
  let startBtn, cancelBtn, clearLogBtn, statusText, taskIdBox, taskIdText, logOutput, uploadFileInput, llmProviderSelect;
  
  // 允许上传的文件扩展名
  const ALLOWED_UPLOAD_EXTENSIONS = ['.md', '.txt', '.pdf', '.png', '.jpg', '.jpeg'];
  
  // 全局状态
  let currentTaskId = null;
  let logIndex = 0;
  let pollTimer = null;
  
  /**
   * 初始化 DOM 元素引用
   * @returns {boolean} 是否成功初始化
   */
  function initElements() {
    startBtn = document.getElementById('start-btn');
    cancelBtn = document.getElementById('cancel-btn');
    clearLogBtn = document.getElementById('clear-log-btn');
    statusText = document.getElementById('status-text');
    taskIdBox = document.getElementById('task-id-box');
    taskIdText = document.getElementById('task-id-text');
    logOutput = document.getElementById('log-output');
    uploadFileInput = document.getElementById('upload-file');
    llmProviderSelect = document.getElementById('llm-provider');

    // 仅“开始抽取”按钮为必需元素。
    // 之前这里强依赖 status-text，但 index.html 没有该元素，导致事件监听器无法绑定，按钮点击无响应。
    return !!startBtn;
  }
  
  /**
   * 设置状态文本和样式
   * @param {string} text - 状态文本
   * @param {string} cssClass - CSS 类名
   */
  function setStatus(text, cssClass = 'text-slate-700') {
    if (statusText) {
      statusText.className = `font-semibold ${cssClass}`;
      statusText.textContent = text;
    }
  }
  
  /**
   * 加载 LLM 选项
   */
  async function loadLlmOptions() {
    const configPath = (document.getElementById('config-path')?.value || 'config.yaml').trim() || 'config.yaml';
    try {
      const url = `${API_BASE_URL}/kg/extract/options?config_path=${encodeURIComponent(configPath)}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data.success) {
        return;
      }
  
      const currentValue = llmProviderSelect.value;
      llmProviderSelect.innerHTML = '<option value="">跟随配置文件默认值</option>';
  
      if (data.llm_providers && Array.isArray(data.llm_providers)) {
        data.llm_providers.forEach(provider => {
          const option = document.createElement('option');
          option.value = provider;
          option.textContent = provider;
          llmProviderSelect.appendChild(option);
        });
      }
  
      if (currentValue && data.llm_providers?.includes(currentValue)) {
        llmProviderSelect.value = currentValue;
      } else if (data.default_provider && data.llm_providers?.includes(data.default_provider)) {
        llmProviderSelect.value = data.default_provider;
      }
    } catch (e) {
      console.error('加载 LLM 选项失败:', e);
    }
  }
  
  /**
   * 追加日志
   * @param {string[]} lines - 日志行
   */
  function appendLogs(lines) {
    if (!Array.isArray(lines) || lines.length === 0) return;
    const text = lines.join('\n') + '\n';
    logOutput.textContent += text;
    logOutput.scrollTop = logOutput.scrollHeight;
  }
  
  /**
   * 停止轮询
   */
  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }
  
  /**
   * 调度轮询
   * @param {number} delay - 延迟时间（毫秒）
   */
  function schedulePolling(delay = 1200) {
    stopPolling();
    pollTimer = setTimeout(pollStatus, delay);
  }
  
  /**
   * 轮询任务状态
   */
  async function pollStatus() {
    if (!currentTaskId) return;
    try {
      const url = `${API_BASE_URL}/kg/extract/status?task_id=${encodeURIComponent(currentTaskId)}&last_index=${logIndex}`;
      const res = await fetch(url);
      const data = await res.json();
  
      if (!data.success || !data.task) {
        setStatus(`查询失败：${data.message || '未知错误'}`, 'text-rose-600');
        schedulePolling(1500);
        return;
      }
  
      const task = data.task;
      appendLogs(task.new_logs || []);
      logIndex = typeof task.next_index === 'number' ? task.next_index : logIndex;
  
      if (task.status === 'running' || task.status === 'pending') {
        setStatus(task.status === 'running' ? '运行中' : '排队中', 'text-amber-600');
        schedulePolling(1000);
        return;
      }
  
      if (task.status === 'completed') {
        setStatus('已完成', 'text-emerald-600');
      } else if (task.status === 'failed') {
        setStatus(`失败：${task.error || '未知错误'}`, 'text-rose-600');
      } else if (task.status === 'cancelled') {
        setStatus('已取消', 'text-slate-600');
      } else {
        setStatus(task.status || '未知状态', 'text-slate-700');
      }
  
      startBtn.disabled = false;
      cancelBtn.disabled = true;
      stopPolling();
    } catch (err) {
      setStatus(`轮询异常：${err}`, 'text-rose-600');
      schedulePolling(1500);
    }
  }
  
  /**
   * 初始化事件监听
   */
  function initEventListeners() {
    // 检查元素是否存在，不存在则延迟初始化
    if (!initElements()) {
      setTimeout(initEventListeners, 100);
      return;
    }
    
    startBtn.addEventListener('click', async () => {
      stopPolling();
      currentTaskId = null;
      logIndex = 0;
      logOutput.textContent = '';
  
      const selectedFile = uploadFileInput.files && uploadFileInput.files[0];
      if (!selectedFile) {
        setStatus('请先选择上传文件', 'text-rose-600');
        return;
      }
  
      const fileName = String(selectedFile.name || '').toLowerCase();
      const dotIndex = fileName.lastIndexOf('.');
      const ext = dotIndex >= 0 ? fileName.slice(dotIndex) : '';
      if (!ALLOWED_UPLOAD_EXTENSIONS.includes(ext)) {
        setStatus('文件类型不支持，仅允许 .md/.txt/.pdf/.png/.jpg', 'text-rose-600');
        return;
      }
  
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('output_dir', document.getElementById('output-dir')?.value.trim() || 'csv_generate');
      form.append('config_path', document.getElementById('config-path')?.value.trim() || 'config.yaml');
      form.append('book_name', document.getElementById('book-name')?.value.trim());
      form.append('llm_provider', llmProviderSelect?.value.trim());
      form.append('llm_api_key', document.getElementById('llm-api-key')?.value.trim());
      form.append('ocr_provider', document.getElementById('ocr-provider')?.value.trim());
      form.append('ocr_endpoint', document.getElementById('ocr-endpoint')?.value.trim());
      form.append('ocr_api_key', document.getElementById('ocr-api-key')?.value.trim());
  
      setStatus('启动中...', 'text-amber-600');
      startBtn.disabled = true;
      cancelBtn.disabled = false;
  
      try {
        const res = await fetch(`${API_BASE_URL}/kg/extract/start`, { method: 'POST', body: form });
        const data = await res.json();
  
        if (!data.success || !data.task_id) {
          setStatus(`启动失败：${data.message || '未知错误'}`, 'text-rose-600');
          startBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }
  
        currentTaskId = data.task_id;
        taskIdBox.classList.remove('hidden');
        taskIdText.textContent = currentTaskId;
        setStatus('已启动，等待执行...', 'text-amber-600');
        schedulePolling(500);
      } catch (err) {
        setStatus(`启动异常：${err}`, 'text-rose-600');
        startBtn.disabled = false;
        cancelBtn.disabled = true;
      }
    });
  
    cancelBtn.addEventListener('click', async () => {
      if (!currentTaskId) return;
      try {
        const form = new FormData();
        form.append('task_id', currentTaskId);
        const res = await fetch(`${API_BASE_URL}/kg/extract/cancel`, { method: 'POST', body: form });
        const data = await res.json();
        if (data.success) {
          setStatus('正在取消...', 'text-slate-600');
        } else {
          setStatus(`取消失败：${data.message || '未知错误'}`, 'text-rose-600');
        }
      } catch (err) {
        setStatus(`取消异常：${err}`, 'text-rose-600');
      }
    });
  
    clearLogBtn.addEventListener('click', () => {
      logOutput.textContent = '';
      logIndex = 0;
    });
  
    const configPathInput = document.getElementById('config-path');
    if (configPathInput) {
      configPathInput.addEventListener('blur', loadLlmOptions);
    }
    loadLlmOptions();
  }
  
  // DOM 加载完成后初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEventListeners);
  } else {
    initEventListeners();
  }
})();
