// 系统配置预览页面 - 配置管理脚本
// 用于查看后端当前配置（敏感字段已脱敏）

(function() {
  'use strict';
  
  const API_BASE_URL = '/api';
  let adminToken = '';
  
  // DOM 元素引用
  let loginPanel, configPanel, tokenInput, loginBtn, refreshBtn, configText;
  
  // 初始化 DOM 元素
  function initElements() {
    loginPanel = document.getElementById('config-login-panel');
    configPanel = document.getElementById('config-panel');
    tokenInput = document.getElementById('config-token-input');
    loginBtn = document.getElementById('config-login-btn');
    refreshBtn = document.getElementById('refresh-config');
    configText = document.getElementById('config-text');
  }
  
  async function loadConfig() {
    const res = await fetch(`${API_BASE_URL}/config/preview`, {
      headers: { 'X-Admin-Token': adminToken }
    });
    const data = await res.json();
    if (!res.ok || data.success === false) {
      throw new Error(data.message || data.detail || `请求失败：${res.status}`);
    }
    configText.textContent = JSON.stringify(data.data?.config || {}, null, 2);
    return data;
  }
  
  async function doLoginAndLoad() {
    const token = tokenInput.value.trim();
    if (!token) {
      alert('请输入管理员密钥');
      return;
    }
  
    adminToken = token;
    try {
      await loadConfig();
      loginPanel.classList.add('hidden');
      configPanel.classList.remove('hidden');
    } catch (e) {
      adminToken = '';
      alert(`认证失败：${e.message}`);
    }
  }
  
  // 初始化事件监听
  function initEventListeners() {
    loginBtn.addEventListener('click', doLoginAndLoad);
    tokenInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        doLoginAndLoad();
      }
    });
    refreshBtn.addEventListener('click', async () => {
      try {
        await loadConfig();
      } catch (e) {
        alert(`刷新失败：${e.message}`);
      }
    });
  }
  
  // DOM 加载完成后初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      initElements();
      initEventListeners();
    });
  } else {
    initElements();
    initEventListeners();
  }
})();
