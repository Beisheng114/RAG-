// 知识库管理中心 - 管理端脚本

(function() {
  'use strict';
  
  const API_BASE_URL = '/api';
  let adminToken = '';
  let nodeTypeChart = null;
  
  // DOM 元素引用
  let el = {};
  
  // 初始化 DOM 元素
  function initElements() {
    el = {
      loginPanel: document.getElementById('login-panel'),
      adminPanel: document.getElementById('admin-panel'),
      tokenInput: document.getElementById('admin-token-input'),
      loginBtn: document.getElementById('admin-login-btn'),
      logoutBtn: document.getElementById('logout-btn'),
      refreshBtn: document.getElementById('refresh-dashboard'),
      backupBtn: document.getElementById('run-backup'),
      restoreBtn: document.getElementById('run-restore'),
      restoreFile: document.getElementById('restore-file'),
      refreshing: document.getElementById('graph-refreshing'),
      resultWrap: document.getElementById('kg-admin-result'),
      resultText: document.getElementById('kg-admin-result-text'),
      kbVersion: document.getElementById('kb-version'),
      kbVectorIndex: document.getElementById('kb-vector-index'),
      kbCollection: document.getElementById('kb-collection'),
      kbTotalEntities: document.getElementById('kb-total-entities'),
      kbNodeTypeCount: document.getElementById('kb-node-type-count'),
      graphNodeCount: document.getElementById('graph-node-count'),
      graphEdgeCount: document.getElementById('graph-edge-count'),
      graphNodeLimit: document.getElementById('graph-node-limit'),
      chartCanvas: document.getElementById('node-type-chart')
    };
  }

function getHeaders() {
  return { 'X-Admin-Token': adminToken };
}

function showResult(data) {
  el.resultWrap.classList.remove('hidden');
  el.resultText.textContent = JSON.stringify(data, null, 2);
}

async function fetchWithAuth(url, options = {}) {
  const merged = {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...getHeaders()
    }
  };
  const res = await fetch(url, merged);
  const data = await res.json();
  if (!res.ok || data.success === false) {
    throw new Error(data.message || data.detail || `请求失败：${res.status}`);
  }
  return data;
}

function renderNodeTypeChart(nodeCounts) {
  const labels = Object.keys(nodeCounts || {});
  const values = Object.values(nodeCounts || {});

  if (!nodeTypeChart) {
    nodeTypeChart = new Chart(el.chartCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: '节点数量',
          data: values,
          backgroundColor: '#3b82f6'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
      }
    });
  } else {
    nodeTypeChart.data.labels = labels;
    nodeTypeChart.data.datasets[0].data = values;
    nodeTypeChart.update();
  }
}

function applyDashboard(data) {
  const version = data.version || {};
  const graph = data.graph || {};
  const nodeCounts = data.node_counts || {};

  el.kbVersion.textContent = version.version || '--';
  el.kbVectorIndex.textContent = version.vector_index_type || '--';
  el.kbCollection.textContent = version.qdrant_collection || '--';
  el.kbTotalEntities.textContent = version.total_entities ?? '--';
  el.kbNodeTypeCount.textContent = version.node_type_count ?? '--';

  el.graphNodeCount.textContent = graph.stats?.total_nodes ?? '--';
  el.graphEdgeCount.textContent = graph.stats?.total_edges ?? '--';
  el.graphNodeLimit.textContent = graph.node_limit ?? '--';

  renderNodeTypeChart(nodeCounts);
}

async function loadDashboard() {
  el.refreshing.classList.remove('hidden');
  try {
    const data = await fetchWithAuth(`${API_BASE_URL}/admin/dashboard`);
    applyDashboard(data.data || {});
    showResult(data);
  } finally {
    el.refreshing.classList.add('hidden');
  }
}

async function tryLogin() {
  const token = el.tokenInput.value.trim();
  if (!token) {
    alert('请输入登录密钥');
    return;
  }
  adminToken = token;
  try {
    await loadDashboard();
    el.loginPanel.classList.add('hidden');
    el.adminPanel.classList.remove('hidden');
  } catch (e) {
    adminToken = '';
    alert(`登录失败：${e.message}`);
  }
}

function logout() {
  adminToken = '';
  el.tokenInput.value = '';
  el.adminPanel.classList.add('hidden');
  el.loginPanel.classList.remove('hidden');
}

async function runBackup() {
  try {
    const data = await fetchWithAuth(`${API_BASE_URL}/admin/backup`, {
      method: 'POST'
    });
    showResult(data);
    alert('备份完成');
    await loadDashboard();
  } catch (e) {
    alert(`备份失败：${e.message}`);
  }
}

async function runRestore() {
  if (!el.restoreFile.files || !el.restoreFile.files[0]) {
    alert('请先选择备份 zip 文件');
    return;
  }

  const form = new FormData();
  form.append('backup_file', el.restoreFile.files[0]);

  try {
    const data = await fetchWithAuth(`${API_BASE_URL}/admin/restore`, {
      method: 'POST',
      body: form
    });
    showResult(data);
    alert('导入完成（已解压）');
  } catch (e) {
    alert(`导入失败：${e.message}`);
  }
}

// 初始化事件监听
function initEventListeners() {
  el.loginBtn.addEventListener('click', tryLogin);
  el.tokenInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      tryLogin();
    }
  });
  el.logoutBtn.addEventListener('click', logout);
  el.refreshBtn.addEventListener('click', loadDashboard);
  el.backupBtn.addEventListener('click', runBackup);
  el.restoreBtn.addEventListener('click', runRestore);
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
