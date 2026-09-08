# 知识图谱可视化优化 + 全站前端风格统一

- 日期：2026-09-08
- 分支：`optimize/kg-visualization`
- 状态：已确认（待用户评审）
- 范围：前端图谱渲染层重写 + 交互增强 + 后端小改（分页/邻居接口）+ 全站设计令牌统一

## 1. 背景与现状

船舶故障诊断问答 RAG 系统（FastAPI + 静态前端）的图谱可视化存在以下问题：

**交互弱**
- 点击节点仅 `console.log`，无详情查看、邻居高亮、聚焦定位（`static/index.html` `renderGraph`）
- 「加载更多」按钮为假实现（弹窗提示开发中）

**视觉糙**
- 无节点类型图例；边标签全部平铺显示导致杂乱
- 统计卡片直接输出原始字段名（`key: value`）
- 统计饼图颜色为随机数组切片，与节点着色不一致

**规模受限**
- UI 滑块上限 350 节点并警告性能问题；vis-network 物理模拟抖动

**风格分裂（全站）**
- 主页面：自定义 CSS（`static/css/index.css` 2734 行），底色 `#F0F8FF`、28px 大圆角、主色 blue-500
- 其余 4 页（home/config/extract/kg_admin）：Tailwind utility，底色 slate-50/100、圆角 12–16px、主色 blue-600
- 死资源：`static/index.css`（2550 行，无人引用）、`font-awesome-6.0.0.min.css`（无人引用）

**技术栈事实**
- `static/libs/echarts.min.js` 为 5.5.0，已在 index.html 加载，graph series / `emphasis.focus` / legend 交互 / `labelLayout.hideOverlap` 全部可用
- vis-network 仅 index.html 使用（backup 文件不计）
- 后端 `query_graph`（`ragmain.py:631`）node_limit clamp 50–500；无节点详情/邻居接口
- 测试为 pytest 冒烟模式（`tests/test_smoke.py`，不依赖外部服务）

## 2. 目标（用户确认的四个方向 + 风格统一）

1. 交互体验：节点详情侧栏、邻居高亮、双击展开邻居、搜索定位、重布局
2. 视觉效果：图例过滤、统一配色、边标签降噪、布局稳定
3. 性能与规模：上限 350→500、大图自动降噪
4. 信息展示：详情侧栏（描述/系统/关联关系列表）、统计卡片结构化
5. 配套：真分页「加载更多」、导出 PNG
6. 全站：设计令牌统一视觉，清理死资源

## 3. 方案选型

| 方案 | 结论 |
|---|---|
| A. 深度调优 vis-network | 备选，未选：库已停维护，交互需手写大量代码 |
| **B. ECharts 重写渲染层（选定）** | 零新增依赖，图例/邻接高亮/tooltip/缩放开箱即得，Canvas 千级节点性能更稳 |
| C. 引入 AntV G6 | 否决：需新增离线依赖入库，YAGNI |

代码结构上用户明确**不拆分** index.html（图谱逻辑保持内联）。

## 4. 架构

```
index.html（图谱视图）
 ├─ 查询面板（保留：关键词/实体类型/系统/节点上限滑块 50→500）
 ├─ #graph-container → echarts.init()（一次初始化，setOption 增量更新）
 ├─ 节点详情侧栏（新增：覆盖在画布右侧的 drawer）
 ├─ 画布工具栏（新增：重布局/适配视野/边标签开关/导出 PNG）
 └─ toast 容器（新增：非阻塞提示，替代 alert）

后端
 ├─ POST /api/graph/query        + offset 参数（分页）
 └─ GET /api/graph/node/{id}/neighbors  （新增，展开邻居）
```

移除 index.html 中 vis-network 的 `<script>/<link>` 引用（`static/libs/vis-network.*` 文件保留，不影响 backup 文件）。

## 5. 渲染层设计（ECharts graph series）

- **布局**：`layout: 'force'`，repulsion 180 / edgeLength 50–120 / gravity 0.1；节点 >500 关闭布局动画
- **categories**：按节点类型动态生成；颜色取自统一语义色板（`--kg-*`，见第 9 节）→ 图例色 = 节点色 = 饼图色
- **symbol 按类型**：Equipment `rect`、Component `circle`、Fault `diamond`、FaultReason `triangle`、MaintenanceAction `pin`、SafetyNotice `roundRect`、KnowledgeSource `square`
- **节点规模**：Equipment 45 / Component 35 / 其余 28（symbolSize），度数加权可选
- **节点标签**：>15 字截断；悬停 tooltip 显示全名 + 类型 + 系统 + 描述
- **边标签**：默认隐藏；节点数 ≤300 或用户开工具栏开关时显示，并开启 `labelLayout: {hideOverlap: true}`
- **邻接高亮**：`emphasis: {focus: 'adjacency'}`，`blurScope: 'coordinateSystem'`
- **交互**：`roam: true`（缩放 + 拖拽）
- **实例管理**：init 一次，查询用 `setOption` 合并（追加模式 `notMerge: false`）；tab 隐藏时跳过 resize，切回恢复
- **状态缓存**：`currentNodes/currentEdges`（原始数据），支撑详情侧栏关联列表与增量合并

## 6. 交互设计

| 操作 | 行为 |
|---|---|
| 单击节点 | 右侧滑出详情侧栏：名称、类型徽章、所属系统、描述、关联关系列表（方向 + 关系类型 + 对端节点；点击列表项定位高亮对端节点） |
| 双击节点 | 调 `GET /api/graph/node/{id}/neighbors` 拉取一度邻居（limit 20）增量入图（按 id 去重），聚焦该节点；节点上显示 loading |
| 点击图例 | 按类型过滤显隐（ECharts legend 内置行为） |
| 悬停节点 | 邻接高亮 + tooltip |
| 工具栏 | 重布局（重触发 force）/ 适配视野 / 边标签开关 / 导出 PNG |
| 「加载更多」 | 真分页：`offset += node_limit` 追加查询；仅浏览模式（无关键词）可用；`has_more=false` 时禁用 |
| 详情侧栏数据 | 前端 `currentNodes/currentEdges` 缓存提取，无额外请求 |

## 7. 后端改动

**`ragmain.py`**
- `query_graph()` 新增 `offset: int = 0`（clamp ≥0）：三种 browse Cypher 增加 `SKIP $offset`（`WITH n, r, m SKIP $offset LIMIT $lim`）；有关键词的查询模式不分页
- `stats` 增加 `has_more: bool`（浏览模式返回边数达到 browse_edge_lim 时为 true）
- 新增 `get_node_neighbors(node_id, limit=20)`：`MATCH (n)-[r]-(m) WHERE id(n) = $nid RETURN n, r, m LIMIT $lim`，复用 `node_to_item` / 去重逻辑，返回 (nodes, edges)

**`routers/graph_routes.py`**
- `GraphQueryRequest` 增加 `offset: int = 0`
- 新增 `GET /graph/node/{node_id}/neighbors`，复用 `GraphQueryResponse` 模型

**`services/graph_service.py`**
- `query_graph_data()` 透传 offset
- 新增 `get_node_neighbors_data()`

## 8. 错误处理

- 查询 / 展开邻居 / 分页失败：画布内浮动 toast（非阻塞），替换现有全页 `alert`
- 展开邻居：双击的节点进入 loading 态（旋转图标）
- 图谱 tab 隐藏时跳过 `resize`，切回时恢复，避免 0 尺寸初始化异常

## 9. 全站风格统一（设计令牌）

新建 `static/css/theme.css`，全站五个页面引用：

```css
:root {
  --color-primary: #2563eb;    /* blue-600，多数页面已用 */
  --color-primary-hover: #1d4ed8;
  --color-page-bg: #F0F8FF;    /* 统一为主页面浅蓝 */
  --color-text: #1e293b;
  --radius-card: 16px;
  --radius-btn: 10px;
  --shadow-card: 0 1px 2px rgba(0, 0, 0, .05);
  /* 图谱类型语义色板（= getNodeColor = 图例 = 饼图） */
  --kg-equipment: #3b82f6; --kg-component: #10b981; --kg-fault: #ef4444;
  --kg-faultreason: #f59e0b; --kg-maintenanceaction: #8b5cf6;
  --kg-safetynotice: #ec4899; --kg-knowledgesource: #6b7280; ...
}
```

**各页面改动**
1. 4 个小页面：引入 theme.css；底色统一浅蓝；卡片圆角对齐 16px；页头结构统一（标题 + 副标题 + 右上角返回按钮组）
2. index.html：`css/index.css` 硬编码色替换为变量引用（改值不改结构）；图谱新组件直接用变量；ECharts categories 颜色由 JS 读 `--kg-*`
3. 主页面卡片圆角 28px → 16px（`--radius-card`）

**明确不做**：全站 Tailwind utility 化重写（主页面 2700+ 行 CSS 迁移风险远超收益）。

**清理项**（实现阶段 grep 验证后执行）
- 删除 `static/index.css`（死文件，确认无引用后）
- 移除 index.html 对 `css/home.css`、`css/config.css`、`css/kg_admin.css`、`css/extract.css` 的引用（先确认无类名依赖）
- 删除 `libs/font-awesome-6.0.0.min.css` 引用（无人使用）

## 10. 视觉统一细节（图谱页）

- 统一色板：`getNodeColor` 为唯一色源，饼图 `createNodeTypeChart` 按类型取同色
- 类型名中文化：图例/详情/统计统一映射（Equipment→设备 等，复用现有 entity-type 映射）
- 统计卡片：`updateGraphStats` 由原始 `key: value` 改为结构化卡片（当前节点数 / 关系数 / 类型数）

## 11. 测试

**后端 pytest**（跟随 test_smoke.py 无外部依赖模式，mock driver）：
- offset 透传进 Cypher 参数、负数 clamp
- neighbors 接口参数构造、响应模型、去重行为
- has_more 判定逻辑

**前端手动验收清单**：
- [ ] 浏览模式查询 / 关键词查询 / 系统过滤
- [ ] 图例点击过滤类型
- [ ] 单击节点 → 详情侧栏 → 点击关联项定位
- [ ] 双击节点 → 展开邻居增量入图（去重）
- [ ] 加载更多 → 追加渲染 → 无更多时禁用
- [ ] 工具栏四按钮（重布局/适配/边标签/导出 PNG）
- [ ] >300 节点自动隐藏边标签、>500 关闭布局动画
- [ ] 五个页面底色/圆角/主色视觉一致
- [ ] toast 替代 alert 生效

## 12. 实施顺序（建议）

1. theme.css + 风格统一 + 清理死资源（独立可交付）
2. 后端：offset / neighbors / has_more + pytest
3. 前端渲染层：ECharts 重写 renderGraph + 图例/高亮/tooltip
4. 前端交互：详情侧栏 / 工具栏 / toast / 分页 / 展开邻居 / 导出
5. 手动验收清单过一遍
