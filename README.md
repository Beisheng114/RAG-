# 船舶故障诊断问答RAG系统

## 项目概述

船舶故障诊断问答RAG系统是一个基于检索增强生成（Retrieval-Augmented Generation）技术的智能问答系统，专为船舶维修领域设计。该系统整合了向量检索和图检索技术，能够提供准确、全面的船舶故障维修信息。

### 核心功能

- **混合检索**：结合向量检索和图检索，提供更精准的信息获取
- **本地模型支持**：支持本地部署的BAAI/bge-base-zh-v1.5嵌入模型
- **多向量数据库支持**：支持Qdrant和Milvus向量数据库
- **Neo4j图数据库**：存储和检索复杂的知识图谱关系
- **智能查询路由**：根据查询类型自动选择最佳检索策略
- **多模型支持**：可选择使用Deepseek、OpenAI或本地Ollama模型提高数据提取准确率
- **知识图谱自动构建**：支持从文档中自动提取实体和关系
- **Web界面**：提供直观的Web界面进行交互

## 系统架构 

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                 │     │                  │     │                  │
│   用户查询        │────>│ 智能查询路由器   │────>│  混合检索模块    │
│                 │     │                  │     │                  │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                 │     │                  │     │                  │
│   生成模块      │<────│  结果集成        │<────│  向量与图检索    │
│                 │     │                  │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

## 目录结构

```
├── app.py                # FastAPI应用入口
├── ragmain.py           # 主程序入口
├── config.py            # 配置文件
├── requirements.txt     # 依赖项
├── 启动.bat            # 启动脚本
├── 关闭.bat            # 关闭脚本
├── data/                # 原始数据文件
├── rag_modules/         # RAG核心模块
│   ├── hybrid_retrieval.py          # 混合检索模块
│   ├── graph_rag_retrieval.py       # 图RAG检索
│   ├── faiss_index_construction.py  # Faiss索引构建
│   ├── qdrant_index_construction.py # Qdrant索引构建
│   ├── milvus_index_construction.py # Milvus索引构建
│   ├── graph_data_preparation.py    # 图数据准备
│   ├── graph_data_insert.py         # 图数据插入
│   ├── graph_indexing.py            # 图索引构建
│   ├── intelligent_query_router.py  # 智能查询路由
│   ├── generation_integration.py    # 生成集成
│   └── context_manager.py           # 上下文管理
├── routers/             # API路由
│   ├── admin_routes.py              # 管理路由
│   ├── graph_routes.py              # 图相关路由
│   ├── kg_import_routes.py          # 知识图谱导入路由
│   └── page_routes.py               # 页面路由
├── services/            # 服务层
│   └── kg_import_service.py         # 知识图谱导入服务
├── static/              # 静态文件
│   ├── index.html                   # 主页面
│   ├── index.css                    # 样式文件
│   └── index.js                     # JavaScript文件
├── csv_generate/        # CSV生成工具
│   ├── config.yaml                  # 配置文件
│   ├── kg_generator_v2.py           # 知识图谱生成器
│   └── llm_client.py                # LLM客户端
├── faiss_index/         # Faiss索引存储
├── models/              # 模型存储
├── uploads/             # 上传文件存储
└── llm_debug/           # LLM调试信息
```

## 快速开始

### 环境要求

- Python 3.8+
- Neo4j 5.0+
- Qdrant 1.0+ (可选)
- Milvus 2.0+ (可选)
- Ollama (可选，用于本地LLM服务)
- CUDA 11.0+ (可选，用于加速模型)

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 下载模型

```bash
# 下载BAAI/bge-base-zh-v1.5模型
python download_model.py

# 可选：下载Ollama模型
# ollama pull qwen2.5:7b
```

### 配置设置

编辑 `config.py` 文件，设置以下参数：

```python
# 模型配置
embedding_model: str = "./models/bge-base-zh-v1.5"  # 本地模型路径
llm_model: str = "http://localhost:8000/v1"  # 本地LLM服务地址

# Neo4j配置
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "your_password"
neo4j_database: str = "neo4j"

# 向量数据库配置
use_qdrant: bool = True
qdrant_url: str = "http://localhost:6333"
qdrant_collection: str = "ship_maintenance"

use_milvus: bool = False
milvus_host: str = "localhost"
milvus_port: int = 19530
milvus_collection: str = "ship_maintenance_knowledge"
```

### 启动服务

1. **使用启动脚本**

```bash
# Windows
双击 启动.bat

# Linux/Mac
chmod +x 启动.sh
./启动.sh
```

启动脚本会自动启动以下服务：
- Ollama服务（用于本地LLM）
- Neo4j服务（知识图谱）
- Qdrant服务（向量数据库）
- FastAPI应用（Web界面）

2. **手动启动**

```bash
# 启动Neo4j服务
# 启动Qdrant服务
# 启动Ollama服务
ollama serve

# 运行FastAPI应用
python app.py
```

### 访问系统

打开浏览器，访问 `http://localhost:8000` 即可使用Web界面。

## 使用指南

### Web界面

- **输入问题**：在输入框中输入您的船舶维修相关问题
- **查看结果**：系统会显示相关的维修知识和建议
- **知识图谱**：查看问题相关的知识图谱关系
- **维修步骤**：查看详细的维修步骤和注意事项

### 示例查询

```
全船停电怎么办
发电机异响如何处理
主机过热的原因有哪些
```

### 知识图谱导入

1. **上传文档**：在Web界面中上传PDF、DOCX等文档
2. **提取知识**：系统会自动提取文档中的实体和关系
3. **构建图谱**：生成知识图谱并存储到Neo4j

## 技术细节

### 1. 向量检索

- 支持FAISS、Qdrant和Milvus向量数据库
- 本地存储索引文件，提高系统可靠性
- 支持增量更新，减少启动时间

### 2. 图检索

- 使用Neo4j图数据库存储知识图谱
- 支持复杂的关系查询
- 优化的Cypher查询语句

### 3. 混合检索策略

- 实体级检索：基于实体关键词的精确匹配
- 主题级检索：基于主题的相关度匹配
- 图结构检索：基于知识图谱的关系推理

### 4. 智能查询路由

- 分析查询类型和意图
- 根据查询特征选择最佳检索策略
- 动态调整检索参数

### 5. 知识图谱构建

- 两阶段LLM提取：高召回候选抽取 + 关系验证
- 实体标准化和去重
- 自动补充关系，避免孤岛

## 数据管理

### 数据准备

1. **原始数据**：将PDF、DOCX等文档放入 `data/` 目录
2. **数据转换**：使用 `csv_generate/` 工具将文档转换为CSV格式
3. **数据导入**：使用Web界面或API将数据导入系统

### 知识库更新

- **增量更新**：系统支持增量添加新数据
- **重建索引**：使用管理界面重建整个知识库

## 故障排除

### 常见问题

1. **Neo4j连接失败**
   - 检查Neo4j服务是否启动
   - 验证连接参数是否正确

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保模型文件完整下载

3. **JSON解析错误**
   - 检查LLM服务是否正常运行
   - 验证LLM响应格式是否正确

4. **Cypher查询警告**
   - 检查节点属性名是否正确
   - 验证关系类型是否存在

5. **知识图谱提取失败**
   - 检查Ollama服务是否正在运行
   - 确保配置文件中的LLM设置正确

## 性能优化

- **索引优化**：定期重建向量索引
- **缓存策略**：使用内存缓存提高响应速度
- **并发处理**：支持多线程处理查询
- **批处理**：批量处理相似查询

## 扩展功能

- **多语言支持**：添加多语言模型
- **多模态支持**：整合图像识别能力
- **实时数据**：接入实时传感器数据
- **个性化推荐**：基于用户历史查询

## 贡献指南

欢迎提交问题和Pull Request来改进系统。请确保：

1. 遵循项目代码风格
2. 添加适当的测试
3. 更新文档

## 许可证

本项目采用 MIT 许可证。



**版本**: 1.0.0
**更新日期**: 2026-04-23
