"""
Milvus索引构建模块

用于构建Milvus索引
"""

import logging
import time
import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

logger = logging.getLogger(__name__)

class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "ship_maintenance_knowledge",
                 dimension: int = 768,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化Milvus索引构建模块

        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            dimension: 向量维度
            model_name: 嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        
        self.client = None
        self.embeddings = None
        self.collection_created = False
        
        self._setup_client()
        self._setup_embeddings()
    
    def _safe_truncate(self, text: str, max_length: int) -> str:
        """
        安全截断文本
        
        Args:
            text: 原始文本
            max_length: 最大长度
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
        return str(text)[:max_length]
    
    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")


            # 测试连接
            collections = self.client.list_collections()

            logger.info(f"连接成功，当前集合: {collections}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
    
    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式
        
        Returns:
            集合模式对象
        """
        # 定义字段
        fields = [
            # ========= 主键 =========
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=150,
                is_primary=True,
                auto_id=False
            ),
            # ========= 向量 =========
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=512  # 依据你的 embedding 模型
            ),
            # ========= 语义内容 =========
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=20000
            ),
            # ========= 图数据库映射 =========
            FieldSchema(
                name="neo4j_node_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="neo4j_label",
                dtype=DataType.VARCHAR,
                max_length=50
            ),
            # ========= 本体类型 =========
            FieldSchema(
                name="entity_type",
                dtype=DataType.VARCHAR,
                max_length=50
                # Equipment / Component / FaultPhenomenon / FaultReason / MaintenanceStep / Attention
            ),
            # ========= 业务分类（用于过滤） =========
            FieldSchema(
                name="equipment_type",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            # ========= 文档结构 =========
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=50
                # entity / chunk
            ),
            # ========= 分块标识（用于去重和更新） =========
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64
            )
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="船舶故障知识图谱向量集合"
        )
        
        return schema
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合
        
        Args:
            force_recreate: 是否强制重新创建集合
        
        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True
            
            # 创建集合
            schema = self._create_collection_schema()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",  # 使用余弦相似度
                consistency_level="Strong"
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self) -> bool:
        """
        创建向量索引
        
        Returns:
            是否创建成功
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")
            
            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()
            
            # 添加向量字段索引
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info("向量索引创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            是否构建成功
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collection(force_recreate=True):
                return False
            
            # 2. 准备数据
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            # 3. 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 20000),
                    "neo4j_node_id": self._safe_truncate(chunk.metadata.get("neo4j_node_id", ""), 100),
                    "neo4j_label": self._safe_truncate(chunk.metadata.get("neo4j_label", ""), 50),
                    "entity_type": self._safe_truncate(chunk.metadata.get("entity_type", ""), 50),
                    "equipment_type": self._safe_truncate(chunk.metadata.get("equipment_type", ""), 100),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 100),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100),
                    "chunk_index": int(chunk.metadata.get("chunk_index", 0))
                }
                entities.append(entity)
            
            # 4. 批量插入数据
            logger.info("正在插入向量数据...")
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")
            
            # 4.5 刷新数据，确保数据被持久化
            logger.info("正在刷新数据...")
            self.client.flush(self.collection_name)
            
            # 5. 创建索引
            if not self.create_index():
                return False
            
            # 6. 加载集合到内存
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")
            
            # 7. 等待索引构建完成
            logger.info("等待索引构建完成...")
            time.sleep(2)
            
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False

    def update_from_graph(self, graph_data_preparation_module, rebuild: bool = False) -> bool:
        """
        从图数据库同步更新向量索引
        
        Args:
            graph_data_preparation_module: 图数据准备模块
            rebuild: 是否重建整个索引
            
        Returns:
            是否更新成功
        """
        logger.info("开始从图数据库同步更新向量索引...")
        
        try:
            # 1. 从图数据库加载数据
            stats = graph_data_preparation_module.load_graph_data()
            logger.info(f"图数据加载完成: {stats}")
            
            # 2. 构建文档
            documents = graph_data_preparation_module.build_documents()
            logger.info(f"文档构建完成，共 {len(documents)} 个文档")
            
            # 3. 文档分块
            chunks = graph_data_preparation_module.chunk_documents()
            logger.info(f"文档分块完成，共 {len(chunks)} 个块")
            
            # 4. 构建或更新向量索引
            if rebuild or not self.collection_created:
                # 重建索引
                logger.info("重建向量索引...")
                success = self.build_vector_index(chunks)
            else:
                # 增量更新
                logger.info("增量更新向量索引...")
                success = self.add_documents(chunks)
            
            if success:
                logger.info("图数据库同步更新成功")
                return True
            else:
                logger.error("图数据库同步更新失败")
                return False
                
        except Exception as e:
            logger.error(f"从图数据库更新失败: {e}")
            return False
    
    def sync_graph_update(self, neo4j_driver, node_ids: List[str]) -> bool:
        """
        同步图数据库中的节点更新到向量索引
        
        Args:
            neo4j_driver: Neo4j驱动
            node_ids: 需要更新的节点ID列表
            
        Returns:
            是否同步成功
        """
        if not self.collection_created:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"开始同步 {len(node_ids)} 个节点的更新...")
        
        try:
            from langchain_core.documents import Document
            
            chunks_to_update = []
            
            with neo4j_driver.session() as session:
                for node_id in node_ids:
                    # 查询节点及其相关信息
                    query = """
                    MATCH (n)
                    WHERE n.id = $node_id
                    OPTIONAL MATCH (n)-[r]->(neighbor)
                    RETURN n, collect(DISTINCT neighbor) as neighbors
                    """
                    
                    result = session.run(query, {"node_id": node_id})
                    record = result.single()
                    
                    if not record:
                        logger.warning(f"节点 {node_id} 不存在，跳过")
                        continue
                    
                    node = record["n"]
                    neighbors = record["neighbors"]
                    
                    # 构建文档内容
                    content_parts = []
                    
                    # 添加节点基本信息
                    if node.get("name"):
                        content_parts.append(f"名称: {node['name']}")
                    if node.get("description"):
                        content_parts.append(f"描述: {node['description']}")
                    
                    # 添加邻居信息
                    if neighbors:
                        neighbor_names = [n.get("name", "未知") for n in neighbors if n.get("name")]
                        if neighbor_names:
                            content_parts.append(f"相关实体: {', '.join(neighbor_names[:5])}")
                    
                    # 创建文档
                    content = "\n".join(content_parts)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "neo4j_node_id": node_id,
                            "neo4j_label": node.labels[0] if node.labels else "Unknown",
                            "entity_type": node.get("entity_type", node.labels[0] if node.labels else "Unknown"),
                            "equipment_type": node.get("equipment_type", ""),
                            "doc_type": "entity",
                            "chunk_id": f"{node_id}_sync_{int(time.time())}",
                            "parent_id": "",
                            "chunk_index": 0
                        }
                    )
                    
                    chunks_to_update.append(doc)
            
            # 删除旧的向量数据
            for node_id in node_ids:
                try:
                    self.client.delete(
                        collection_name=self.collection_name,
                        filter=f"neo4j_node_id == '{node_id}'"
                    )
                except Exception as e:
                    logger.warning(f"删除节点 {node_id} 的旧向量数据失败: {e}")
            
            # 添加新的向量数据
            if chunks_to_update:
                success = self.add_documents(chunks_to_update)
                if success:
                    logger.info(f"节点同步更新成功: {len(chunks_to_update)} 个")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"同步图更新失败: {e}")
            return False

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
            
        Returns:
            是否添加成功
        """
        if not self.collection_created:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        
        try:
            # 生成向量
            texts = [chunk.page_content for chunk in new_chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            # 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(new_chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 20000),
                    "neo4j_node_id": self._safe_truncate(chunk.metadata.get("neo4j_node_id", ""), 100),
                    "neo4j_label": self._safe_truncate(chunk.metadata.get("neo4j_label", ""), 50),
                    "entity_type": self._safe_truncate(chunk.metadata.get("entity_type", ""), 50),
                    "equipment_type": self._safe_truncate(chunk.metadata.get("equipment_type", ""), 100),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 100),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100),
                    "chunk_index": int(chunk.metadata.get("chunk_index", 0))
                }
                entities.append(entity)
            
            # 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=entities
            )
            
            logger.info("新文档添加完成")
            return True
            
        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")
        
        try:
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)
            
            # 构建过滤表达式
            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list):
                        # 支持IN操作
                        if all(isinstance(v, str) for v in value):
                            value_str = '", "'.join(value)
                            filter_conditions.append(f'{key} in ["{value_str}"]')
                        else:
                            value_str = ', '.join(map(str, value))
                            filter_conditions.append(f'{key} in [{value_str}]')
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # 执行搜索 - 修复参数传递
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 构建搜索参数，避免重复传递
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": "vector",
                "limit": k,
                "output_fields": ["text", "neo4j_node_id", "neo4j_label", "entity_type",
                                "equipment_type", "doc_type", "chunk_id"],
                "search_params": search_params
            }
            
            # 只在有过滤条件时添加filter参数
            if filter_expr:
                search_kwargs["filter"] = filter_expr
                
            results = self.client.search(**search_kwargs)
            
            # 处理结果
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:  # results[0]因为我们只发送了一个查询向量
                    neo4j_node_id = hit["entity"]["neo4j_node_id"]
                    entity_name = hit["entity"].get("entity_name", "未知设备")
                    
                    result = {
                        "id": hit["id"],
                        "score": hit["distance"],  # 注意：在COSINE距离中，值越大相似度越高
                        "text": hit["entity"]["text"],
                        "metadata": {
                            "neo4j_node_id": neo4j_node_id,
                            "neo4j_label": hit["entity"]["neo4j_label"],
                            "entity_type": hit["entity"]["entity_type"],
                            "equipment_type": hit["entity"]["equipment_type"],
                            "doc_type": hit["entity"]["doc_type"],
                            "chunk_id": hit["entity"]["chunk_id"],
                            "entity_name": entity_name  # 添加设备名称
                        }
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def full_text_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        全文搜索（关键词检索）- 使用Milvus的查询
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")
        
        try:
            # 提取关键词
            keywords = query.lower().split()
            
            # 构建查询表达式
            conditions = []
            for keyword in keywords:
                conditions.append(f'text like "%{keyword}%"')
            
            if conditions:
                query_expr = " or ".join(conditions)
            else:
                query_expr = ""
            
            # 构建过滤表达式
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
                    if query_expr:
                        query_expr = f"({query_expr}) and ({filter_expr})"
                    else:
                        query_expr = filter_expr
            
            # 使用query方法进行文本搜索
            query_params = {
                "collection_name": self.collection_name,
                "output_fields": ["text", "neo4j_node_id", "neo4j_label", "entity_type",
                                "equipment_type", "doc_type", "chunk_id"],
                "limit": k
            }
            
            if query_expr:
                query_params["filter"] = query_expr
            
            results = self.client.query(**query_params)
            
            # 简单的关键词匹配打分
            scored_results = []
            for doc in results:
                text = doc.get("text", "").lower()
                score = 0
                for keyword in keywords:
                    if keyword in text:
                        score += text.count(keyword)
                
                if score > 0:
                    scored_results.append({
                        "id": doc.get("id"),
                        "score": score,
                        "text": doc.get("text"),
                        "metadata": {
                            "neo4j_node_id": doc.get("neo4j_node_id"),
                            "neo4j_label": doc.get("neo4j_label"),
                            "entity_type": doc.get("entity_type"),
                            "equipment_type": doc.get("equipment_type"),
                            "doc_type": doc.get("doc_type"),
                            "chunk_id": doc.get("chunk_id"),
                            "entity_name": doc.get("entity_name", "未知设备"),
                            "search_type": "full_text"
                        }
                    })
            
            # 按分数排序并返回Top-K
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"全文搜索完成，返回 {len(scored_results[:k])} 个结果")
            return scored_results[:k]
            
        except Exception as e:
            logger.error(f"全文搜索失败: {e}")
            # 回退到简单的关键词匹配
            return self._fallback_keyword_search(query, k, filters)
    
    def _fallback_keyword_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        回退关键词搜索 - 当BM25不可用时使用
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            logger.info("使用回退关键词搜索...")
            
            # 提取关键词
            keywords = query.lower().split()
            
            # 构建过滤表达式
            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # 查询所有文档
            query_params = {
                "collection_name": self.collection_name,
                "output_fields": ["text", "neo4j_node_id", "neo4j_label", "entity_type",
                                "equipment_type", "doc_type", "chunk_id"],
                "limit": 1000  # 获取足够多的文档进行筛选
            }
            if filter_expr:
                query_params["filter"] = filter_expr
            
            results = self.client.query(**query_params)
            
            # 简单的关键词匹配打分
            scored_results = []
            for doc in results:
                text = doc.get("text", "").lower()
                score = 0
                for keyword in keywords:
                    if keyword in text:
                        score += text.count(keyword)
                
                if score > 0:
                    scored_results.append({
                        "id": doc.get("id"),
                        "score": score,
                        "text": doc.get("text"),
                        "metadata": {
                            "neo4j_node_id": doc.get("neo4j_node_id"),
                            "neo4j_label": doc.get("neo4j_label"),
                            "entity_type": doc.get("entity_type"),
                            "equipment_type": doc.get("equipment_type"),
                            "doc_type": doc.get("doc_type"),
                            "chunk_id": doc.get("chunk_id"),
                            "entity_name": doc.get("entity_name", "未知设备"),
                            "search_type": "keyword_fallback"
                        }
                    })
            
            # 按分数排序并返回Top-K
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:k]
            
        except Exception as e:
            logger.error(f"回退关键词搜索失败: {e}")
            return []
    
    def vector_keyword_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        向量+关键词搜索：结合向量搜索和全文搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 1. 执行向量搜索
            vector_results = self.similarity_search(query, k=k*2, filters=filters)
            
            # 2. 执行全文搜索
            text_results = self.full_text_search(query, k=k*2, filters=filters)
            
            # 3. 合并结果（Round-robin策略）
            merged_results = []
            seen_ids = set()
            
            max_len = max(len(vector_results), len(text_results))
            
            for i in range(max_len):
                # 添加向量搜索结果
                if i < len(vector_results):
                    result = vector_results[i]
                    result_id = result.get("id")
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        result["search_method"] = "vector"
                        merged_results.append(result)
                
                # 添加全文搜索结果
                if i < len(text_results):
                    result = text_results[i]
                    result_id = result.get("id")
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        result["search_method"] = "full_text"
                        merged_results.append(result)
            
            logger.info(f"向量+关键词搜索完成，返回 {len(merged_results[:k])} 个结果")
            return merged_results[:k]
            
        except Exception as e:
            logger.error(f"向量+关键词搜索失败: {e}")
            # 如果搜索失败，回退到向量搜索
            return self.similarity_search(query, k=k, filters=filters)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}
            
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0),
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def delete_by_neo4j_node_ids(self, neo4j_node_ids: List[str]) -> bool:
        """
        按 payload 中的 neo4j_node_id 删除向量点（用于增量更新替换）。
        """
        ok = True
        for node_id in neo4j_node_ids or []:
            node_id = str(node_id)
            if not node_id:
                continue
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=f"neo4j_node_id == '{node_id}'"
                )
            except Exception as e:
                logger.warning(f"删除节点 {node_id} 的旧向量数据失败: {e}")
                ok = False
        return ok
    
    def has_collection(self) -> bool:
        """
        检查集合是否存在
        
        Returns:
            集合是否存在
        """
        try:
            return self.client.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合失败: {e}")
            return False
    
    def export_to_csv(self, output_file: str = "milvus_data.csv", output_dir: str = "backups") -> bool:
        """
        导出Milvus集合数据到CSV文件
        
        Args:
            output_file: 输出CSV文件路径
            output_dir: 输出目录，默认为backups
            
        Returns:
            是否导出成功
        """
        try:
            if not self.collection_created:
                logger.error("集合未创建，无法导出")
                return False
            
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"创建备份目录: {output_dir}")
            
            # 构建完整的文件路径
            full_path = os.path.join(output_dir, output_file)
            
            # 查询所有数据
            results = self.client.query(
                collection_name=self.collection_name,
                output_fields=["text", "neo4j_node_id", "neo4j_label", "entity_type",
                              "equipment_type", "doc_type", "chunk_id"]
            )
            
            if not results:
                logger.info("集合为空，无需导出")
                return True
            
            # 写入CSV文件
            with open(full_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["text", "neo4j_node_id", "neo4j_label", "entity_type",
                               "equipment_type", "doc_type", "chunk_id"])
                # 写入数据
                for doc in results:
                    writer.writerow([
                        doc.get("text", ""),
                        doc.get("neo4j_node_id", ""),
                        doc.get("neo4j_label", ""),
                        doc.get("entity_type", ""),
                        doc.get("equipment_type", ""),
                        doc.get("doc_type", ""),
                        doc.get("chunk_id", "")
                    ])
            
            logger.info(f"成功导出 {len(results)} 条数据到 {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出到CSV失败: {e}")
            return False
    
    def import_from_csv(self, input_file: str = "milvus_data.csv", input_dir: str = "backups") -> bool:
        """
        从CSV文件导入数据到Milvus集合
        
        Args:
            input_file: 输入CSV文件路径
            input_dir: 输入目录，默认为backups
            
        Returns:
            是否导入成功
        """
        try:
            if not self.collection_created:
                logger.error("集合未创建，无法导入")
                return False
            
            # 构建完整的文件路径
            full_path = os.path.join(input_dir, input_file)
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                logger.error(f"备份文件不存在: {full_path}")
                return False
            
            documents = []
            
            # 读取CSV文件
            with open(full_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 创建Document对象
                    doc = Document(
                        page_content=row.get("text", ""),
                        metadata={
                            "neo4j_node_id": row.get("neo4j_node_id"),
                            "neo4j_label": row.get("neo4j_label"),
                            "entity_type": row.get("entity_type"),
                            "equipment_type": row.get("equipment_type"),
                            "doc_type": row.get("doc_type"),
                            "chunk_id": row.get("chunk_id")
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                logger.info("CSV文件为空，无需导入")
                return True
            
            # 批量添加文档
            success = self.add_documents(documents)
            logger.info(f"从CSV导入 {len(documents)} 条数据")
            return success
            
        except Exception as e:
            logger.error(f"从CSV导入失败: {e}")
            return False
    
    def backup_collection(self, backup_name: str = None, backup_dir: str = "backups") -> bool:
        """
        备份Milvus集合
        
        Args:
            backup_name: 备份名称
            backup_dir: 备份目录，默认为backups
            
        Returns:
            是否备份成功
        """
        try:
            if not self.collection_created:
                logger.error("集合未创建，无法备份")
                return False
            
            # 确保备份目录存在
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                logger.info(f"创建备份目录: {backup_dir}")
            
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{self.collection_name}_backup_{timestamp}"
            
            # 导出到CSV作为备份
            backup_file = f"{backup_name}.csv"
            success = self.export_to_csv(backup_file, backup_dir)
            
            if success:
                # 创建备份信息文件
                backup_info = {
                    "backup_name": backup_name,
                    "backup_time": datetime.now().isoformat(),
                    "collection_name": self.collection_name,
                    "backup_file": backup_file
                }
                info_file = os.path.join(backup_dir, f"{backup_name}.json")
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_info, f, ensure_ascii=False, indent=2)
                
                logger.info(f"成功备份集合到 {backup_dir}/{backup_file}")
            
            return success
            
        except Exception as e:
            logger.error(f"备份集合失败: {e}")
            return False
    
    def restore_collection(self, backup_file: str, backup_dir: str = "backups") -> bool:
        """
        从备份恢复Milvus集合
        
        Args:
            backup_file: 备份文件路径
            backup_dir: 备份目录，默认为backups
            
        Returns:
            是否恢复成功
        """
        try:
            # 构建完整的文件路径
            full_path = os.path.join(backup_dir, backup_file)
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                logger.error(f"备份文件不存在: {full_path}")
                return False
            
            # 从CSV导入数据
            success = self.import_from_csv(backup_file, backup_dir)
            
            if success:
                logger.info(f"成功从 {backup_dir}/{backup_file} 恢复集合")
            
            return success
            
        except Exception as e:
            logger.error(f"恢复集合失败: {e}")
            return False
    
    def list_backups(self, backup_dir: str = "backups") -> List[Dict[str, Any]]:
        """
        列出所有可用的备份
        
        Args:
            backup_dir: 备份目录，默认为backups
            
        Returns:
            备份列表
        """
        try:
            if not os.path.exists(backup_dir):
                logger.info(f"备份目录不存在: {backup_dir}")
                return []
            
            backups = []
            for filename in os.listdir(backup_dir):
                if filename.endswith('.json'):
                    info_path = os.path.join(backup_dir, filename)
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            backup_info = json.load(f)
                            backups.append(backup_info)
                    except Exception as e:
                        logger.warning(f"读取备份信息失败 {filename}: {e}")
            
            # 按备份时间排序
            backups.sort(key=lambda x: x.get('backup_time', ''), reverse=True)
            logger.info(f"找到 {len(backups)} 个备份")
            return backups
            
        except Exception as e:
            logger.error(f"列出备份失败: {e}")
            return []
    
    def auto_backup_on_error(self, backup_dir: str = "backups") -> bool:
        """
        在发生错误时自动备份当前数据
        
        Args:
            backup_dir: 备份目录，默认为backups
            
        Returns:
            是否备份成功
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self.collection_name}_auto_backup_{timestamp}"
            success = self.backup_collection(backup_name, backup_dir)
            
            if success:
                logger.info(f"自动备份成功: {backup_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"自动备份失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """
        加载集合到内存
        
        Returns:
            是否加载成功
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False
            
            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'client') and self.client:
            # Milvus客户端不需要显式关闭
            logger.info("Milvus连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close() 