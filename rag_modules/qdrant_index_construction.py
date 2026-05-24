"""
Qdrant向量索引模块

充分利用Qdrant的高级特性：
1. 高效向量检索 - HNSW算法
2. 过滤检索 - 基于payload的过滤
3. 混合检索 - 向量+关键词
4. 批量操作 - 高效批量索引
5. 持久化存储 - 数据持久化
6. 集合管理 - 多集合支持
7. 量化支持 - 内存优化
8. 分布式支持 - 水平扩展
"""

import json
import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, MatchText,
        Range, IsEmptyCondition, IsNullCondition,
        SearchRequest, RecommendRequest,
        HasIdCondition
    )
    IsEmpty = IsEmptyCondition
    IsNull = IsNullCondition
    HasId = HasIdCondition
    QDRANT_AVAILABLE = True
except ImportError as e:
    QDRANT_AVAILABLE = False
    logger.warning(f"qdrant-client未安装，请运行: pip install qdrant-client。错误: {e}")
except Exception as e:
    QDRANT_AVAILABLE = False
    logger.warning(f"qdrant-client导入失败: {type(e).__name__}: {e}")


@dataclass
class QdrantConfig:
    """Qdrant配置"""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    collection_name: str = "ship_maintenance_knowledge"
    vector_size: int = 768
    distance: str = "Cosine"
    
    hnsw_config: Dict[str, Any] = field(default_factory=lambda: {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000
    })
    
    quantization_config: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    
    optimizers_config: Dict[str, Any] = field(default_factory=lambda: {
        "deleted_threshold": 0.2,
        "vacuum_min_vector_number": 1000,
        "default_segment_number": 0,
        "max_segment_size": None,
        "memmap_threshold": None,
        "indexing_threshold": 20000,
        "flush_interval_sec": 5,
        "max_optimization_threads": 1
    })
    
    wal_config: Dict[str, Any] = field(default_factory=lambda: {
        "wal_capacity_mb": 32,
        "wal_segments_ahead": 0
    })

    # 查询阶段 HNSW 的 ef；略小于默认时可缩短 Qdrant 检索耗时，过大则变慢。None 表示使用服务端默认。
    hnsw_ef_search: Optional[int] = 128


@dataclass
class PerformanceStats:
    """性能统计"""
    total_queries: int = 0
    total_insertions: int = 0
    avg_query_time: float = 0.0
    avg_insert_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_points: int = 0


class QdrantIndexConstructionModule:
    """
    Qdrant向量索引模块
    
    核心特性：
    1. 高效HNSW索引
    2. 丰富的过滤条件
    3. 批量操作优化
    4. 持久化存储
    5. 量化支持
    6. 多种距离度量
    """
    
    def __init__(self,
                 config: Optional[QdrantConfig] = None,
                 embedding_model_path: str = "./models/bge-small-zh-v1.5",
                 **kwargs):
        """
        初始化Qdrant索引模块
        
        Args:
            config: Qdrant配置
            embedding_model_path: 嵌入模型路径
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("请先安装qdrant-client: pip install qdrant-client")
        
        self.config = config or QdrantConfig(**kwargs)
        self.embedding_model_path = embedding_model_path
        
        self.client: Optional[QdrantClient] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.collection_created = False
        
        self.query_cache: Dict[str, Tuple[List[Dict], float]] = {}
        self.cache_max_size = 1000
        self.cache_ttl = 3600
        
        self.performance_stats = PerformanceStats()
        
        self._setup_embeddings()
        self._connect()
        
        logger.info(f"Qdrant索引模块初始化完成 - 集合: {self.config.collection_name}")
    
    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.embedding_model_path}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 64},
        )
        
        test_embedding = self.embeddings.embed_query("测试")
        self.config.vector_size = len(test_embedding)
        logger.info(f"嵌入模型初始化完成，向量维度: {self.config.vector_size}")
    
    def _connect(self):
        """连接Qdrant服务器"""
        try:
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                prefer_grpc=bool(self.config.prefer_grpc),
            )
            
            collections = self.client.get_collections()
            logger.info(f"成功连接Qdrant服务器: {self.config.host}:{self.config.port}")
            
            if self.config.collection_name in [c.name for c in collections.collections]:
                self.collection_created = True
                logger.info(f"集合 '{self.config.collection_name}' 已存在")
            else:
                logger.info(f"集合 '{self.config.collection_name}' 不存在，需要创建")
                
        except Exception as e:
            logger.error(f"连接Qdrant服务器失败: {e}")
            raise
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建集合
        
        Args:
            force_recreate: 是否强制重建
            
        Returns:
            是否创建成功
        """
        try:
            if self.client.collection_exists(self.config.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.config.collection_name}")
                    self.client.delete_collection(self.config.collection_name)
                else:
                    logger.info(f"集合已存在: {self.config.collection_name}")
                    self.collection_created = True
                    return True
            
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            hnsw_config = models.HnswConfigDiff(
                m=self.config.hnsw_config.get("m", 16),
                ef_construct=self.config.hnsw_config.get("ef_construct", 100),
                full_scan_threshold=self.config.hnsw_config.get("full_scan_threshold", 10000)
            )
            
            quantization_config = None
            if self.config.quantization_config and "scalar" in self.config.quantization_config:
                try:
                    scalar_config = self.config.quantization_config.get("scalar", {})
                    quantization_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=scalar_config.get("quantile", 0.99)
                        )
                    )
                except Exception as qe:
                    logger.warning(f"量化配置创建失败，将不使用量化: {qe}")
                    quantization_config = None
            
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map.get(self.config.distance, Distance.COSINE)
                ),
                hnsw_config=hnsw_config,
                quantization_config=quantization_config
            )
            
            self.collection_created = True
            logger.info(f"成功创建集合: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def build_vector_index(self, chunks: List[Document], batch_size: int = 100) -> bool:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            batch_size: 批量大小
            
        Returns:
            是否构建成功
        """
        if not self.collection_created:
            self.create_collection()
        
        try:
            logger.info(f"开始构建向量索引，文档数: {len(chunks)}")
            start_time = time.time()
            
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch = chunks[batch_idx:batch_idx + batch_size]
                
                texts = [chunk.page_content for chunk in batch]
                vectors = self.embeddings.embed_documents(texts)
                
                points = []
                for i, (chunk, vector) in enumerate(zip(batch, vectors)):
                    point_id = str(uuid.uuid4())
                    
                    payload = {
                        "text": chunk.page_content,
                        "neo4j_node_id": chunk.metadata.get("neo4j_node_id", ""),
                        "neo4j_label": chunk.metadata.get("neo4j_label", ""),
                        "entity_type": chunk.metadata.get("entity_type", ""),
                        "equipment_type": chunk.metadata.get("equipment_type", ""),
                        "doc_type": chunk.metadata.get("doc_type", ""),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "parent_id": chunk.metadata.get("parent_id", ""),
                        "sources": chunk.metadata.get("sources", []),
                        "created_at": datetime.now().isoformat()
                    }
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    ))
                
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points
                )
                
                batch_num = batch_idx // batch_size + 1
                logger.info(f"已处理批次 {batch_num}/{total_batches}")
            
            elapsed = time.time() - start_time
            self.performance_stats.total_insertions = len(chunks)
            self.performance_stats.avg_insert_time = elapsed / len(chunks)
            
            logger.info(f"向量索引构建完成，耗时: {elapsed:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False
    
    def similarity_search(self,
                         query: str,
                         k: int = 5,
                         filters: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            use_cache: 是否使用缓存
            
        Returns:
            搜索结果列表
        """
        start_time = time.time()
        
        cache_key = f"{query}_{k}_{json.dumps(filters, sort_keys=True) if filters else ''}"
        
        if use_cache and cache_key in self.query_cache:
            cache_data, cache_time = self.query_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                self.performance_stats.cache_hits += 1
                return cache_data
        
        self.performance_stats.cache_misses += 1
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            filter_obj = self._build_filter(filters) if filters else None

            query_kwargs: Dict[str, Any] = {
                "collection_name": self.config.collection_name,
                "query": query_vector,
                "query_filter": filter_obj,
                "limit": k,
                "with_payload": True,
                "with_vectors": False,
            }
            ef = getattr(self.config, "hnsw_ef_search", None)
            if ef is not None and hasattr(models, "SearchParams"):
                query_kwargs["search_params"] = models.SearchParams(hnsw_ef=int(ef))

            search_result = self.client.query_points(**query_kwargs)
            
            results = []
            for scored_point in search_result.points:
                result = {
                    "id": str(scored_point.id),
                    "score": scored_point.score,
                    "text": scored_point.payload.get("text", "") if scored_point.payload else "",
                    "metadata": {
                        "neo4j_node_id": scored_point.payload.get("neo4j_node_id", "") if scored_point.payload else "",
                        "neo4j_label": scored_point.payload.get("neo4j_label", "") if scored_point.payload else "",
                        "entity_type": scored_point.payload.get("entity_type", "") if scored_point.payload else "",
                        "equipment_type": scored_point.payload.get("equipment_type", "") if scored_point.payload else "",
                        "doc_type": scored_point.payload.get("doc_type", "") if scored_point.payload else "",
                        "chunk_index": scored_point.payload.get("chunk_index", 0) if scored_point.payload else 0,
                        "parent_id": scored_point.payload.get("parent_id", "") if scored_point.payload else "",
                        "sources": scored_point.payload.get("sources", []) if scored_point.payload else []
                    }
                }
                results.append(result)
            
            if use_cache:
                self._update_cache(cache_key, results)
            
            query_time = time.time() - start_time
            self._update_performance_stats(query_time)
            
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """
        构建Qdrant过滤条件
        
        支持的过滤类型：
        - match: 精确匹配
        - range: 范围过滤
        - has_id: ID过滤
        - is_empty/is_null: 空值过滤
        - text_match: 文本匹配
        """
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                if "match" in value:
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value["match"])
                    ))
                elif "text" in value:
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchText(text=value["text"])
                    ))
                elif "range" in value:
                    range_val = value["range"]
                    conditions.append(FieldCondition(
                        key=key,
                        range=Range(
                            gt=range_val.get("gt"),
                            gte=range_val.get("gte"),
                            lt=range_val.get("lt"),
                            lte=range_val.get("lte")
                        )
                    ))
                elif "is_empty" in value:
                    conditions.append(IsEmpty(is_empty=value["is_empty"]))
                elif "is_null" in value:
                    conditions.append(IsNull(is_null=value["is_null"]))
            elif isinstance(value, list):
                for v in value:
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=v)
                    ))
            else:
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def filtered_search(self,
                       query: str,
                       filters: Dict[str, Any],
                       k: int = 5) -> List[Dict[str, Any]]:
        """
        过滤检索
        
        Args:
            query: 查询文本
            filters: 过滤条件
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        return self.similarity_search(query, k=k, filters=filters)
    
    def multi_vector_search(self,
                           queries: List[str],
                           k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        多向量批量搜索
        
        Args:
            queries: 查询列表
            k: 每个查询返回的结果数
            
        Returns:
            每个查询的搜索结果列表
        """
        try:
            query_vectors = self.embeddings.embed_documents(queries)
            
            search_requests = []
            for i, vector in enumerate(query_vectors):
                search_requests.append(models.QueryRequest(
                    query=vector,
                    limit=k,
                    with_payload=True
                ))
            
            results = self.client.query_batch_points(
                collection_name=self.config.collection_name,
                requests=search_requests
            )
            
            all_results = []
            for search_result in results:
                query_results = []
                for scored_point in search_result.points:
                    query_results.append({
                        "id": str(scored_point.id),
                        "score": scored_point.score,
                        "text": scored_point.payload.get("text", "") if scored_point.payload else "",
                        "metadata": scored_point.payload if scored_point.payload else {}
                    })
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"多向量搜索失败: {e}")
            return [[] for _ in queries]
    
    def recommend_search(self,
                        positive_ids: List[str],
                        negative_ids: Optional[List[str]] = None,
                        k: int = 5) -> List[Dict[str, Any]]:
        """
        推荐搜索（基于已有向量）
        
        Args:
            positive_ids: 正例ID列表
            negative_ids: 负例ID列表
            k: 返回结果数量
            
        Returns:
            推荐结果列表
        """
        try:
            positive_uuids = [uuid.UUID(pid) for pid in positive_ids]
            negative_uuids = [uuid.UUID(nid) for nid in negative_ids] if negative_ids else []
            
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                query=models.RecommendQuery(
                    positive=positive_uuids,
                    negative=negative_uuids
                ),
                limit=k,
                with_payload=True
            )
            
            recommendations = []
            for scored_point in results.points:
                recommendations.append({
                    "id": str(scored_point.id),
                    "score": scored_point.score,
                    "text": scored_point.payload.get("text", "") if scored_point.payload else "",
                    "metadata": scored_point.payload if scored_point.payload else {}
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"推荐搜索失败: {e}")
            return []
    
    def scroll_points(self,
                     offset: Optional[str] = None,
                     limit: int = 100,
                     filters: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        滚动获取点（用于批量导出或遍历）
        
        Args:
            offset: 偏移量
            limit: 每次获取的数量
            filters: 过滤条件
            
        Returns:
            (点列表, 下一个偏移量)
        """
        try:
            filter_obj = self._build_filter(filters) if filters else None
            
            points, next_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=filter_obj
            )
            
            results = []
            for point in points:
                results.append({
                    "id": str(point.id),
                    "payload": point.payload
                })
            
            return results, next_offset
            
        except Exception as e:
            logger.error(f"滚动获取点失败: {e}")
            return [], None
    
    def vector_keyword_search(self,
                             query: str,
                             k: int = 5,
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        向量+关键词混合搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        wide = max(k * 2, k)
        results = self.similarity_search(query, k=wide, filters=filters)
        return results[:k]
    
    def _keyword_search(self,
                       query: str,
                       k: int = 5,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        关键词通道的占位实现（当前等同于向量检索）。
        若需真正 BM25/全文，应对 payload 建文本索引或走 LangChain BM25Retriever，避免与本模块向量检索重复调用。
        """
        try:
            return self.similarity_search(query, k=k, filters=filters)
        except Exception as e:
            logger.warning(f"关键词搜索失败: {e}")
            return []
    
    def _merge_results(self,
                       vector_results: List[Dict],
                       keyword_results: List[Dict],
                       k: int) -> List[Dict]:
        """
        合并向量和关键词结果（RRF算法）
        """
        doc_scores = {}
        doc_info = {}
        rrf_k = 60
        
        for rank, result in enumerate(vector_results, 1):
            doc_id = result["id"]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result["id"]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_results = []
        for doc_id, score in sorted_docs[:k]:
            result = doc_info[doc_id].copy()
            result["merged_score"] = score
            merged_results.append(result)
        
        return merged_results
    
    def _merge_multi_results(self,
                            results_list: List[List[Dict]],
                            k: int) -> List[Dict[str, Any]]:
        """
        合并多个搜索结果（RRF算法）
        """
        doc_scores = {}
        doc_info = {}
        rrf_k = 60
        
        for results in results_list:
            for rank, result in enumerate(results, 1):
                doc_id = result["id"]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
                if doc_id not in doc_info:
                    doc_info[doc_id] = result
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_results = []
        for doc_id, score in sorted_docs[:k]:
            result = doc_info[doc_id].copy()
            result["merged_score"] = score
            merged_results.append(result)
        
        return merged_results
    
    def bm25_search(self,
                   query: str,
                   k: int = 5,
                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        BM25风格文本检索（本地实现）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        return self._keyword_search(query, k=k, filters=filters)
    
    def rrf_hybrid_search(self,
                         query: str,
                         k: int = 5,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        RRF融合检索（向量+关键词，本地实现）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            融合后的搜索结果列表
        """
        try:
            wide = max(k * 2, k)
            results = self.similarity_search(query, k=wide, filters=filters)
            return results[:k]
        except Exception as e:
            logger.warning(f"RRF融合检索失败，回退到向量检索: {e}")
            return self.similarity_search(query, k=k, filters=filters)
    
    def multi_query_fusion_search(self,
                                  queries: List[str],
                                  k: int = 5,
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        多查询融合检索（本地实现）
        
        Args:
            queries: 查询文本列表
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            融合后的搜索结果列表
        """
        try:
            all_results = []
            for q in queries:
                results = self.similarity_search(q, k=k*2, filters=filters)
                all_results.append(results)
            
            return self._merge_multi_results(all_results, k)
            
        except Exception as e:
            logger.warning(f"多查询融合检索失败: {e}")
            if queries:
                return self.similarity_search(queries[0], k=k, filters=filters)
            return []
    
    def semantic_keyword_search(self,
                               query: str,
                               k: int = 5,
                               filters: Optional[Dict[str, Any]] = None,
                               semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        语义+关键词加权检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            semantic_weight: 语义搜索权重 (0-1)
            
        Returns:
            搜索结果列表
        """
        try:
            wide = max(k * 2, k)
            results = self.similarity_search(query, k=wide, filters=filters)[:k]
            for r in results:
                r["weighted_score"] = r.get("score", 0.0)
            return results
        except Exception as e:
            logger.warning(f"语义关键词加权检索失败: {e}")
            return self.similarity_search(query, k=k, filters=filters)
    
    def hybrid_search(self,
                     query: str,
                     k: int = 5,
                     filters: Optional[Dict[str, Any]] = None,
                     mode: str = "rrf") -> List[Dict[str, Any]]:
        """
        高级混合检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            mode: 检索模式
                - "rrf": RRF融合检索（推荐）
                - "weighted": 加权混合检索
                - "simple": 简单向量+关键词检索
                
        Returns:
            搜索结果列表
        """
        if mode == "rrf":
            return self.rrf_hybrid_search(query, k=k, filters=filters)
        elif mode == "weighted":
            return self.semantic_keyword_search(query, k=k, filters=filters)
        else:
            return self.vector_keyword_search(query, k=k, filters=filters)
    
    def batch_search(self,
                    queries: List[str],
                    k: int = 5,
                    filters: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """
        批量搜索
        
        Args:
            queries: 查询列表
            k: 每个查询返回的结果数
            filters: 过滤条件
            
        Returns:
            每个查询的搜索结果列表
        """
        return self.multi_vector_search(queries, k)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加新文档
        
        Args:
            documents: 文档列表
            
        Returns:
            是否添加成功
        """
        return self.build_vector_index(documents, batch_size=50)
    
    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        根据ID删除点
        
        Args:
            ids: 点ID列表
            
        Returns:
            是否删除成功
        """
        try:
            uuids = [uuid.UUID(id) for id in ids]
            
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(
                    points=uuids
                )
            )
            
            logger.info(f"已删除 {len(ids)} 个点")
            return True
            
        except Exception as e:
            logger.error(f"删除点失败: {e}")
            return False
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> bool:
        """
        根据过滤条件删除点
        
        Args:
            filters: 过滤条件
            
        Returns:
            是否删除成功
        """
        try:
            filter_obj = self._build_filter(filters)
            
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=filter_obj
            )
            
            logger.info(f"已根据过滤条件删除点")
            return True
            
        except Exception as e:
            logger.error(f"根据过滤条件删除失败: {e}")
            return False

    def delete_by_neo4j_node_ids(self, neo4j_node_ids: List[str]) -> bool:
        """
        按 payload 中的 neo4j_node_id 删除向量点（用于增量更新替换）。

        说明：这里按 node_id 循环删除，避免构建复杂的 OR 过滤。
        """
        ok = True
        for node_id in neo4j_node_ids or []:
            node_id = str(node_id)
            if not node_id:
                continue
            try:
                self.delete_by_filter({"neo4j_node_id": {"match": node_id}})
            except Exception as e:
                logger.warning(f"删除 neo4j_node_id={node_id} 失败: {e}")
                ok = False
        return ok
    
    def update_payload(self, point_id: str, payload: Dict[str, Any]) -> bool:
        """
        更新点的payload
        
        Args:
            point_id: 点ID
            payload: 新的payload
            
        Returns:
            是否更新成功
        """
        try:
            self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=payload,
                points=[uuid.UUID(point_id)]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"更新payload失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            info = self.client.get_collection(self.config.collection_name)
            
            return {
                "collection_name": self.config.collection_name,
                "points_count": getattr(info, 'points_count', 0),
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
                "segments_count": getattr(info, 'segments_count', 0),
                "status": info.status.value if hasattr(info, 'status') else 'unknown',
                "optimizer_status": str(getattr(info, 'optimizer_status', 'unknown')),
                "collection_created": self.collection_created
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def has_collection(self) -> bool:
        """
        检查集合是否存在
        
        Returns:
            集合是否存在
        """
        try:
            return self.client.collection_exists(self.config.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """
        加载集合（Qdrant自动加载）
        
        Returns:
            是否加载成功
        """
        try:
            if self.has_collection():
                self.collection_created = True
                return True
            return False
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection_created = False
            logger.info(f"已删除集合: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def optimize_collection(self) -> bool:
        """
        优化集合
        
        Returns:
            是否优化成功
        """
        try:
            self.client.update_collection(
                collection_name=self.config.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            
            logger.info("集合优化已触发")
            return True
            
        except Exception as e:
            logger.error(f"优化集合失败: {e}")
            return False
    
    def create_payload_index(self, field_name: str, field_type: str = "keyword") -> bool:
        """
        创建payload索引
        
        Args:
            field_name: 字段名
            field_type: 字段类型 (keyword, integer, float, text, bool)
            
        Returns:
            是否创建成功
        """
        try:
            type_map = {
                "keyword": models.PayloadSchemaType.KEYWORD,
                "integer": models.PayloadSchemaType.INTEGER,
                "float": models.PayloadSchemaType.FLOAT,
                "text": models.PayloadSchemaType.TEXT,
                "bool": models.PayloadSchemaType.BOOL
            }
            
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name=field_name,
                field_schema=type_map.get(field_type, models.PayloadSchemaType.KEYWORD)
            )
            
            logger.info(f"已创建payload索引: {field_name} ({field_type})")
            return True
            
        except Exception as e:
            logger.error(f"创建payload索引失败: {e}")
            return False
    
    def create_fulltext_index(self, field_name: str = "text") -> bool:
        """
        创建全文索引（用于BM25检索）
        
        Args:
            field_name: 要创建全文索引的字段名，默认为"text"
            
        Returns:
            是否创建成功
        """
        try:
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name=field_name,
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    min_token_len=1,
                    max_token_len=20,
                    lowercase=True
                )
            )
            
            logger.info(f"已创建全文索引: {field_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建全文索引失败: {e}")
            return False
    
    def setup_advanced_indexes(self) -> Dict[str, bool]:
        """
        设置高级索引（全文索引、payload索引等）
        
        Returns:
            各索引创建结果
        """
        results = {}
        
        results["text_fulltext"] = self.create_fulltext_index("text")
        results["neo4j_label_index"] = self.create_payload_index("neo4j_label", "keyword")
        results["entity_type_index"] = self.create_payload_index("entity_type", "keyword")
        results["equipment_type_index"] = self.create_payload_index("equipment_type", "keyword")
        results["doc_type_index"] = self.create_payload_index("doc_type", "keyword")
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"高级索引设置完成: {success_count}/{len(results)} 成功")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计字典
        """
        cache_hit_rate = (
            self.performance_stats.cache_hits /
            (self.performance_stats.cache_hits + self.performance_stats.cache_misses)
            if (self.performance_stats.cache_hits + self.performance_stats.cache_misses) > 0
            else 0.0
        )
        
        return {
            "total_queries": self.performance_stats.total_queries,
            "total_insertions": self.performance_stats.total_insertions,
            "avg_query_time": f"{self.performance_stats.avg_query_time:.4f}s",
            "avg_insert_time": f"{self.performance_stats.avg_insert_time:.4f}s",
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "cache_size": len(self.query_cache)
        }
    
    def clear_cache(self):
        """清空查询缓存"""
        self.query_cache.clear()
        logger.info("查询缓存已清空")
    
    def _update_cache(self, cache_key: str, results: List[Dict]):
        """更新缓存"""
        if len(self.query_cache) >= self.cache_max_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = (results, time.time())
    
    def _update_performance_stats(self, query_time: float):
        """更新性能统计"""
        self.performance_stats.total_queries += 1
        
        if self.performance_stats.total_queries == 1:
            self.performance_stats.avg_query_time = query_time
        else:
            self.performance_stats.avg_query_time = (
                (self.performance_stats.avg_query_time * (self.performance_stats.total_queries - 1) + query_time) /
                self.performance_stats.total_queries
            )
    
    def close(self):
        """关闭连接"""
        try:
            if self.client:
                self.client.close()
            logger.info("Qdrant连接已关闭")
        except Exception as e:
            logger.warning(f"关闭Qdrant连接时出错: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
