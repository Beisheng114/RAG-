"""
基于图数据库的RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "myrag123456"
    neo4j_database: str = "neo4j"

    # 向量索引类型:"milvus", "qdrant"
    vector_index_type: str = "qdrant"
    
    # Milvus配置（当vector_index_type为milvus时使用）
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "ship_maintenance_knowledge"
    milvus_dimension: int = 512  # BGE-base-zh-v1.5的向量维度

    
    # Qdrant配置（当vector_index_type为qdrant时使用）
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_collection_name: str = "ship_maintenance_knowledge"
    qdrant_vector_size: int = 768  # BGE-small-zh-v1.5的向量维度
    qdrant_distance: str = "Cosine"  # Cosine, Euclidean, Dot
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
    # 检索时 HNSW ef（约 64–128）；略小更快、略降召回，与构建 ef_construct 无关
    qdrant_hnsw_ef_search: int = 128

    # FAISS兼容字段（旧配置保留，避免管理页面预览时报错）
    faiss_index_path: str = "./faiss_index"
    faiss_dimension: int = 768

    # 模型配置
    embedding_model: str = "./models/bge-base-zh-v1.5"
    
    # LLM配置
    # 使用本地vLLM服务器（默认）
    llm_provider: str = "ollama"  # 可选: "vllm", "ollama"
    llm_model: str = "qwen3.5:2b"  ##最准

    # DeepSeek 配置（当 llm_provider 为 deepseek 时使用）
    deepseek_base_url: str = "https://api.deepseek.com"

    # Ollama配置（当llm_provider为ollama时使用）
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "kamekichi128/qwen3-4b-instruct-2507"  # Ollama中的模型名称
    # ollama_model:str = "qwen3.5:2b"
    
    # vLLM配置（当llm_provider为vllm时使用）
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
        
    # 检索配置（LightRAG Round-robin策略）；多原因/多片段问题建议 >=5
    top_k: int = 5

    # 生成配置
    temperature: float = 0.3
    max_tokens: int = 4096  # 适配 Qwen3-4B 模型的最大上下文 1024

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 4  # 图遍历最大深度

    # 图索引配置
    enable_llm_relation_keys: bool = False  # 是否使用LLM增强关系索引键

    # 检索策略配置
    use_intelligent_router: bool = True  # 是否使用智能查询路由
    hybrid_search_weight: float = 0.5  # 混合检索权重

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'vector_index_type': self.vector_index_type,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'faiss_index_path': self.faiss_index_path,
            'faiss_dimension': self.faiss_dimension,
            'qdrant_host': self.qdrant_host,
            'qdrant_port': self.qdrant_port,
            'qdrant_grpc_port': self.qdrant_grpc_port,
            'qdrant_prefer_grpc': self.qdrant_prefer_grpc,
            'qdrant_collection_name': self.qdrant_collection_name,
            'qdrant_vector_size': self.qdrant_vector_size,
            'qdrant_distance': self.qdrant_distance,
            'qdrant_hnsw_m': self.qdrant_hnsw_m,
            'qdrant_hnsw_ef_construct': self.qdrant_hnsw_ef_construct,
            'qdrant_hnsw_ef_search': self.qdrant_hnsw_ef_search,
            'embedding_model': self.embedding_model,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'ollama_base_url': self.ollama_base_url,
            'ollama_model': self.ollama_model,
            'vllm_base_url': self.vllm_base_url,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'enable_llm_relation_keys': self.enable_llm_relation_keys,
            'use_intelligent_router': self.use_intelligent_router,
            'hybrid_search_weight': self.hybrid_search_weight
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 