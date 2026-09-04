"""
基于图数据库的RAG系统配置文件

敏感凭证（Neo4j 密码、管理口令、API Key 等）一律通过环境变量或 .env 文件
注入，代码中不保留任何默认密码。参见 .env.example。

注意：本模块是配置链最底层，在此处调用 load_dotenv()，确保任何入口
（app.py / ragmain.py / csv_to_neo4j.py）import 配置时 .env 已生效。
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv

# 先于字段默认值求值加载 .env（幂等，重复调用无副作用）
load_dotenv()

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置（凭证通过环境变量注入，见 .env.example；不内置默认密码）
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # 向量索引类型：当前仅支持 "qdrant"（Milvus 后端已移除）
    vector_index_type: str = "qdrant"

    # Qdrant配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_collection_name: str = "ship_maintenance_knowledge"
    qdrant_vector_size: int = 768  # BGE-base-zh-v1.5的向量维度（bge-base 为 768 维）
    qdrant_distance: str = "Cosine"  # Cosine, Euclidean, Dot
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
    # 检索时 HNSW ef（约 64–128）；略小更快、略降召回，与构建 ef_construct 无关
    qdrant_hnsw_ef_search: int = 128

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
    max_tokens: int = 4096  # 生成回答的最大 token 数（需不超过所用LLM的上下文窗口）

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
        # Milvus 后端已移除，仅支持 Qdrant
        if self.vector_index_type != "qdrant":
            raise ValueError(
                f"vector_index_type 仅支持 'qdrant'，当前值: {self.vector_index_type!r}（Milvus 后端已移除）"
            )
    
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