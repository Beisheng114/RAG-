"""
基于图RAG的船舶故障维修系统 - 主程序
整合传统检索和图RAG检索，实现真正的图数据优势
"""

import os
import sys
import time
import logging
from typing import List, Optional, Set, Tuple

# 知识图谱可视化：允许的故障树关系类型（与 graph_data_insert 中 MERGE 一致）
FAULT_GRAPH_REL_TYPES = [
    "CONTAINS",
    "CONSISTS_OF",
    "HAS_FAULT",
    "PRESENTS_AS",
    "CAUSED_BY",
    "RELATES_TO",
    "FIXED_BY",
    "HAS_NOTICE",
    "COMES_FROM",
]
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules import (
    GraphDataPreparationModule,
    GenerationIntegrationModule
)
from rag_modules.qdrant_index_construction import QdrantIndexConstructionModule, QdrantConfig
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis
from rag_modules.graph_data_insert import GraphDataInsert
from rag_modules.reranker import RerankModule

load_dotenv()


class AdvancedGraphRAGSystem:
    """
    图RAG系统

    核心特性：
    1. 智能路由：自动选择最适合的检索策略
    2. 双引擎检索：传统混合检索 + 图RAG检索
    3. 图结构推理：多跳遍历、子图提取、关系推理
    4. 查询复杂度分析：深度理解用户意图
    5. 多轮查询改写：检索前对指代问题做指代消解
    6. Rerank精排：RRF融合后用bge-reranker精排候选
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # 核心模块
        self.data_module = None
        self.index_module = None
        self.generation_module = None

        # 检索引擎
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        self.rerank_module = None

        # 系统状态
        self.system_ready = False

    def initialize_system(self):
        """初始化高级图RAG系统"""
        logger.info("启动高级图RAG系统...")

        try:
            # 1. 数据准备模块
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )

            # 2. 向量索引模块（当前仅支持 Qdrant）
            print("初始化Qdrant向量索引...")
            qdrant_config = QdrantConfig(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                grpc_port=self.config.qdrant_grpc_port,
                prefer_grpc=self.config.qdrant_prefer_grpc,
                collection_name=self.config.qdrant_collection_name,
                vector_size=self.config.qdrant_vector_size,
                distance=self.config.qdrant_distance,
                hnsw_config={
                    "m": self.config.qdrant_hnsw_m,
                    "ef_construct": self.config.qdrant_hnsw_ef_construct
                },
                hnsw_ef_search=int(getattr(self.config, "qdrant_hnsw_ef_search", 128)),
            )
            self.index_module = QdrantIndexConstructionModule(
                config=qdrant_config,
                embedding_model_path=self.config.embedding_model
            )

            # 3. 生成模块
            print("初始化生成模块...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                llm_provider=self.config.llm_provider,
                ollama_base_url=self.config.ollama_base_url,
                ollama_model=self.config.ollama_model,
                vllm_base_url=self.config.vllm_base_url
            )

            # 4. 传统混合检索模块
            print("初始化传统混合检索...")
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                vector_module=self.index_module,
                data_module=self.data_module,
                llm_client=self.generation_module.client
            )

            # 5. 图RAG检索模块
            print("初始化图RAG检索引擎...")
            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client
            )

            # 6. 智能查询路由器
            print("初始化智能查询路由器...")
            self.query_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                llm_client=self.generation_module.client,
                config=self.config
            )

            # 7. Rerank 精排模块（懒加载，模型缺失时自动降级）
            self.rerank_module = RerankModule(self.config)
            if self.rerank_module.is_enabled():
                print("初始化Rerank精排模块（懒加载）...")

            print("初始化数据插入系统")

            self.graph_data_insert = GraphDataInsert(
                llm_client=self.generation_module.client,
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                config=self.config,
                use_deepseek=False,
                llm_provider=self.config.llm_provider,
                ollama_model=self.config.ollama_model,
                ollama_base_url=self.config.ollama_base_url
            )


            print("✅ 高级图RAG系统初始化完成！")

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def build_knowledge_base(self):
        """构建知识库（如果需要）"""
        print("\n检查知识库状态...")

        try:
            if self.index_module.has_collection():
                print("✅ 发现已存在的Qdrant知识库，尝试加载...")
                if self.index_module.load_collection():
                    print("知识库加载成功！")

                    print("加载图数据以支持图检索...")
                    self.data_module.load_graph_data()
                    print("构建设备文档...")
                    self.data_module.build_documents()
                    print("进行文档分块...")
                    chunks = self.data_module.chunk_documents(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )

                    self._initialize_retrievers(chunks)
                    return
                else:
                    print("❌ 知识库加载失败，开始重建...")

            print("未找到已存在的Qdrant集合，开始构建新的知识库...")

            print("从Neo4j加载图数据...")
            self.data_module.load_graph_data()

            print("构建设备文档...")
            self.data_module.build_documents()

            print("进行文档分块...")
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            print("构建Qdrant向量索引...")
            if not self.index_module.build_vector_index(chunks):
                raise Exception("构建向量索引失败")

            print("创建高级索引（全文索引等）...")
            index_results = self.index_module.setup_advanced_indexes()
            if any(index_results.values()):
                print(f"✅ 高级索引创建成功")

            self._initialize_retrievers(chunks)

            self._show_knowledge_base_stats()

            print("✅ 知识库构建完成！")

        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise

    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器"""
        print("初始化检索引擎...")

        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []

        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)

        # 初始化图RAG检索器
        self.graph_rag_retrieval.initialize()

        self.system_ready = True
        print("✅ 检索引擎初始化完成！")

    def _show_knowledge_base_stats(self):
        """显示知识库统计信息"""
        print(f"\n知识库统计:")

        stats = self.data_module.get_statistics()
        print(f"   设备数量: {stats.get('total_equipments', 0)}")
        print(f"   部件数量: {stats.get('total_components', 0)}")
        print(f"   故障现象: {stats.get('total_fault_phenomenons', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")

        index_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {index_stats.get('points_count', 0)} 条记录 (Qdrant)")

        route_stats = self.query_router.get_route_statistics()
        print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")

        if stats.get('categories'):
            categories = list(stats['categories'].keys())[:10]
            print(f"   🏷️ 主要分类: {', '.join(categories)}")

    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False, conversation_history: list = None, search_mode: str = "intelligent"):
        """
        智能问答：自动选择最佳检索策略
        
        Args:
            question: 用户问题
            stream: 是否使用流式输出
            explain_routing: 是否解释路由决策
            conversation_history: 对话历史（可选）
            search_mode: 搜索模式 ("intelligent", "traditional", "graph", "combined")
            
        Returns:
            (result, analysis) 元组
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")

        print(f"\n❓ 用户问题: {question}")
        print(f"🔧 搜索模式: {search_mode}")

        # 多轮对话查询改写（指代消解）：用改写后的独立问题做检索
        retrieval_query = question
        if conversation_history and getattr(self.config, "enable_query_rewrite", True):
            retrieval_query = self.generation_module.rewrite_query(question, conversation_history)
            if retrieval_query != question:
                print(f"🔄 查询改写: {question} → {retrieval_query}")

        # 精排候选数：启用rerank时扩大召回，检索后统一精排截断到top_k
        recall_k = self.config.top_k
        if self.rerank_module and self.rerank_module.is_enabled():
            recall_k = max(self.config.top_k, int(getattr(self.config, "rerank_candidate_k", 20)))

        # 显示路由决策解释（可选）
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)

        start_time = time.time()

        try:
            # 根据搜索模式选择检索策略
            if search_mode == "traditional":
                print("🔍 使用传统双搜索模式...")
                relevant_docs = self.traditional_retrieval.hybrid_search(retrieval_query, recall_k)
                # 创建默认的 QueryAnalysis
                from rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy
                analysis = QueryAnalysis(
                    query_complexity=0.3,
                    relationship_intensity=0.3,
                    reasoning_required=False,
                    entity_count=1,
                    recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
                    confidence=1.0,
                    reasoning="用户选择传统双搜索模式"
                )
            elif search_mode == "graph":
                print("🕸️ 使用图搜索模式...")
                relevant_docs = self.graph_rag_retrieval.graph_rag_search(retrieval_query, recall_k)
                # 创建默认的 QueryAnalysis
                from rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy
                analysis = QueryAnalysis(
                    query_complexity=0.8,
                    relationship_intensity=0.8,
                    reasoning_required=True,
                    entity_count=2,
                    recommended_strategy=SearchStrategy.GRAPH_RAG,
                    confidence=1.0,
                    reasoning="用户选择图搜索模式"
                )
            elif search_mode == "combined":
                print("🔄 使用组合搜索模式...")
                # 并行执行传统检索和图检索
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_traditional = executor.submit(
                        self.traditional_retrieval.hybrid_search, retrieval_query, recall_k // 2
                    )
                    future_graph = executor.submit(
                        self.graph_rag_retrieval.graph_rag_search, retrieval_query, recall_k // 2
                    )

                    traditional_docs = future_traditional.result()
                    graph_docs = future_graph.result()

                # 合并结果
                relevant_docs = traditional_docs + graph_docs
                # 根据分数排序
                relevant_docs.sort(
                    key=lambda x: x.metadata.get('final_score', x.metadata.get('relevance_score', 0)),
                    reverse=True
                )
                relevant_docs = relevant_docs[:recall_k]
                # 创建默认的 QueryAnalysis
                from rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy
                analysis = QueryAnalysis(
                    query_complexity=0.6,
                    relationship_intensity=0.6,
                    reasoning_required=True,
                    entity_count=2,
                    recommended_strategy=SearchStrategy.COMBINED,
                    confidence=1.0,
                    reasoning="用户选择组合搜索模式"
                )
            else:
                # 智能路由模式（默认）
                print("🧠 使用智能路由模式...")
                relevant_docs, analysis = self.query_router.route_query(retrieval_query, recall_k)

            # Rerank 精排：多路候选 → 精排截断到 top_k（模型不可用时自动跳过）
            if self.rerank_module and self.rerank_module.is_enabled() and relevant_docs:
                relevant_docs = self.rerank_module.rerank(retrieval_query, relevant_docs, self.config.top_k)

            # 2. 显示路由信息
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🕸️",
                "combined": "🔄"
            }

            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")

            # 3. 显示检索结果信息
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    entity_name = doc.metadata.get('entity_name', '未知内容')
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"{entity_name}({search_type}, {score:.3f})")

                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                # 保持返回值签名一致：始终返回 (result, analysis)
                return "抱歉，没有找到相关的故障维修信息。请尝试其他问题。", analysis

            # 4. 生成回答
            print("🎯 智能生成回答...")

            if stream:
                # 返回生成器用于流式输出
                def stream_generator():
                    try:
                        full_response = ""
                        for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs, conversation_history):
                            if chunk_text:
                                full_response += chunk_text
                                yield chunk_text
                        # 性能统计
                        end_time = time.time()
                        print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")
                    except Exception as stream_error:
                        logger.error(f"流式输出过程中出现错误: {stream_error}")
                        error_msg = f"抱歉，流式输出出现错误：{str(stream_error)}"
                        yield error_msg
                
                return stream_generator(), analysis
            else:
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs, conversation_history)

                # 5. 性能统计
                end_time = time.time()
                print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")

                return result, analysis

        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None

    def run_interactive(self):
        """运行交互式问答"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return

        print("\n欢迎使用船舶故障维修RAG系统！")
        print("可用功能：")
        print("   - 'stats' : 查看系统统计")
        print("   - 'rebuild' : 重建知识库")
        print("   - 'quit' : 退出系统")
        print("\n" + "=" * 50)

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'rebuild':
                    self._rebuild_knowledge_base()
                    continue
                elif user_input.lower() == 'insert':
                    print("请输入插入的案例")
                    case = input("案例: ")
                    use_deepseek_input = input("是否使用Deepseek提取数据？(y/N): ").strip().lower()
                    use_deepseek = use_deepseek_input == 'y'
                    result = self.graph_data_insert.insert_case(case, use_deepseek=use_deepseek)
                    if result.get("success"):
                        neo4j_node_ids = result.get("neo4j_node_ids", []) or []
                        self.update_knowledge_base_incremental(neo4j_node_ids)
                    continue

                # 普通问答 - 使用默认设置
                use_stream = True  # 默认使用流式输出
                explain_routing = False  # 默认不显示路由决策

                print("\n回答:")

                result, analysis = self.ask_question_with_routing(
                    user_input,
                    stream=use_stream,
                    explain_routing=explain_routing
                )

                if not use_stream and result:
                    print(f"{result}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n👋 感谢使用船舶故障维修RAG系统！")
        self._cleanup()

    def _show_system_stats(self):
        """显示系统统计信息"""
        print("\n系统运行统计")
        print("=" * 40)

        # 路由统计
        route_stats = self.query_router.get_route_statistics()
        total_queries = route_stats.get('total_queries', 0)

        if total_queries > 0:
            print(f"总查询次数: {total_queries}")
            print(
                f"传统检索: {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})")
            print(f"图RAG检索: {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})")
            print(f"组合策略: {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})")
        else:
            print("暂无查询记录")

        # 知识库统计
        self._show_knowledge_base_stats()

    def _rebuild_knowledge_base(self,key:bool=False):
        """重建知识库"""
        print("\n准备重建知识库...")
        if key == False:
            confirm = input("⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ 重建操作已取消")
                return

        try:
            print("删除现有的Qdrant集合...")
            if self.index_module.delete_collection():
                print("✅ 现有Qdrant集合已删除")
            else:
                print("删除集合时出现问题，继续重建...")

            print("开始重建知识库...")
            self.build_knowledge_base()

            print("✅ 知识库重建完成！")

        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            print(f"❌ 重建失败: {e}")
            print("建议：请检查Qdrant服务状态后重试")

    def update_knowledge_base_incremental(self, neo4j_node_ids: List[str]):
        """
        增量更新知识库向量索引（不删除/重建整个集合）。

        流程：
        1) 仅对本次新增的 neo4j_node_id 构建文档/切块并嵌入
        2) 先删除旧的（同 neo4j_node_id）向量点，再添加新向量点
        3) 刷新 BM25（传统检索）与图RAG的索引缓存（不涉及向量库全量重建）
        """
        if not neo4j_node_ids:
            logger.info("update_knowledge_base_incremental: 未提供节点 id，跳过")
            return

        try:
            logger.info(f"开始增量更新向量索引：节点数={len(neo4j_node_ids)}")

            # 如果向量集合不存在，直接走全量构建
            if not getattr(self.index_module, "has_collection", lambda: True)():
                logger.info("向量集合不存在，执行全量构建")
                self.build_knowledge_base()
                return

            # 1) 增量构建文档 + 切块（仅指定节点）
            self.data_module.load_graph_data()
            subset_documents = self.data_module.build_documents_for_node_ids(neo4j_node_ids)
            if not subset_documents:
                logger.info("本次节点没有可构建文档，跳过向量更新")
            else:
                subset_chunks = self.data_module.chunk_documents(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

                # 2) 删除旧向量点（按 neo4j_node_id）
                if hasattr(self.index_module, "delete_by_neo4j_node_ids"):
                    self.index_module.delete_by_neo4j_node_ids(neo4j_node_ids)

                # 3) 添加新向量点
                if hasattr(self.index_module, "add_documents"):
                    self.index_module.add_documents(subset_chunks)
                else:
                    # 兜底：若没有 add_documents，则回退到 build_vector_index
                    if hasattr(self.index_module, "build_vector_index"):
                        self.index_module.build_vector_index(subset_chunks)

            # 4) 刷新传统检索与图RAG索引缓存（不触碰向量库全量重建）
            self.data_module.load_graph_data()
            self.data_module.build_documents()
            full_chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            self._initialize_retrievers(full_chunks)

            logger.info("增量更新完成")
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            # 兜底：回退到全量重建，确保系统可用
            try:
                logger.info("增量更新失败，回退执行全量重建")
                self._rebuild_knowledge_base(key=True)
            except Exception:
                logger.exception("全量重建也失败")

    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()

    def query_graph(self, query: str, entity_type: str = "all", node_limit: int = 200, system_name: str = "all"):
        """
        查询知识图谱
        
        Args:
            query: 查询关键词
            entity_type: 实体类型，默认为"all"
            node_limit: 节点数量限制，默认为200
            system_name: 所属系统过滤（如 动力系统/电力系统），默认为"all"
            
        Returns:
            (nodes, edges, stats) 元组
        """
        try:
            driver = self.data_module.driver

            nodes: List[dict] = []
            edges: List[dict] = []
            stats = {}

            node_limit = min(max(int(node_limit), 50), 500)
            depth = min(max(int(getattr(self.config, "max_graph_depth", 4)), 1), 8)
            anchor_lim = min(25, max(5, node_limit // 20))
            path_lim = min(200, max(40, node_limit))
            # 浏览模式：按「边」采样，保证每条边两端节点成对出现（修复原先节点/边两次独立 LIMIT 导致的不连通）
            browse_edge_lim = min(4000, max(200, node_limit * 5))

            def node_to_item(n) -> dict:
                n_id = n.id
                node_type = list(n.labels)[0] if n.labels else "Unknown"
                if node_type == "FaultReason":
                    label = n.get("cause_name") or str(n_id)
                elif node_type in ("MaintenanceAction", "FaultPhenomenon", "SafetyNotice"):
                    label = n.get("description") or str(n_id)
                elif node_type == "Fault":
                    label = n.get("name") or str(n_id)
                else:
                    label = n.get("name") or str(n_id)
                extra = n.get("description") or n.get("cause_name") or ""
                node_system = n.get("system_name")
                title_lines = [f"{label} ({node_type})"]
                if node_system:
                    title_lines.append(f"所属系统: {node_system}")
                if extra and extra != label:
                    title_lines.append(f"描述: {extra[:300]}")
                title = "\n".join(title_lines)
                return {
                    "id": str(n_id),
                    "label": label,
                    "type": node_type,
                    "title": title,
                    "system_name": node_system,
                }

            def ingest_triplet(
                n,
                m,
                r,
                seen_nodes: Set[int],
                seen_edges: Set[Tuple[int, int, str]],
            ) -> None:
                # 强制使用关系真实方向，避免无向 MATCH 导致前端看起来“双向箭头”
                start_n = r.start_node
                end_n = r.end_node
                n_id, m_id = start_n.id, end_n.id

                if n_id not in seen_nodes:
                    seen_nodes.add(n_id)
                    nodes.append(node_to_item(start_n))
                if m_id not in seen_nodes:
                    seen_nodes.add(m_id)
                    nodes.append(node_to_item(end_n))

                # 关系去重（无向可视化去重）：将 A->B 与 B->A 视为同一条边，避免前端出现双向重复连线
                edge_key = (min(n_id, m_id), max(n_id, m_id), r.type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    # 保留首次出现方向用于展示
                    edges.append({"from": str(n_id), "to": str(m_id), "label": r.type})

            system_name = (system_name or "all").strip()
            sys_lower = system_name.lower()

            if not query or query.strip() == "":
                if sys_lower != "all":
                    # 系统过滤浏览模式：从该系统设备出发扩展子图，避免只显示设备孤点
                    browse_cypher = """
                    MATCH (e:Equipment)
                    WHERE toLower(coalesce(e.system_name, '')) = toLower($sys)
                    WITH DISTINCT e
                    LIMIT $anchor_lim
                    MATCH p = (e)-[*1..2]-(x)
                    UNWIND relationships(p) AS rel
                    RETURN startNode(rel) AS n, rel AS r, endNode(rel) AS m
                    LIMIT $lim
                    """
                elif entity_type == "all":
                    browse_cypher = """
                    MATCH (n)-[r]-(m)
                    WITH n, r, m
                    LIMIT $lim
                    RETURN n, r, m
                    """
                else:
                    browse_cypher = """
                    MATCH (n)-[r]-(m)
                    WHERE $etype IN labels(n) OR $etype IN labels(m)
                    WITH n, r, m
                    LIMIT $lim
                    RETURN n, r, m
                    """
                with driver.session() as session:
                    seen_nodes: Set[int] = set()
                    seen_edges: Set[Tuple[int, int, str]] = set()
                    result = session.run(
                        browse_cypher,
                        {
                            "lim": browse_edge_lim,
                            "etype": entity_type,
                            "sys": system_name,
                            "anchor_lim": anchor_lim,
                        },
                    )
                    for record in result:
                        ingest_triplet(record["n"], record["m"], record["r"], seen_nodes, seen_edges)
            else:
                q = query.strip()
                # 优先走全文索引召回 anchor，避免大范围 CONTAINS + OR 扫描
                subgraph_cypher = f"""
                CALL db.index.fulltext.queryNodes('idx_fulltext_anchor', $q) YIELD node AS anchor, score
                WHERE ($etype = 'all' OR $etype IN labels(anchor))
                  AND ($sys = 'all' OR toLower(coalesce(anchor.system_name, '')) = toLower($sys))
                WITH DISTINCT anchor, score
                ORDER BY score DESC
                LIMIT $anchor_lim
                MATCH p = (anchor)-[*1..{depth}]-(x)
                WHERE ALL(rel IN relationships(p) WHERE type(rel) IN $rel_types)
                WITH p
                LIMIT $path_lim
                UNWIND relationships(p) AS rel
                RETURN startNode(rel) AS n, rel AS r, endNode(rel) AS m
                """
                fallback_cypher = """
                CALL db.index.fulltext.queryNodes('idx_fulltext_anchor', $q) YIELD node AS anchor, score
                WHERE ($etype = 'all' OR $etype IN labels(anchor))
                  AND ($sys = 'all' OR toLower(coalesce(anchor.system_name, '')) = toLower($sys))
                WITH anchor, score
                ORDER BY score DESC
                LIMIT $anchor_lim
                MATCH (anchor)-[r]-(m)
                RETURN anchor AS n, r, m
                LIMIT $onehop_lim
                """
                # 兜底：全文索引不存在时，退回原 CONTAINS 逻辑
                legacy_contains_cypher = f"""
                MATCH (anchor)
                WHERE ($etype = 'all' OR $etype IN labels(anchor))
                AND ($sys = 'all' OR toLower(coalesce(anchor.system_name, '')) = toLower($sys))
                AND (
                  toLower(coalesce(anchor.name, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.description, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.cause_name, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.action_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.notice_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.source_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.equipment_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.component_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.fault_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.phenomenon_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.cause_id, '')) CONTAINS toLower($q)
                  OR toLower(coalesce(anchor.category_id, '')) CONTAINS toLower($q)
                )
                WITH DISTINCT anchor
                LIMIT $anchor_lim
                MATCH p = (anchor)-[*1..{depth}]-(x)
                WHERE ALL(rel IN relationships(p) WHERE type(rel) IN $rel_types)
                WITH p
                LIMIT $path_lim
                UNWIND relationships(p) AS rel
                RETURN startNode(rel) AS n, rel AS r, endNode(rel) AS m
                """
                params = {
                    "q": q,
                    "etype": entity_type,
                    "sys": system_name,
                    "anchor_lim": anchor_lim,
                    "path_lim": path_lim,
                    "rel_types": FAULT_GRAPH_REL_TYPES,
                    "onehop_lim": min(300, node_limit * 2),
                }
                with driver.session() as session:
                    seen_nodes = set()
                    seen_edges = set()
                    try:
                        records = list(session.run(subgraph_cypher, params))
                        if not records:
                            records = list(session.run(fallback_cypher, params))
                    except Exception:
                        records = list(session.run(legacy_contains_cypher, params))
                    for record in records:
                        ingest_triplet(record["n"], record["m"], record["r"], seen_nodes, seen_edges)

            # 统计信息
            stats["total_nodes"] = len(nodes)
            stats["total_edges"] = len(edges)
            
            # 按类型统计节点
            node_types = {}
            for node in nodes:
                node_type = node["type"]
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
            stats["node_types"] = node_types
            
            return nodes, edges, stats
        except Exception as e:
            logger.error(f"图谱查询失败: {e}")
            import traceback
            traceback.print_exc()
            return [], [], {}
    
    def get_node_type_counts(self):
        """
        获取各类节点的总数
        
        Returns:
            Dict[str, int]: 节点类型到数量的映射
        """
        try:
            driver = self.data_module.driver
            counts = {}
            
            # 获取所有节点标签
            with driver.session() as session:
                # 查询所有不同的节点标签
                result = session.run("""
                    CALL db.labels() YIELD label
                    RETURN label
                """)
                labels = [record["label"] for record in result]
                
                # 统计每种类型的节点数量
                for label in labels:
                    count_result = session.run(f"""
                        MATCH (n:{label})
                        RETURN count(n) as count
                    """)
                    count = count_result.single()["count"]
                    counts[label] = count
            
            return counts
        except Exception as e:
            logger.error(f"获取节点类型总数失败: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """主函数"""
    try:
        print("启动高级图RAG系统...")

        # 创建高级图RAG系统
        rag_system = AdvancedGraphRAGSystem()

        # 初始化系统
        rag_system.initialize_system()

        # 构建知识库
        rag_system.build_knowledge_base()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")


if __name__ == "__main__":
    main()