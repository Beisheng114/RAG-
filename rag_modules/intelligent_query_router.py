"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现

需要联网，需要修改部分逻辑。
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要推理
    entity_count: int  # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由

class IntelligentQueryRouter:
    """
    智能查询路由器
    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """
    
    def __init__(self, 
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图RAG检索模块
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        
        # Ollama支持
        self.llm_provider = config.llm_provider
        self.ollama_base_url = config.ollama_base_url.rstrip('/')
        self.ollama_model = config.ollama_model
        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        深度分析查询特征，决定最佳检索策略
        """
        logger.info(f"分析查询特征: {query}")
        if self.config.use_intelligent_router != True:
            return self._rule_based_analysis(query)
        # 使用LLM进行智能分析
        analysis_prompt = f"""
        作为RAG系统的查询分析专家，请深度分析以下查询的特征：
        查询：{query}
        请从以下维度分析：
        1. 查询复杂度 (0-1)：
           - 0.0-0.3: 简单信息查找（如：轴承的型号有哪些？）
           - 0.4-0.7: 中等复杂度（如：船舶主柴油机有哪些部件？）
           - 0.8-1.0: 高复杂度推理（如：为什么船舶主柴油机的曲轴轴承容易发生故障？）
        2. 关系密集度 (0-1)：
           - 0.0-0.3: 单一实体信息（如：传感器的工作原理）
           - 0.4-0.7: 实体间关系（如：轴承故障与哪些部件相关？）
           - 0.8-1.0: 复杂关系网络（如：故障现象、原因与维修步骤之间的关系）
        3. 推理需求：
           - 是否需要多跳推理？
           - 是否需要因果分析？
           - 是否需要对比分析？
        4. 实体识别：
           - 查询中包含多少个明确实体？
           - 实体类型是什么？
        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的信息查找
        - graph_rag: 适合复杂关系推理和知识发现
        - combined: 需要两种策略结合
        返回JSON格式：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 3,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
        }}
        """
        
        try:
            if self.llm_provider == "ollama":
                # 使用Ollama原生API
                import requests
                url = f"{self.ollama_base_url}/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512
                    }
                }
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result_json = response.json()
                content = result_json["message"]["content"].strip()
            else:
                # 使用vLLM OpenAI兼容API
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                content = response.choices[0].message.content.strip()
            
            # 处理Qwen3-0.6B返回的<|think|>标签
            if "<|think|>" in content:
                # 移除<|think|>...</|think|>标签及其内容
                content = content.split("<|think|>")[-1].split("</|think>")[-1].strip()
            
            # 清理Markdown代码块标记
            if content.startswith("```json"):
                content = content[7:]  # 移除```json
            elif content.startswith("```"):
                content = content[3:]   # 移除```
            
            if content.endswith("```"):
                content = content[:-3]  # 移除结尾的```
            
            content = content.strip()
            
            result = json.loads(content)
            
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )
            
            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            #当出现无法联网的时候也可以利用这个来回答，或者后期通过接入模型来改善本地回答的质量.
            # 降级方案：基于规则的简单分析
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析"""
        # 简单的规则判断
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别", "故障", "维修", "步骤", "注意事项"]
        relation_keywords = ["属于", "包含", "组成", "相关", "联系", "连接", "导致", "表现为", "需通过", "附带"]
        
        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)
        
        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL
            
        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )
    
    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎
        """
        logger.info(f"开始智能路由: {query}")
        # 1. 分析查询特征
        analysis = self.analyze_query(query)
        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)
        # 3. 根据策略执行检索
        documents = []
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)
            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, top_k)
            # 4. 结果后处理
            documents = self._post_process_results(documents, analysis)
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis
    
    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：结合传统检索和图RAG的优势（多线程优化版）
        """
        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k
        
        # 使用多线程并行执行两种检索
        traditional_docs = []
        graph_docs = []
        
        def traditional_search():
            start_time = time.time()
            docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
            elapsed = time.time() - start_time
            logger.info(f"传统检索完成: {len(docs)} 个结果, 耗时: {elapsed:.2f}s")
            return docs, elapsed
        
        def graph_search():
            start_time = time.time()
            docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)
            elapsed = time.time() - start_time
            logger.info(f"图RAG检索完成: {len(docs)} 个结果, 耗时: {elapsed:.2f}s")
            return docs, elapsed
        
        # 并行执行两种检索
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_traditional = executor.submit(traditional_search)
            future_graph = executor.submit(graph_search)
            
            # 获取结果
            try:
                traditional_docs, traditional_time = future_traditional.result(timeout=8.0)
            except Exception as e:
                logger.error(f"传统检索失败: {e}")
                traditional_docs, traditional_time = [], 0
            
            try:
                graph_docs, graph_time = future_graph.result(timeout=8.0)
            except Exception as e:
                logger.error(f"图RAG检索失败: {e}")
                graph_docs, graph_time = [], 0
        
        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    # 添加文本相关信息
                    doc.metadata["text_info"] = {
                        "char_count": len(doc.page_content),
                        "word_count": len(doc.page_content.split()),
                        "line_count": doc.page_content.count('\n') + 1,
                        "retrieval_time": graph_time
                    }
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    # 添加文本相关信息
                    doc.metadata["text_info"] = {
                        "char_count": len(doc.page_content),
                        "word_count": len(doc.page_content.split()),
                        "line_count": doc.page_content.count('\n') + 1,
                        "retrieval_time": traditional_time
                    }
                    combined_docs.append(doc)
        
        # 记录合并后的统计信息
        logger.info(f"组合搜索完成: 共 {len(combined_docs)} 个结果 (传统: {len(traditional_docs)}, 图RAG: {len(graph_docs)})")
        
        return combined_docs[:top_k]
    
    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })
        
        return documents
    
    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1
        
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }
    
    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)
        
        explanation = f"""
        查询路由分析报告
        查询：{query}
        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}
        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}
        决策理由：{analysis.reasoning}
        """
        
        return explanation
 