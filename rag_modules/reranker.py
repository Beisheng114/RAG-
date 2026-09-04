"""
Rerank 精排模块（基于 CrossEncoder，默认 BAAI/bge-reranker-base）

解决 RRF 融合后直接取 top_k 导致排序精度不足的问题：
多路召回 rerank_candidate_k（默认 20）条候选 → 精排 → 截断到 top_k（默认 5）。

设计原则：
- 懒加载：首次调用时才加载模型，避免拖慢启动
- 自动降级：模型未下载/加载失败/推理异常时跳过精排，原样返回候选前 top_k，
  不影响主链路可用性（仅记录一次告警）
- 可通过 config.enable_rerank 一键关闭

模型下载：python download_model.py --rerank
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class RerankModule:
    """CrossEncoder 精排模块"""

    def __init__(self, config):
        """
        Args:
            config: GraphRAGConfig，读取 enable_rerank / rerank_model /
                rerank_candidate_k 字段
        """
        self.enabled = bool(getattr(config, "enable_rerank", False))
        self.model_path = getattr(config, "rerank_model", "./models/bge-reranker-base")
        self.candidate_k = int(getattr(config, "rerank_candidate_k", 20))
        self._model = None
        self._load_failed = False

    def is_enabled(self) -> bool:
        """是否启用精排（配置开关）"""
        return self.enabled

    def _ensure_model(self):
        """懒加载精排模型；失败后本次进程不再重试"""
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_path, max_length=512)
            logger.info(f"Rerank 模型加载完成: {self.model_path}")
        except Exception as e:
            self._load_failed = True
            logger.warning(
                f"Rerank 模型不可用（{e}），将跳过精排。"
                f"如需启用：python download_model.py --rerank"
            )
        return self._model

    def rerank(self, query: str, documents: List, top_k: int) -> List:
        """
        对候选文档精排并截断到 top_k

        Args:
            query: 查询文本（建议使用改写后的独立问题）
            documents: 候选 Document 列表（已按粗排分数排序）
            top_k: 精排后保留数量

        Returns:
            精排后的 Document 列表；不可用/异常时返回前 top_k 原顺序
        """
        if not documents:
            return documents
        # 候选数不超过目标数时无需精排
        if len(documents) <= top_k:
            return documents

        model = self._ensure_model()
        if model is None:
            return documents[:top_k]

        try:
            pairs = [
                (query, str(getattr(doc, "page_content", ""))[:1500])
                for doc in documents
            ]
            scores = model.predict(pairs)
            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

            results = []
            for rank, (doc, score) in enumerate(scored[:top_k], start=1):
                metadata = getattr(doc, "metadata", None)
                if metadata is not None:
                    metadata["rerank_score"] = float(score)
                    metadata["rerank_rank"] = rank
                results.append(doc)

            logger.info(f"Rerank 精排完成: {len(documents)} 候选 -> top {len(results)}")
            return results
        except Exception as e:
            logger.warning(f"Rerank 执行失败，降级为粗排顺序: {e}")
            return documents[:top_k]
