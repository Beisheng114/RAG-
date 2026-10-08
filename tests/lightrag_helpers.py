"""Load real retrieval code with import-only stubs, without models/services."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_modules():
    package = ModuleType("lightrag_test")
    package.__path__ = [str(ROOT / "rag_modules")]
    documents = ModuleType("langchain_core.documents")
    documents.Document = Document
    model_names = ("Distance VectorParams PointStruct Filter FieldCondition MatchValue MatchText "
                   "Range IsEmptyCondition IsNullCondition HasIdCondition HnswConfigDiff "
                   "SearchParams PointIdsList").split()
    models = ModuleType("qdrant_client.http.models")
    for name in model_names:
        setattr(models, name, Model)
    models.Distance = SimpleNamespace(COSINE="Cosine", EUCLID="Euclidean", DOT="Dot")
    stubs = {
        "lightrag_test": package,
        "langchain_core": ModuleType("langchain_core"),
        "langchain_core.documents": documents,
        "langchain_community": ModuleType("langchain_community"),
        "langchain_community.retrievers": SimpleNamespace(BM25Retriever=MagicMock()),
        "langchain_huggingface": SimpleNamespace(HuggingFaceEmbeddings=MagicMock()),
        "neo4j": SimpleNamespace(GraphDatabase=MagicMock()),
        "qdrant_client": SimpleNamespace(QdrantClient=MagicMock()),
        "qdrant_client.http": SimpleNamespace(models=models),
        "qdrant_client.http.models": models,
    }
    loaded = {}
    with patch.dict(sys.modules, stubs):
        for name in ("graph_indexing", "hybrid_retrieval", "intelligent_query_router",
                     "qdrant_index_construction", "reranker"):
            spec = importlib.util.spec_from_file_location("lightrag_test." + name, ROOT / "rag_modules" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        rag_package = ModuleType("rag_modules")
        rag_package.GraphDataPreparationModule = MagicMock()
        rag_package.GenerationIntegrationModule = MagicMock()
        sys.modules["rag_modules"] = rag_package
        for name, module in loaded.items():
            sys.modules["rag_modules." + name] = module
        sys.modules["rag_modules.graph_rag_retrieval"] = SimpleNamespace(GraphRAGRetrieval=MagicMock())
        sys.modules["rag_modules.graph_data_insert"] = SimpleNamespace(GraphDataInsert=MagicMock())
        spec = importlib.util.spec_from_file_location("lightrag_test.ragmain", ROOT / "ragmain.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded["ragmain"] = module
    return SimpleNamespace(**loaded)


def make_hybrid(modules):
    from config import GraphRAGConfig
    hybrid = modules.hybrid_retrieval.HybridRetrievalModule(
        GraphRAGConfig(), MagicMock(), MagicMock(), MagicMock()
    )
    hybrid._get_node_neighbors = MagicMock(return_value=[])
    hybrid._get_node_name_from_neo4j = MagicMock(return_value="name")
    hybrid._get_node_source_info = MagicMock(return_value=[])
    hybrid._neo4j_entity_level_search = MagicMock(return_value=[])
    hybrid._neo4j_topic_level_search = MagicMock(return_value=[])
    hybrid.graph_indexing.vector_module = hybrid.vector_module
    hybrid.graph_indexing.vector_module.search_kv_vectors.return_value = []
    return hybrid


def seed_graph(modules, hybrid):
    graph = hybrid.graph_indexing
    entity = modules.graph_indexing.EntityKeyValue
    graph.entity_kv_store = {
        "e1": entity("主柴油机", ["主柴油机"], "主柴油机说明", "Equipment", {"neo4j_node_id": "e1"}),
        "e2": entity("排气温度过高", ["排气温度过高"], "温度异常", "FaultPhenomenon", {"neo4j_node_id": "e2"}),
    }
    graph._rebuild_key_mappings()
    graph.create_relation_key_values([("e1", "HAS_FAULT", "e2")])
    return graph
