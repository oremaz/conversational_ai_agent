"""RAG helpers and logging wrappers for the LlamaIndex agent."""

import logging
from typing import List, Optional

import weave
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

from .custom_models import (
    get_or_create_jina_embedder,
    get_or_create_jina_reranker,
    get_or_create_qwen_embedder,
    get_or_create_qwen_reranker,
)
from . import models

logger = logging.getLogger(__name__)


class LoggingQueryEngine(BaseQueryEngine):
    """Wrap a BaseQueryEngine to add structured logging."""

    def __init__(self, inner_engine: BaseQueryEngine, name: str = "dynamic_hybrid_multimodal_rag_tool"):
        self._inner_engine = inner_engine
        self._logger = logging.getLogger(__name__)
        self._tool_name = name

    def query(self, query: str, **kwargs):
        self._logger.info("%s query received: %s", self._tool_name, query)
        try:
            response = self._inner_engine.query(query, **kwargs)
            self._logger.info("%s query succeeded", self._tool_name)
            return response
        except Exception:
            self._logger.exception("%s query failed", self._tool_name)
            raise

    async def aquery(self, query: str, **kwargs):
        self._logger.info("%s async query received: %s", self._tool_name, query)
        try:
            response = await self._inner_engine.aquery(query, **kwargs)
            self._logger.info("%s async query succeeded", self._tool_name)
            return response
        except Exception:
            self._logger.exception("%s async query failed", self._tool_name)
            raise

    def _query(self, query: str, **kwargs):
        if hasattr(self._inner_engine, "_query"):
            return self._inner_engine._query(query, **kwargs)
        return self._inner_engine.query(query, **kwargs)

    async def _aquery(self, query: str, **kwargs):
        if hasattr(self._inner_engine, "_aquery"):
            return await self._inner_engine._aquery(query, **kwargs)
        return await self._inner_engine.aquery(query, **kwargs)

    def _get_prompt_modules(self):
        if hasattr(self._inner_engine, "_get_prompt_modules"):
            return self._inner_engine._get_prompt_modules()
        return {}

    def __getattr__(self, item):
        return getattr(self._inner_engine, item)


def get_rag_embedder():
    """Return the configured embedder for local retrieval."""
    if models.RAG_PROVIDER == "qwen":
        return get_or_create_qwen_embedder()
    return get_or_create_jina_embedder()


def get_rag_reranker(top_n: int = 5):
    """Return the configured reranker for local retrieval."""
    if models.RAG_PROVIDER == "qwen":
        return get_or_create_qwen_reranker(model_name="Qwen/Qwen3-VL-Reranker-2B", top_n=top_n)
    return get_or_create_jina_reranker(model_name="jinaai/jina-reranker-m0", top_n=top_n)


@weave.op
def create_temporary_web_search_index(documents: List[Document], llm=None) -> Optional[VectorStoreIndex]:
    """Create a temporary in-memory RAG index for web search results."""
    if not documents:
        logger.warning("No documents provided for temporary index")
        return None

    try:
        embed_model = get_rag_embedder()

        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 256],
            chunk_overlap=128,
        )

        all_nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)

        logger.info("Created temporary index with %d leaf nodes", len(leaf_nodes))

        index = VectorStoreIndex(
            nodes=leaf_nodes,
            embed_model=embed_model,
        )

        return index

    except Exception as exc:
        logger.exception("Failed to create temporary web search index: %s", exc)
        return None
