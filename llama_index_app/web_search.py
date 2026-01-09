"""Web search helpers for the LlamaIndex agent."""

import logging
from typing import Dict, Any, List
from urllib.parse import urlparse

from ddgs import DDGS
from llama_index.core import Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

from . import models
from .rag import create_temporary_web_search_index, get_rag_reranker
from .model_wrappers.utils import offload_rag_models
from .utils.document_processor import UserAgentWebPageReader

logger = logging.getLogger(__name__)


def _extract_urls_from_results(results: List[Dict[str, Any]], max_results: int) -> List[str]:
    urls: List[str] = []
    for result in results:
        url = (
            result.get("href")
            or result.get("link")
            or result.get("url")
            or result.get("FirstURL")
            or result.get("first_url")
        )
        if not url:
            continue
        url = str(url).rstrip(").,;:'\"")
        if url not in urls:
            urls.append(url)
        if len(urls) >= max_results:
            break
    return urls


def search_for_urls(query: str, max_results: int = 3) -> List[str]:
    logger.info("[web] start search: %s", query)
    ddgs_errors: List[str] = []
    backend_attempts = [
        ("google", {"backend": "google"}),
        ("default", {}),
    ]

    for backend_name, backend_kwargs in backend_attempts:
        try:
            kwargs = {"max_results": max_results}
            kwargs.update(backend_kwargs)
            with DDGS() as ddg:
                results = list(ddg.text(query, **kwargs))
            logger.info("[web] ddgs backend='%s' results: %d", backend_name, len(results))
            urls = _extract_urls_from_results(results, max_results)
            if urls:
                return urls
        except Exception as exc:
            ddgs_errors.append(f"{backend_name}: {exc}")
            logger.warning("[web] ddgs failed (backend=%s): %s", backend_name, exc)
    if ddgs_errors:
        logger.warning("[web] ddgs attempts exhausted: %s", "; ".join(ddgs_errors))

    logger.info("[web] no URLs extracted (blocked/timeout/empty)")
    return []


def extract_documents_from_url(url: str) -> List[Document]:
    url = str(url).rstrip(").,;:'\"")
    logger.info("[web] fetching url: %s", url)
    try:
        netloc = urlparse(url).netloc.lower()
        if "youtube" in netloc or "youtu.be" in netloc:
            logger.info("[web] using YoutubeTranscriptReader()")
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[url])
        else:
            logger.info("[web] using SimpleWebPageReader()")
            default_loader = SimpleWebPageReader(html_to_text=True, fail_on_error=True)
            ua_loader = UserAgentWebPageReader(html_to_text=True, fail_on_error=True)
            try:
                documents = default_loader.load_data(urls=[url])
            except Exception as exc:
                logger.warning("[web] default web reader failed; retrying with User-Agent: %s", exc)
                documents = ua_loader.load_data(urls=[url])

        for doc in documents:
            if not getattr(doc, "metadata", None):
                doc.metadata = {}
            doc.metadata["source"] = url
            doc.metadata["type"] = "web_text"

        logger.info("[web] extracted %d documents from %s", len(documents), url)
        return documents
    except Exception as exc:
        logger.warning("[web] fetch failed for %s: %s", url, exc)
        return [
            Document(
                text=f"Error extracting content from URL: {exc}",
                metadata={"type": "web_text", "source": url, "error": True},
            )
        ]


def search_and_extract_content_from_url(query: str, max_results: int = 3) -> List[Document]:
    urls = search_for_urls(query, max_results=max_results)
    if not urls:
        return [
            Document(
                text="No URL could be extracted from the search results.",
                metadata={"type": "web_text", "source": "search", "error": True},
            )
        ]

    documents: List[Document] = []
    for url in urls:
        documents.extend(extract_documents_from_url(url))
    return documents


def format_web_search_documents(documents: List[Document]) -> str:
    sources: List[str] = []
    text_by_source: Dict[str, List[str]] = {}
    for doc in documents:
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        if source not in text_by_source:
            text_by_source[source] = []
            sources.append(source)
        text_by_source[source].append(doc.text or "")

    parts: List[str] = []
    for source in sources:
        body = "\n\n".join(text_by_source[source]).strip()
        if not body:
            continue
        parts.append(f"Source: {source}\n\n{body}")
    return "\n\n---\n\n".join(parts)


def enhanced_web_search_and_query(query: str) -> str:
    """Run web search and return either raw pages or a RAG answer."""
    documents = search_and_extract_content_from_url(query, max_results=3)

    valid_documents = [
        doc for doc in documents
        if doc.text and not (doc.metadata or {}).get("error")
    ]
    if not valid_documents:
        error_msg = documents[0].text if documents else "No content extracted"
        logger.warning("Web search failed for query '%s': %s", query, error_msg)
        return f"Failed to extract web content: {error_msg}"

    if models.USE_API_MODE:
        logger.info("API mode web search: returning raw content for %d documents", len(valid_documents))
        return format_web_search_documents(valid_documents)

    logger.info("Creating temporary index for %d documents from web search", len(valid_documents))

    try:
        temp_index = create_temporary_web_search_index(valid_documents, llm=models.proj_llm)

        if temp_index is None:
            return "Failed to create temporary search index"

        try:
            reranker = get_rag_reranker(top_n=5)

            query_engine = temp_index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[reranker],
                response_mode="tree_summarize",
                llm=models.proj_llm,
            )

            response = query_engine.query(query)
            result_text = str(response)

            logger.info("Temporary web search RAG completed successfully")
            return result_text

        except Exception as exc:
            logger.exception("Failed to query temporary index: %s", exc)
            text_docs = [doc for doc in documents if doc.metadata.get("type") == "web_text"]
            if text_docs:
                return text_docs[0].text[:2000]
            return "Error processing web search results"
    finally:
        try:
            offload_rag_models()
        except Exception as exc:
            logger.debug("RAG offload skipped: %s", exc)
