"""Web search tool for smolagents."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from ddgs import DDGS
from smolagents import Tool

from .prompts import WEB_SEARCH_TOOL_DESCRIPTION

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    name = "web_search"
    description = WEB_SEARCH_TOOL_DESCRIPTION
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.ddgs = DDGS(**kwargs)

    def _perform_search(self, query: str):
        return self.ddgs.text(query, max_results=self.max_results)

    def forward(self, query: str) -> str:
        results = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self._perform_search, query)
                results = future.result(timeout=30)
            except TimeoutError:
                logger.warning("First search attempt timed out after 30 seconds, retrying...")
                results = []

        if len(results) == 0:
            logger.info("Retrying search...")
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(self._perform_search, query)
                    results = future.result(timeout=30)
                except TimeoutError:
                    raise Exception("Search timed out after 30 seconds on both attempts. Try a different query.")

        if len(results) == 0:
            raise Exception("No results found after two attempts! Try a less restrictive/shorter query.")

        postprocessed_results = [
            f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
        ]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
