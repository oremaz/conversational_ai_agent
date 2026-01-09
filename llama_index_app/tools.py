"""Tool helpers for the LlamaIndex agent."""

import contextlib
import io
import logging
from typing import Dict

from .web_search import enhanced_web_search_and_query

logger = logging.getLogger(__name__)


def make_enhanced_web_search_tool():
    """Create a web search tool with API/raw and local/RAG behavior."""
    def enhanced_web_search(query: str) -> str:
        """Perform web search and return raw pages (API) or a RAG answer (local)."""
        logger.info("enhanced_web_search called with query: %s", query)
        return enhanced_web_search_and_query(query)

    enhanced_web_search.__name__ = "enhanced_web_search"
    return enhanced_web_search


def safe_import(module_name):
    """Import a module by name, returning None on ImportError."""
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError:
        return None


safe_globals: Dict[str, object] = {
    "__builtins__": {
        k: v for k, v in __builtins__.items()
        if k in {
            "len", "str", "int", "float", "list", "dict", "sum", "max", "min",
            "round", "abs", "sorted", "enumerate", "range", "zip", "map", "filter",
            "any", "all", "type", "isinstance", "print", "bool", "set", "tuple",
        }
    }
}

modules_to_load = {
    "math": "math", "datetime": "datetime", "re": "re", "os": "os", "sys": "sys",
    "json": "json", "csv": "csv", "random": "random", "collections": "collections",
    "itertools": "itertools", "functools": "functools", "pathlib": "pathlib",
    "np": "numpy", "pd": "pandas", "plt": "matplotlib.pyplot", "sns": "seaborn",
    "sklearn": "sklearn", "scipy": "scipy", "requests": "requests", "bs4": "bs4",
    "PIL": "PIL", "yaml": "yaml", "tqdm": "tqdm",
}

for alias, name in modules_to_load.items():
    mod = safe_import(name)
    if mod:
        safe_globals[alias] = mod
        if alias != name:
            safe_globals[name] = mod

if "bs4" in safe_globals:
    safe_globals["BeautifulSoup"] = safe_globals["bs4"].BeautifulSoup
if "PIL" in safe_globals:
    from PIL import Image
    safe_globals["Image"] = Image


def execute_python_code(code: str) -> str:
    """Execute Python code in a restricted global namespace."""
    output_buffer = io.StringIO()
    try:
        exec_locals = {}
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, exec_locals)

        stdout = output_buffer.getvalue().strip()
        result_val = exec_locals.get('result')

        response_parts = []
        if stdout:
            response_parts.append(stdout)
        if result_val is not None:
            response_parts.append(f"Result: {result_val}")

        if not response_parts:
            return "Code executed successfully (no output)"
        return "\n".join(response_parts)

    except Exception as exc:
        return f"Code execution failed: {str(exc)}"
