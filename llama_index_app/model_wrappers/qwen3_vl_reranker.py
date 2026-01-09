from typing import Any, Dict, Optional
import logging

from transformers import AutoModel

_logger = logging.getLogger(__name__)


class Qwen3VLReranker:
    """Qwen3-VL reranker wrapper."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        top_n: int = 5,
        instruction: str = "Retrieve images or text relevant to the user's query.",
        fps: float = 1.0,
        torch_dtype: Optional[Any] = None,
        attn_implementation: Optional[str] = None,
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.instruction = instruction
        self.fps = fps
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self._model = None
        self._loaded = False

    def _load_model(self):
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._loaded = True

    def postprocess_nodes(self, nodes, query_bundle):
        if not nodes:
            return []

        if not self._loaded:
            try:
                self._load_model()
            except Exception as e:
                _logger.exception("Qwen3-VL reranker lazy-load failed: %s", e)
                return nodes[:self.top_n]

        query = getattr(query_bundle, "query_str", "") or ""
        documents = []
        doc_indices = []

        def _unwrap(node_like):
            return getattr(node_like, "node", None) or node_like

        for i, node_wrapper in enumerate(nodes):
            node = _unwrap(node_wrapper)
            entry: Dict[str, Any] = {}

            text_content = ""
            if hasattr(node, "get_content"):
                try:
                    text_content = node.get_content() or ""
                except Exception:
                    text_content = ""
            if not text_content:
                text_content = getattr(node, "text", "") or ""
            if text_content:
                entry["text"] = text_content

            if self._node_has_image(node):
                image_path = self._extract_image_path(node)
                if image_path:
                    entry["image"] = image_path

            if entry:
                documents.append(entry)
                doc_indices.append(i)

        if not documents:
            return nodes[:self.top_n]

        inputs = {
            "instruction": self.instruction,
            "query": {"text": query},
            "documents": documents,
            "fps": self.fps,
        }

        try:
            if hasattr(self._model, "process"):
                scores = self._model.process(inputs)
            elif hasattr(self._model, "encode"):
                scores = self._model.encode(inputs)
            else:
                raise RuntimeError("Qwen3-VL reranker backend does not expose a process/encode method.")
        except Exception as e:
            _logger.exception("Qwen3-VL reranker scoring failed: %s", e)
            return nodes[:self.top_n]

        if not isinstance(scores, list):
            scores = [scores]

        ranked = []
        for score, idx in zip(scores, doc_indices):
            ranked.append((float(score), idx))

        ranked.sort(key=lambda x: x[0], reverse=True)
        reranked_nodes = []
        for score, node_idx in ranked[:self.top_n]:
            if node_idx < len(nodes):
                node = nodes[node_idx]
                if hasattr(node, "score"):
                    node.score = score
                reranked_nodes.append(node)

        return reranked_nodes

    def _node_has_image(self, node) -> bool:
        metadata = getattr(node, "metadata", {})
        file_type = metadata.get("file_type", "").lower()
        if file_type in ["jpg", "jpeg", "png", "gif", "bmp", "webp", "pdf"]:
            return True
        content_type = metadata.get("type", "").lower()
        if content_type in ["image", "web_image"]:
            return True
        if hasattr(node, "image_path") and node.image_path:
            return True
        if "image_data" in metadata:
            return True
        source = metadata.get("source", "").lower()
        if any(ext in source for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            return True
        return False

    def _extract_image_path(self, node) -> Optional[str]:
        if hasattr(node, "image_path") and node.image_path:
            return node.image_path
        metadata = getattr(node, "metadata", {})
        if "path" in metadata:
            return metadata["path"]
        source = metadata.get("source", "")
        if source and any(ext in source.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            return source
        return None
