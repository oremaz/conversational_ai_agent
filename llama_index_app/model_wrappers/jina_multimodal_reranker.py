from typing import List, Optional
import logging

import torch
from transformers import AutoModel, BitsAndBytesConfig

_logger = logging.getLogger(__name__)


class JinaMultimodalReranker:
    """Jina multimodal reranker using jinaai/jina-reranker-m0 (GPU only)."""

    def __init__(self, model_name: str = "jinaai/jina-reranker-m0", top_n: int = 5, device: str = "auto"):
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self._model = None
        self._loaded = False

    def _load_model(self):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("Jina reranker requires CUDA; CPU mode is not supported.")

            if self.device == "auto":
                try:
                    dev_count = torch.cuda.device_count()
                except Exception:
                    dev_count = 1
                target_dev = "cuda:1" if dev_count > 1 else "cuda:0"
            else:
                if str(self.device).startswith("cpu"):
                    raise RuntimeError("Jina reranker requires CUDA; CPU device is not supported.")
                target_dev = self.device

            device_map = {"": target_dev}

            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                )
            except Exception as e:
                _logger.warning("4-bit load failed for reranker: %s. Falling back to non-4bit load.", e)
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device_map,
                )

            if hasattr(self._model, "eval"):
                self._model.eval()
            _logger.info("Loaded Jina reranker model: %s on %s", self.model_name, target_dev)

        except Exception as e:
            _logger.error("Error loading Jina reranker model: %s", e)
            raise

    def postprocess_nodes(self, nodes, query_bundle):
        if not nodes:
            return []

        if not getattr(self, "_loaded", False):
            try:
                self._load_model()
                self._loaded = True
            except Exception as e:
                _logger.exception("Reranker lazy-load failed: %s", e)
                return nodes[:self.top_n]

        query = getattr(query_bundle, "query_str", "") or ""
        _logger.debug("Reranker received %d nodes for query='%s'", len(nodes), query)

        text_pairs = []
        text_indices = []
        image_pairs = []
        image_indices = []

        def _unwrap(node_like):
            return getattr(node_like, "node", None) or node_like

        for i, node_wrapper in enumerate(nodes):
            node = _unwrap(node_wrapper)
            has_image = self._node_has_image(node)

            if has_image:
                image_path = self._extract_image_path(node)
                if image_path:
                    image_pairs.append([query, image_path])
                    image_indices.append(i)
                else:
                    _logger.debug("Image node missing path for index %s", i)
            else:
                text_content = ""
                if hasattr(node, "get_content"):
                    try:
                        text_content = node.get_content() or ""
                    except Exception as err:
                        _logger.debug("get_content failed for node %s: %s", i, err)
                        text_content = ""
                if not text_content:
                    text_content = getattr(node, "text", "") or ""
                if text_content:
                    text_pairs.append([query, text_content])
                    text_indices.append(i)
                else:
                    _logger.debug("Skipping empty text node at index %s", i)

        all_scores = []

        try:
            if text_pairs:
                text_scores = self._model.compute_score(
                    text_pairs,
                    max_length=1024,
                    doc_type="text"
                )
                for score, idx in zip(text_scores, text_indices):
                    all_scores.append((score, idx))

            if image_pairs:
                image_scores = self._model.compute_score(
                    image_pairs,
                    max_length=2048,
                    doc_type="image"
                )
                for score, idx in zip(image_scores, image_indices):
                    all_scores.append((score, idx))

        except Exception as e:
            _logger.exception("Error during reranking: %s", e)
            return nodes[:self.top_n]

        if not all_scores:
            _logger.debug("No reranker scores computed; returning original nodes")
            return nodes[:self.top_n]

        all_scores.sort(key=lambda x: x[0], reverse=True)

        reranked_nodes = []
        for score, node_idx in all_scores[:self.top_n]:
            if node_idx < len(nodes):
                node = nodes[node_idx]
                if hasattr(node, "score"):
                    node.score = score
                reranked_nodes.append(node)

        _logger.debug("Reranker returning %d nodes", len(reranked_nodes))
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

    def score_text_pairs(self, pairs: List[List[str]], max_length: int = 1024) -> List[float]:
        return self._model.compute_score(pairs, max_length=max_length, doc_type="text")
