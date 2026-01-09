import logging

import torch
from PIL import Image

from .utils import QwenImageEditPlusPipeline, DIFFUSERS_AVAILABLE, unload_diffusion_pipeline

_logger = logging.getLogger(__name__)


class QwenImageEditor:
    """Image editing using the Qwen-Image-Edit-2511 model."""

    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2511"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available.")
        self.model_name = model_name
        self.pipeline = None

    def _ensure_pipeline(self):
        if self.pipeline is None:
            _logger.info("Loading image editing pipeline: %s", self.model_name)
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.set_progress_bar_config(disable=None)

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str = "edited_image.png",
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
    ) -> str:
        self._ensure_pipeline()
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = {
                "prompt": prompt,
                "image": image,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }

            output = self.pipeline(**inputs)

            output_image = output.images[0]
            output_image.save(output_path)

            _logger.info("Edited image saved to %s", output_path)
            return output_path
        finally:
            if self.pipeline is not None:
                unload_diffusion_pipeline(self.pipeline)
                self.pipeline = None
