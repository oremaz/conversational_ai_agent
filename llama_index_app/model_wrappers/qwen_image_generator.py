import logging

import torch
from .utils import DiffusionPipeline, DIFFUSERS_AVAILABLE, unload_diffusion_pipeline

_logger = logging.getLogger(__name__)


class QwenImageGenerator:
    """Image generation using the Qwen-Image-2512 diffusion model."""

    def __init__(self, model_name: str = "Qwen/Qwen-Image-2512"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available.")
        self.model_name = model_name
        self.pipeline = None

    def _ensure_pipeline(self):
        if self.pipeline is None:
            _logger.info("Loading image generation pipeline: %s", self.model_name)
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.set_progress_bar_config(disable=None)

    def generate(
        self,
        prompt: str,
        output_path: str = "generated_image.png",
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
    ) -> str:
        self._ensure_pipeline()
        try:
            output = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )

            image = output.images[0]
            image.save(output_path)
            _logger.info("Generated image saved to %s", output_path)
            return output_path
        finally:
            if self.pipeline is not None:
                unload_diffusion_pipeline(self.pipeline)
                self.pipeline = None
