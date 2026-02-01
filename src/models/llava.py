"""LLaVA model implementation."""

import torch
import time
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import logging

from transformers import (
    LlavaProcessor, LlavaForConditionalGeneration,
    BitsAndBytesConfig
)

from .base import VisionLanguageModel

logger = logging.getLogger(__name__)


class LLaVAModel(VisionLanguageModel):
    """LLaVA 1.5 model implementation with quantization support."""
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "auto",
        quantization: str = "auto",
        **kwargs
    ):
        """
        Initialize LLaVA model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('auto', 'cpu', 'cuda')
            quantization: Quantization mode ('auto', 'none', '4bit', '8bit')
            **kwargs: Additional arguments
        """
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.quantization = quantization
        self._device_map = None
        self._quantization_config = None
        
        # Determine actual device and quantization settings
        self._configure_device_and_quantization()
    
    def _configure_device_and_quantization(self):
        """Configure device and quantization settings based on hardware."""
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        # Determine quantization
        if self.device == "cuda" and self.quantization == "auto":
            # Auto-select quantization based on available VRAM
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Available VRAM: {vram_gb:.1f} GB")
                
                if vram_gb < 8:
                    self.quantization = "4bit"
                    logger.info("Auto-selected 4-bit quantization (VRAM < 8GB)")
                elif vram_gb < 16:
                    self.quantization = "8bit"
                    logger.info("Auto-selected 8-bit quantization (VRAM < 16GB)")
                else:
                    self.quantization = "none"
                    logger.info("No quantization needed (VRAM >= 16GB)")
            except Exception as e:
                logger.warning(f"Could not determine VRAM, using 4-bit: {e}")
                self.quantization = "4bit"
        elif self.device == "cpu":
            self.quantization = "none"
            logger.info("CPU mode: quantization disabled")
        elif self.quantization == "auto":
            self.quantization = "none"
        
        # Configure quantization
        if self.quantization == "4bit" and self.device == "cuda":
            self._quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self._device_map = "auto"
            logger.info("Configured 4-bit quantization")
        elif self.quantization == "8bit" and self.device == "cuda":
            self._quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self._device_map = "auto"
            logger.info("Configured 8-bit quantization")
        else:
            self._quantization_config = None
            self._device_map = self.device
            logger.info(f"No quantization, using device: {self.device}")
    
    def load(self) -> None:
        """Load the LLaVA model and processor."""
        try:
            logger.info("=== Starting Model Load ===")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Requested device: {self.device}, Quantization: {self.quantization}")
            
            # Check GPU status if using CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                logger.info(f"CUDA total memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            else:
                logger.info("CUDA not available, will use CPU")
            
            logger.info(f"Device map: {self._device_map}")
            logger.info(f"Quantization config: {self._quantization_config}")
            
            # Load processor
            logger.info(f"Loading processor from {self.model_name}...")
            self.processor = LlavaProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=False
            )
            logger.info("✓ Processor loaded successfully")
            
            # Load model with quantization config
            logger.info("Loading model weights... (this may take 10-60 seconds)")
            if self._quantization_config:
                logger.info("Using quantized model loading")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=self._quantization_config,
                    device_map=self._device_map,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )
                logger.info("✓ Quantized model loaded")
            else:
                logger.info("Using full precision model loading")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )
                logger.info("✓ Model loaded, moving to device...")
                self.model.to(self.device)
                logger.info(f"✓ Model moved to {self.device}")
            
            # Set model to eval mode
            self.model.eval()
            logger.info("✓ Model set to eval mode")
            
            logger.info("✓ Model loaded successfully")
            logger.info("=== Model Load Complete ===")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA out of memory!")
            raise RuntimeError(
                "GPU out of memory. Try:\n"
                "1. Enable 4-bit or 8-bit quantization\n"
                "2. Close other GPU applications\n"
                "3. Use a smaller model\n"
                "4. Use CPU mode (very slow)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_caption(
        self,
        prompt: str = "Describe this image in detail:",
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            prompt: Text prompt for caption generation
            image_path: Path to the image file (if image not provided)
            image: PIL Image object (if path not provided)
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)
            num_beams: Number of beams for beam search
            **kwargs: Additional generation parameters
            
        Returns:
            Generated caption as string
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Accept either path or PIL Image
        if image is None and image_path is None:
            raise ValueError("Must provide either image_path or image")
        
        try:
            # Load image if path provided
            if image is None:
                image = Image.open(image_path).convert('RGB')
                logger.debug(f"Loaded image from path: {Path(image_path).name}")
            else:
                # Ensure RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                logger.debug(f"Using provided PIL Image: {image.size}")
            
            # For LLaVA 1.5, use simple USER/ASSISTANT format
            # Note: The processor adds <image> tokens automatically
            text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            logger.debug(f"Prompt: {prompt[:100]}")
            logger.debug(f"Formatted prompt length: {len(text_prompt)} chars")
            
            # Process inputs
            image_name = Path(image_path).name if image_path else "PIL Image"
            logger.debug(f"Processing inputs for {image_name}")
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # CRITICAL: Always move inputs to correct device
            target_device = "cuda" if self.device == "cuda" else "cpu"
            logger.debug(f"Moving inputs to {target_device}")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            # Verify inputs are on correct device
            for k, v in inputs.items():
                if hasattr(v, 'device'):
                    logger.debug(f"  {k}: device={v.device}, dtype={v.dtype}, shape={v.shape}")
            
            # Generate caption
            logger.debug("Starting generation...")
            gen_start = time.time()
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **kwargs
                )
            
            gen_time = time.time() - gen_start
            image_name = Path(image_path).name if image_path else "PIL Image"
            logger.info(f"⚡ Generation took {gen_time:.2f}s for {image_name}")
            
            # Decode output
            generated_text = self.processor.decode(
                output[0],
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            # LLaVA output format: [INST] prompt [/INST] response
            if "[/INST]" in generated_text:
                caption = generated_text.split("[/INST]")[-1].strip()
            elif "ASSISTANT:" in generated_text.upper():
                caption = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                # If no clear delimiter, try to extract after the prompt
                caption = generated_text.replace(prompt, "").strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded")
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "quantization": self.quantization,
            "device_map": str(self._device_map) if self._device_map else None,
        })
        
        if self.device == "cuda" and torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
        
        return info
    
    def clear_cache(self):
        """Clear CUDA cache to free memory."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
