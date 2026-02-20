"""Worker thread for batch image processing."""

from PySide6.QtCore import QThread, Signal
from pathlib import Path
from typing import List, Dict, Any
import logging
import torch

from models.llava import LLaVAModel
from models.downloader import ModelDownloader
from processing.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Background worker for processing images with LLaVA model."""
    
    # Signals
    progress_updated = Signal(int, int, str)  # current, total, image_name
    caption_generated = Signal(str, str, float, int, str, str)  # image_name, caption, gen_time, file_size, dimensions, img_type
    error_occurred = Signal(str, str)  # image_name, error_message
    status_message = Signal(str)  # status message
    finished = Signal(bool, dict)  # success, summary
    
    def __init__(
        self,
        directory: Path,
        model_config: Dict[str, Any],
        inference_config: Dict[str, Any],
        prompt: str,
        trigger_word: str = "",
        processing_config: Dict[str, Any] = None,
        parent=None
    ):
        """
        Initialize inference worker.
        
        Args:
            directory: Directory containing images
            model_config: Model configuration dictionary
            inference_config: Inference parameters dictionary
            prompt: Prompt text for caption generation
            trigger_word: Optional trigger word/prefix to prepend to captions
            processing_config: Processing configuration (resize settings)
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.directory = directory
        self.model_config = model_config
        self.inference_config = inference_config
        self.prompt = prompt
        self.trigger_word = trigger_word
        self.processing_config = processing_config or {}
        self.export_dir = None  # Set by caller for resize caching
        
        self.batch_processor = BatchProcessor(skip_errors=True)
        self.model = None
        self._is_cancelled = False
        self._cache_clear_interval = 10
    
    def run(self):
        """
        Run the batch processing in background thread.
        """
        try:
            logger.info("=== Inference Worker Starting ===")
            logger.info(f"Model config: {self.model_config}")
            logger.info(f"Inference config: {self.inference_config}")
            logger.info(f"Directory: {self.directory}")
            
            # Check if model is cached
            logger.info("Checking model cache...")
            self.status_message.emit("Checking model availability...")
            downloader = ModelDownloader()
            
            if not downloader.is_model_cached():
                self.status_message.emit("Model not cached. Please download first.")
                self.finished.emit(False, {"error": "Model not cached"})
                return
            
            # Prepare batch
            self.status_message.emit("Scanning directory for images...")
            image_paths = self.batch_processor.prepare_batch(
                self.directory,
                recursive=False
            )
            
            if not image_paths:
                self.status_message.emit("No valid images found")
                self.finished.emit(False, {"error": "No valid images"})
                return
            
            total_images = len(image_paths)
            logger.info(f"Starting batch processing of {total_images} images")
            
            # Load model
            logger.info("Initializing LLaVA model...")
            logger.info(f"Device: {self.model_config.get('device', 'auto')}, Quantization: {self.model_config.get('quantization', 'auto')}")
            self.status_message.emit("Loading model...")
            self.model = LLaVAModel(
                device=self.model_config.get("device", "auto"),
                quantization=self.model_config.get("quantization", "auto")
            )
            
            try:
                logger.info("Calling model.load()... (this may take 10-30 seconds)")
                self.model.load()
                self.status_message.emit("Model loaded successfully")
                model_info = self.model.get_model_info()
                logger.info(f"Model loaded successfully: {model_info}")
                logger.info("=== Model Ready for Inference ===")
            except RuntimeError as e:
                # Handle CUDA out-of-memory errors with automatic fallback
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    current_quant = self.model_config.get("quantization", "auto")
                    
                    if current_quant in ["8bit", "auto"]:
                        # Fallback to 4-bit quantization
                        logger.warning("CUDA OOM detected, retrying with 4-bit quantization...")
                        self.status_message.emit("GPU memory full - switching to 4-bit mode...")
                        self.model_config["quantization"] = "4bit"
                        self.model = LLaVAModel(
                            device=self.model_config.get("device", "auto"),
                            quantization="4bit"
                        )
                        try:
                            self.model.load()
                            self.status_message.emit("Model loaded in 4-bit mode")
                            logger.info("Successfully loaded model with 4-bit quantization")
                        except Exception as e2:
                            error_msg = f"Failed to load model even with 4-bit: {str(e2)}"
                            logger.error(error_msg)
                            self.status_message.emit(error_msg)
                            self.finished.emit(False, {"error": error_msg})
                            return
                    elif current_quant == "4bit":
                        # Fallback to CPU mode
                        logger.warning("4-bit OOM detected, falling back to CPU (slow)...")
                        self.status_message.emit("Switching to CPU mode (processing will be slower)...")
                        self.model_config["device"] = "cpu"
                        self.model_config["quantization"] = "none"
                        self.model = LLaVAModel(device="cpu", quantization="none")
                        try:
                            self.model.load()
                            self.status_message.emit("Model loaded in CPU mode (slower)")
                            logger.info("Successfully loaded model on CPU")
                        except Exception as e2:
                            error_msg = f"Failed to load model on CPU: {str(e2)}"
                            logger.error(error_msg)
                            self.status_message.emit(error_msg)
                            self.finished.emit(False, {"error": error_msg})
                            return
                    else:
                        # Already on optimal settings, re-raise
                        error_msg = f"Failed to load model: {str(e)}"
                        logger.error(error_msg)
                        self.status_message.emit(error_msg)
                        self.finished.emit(False, {"error": error_msg})
                        return
                else:
                    # Non-OOM RuntimeError, re-raise
                    error_msg = f"Failed to load model: {str(e)}"
                    logger.error(error_msg)
                    self.status_message.emit(error_msg)
                    self.finished.emit(False, {"error": error_msg})
                    return
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg)
                self.status_message.emit(error_msg)
                self.finished.emit(False, {"error": error_msg})
                return
            
            # Process images
            for idx, image_path in enumerate(image_paths):
                if self._is_cancelled:
                    logger.info("Processing cancelled by user")
                    break
                
                image_name = image_path.name
                
                # Update progress
                self.progress_updated.emit(idx + 1, total_images, image_name)
                
                try:
                    processed_img, metadata = self.batch_processor.image_processor.prepare_image_for_inference(
                        image_path=image_path,
                        processing_config=self.processing_config,
                        export_dir=self.export_dir
                    )

                    file_size = metadata["file_size"]
                    dimensions_display = metadata["dimensions"]
                    img_format = metadata["img_format"]

                    # Generate caption with timing
                    import time
                    start_time = time.time()

                    # Pass PIL Image directly to model
                    with torch.inference_mode():
                        caption = self.model.generate_caption(
                            image=processed_img,
                            prompt=self.prompt,
                            **self.inference_config
                        )

                    # Apply trigger word/prefix if provided (validate non-empty)
                    if self.trigger_word:
                        caption = f"{self.trigger_word}{caption}"

                    generation_time = time.time() - start_time
                    
                    # Store result
                    self.batch_processor.add_result(image_path, caption)
                    
                    # Emit signal with full metadata
                    self.caption_generated.emit(
                        image_name,
                        caption,
                        generation_time,
                        file_size,
                        dimensions_display,
                        img_format
                    )
                    
                    logger.debug(f"Generated caption for {image_name}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing {image_name}: {error_msg}")
                    
                    # Store error
                    self.batch_processor.add_error(image_path, error_msg)
                    
                    # Emit error signal
                    self.error_occurred.emit(image_name, error_msg)
                
                # Clear CUDA cache periodically
                if (idx + 1) % self._cache_clear_interval == 0:
                    if self.model:
                        self.model.clear_cache()
                        logger.debug("Cleared CUDA cache")
            
            # Cleanup
            self.status_message.emit("Cleaning up...")
            if self.model:
                self.model.unload()
            
            # Get summary
            summary = self.batch_processor.get_summary()
            summary["results"] = self.batch_processor.get_results()
            summary["errors"] = self.batch_processor.get_errors()
            summary["cancelled"] = self._is_cancelled
            
            success = not self._is_cancelled and summary["total_errors"] < total_images
            
            logger.info(
                f"Batch processing finished: "
                f"{summary['total_processed']} processed, "
                f"{summary['total_errors']} errors"
            )
            
            self.finished.emit(success, summary)
            
        except Exception as e:
            error_msg = f"Unexpected error during processing: {str(e)}"
            logger.exception(error_msg)
            self.status_message.emit(error_msg)
            self.finished.emit(False, {"error": error_msg})
        
        finally:
            # Ensure cleanup
            if self.model:
                try:
                    self.model.unload()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
    
    def cancel(self):
        """Cancel the processing."""
        self._is_cancelled = True
        logger.info("Cancellation requested")
