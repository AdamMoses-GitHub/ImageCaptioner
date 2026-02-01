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
                    # Get file info
                    file_size = image_path.stat().st_size
                    
                    # Load image and get metadata
                    from PIL import Image
                    from processing.image_processor import ImageProcessor
                    
                    with Image.open(image_path) as img:
                        original_width, original_height = img.size
                        img_format = img.format or 'Unknown'
                        original_dimensions = f"{original_width}x{original_height}"
                        
                        # Apply resize if enabled
                        resize_enabled = self.processing_config.get("resize_before_inference", True)
                        max_dimension = self.processing_config.get("max_dimension", 1024)
                        cache_resized = self.processing_config.get("cache_resized_images", False)
                        
                        # Work with copy to avoid modifying original
                        processed_img = img.copy()
                        was_resized = False
                        resized_dimensions = original_dimensions
                        
                        # Apply resize for inference if enabled (smart resize - downscale only)
                        if resize_enabled:
                            processed_img, was_resized = ImageProcessor.resize_image_smart(
                                processed_img, 
                                max_dimension,
                                method="lanczos",
                                allow_upscale=False
                            )
                            if was_resized:
                                new_width, new_height = processed_img.size
                                resized_dimensions = f"{new_width}x{new_height}"
                                logger.debug(f"Resized {image_name}: {original_dimensions} → {resized_dimensions}")
                        
                        # Cache resized image if option enabled (always resize to target dimension)
                        if cache_resized and hasattr(self, 'export_dir'):
                            cache_format = self.processing_config.get("cache_format", "original")
                            jpeg_quality = self.processing_config.get("jpeg_quality", 95)
                            
                            # Always resize cached images to target dimension (allow upscale)
                            cache_img, was_cache_resized = ImageProcessor.resize_image_smart(
                                img.copy(),
                                max_dimension,
                                method="lanczos",
                                allow_upscale=True
                            )
                            
                            # Save and get size statistics
                            _, original_size, saved_size = ImageProcessor.save_resized_image(
                                cache_img,
                                image_path,
                                self.export_dir,
                                cache_format=cache_format,
                                jpeg_quality=jpeg_quality
                            )
                        
                        # Generate caption with timing
                        import time
                        start_time = time.time()
                        
                        # Pass PIL Image directly to model
                        caption = self.model.generate_caption(
                            image=processed_img,
                            prompt=self.prompt,
                            **self.inference_config
                        )
                        
                        # Apply trigger word/prefix if provided (validate non-empty)
                        if self.trigger_word:
                            caption = f"{self.trigger_word}{caption}"
                        
                        generation_time = time.time() - start_time
                        
                        # Prepare dimensions display
                        if was_resized:
                            dimensions_display = f"{original_dimensions}→{resized_dimensions}"
                        else:
                            dimensions_display = original_dimensions
                    
                    # Store result
                    self.batch_processor.add_result(image_path, caption)
                    
                    # Emit signal with full metadata
                    self.caption_generated.emit(image_name, caption, generation_time, file_size, dimensions_display, img_format)
                    
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
