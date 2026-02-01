"""Worker thread for checking model cache status."""

from PySide6.QtCore import QThread, Signal
from typing import Dict, Any
import logging

from models.downloader import ModelDownloader

logger = logging.getLogger(__name__)


class ModelCheckerWorker(QThread):
    """Background worker for checking model cache without blocking GUI."""
    
    check_complete = Signal(bool, dict)  # is_cached, model_info
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", parent=None):
        """
        Initialize model checker worker.
        
        Args:
            model_name: Hugging Face model identifier
            parent: Parent QObject
        """
        super().__init__(parent)
        self.model_name = model_name
    
    def run(self):
        """
        Check model cache status in background thread.
        """
        try:
            logger.info(f"Checking cache for model: {self.model_name}")
            
            # Create downloader
            downloader = ModelDownloader(self.model_name)
            
            # Check if cached
            is_cached = downloader.is_model_cached()
            
            # Get additional info
            model_size_gb, model_size_str = downloader.get_model_size()
            cache_info = downloader.get_cache_info()
            
            model_info = {
                "model_name": self.model_name,
                "size_gb": model_size_gb,
                "size_str": model_size_str,
                "cache_location": cache_info.get("cache_location", ""),
                "available_space_gb": cache_info.get("available_space_gb", 0)
            }
            
            logger.info(f"Model cache check complete: cached={is_cached}")
            self.check_complete.emit(is_cached, model_info)
            
        except Exception as e:
            logger.error(f"Error checking model cache: {e}")
            self.check_complete.emit(False, {
                "error": str(e),
                "model_name": self.model_name,
                "size_str": "~13 GB"
            })
