"""Model downloader with progress tracking."""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple
from huggingface_hub import snapshot_download, model_info
from huggingface_hub.utils import HfHubHTTPError
import logging

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handle model downloading from Hugging Face Hub with progress tracking."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize the downloader.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.cache_dir = self._get_cache_dir()
        self._cancelled = False
        
    def _get_cache_dir(self) -> Path:
        """Get the Hugging Face cache directory."""
        # Use HF_HOME or default cache location
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home)
        
        # Default cache locations by platform
        if os.name == "nt":  # Windows
            cache_dir = Path.home() / "AppData" / "Local" / "huggingface"
        else:  # Unix-like
            cache_dir = Path.home() / ".cache" / "huggingface"
        
        return cache_dir
    
    def is_model_cached(self) -> bool:
        """
        Check if the model is already cached locally.
        
        Returns:
            True if model is cached, False otherwise
        """
        try:
            # Try to get model info without downloading
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            return True
        except Exception:
            return False
    
    def get_model_size(self) -> Tuple[float, str]:
        """
        Get the estimated size of the model download.
        
        Returns:
            Tuple of (size_in_gb, size_string)
        """
        try:
            info = model_info(self.model_name)
            
            # Calculate total size from siblings (model files)
            total_bytes = 0
            for sibling in info.siblings:
                if hasattr(sibling, 'size') and sibling.size:
                    total_bytes += sibling.size
            
            # Convert to GB
            size_gb = total_bytes / (1024 ** 3)
            
            if size_gb < 1:
                size_mb = total_bytes / (1024 ** 2)
                size_str = f"{size_mb:.1f} MB"
            else:
                size_str = f"{size_gb:.1f} GB"
            
            return size_gb, size_str
        except Exception as e:
            logger.warning(f"Could not determine model size: {e}")
            # Default estimate for LLaVA 1.5 7B
            return 13.0, "~13 GB"
    
    def download_model(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """
        Download the model from Hugging Face Hub.
        
        Args:
            progress_callback: Optional callback function(current_mb, total_mb, status_msg)
            
        Returns:
            True if download successful, False if cancelled or failed
        """
        try:
            self._cancelled = False
            
            if progress_callback:
                progress_callback(0, 0, "Starting download...")
            
            # Download the model
            # Note: snapshot_download doesn't provide fine-grained progress,
            # so we'll just show status updates
            logger.info(f"Downloading model: {self.model_name}")
            
            if progress_callback:
                progress_callback(0, 0, "Downloading model files...")
            
            snapshot_download(
                repo_id=self.model_name,
                local_files_only=False,
                resume_download=True,
            )
            
            if self._cancelled:
                if progress_callback:
                    progress_callback(0, 0, "Download cancelled")
                return False
            
            if progress_callback:
                progress_callback(100, 100, "Download complete!")
            
            logger.info("Model download complete")
            return True
            
        except HfHubHTTPError as e:
            error_msg = f"HTTP error downloading model: {e}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(0, 0, f"Error: {error_msg}")
            return False
            
        except Exception as e:
            error_msg = f"Error downloading model: {e}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(0, 0, f"Error: {error_msg}")
            return False
    
    def cancel_download(self):
        """Cancel the ongoing download."""
        self._cancelled = True
        logger.info("Download cancellation requested")
    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache directory.
        
        Returns:
            Dictionary with cache location and available space
        """
        cache_path = self.cache_dir / "hub"
        
        # Get available disk space
        try:
            if os.name == "nt":  # Windows
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(cache_path.drive)),
                    None, None,
                    ctypes.pointer(free_bytes)
                )
                available_gb = free_bytes.value / (1024 ** 3)
            else:  # Unix-like
                stat = os.statvfs(str(cache_path.parent))
                available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not determine available space: {e}")
            available_gb = 0
        
        return {
            "cache_location": str(cache_path),
            "available_space_gb": available_gb,
            "exists": cache_path.exists(),
        }
