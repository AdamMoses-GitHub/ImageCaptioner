"""Batch processing logic for multiple images."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Manage batch processing of multiple images."""
    
    def __init__(self, skip_errors: bool = True):
        """
        Initialize batch processor.
        
        Args:
            skip_errors: Whether to skip errors and continue processing
        """
        self.image_processor = ImageProcessor()
        self.skip_errors = skip_errors
        self.results = []
        self.errors = []
    
    def prepare_batch(self, directory: Path, recursive: bool = False) -> List[Path]:
        """
        Prepare a batch of images for processing.
        
        Args:
            directory: Directory containing images
            recursive: Whether to scan subdirectories
            
        Returns:
            List of valid image paths
        """
        # Scan directory
        all_images = self.image_processor.scan_directory(directory, recursive)
        
        if not all_images:
            logger.warning(f"No images found in {directory}")
            return []
        
        # Validate images
        valid_images = []
        for img_path in all_images:
            is_valid, error_msg = self.image_processor.validate_image(img_path)
            
            if is_valid:
                valid_images.append(img_path)
            else:
                logger.warning(f"Invalid image {img_path.name}: {error_msg}")
                self.errors.append({
                    "path": img_path,
                    "error": error_msg,
                    "stage": "validation"
                })
        
        logger.info(f"Prepared {len(valid_images)} valid images out of {len(all_images)} total")
        return valid_images
    
    def add_result(self, image_path: Path, caption: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a successful result.
        
        Args:
            image_path: Path to the image
            caption: Generated caption
            metadata: Optional metadata dictionary
        """
        result = {
            "path": image_path,
            "caption": caption,
            "success": True
        }
        
        if metadata:
            result.update(metadata)
        
        self.results.append(result)
    
    def add_error(self, image_path: Path, error: str, stage: str = "processing"):
        """
        Add an error.
        
        Args:
            image_path: Path to the image
            error: Error message
            stage: Processing stage where error occurred
        """
        self.errors.append({
            "path": image_path,
            "error": error,
            "stage": stage
        })
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all successful results."""
        return self.results.copy()
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors."""
        return self.errors.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        total = len(self.results) + len(self.errors)
        success_rate = len(self.results) / total if total > 0 else 0.0
        return {
            "total_processed": len(self.results),
            "total_errors": len(self.errors),
            "success_rate": success_rate,
        }
    
    def reset(self):
        """Reset processor state."""
        self.results.clear()
        self.errors.clear()
