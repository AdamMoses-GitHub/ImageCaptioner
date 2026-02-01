"""Image validation and directory scanning."""

from pathlib import Path
from typing import List, Tuple
from PIL import Image
import logging

from utils.validators import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handle image scanning, validation, and loading."""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
    
    def scan_directory(self, directory: Path, recursive: bool = False) -> List[Path]:
        """
        Scan directory for supported image files.
        
        Args:
            directory: Path to directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of image file paths
        """
        images = []
        
        try:
            if recursive:
                # Recursive scan
                for ext in self.supported_formats:
                    images.extend(directory.rglob(f"*{ext}"))
                    images.extend(directory.rglob(f"*{ext.upper()}"))
            else:
                # Non-recursive scan (only top-level)
                for item in directory.iterdir():
                    if item.is_file() and item.suffix.lower() in self.supported_formats:
                        images.append(item)
            
            # Remove duplicates and sort
            images = sorted(set(images))
            
            logger.info(f"Found {len(images)} images in {directory}")
            return images
            
        except PermissionError as e:
            logger.error(f"Permission denied accessing {directory}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return []
    
    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate that an image file is readable and not corrupted.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check extension
        if image_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format: {image_path.suffix}"
        
        # Check file exists
        if not image_path.exists():
            return False, "File not found"
        
        # Check file is readable
        if not image_path.is_file():
            return False, "Not a file"
        
        # Try to open and verify the image
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify image integrity
            
            # Re-open to check it's actually loadable
            with Image.open(image_path) as img:
                img.load()  # Force load to catch any errors
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Corrupted or invalid image: {str(e)}"
    
    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load and return an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
            
        Raises:
            Exception if image cannot be loaded
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def get_image_info(self, image_path: Path) -> dict:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "path": str(image_path),
                    "name": image_path.name,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                }
        except Exception as e:
            logger.error(f"Failed to get info for {image_path}: {e}")
            return {
                "path": str(image_path),
                "name": image_path.name,
                "error": str(e)
            }
    
    @staticmethod
    def resize_image_smart(
        image: Image.Image, 
        max_dimension: int,
        method: str = "lanczos",
        allow_upscale: bool = False
    ) -> Tuple[Image.Image, bool]:
        """
        Smart resize - downsize if image exceeds max dimension, optionally upscale smaller images.
        
        Args:
            image: PIL Image to resize
            max_dimension: Target maximum width or height
            method: Resize method (lanczos, bilinear, bicubic)
            allow_upscale: If True, upscale images smaller than max_dimension
            
        Returns:
            Tuple of (resized_image, was_resized)
        """
        width, height = image.size
        max_current = max(width, height)
        
        # Only resize if exceeds threshold or upscaling is allowed
        if max_current == max_dimension:
            return image, False
        if not allow_upscale and max_current < max_dimension:
            return image, False
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        # Select resize method
        resample_methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }
        resample = resample_methods.get(method.lower(), Image.Resampling.LANCZOS)
        
        resized = image.resize((new_width, new_height), resample)
        action = "Upscaled" if max_current < max_dimension else "Downscaled"
        logger.debug(f"{action} image from {width}x{height} to {new_width}x{new_height}")
        return resized, True
    
    @staticmethod
    def save_resized_image(
        image: Image.Image,
        original_path: Path,
        output_dir: Path,
        cache_format: str = "original",
        jpeg_quality: int = 95
    ) -> Tuple[Path, int, int]:
        """
        Save resized image to cache directory.
        
        Args:
            image: PIL Image to save
            original_path: Original image path
            output_dir: Directory to save resized images
            cache_format: Format to save ("original", "png", "jpeg")
            jpeg_quality: JPEG quality (1-100)
            
        Returns:
            Tuple of (saved_path, original_size_bytes, saved_size_bytes)
        """
        # Create resized_images subdirectory
        resized_dir = output_dir / "resized_images"
        resized_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine format and extension
        if cache_format == "jpeg":
            save_format = "JPEG"
            new_extension = ".jpg"
            # JPEG doesn't support transparency, ensure RGB
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
        elif cache_format == "png":
            save_format = "PNG"
            new_extension = ".png"
        else:  # "original"
            save_format = image.format if hasattr(image, 'format') and image.format else 'PNG'
            new_extension = original_path.suffix  # Keep original extension
        
        # Create output path with correct extension
        output_filename = original_path.stem + new_extension
        output_path = resized_dir / output_filename
        
        # Calculate original size (in-memory estimate)
        import io
        original_buffer = io.BytesIO()
        original_format = image.format if hasattr(image, 'format') and image.format else 'PNG'
        image.save(original_buffer, format=original_format)
        original_size = original_buffer.tell()
        
        # Save with format-specific options
        save_kwargs = {}
        if save_format == "JPEG":
            save_kwargs = {"quality": jpeg_quality, "optimize": True}
        elif save_format == "PNG":
            save_kwargs = {"optimize": True}
        
        image.save(output_path, format=save_format, **save_kwargs)
        
        # Get actual saved file size
        saved_size = output_path.stat().st_size
        
        logger.info(f"Cached {original_path.name} → {output_filename}: {original_size/1024:.1f}KB → {saved_size/1024:.1f}KB ({save_format})")
        return output_path, original_size, saved_size

