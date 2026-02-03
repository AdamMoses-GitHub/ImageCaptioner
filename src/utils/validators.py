"""Input validation utilities."""

from pathlib import Path
from typing import Tuple, Set


# Supported image formats
SUPPORTED_FORMATS: Set[str] = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'
}


def is_valid_image_format(file_path: Path) -> bool:
    """
    Check if file has a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if format is supported, False otherwise
    """
    return file_path.suffix.lower() in SUPPORTED_FORMATS


def validate_directory(dir_path: Path) -> Tuple[bool, str]:
    """
    Validate that a directory exists and is readable.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not dir_path.exists():
        return False, "Directory does not exist"
    
    if not dir_path.is_dir():
        return False, "Path is not a directory"
    
    try:
        # Test if directory is readable
        list(dir_path.iterdir())
        return True, ""
    except PermissionError:
        return False, "Permission denied - cannot read directory"
    except Exception as e:
        return False, f"Error accessing directory: {str(e)}"


def validate_temperature(value: float) -> Tuple[bool, str]:
    """
    Validate temperature parameter.
    
    Args:
        value: Temperature value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 0.0 <= value <= 2.0:
        return False, "Temperature must be between 0.0 and 2.0"
    return True, ""


def validate_max_tokens(value: int) -> Tuple[bool, str]:
    """
    Validate max_new_tokens parameter.
    
    Args:
        value: Max tokens value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 1 <= value <= 2048:
        return False, "Max tokens must be between 1 and 2048"
    return True, ""


def validate_top_p(value: float) -> Tuple[bool, str]:
    """
    Validate top_p parameter.
    
    Args:
        value: Top-p value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 0.0 <= value <= 1.0:
        return False, "Top P must be between 0.0 and 1.0"
    return True, ""


def validate_repetition_penalty(value: float) -> Tuple[bool, str]:
    """
    Validate repetition penalty parameter.
    
    Args:
        value: Repetition penalty value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 1.0 <= value <= 2.0:
        return False, "Repetition penalty must be between 1.0 and 2.0"
    return True, ""
