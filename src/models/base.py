"""Abstract base class for vision-language models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class VisionLanguageModel(ABC):
    """Abstract base class for vision-language models."""
    
    def __init__(self, device: str = "auto", **kwargs):
        """
        Initialize the model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            **kwargs: Additional model-specific arguments
        """
        self.device = device
        self.model = None
        self.processor = None
        self.model_name = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def generate_caption(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail:",
        **kwargs
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for caption generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated caption as string
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "device": self.device,
            "loaded": self.is_loaded()
        }
