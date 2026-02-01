"""Default configuration values."""

from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "llava-hf/llava-1.5-7b-hf",
        "device": "auto",  # auto, cpu, cuda
        "quantization": "auto",  # auto, none, 4bit, 8bit
    },
    "inference": {
        "temperature": 0.2,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "num_beams": 1,
        "trigger_word": "",
    },
    "processing": {
        "recursive": False,
        "skip_errors": True,
        "clear_cache_interval": 10,
        "resize_before_inference": True,
        "max_dimension": 1024,
        "resize_method": "lanczos",
        "cache_resized_images": False,
        "cache_format": "original",  # original, png, jpeg
        "jpeg_quality": 95,
    },
    "export": {
        "formats": ["txt_individual"],  # txt_individual, csv, json, txt_batch
        "csv_relative_paths": True,
        "output_directory": None,  # None means same as input
    },
    "ui": {
        "last_input_directory": None,
        "last_output_directory": None,
        "window_geometry": None,
    },
    "custom_prompts": [],  # User-saved custom prompts
}
