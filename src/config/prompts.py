"""Preset prompt templates for image captioning."""

from typing import List, Dict


PRESET_PROMPTS: List[Dict[str, str]] = [
    {
        "name": "Detailed Description (Default)",
        "prompt": "Describe this image in detail:",
    },
    {
        "name": "Short Description",
        "prompt": "Provide a short description:",
    },
    {
        "name": "List Objects and People",
        "prompt": "List all objects and people in this image:",
    },
    {
        "name": "Scene and Mood",
        "prompt": "Describe the scene, mood, and setting:",
    },
    {
        "name": "What's Happening",
        "prompt": "What is happening in this image?",
    },
    {
        "name": "Technical Details",
        "prompt": "Describe the composition, colors, and technical aspects of this image:",
    },
]


def get_preset_prompts() -> List[Dict[str, str]]:
    """Get list of preset prompt templates."""
    return PRESET_PROMPTS.copy()


def get_default_prompt() -> str:
    """Get the default prompt text."""
    return PRESET_PROMPTS[0]["prompt"]
