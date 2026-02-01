# Image Captioning GUI Application

A desktop GUI application for batch image captioning using LLaVA 1.5 7B model with GPU acceleration support.

## Features

- **LLaVA 1.5 Model Integration**: State-of-the-art vision-language model for detailed image captioning
- **GPU Acceleration**: Supports CUDA with automatic 4-bit/8-bit quantization for memory efficiency
- **Batch Processing**: Process entire directories of images with real-time progress tracking
- **Customizable Prompts**: Preset templates plus custom prompt creation and saving
- **Multiple Export Formats**: Individual .txt files, CSV, JSON, or combined text file
- **User-Friendly GUI**: Built with PySide6 for modern, responsive interface

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python src/main.py
   ```

## Requirements

- Python 3.9+
- ~13GB disk space for model cache (first run)
- GPU with 4GB+ VRAM recommended (CPU fallback available)

## Configuration

Settings are saved to `config/settings.yaml` and include:
- Model selection and quantization settings
- Inference parameters (temperature, max tokens, etc.)
- Custom prompt templates
- Export preferences

## Usage

1. Select a directory containing images
2. Configure model settings (device, quantization, parameters)
3. Choose or create a custom prompt template
4. Select export format(s)
5. Click "Start Processing"
6. Captions will be generated and exported automatically

## Supported Image Formats

.jpg, .jpeg, .png, .bmp, .gif, .webp, .tiff, .tif

## License

MIT License
