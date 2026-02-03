# ImageCaptioner

A modern desktop application for batch image captioning using the LLaVA 1.5 7B vision-language model with GPU acceleration support.

## Features

- **Advanced Vision-Language AI**: LLaVA 1.5 7B model for detailed, contextual image descriptions
- **GPU Acceleration**: CUDA support with automatic 4-bit/8-bit quantization for efficient memory usage
- **Batch Processing**: Process entire directories with real-time progress tracking and cancellation
- **Flexible Export Options**: TXT, CSV, JSON, or combined formats
- **Customizable Prompts**: Pre-built templates with ability to create and save custom prompts
- **Model Management**: Download, cache, and status indicators for seamless model handling
- **Cross-Platform**: Windows, macOS, and Linux support
- **Verbose Logging**: Debug mode for troubleshooting and development

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ImageCaptioner.git
cd ImageCaptioner
```

### 2. Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

## Requirements

- **Python**: 3.9 or higher
- **GPU**: 4GB+ VRAM recommended (NVIDIA CUDA 12.1+)
  - CPU fallback supported but significantly slower
- **Disk Space**: ~13GB for model cache
- **RAM**: 8GB+ recommended for optimal performance
- **Operating System**: Windows 10+, macOS 10.15+, or Linux

### Dependencies

Core libraries:
- `torch>=2.1.0` with CUDA support
- `torchvision>=0.16.0`
- `transformers>=4.37.0`
- `PySide6>=6.6.0` (GUI framework)
- `accelerate>=0.25.0` (model optimization)
- `bitsandbytes>=0.42.0` (quantization)
- `huggingface-hub>=0.20.0` (model management)
- `pyyaml>=6.0` (configuration)

## Usage

### Run the Application

```bash
python src/main.py
```

### Run with Verbose Logging

```bash
python src/main.py --verbose
```

### Debug Mode (VS Code)

Select "Debug Application" from the Run menu in VS Code.

## Application Workflow

1. **Configure Model Settings**
   - Select model variant (full precision or quantized)
   - Choose inference device (GPU/CPU)
   - Set batch size and other parameters

2. **Select Images**
   - Choose input folder or individual images
   - Supported formats: JPG, PNG, BMP, GIF, WebP, TIFF

3. **Customize Caption Prompt**
   - Use preset templates or create custom prompts
   - Save frequently used prompts for reuse

4. **Run Captioning**
   - Monitor real-time progress with visual indicators
   - Cancel anytime if needed

5. **Export Results**
   - Choose export format (TXT, CSV, JSON, or Combined)
   - Captions automatically saved to specified directory

## Project Structure

```
ImageCaptioner/
├── src/
│   ├── main.py                 # Application entry point
│   ├── gui/
│   │   ├── main_window.py      # Main window and layout
│   │   ├── panels/             # UI panels (config, input, output, prompts)
│   │   └── workers/            # Worker threads for async operations
│   ├── models/
│   │   ├── llava.py            # LLaVA model wrapper
│   │   ├── base.py             # Base model interface
│   │   └── downloader.py       # Model downloading and caching
│   ├── processing/
│   │   ├── batch_processor.py  # Batch image processing
│   │   ├── export.py           # Export to various formats
│   │   └── image_processor.py  # Image preprocessing and validation
│   ├── config/
│   │   ├── app_config.py       # Configuration management
│   │   ├── defaults.py         # Default application settings
│   │   └── prompts.py          # Prompt templates
│   └── utils/
│       ├── logger.py           # Logging utilities
│       └── validators.py       # Input validation
├── config/
│   └── settings.yaml           # User settings (auto-generated)
├── .vscode/
│   └── launch.json             # VS Code debug configurations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

Application settings are stored in `config/settings.yaml`:

```yaml
model:
  name: "llava-hf/llava-1.5-7b-hf"
  device: "cuda"
  quantization: "4bit"

processing:
  batch_size: 4
  num_workers: 2

export:
  default_format: "txt"
  output_directory: "./captions"
```

Modify settings through the GUI or edit the file directly.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open image folder |
| `Ctrl+S` | Save results |
| `Ctrl+Q` | Quit application |
| `F1` | View help |

## Troubleshooting

### Out of Memory (CUDA)
- Reduce batch size in Model Settings
- Enable 8-bit quantization instead of 4-bit
- Use CPU processing (slower but uses less VRAM)

### Model Download Fails
- Check internet connection
- Verify Hugging Face Hub access: `huggingface-cli login`
- Model cache location: `~/.cache/huggingface/hub/`

### PySide6 Import Error
```bash
pip install --upgrade PySide6
```

### Slow Processing
- Enable GPU acceleration (check CUDA availability)
- Reduce image resolution
- Increase batch size (if VRAM allows)

## Performance Tips

- **Batch Size**: Higher = faster but more VRAM (default: 4)
- **Quantization**: 8-bit faster than 4-bit, 4-bit uses less VRAM
- **GPU**: NVIDIA GPUs significantly faster than CPU
- **Images**: Resize large images before processing for faster inference

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [LLaVA Model](https://github.com/haotian-liu/LLaVA) - Vision-language model
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers library
- [PySide6](https://wiki.qt.io/Qt_for_Python) - Qt Python bindings

## Support

For issues, feature requests, or questions:
- Open an [Issue](https://github.com/yourusername/ImageCaptioner/issues)
- Check existing [Discussions](https://github.com/yourusername/ImageCaptioner/discussions)
