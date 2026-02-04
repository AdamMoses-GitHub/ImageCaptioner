# ImageCaptioner

*Because describing thousands of images manually is like debugging without Stack Overflow.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)

## 🎯 About

Tired of staring at hundreds of images wondering what they're about? Meet **ImageCaptioner**—a desktop application that does the heavy lifting for you. Using the powerful LLaVA 1.5 7B vision-language model, it generates detailed, contextual captions for your entire image collection in minutes, not weeks. Whether you're building datasets, archiving photos, or just trying to remember what's in that folder from 2019, this app has your back.

The goal here is simple: make AI-powered image captioning accessible without needing a PhD in machine learning or a server farm. No API calls, no monthly bills, no nonsense. Just drop your images in, tweak a prompt if you want, and get accurate captions in the format you need.

**GitHub**: [https://github.com/AdamMoses-GitHub/ImageCaptioner](https://github.com/AdamMoses-GitHub/ImageCaptioner)

---

## ✨ What It Does

### User-Facing Features

- **🚀 GPU-Accelerated Captioning**: CUDA support with automatic 4-bit/8-bit quantization keeps memory usage sane on consumer hardware
- **📦 Batch Processing That Works**: Drop entire folders—the app processes them all with real-time progress and lets you cancel anytime (yes, actually cancel)
- **🎨 Flexible Export Formats**: Get results as TXT, CSV, JSON, or a combo pack—whatever your pipeline needs
- **🧠 Prompt Templates**: Pre-built prompts for common tasks, plus the freedom to craft custom ones and save them for later
- **🔧 Smart Model Management**: Handles downloading, caching, and checking model status so you don't have to SSH into a server
- **🖥️ Cross-Platform GUI**: Windows, macOS, and Linux—no terminal wizardry required (though `--verbose` logging is there if you want it)

### The Nerdy Stuff

- **Quantization Smarts**: Seamlessly switches between 4-bit and 8-bit modes based on your hardware without sacrificing quality
- **Processor-Agnostic**: Built on Hugging Face transformers and accelerate libraries—works with any device PyTorch supports
- **No Memory Bloat**: Intelligent cache management and batch size tuning prevent your 8GB GPU from catching fire
- **Typed Python**: Full type hints throughout the codebase because runtime errors at 3 AM are nobody's friend
- **Modular Architecture**: Vision model abstraction, separate inference workers, and clean dependency injection—easy to extend or swap in a better model tomorrow

---

## 🚀 Quick Start

For detailed setup instructions, see [INSTALL_AND_USAGE.md](INSTALL_AND_USAGE.md).

### TL;DR

```bash
# Clone the repo
git clone https://github.com/AdamMoses-GitHub/ImageCaptioner.git
cd ImageCaptioner

# Install PyTorch with CUDA support (required first!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r requirements.txt

# Run the app
python src/main.py
```

### With Verbose Logging (for when things get weird)

```bash
python src/main.py --verbose
```

---

## 🔧 Tech Stack

| Component | Purpose | Why This One |
|-----------|---------|--------------|
| **PyTorch 2.1+** | Deep learning framework | The gold standard for neural networks; CUDA integration is bulletproof |
| **Transformers (4.37+)** | Model loading & inference | Hugging Face's library is the ecosystem standard; LLaVA support is first-class |
| **LLaVA 1.5 7B** | Vision-language model | Best open-source image-to-text model at 7B params; context-aware and fast enough |
| **PySide6 (6.6+)** | GUI framework | Qt bindings; native look on all platforms without the GPL headache |
| **Accelerate (0.25+)** | Model optimization | Simplifies multi-GPU / mixed-precision / quantization setup |
| **bitsandbytes (0.42+)** | 4-bit/8-bit quantization | SOTA quantization; keeps models on consumer GPUs |
| **Pillow (10.0+)** | Image I/O | Industry standard; handles every image format you'll throw at it |
| **PyYAML (6.0+)** | Configuration files | Human-readable config without TOML debates |
| **Hugging Face Hub (0.20+)** | Model downloading & caching | Seamless model management with progress tracking |

---

## 📚 What Else?

### Contributing

Found a bug? Have an idea? [Open an issue](https://github.com/yourusername/ImageCaptioner/issues) or submit a pull request. Contributions are welcome.

### Known Limitations (Let's Be Honest)

- **Memory Requirements**: The model is ~13GB. You'll need VRAM or patience (CPU mode is *slow*).
- **Image Diversity**: LLaVA 1.5 7B is good, not perfect—mileage varies on highly specialized or artistic images.
- **Language**: English-centric model; non-English captions are rough around the edges.
- **Single-GPU**: Multi-GPU support isn't implemented yet (patches welcome).

### License

MIT License—use it, fork it, break it, fix it. See [LICENSE](LICENSE) for details.

---

<sub>
**Keywords**: image captioning, vision-language model, LLaVA, batch processing, GPU acceleration, CUDA quantization, AI image analysis, desktop GUI, PyTorch, Hugging Face transformers, computer vision, deep learning, image description generation, open-source ML
</sub>

For issues, feature requests, or questions:
- Open an [Issue](https://github.com/yourusername/ImageCaptioner/issues)
- Check existing [Discussions](https://github.com/yourusername/ImageCaptioner/discussions)
