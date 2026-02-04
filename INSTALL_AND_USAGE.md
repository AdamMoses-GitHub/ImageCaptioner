# ImageCaptioner: Installation & Usage Guide

## What This Tool Does

ImageCaptioner empowers you to:

- **Batch caption thousands of images** without manual effort or cloud APIs
- **Choose your export format**: Plain text, CSV, JSON, or any combination
- **Customize prompts** to shape how the AI describes your images
- **Control hardware usage**: Switch between GPU acceleration or CPU-only processing
- **Optimize for your hardware**: Automatic quantization (4-bit/8-bit) keeps everything running smoothly
- **Work offline**: No internet required after initial model download

---

## Installation

### Method A: Recommended (Clean Install with Virtual Environment)

This approach sets up an isolated Python environment—best for avoiding dependency conflicts.

**Prerequisites:**
- Python 3.9 or higher
- 15GB free disk space (for the model + dependencies)
- NVIDIA GPU with 4GB+ VRAM (highly recommended; CPU mode supported but slow)

**Steps:**

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdamMoses-GitHub/ImageCaptioner.git
   cd ImageCaptioner
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install PyTorch with CUDA support** (required first—do not skip)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   
   **For CPU-only:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Method B: Quick Install (Conda)

If you already use Conda, this is the fastest path.

```bash
# Clone the repo
git clone https://github.com/AdamMoses-GitHub/ImageCaptioner.git
cd ImageCaptioner

# Create Conda environment with PyTorch pre-installed
conda create -n imagecaptioner python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Activate environment
conda activate imagecaptioner

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage: Getting Started

### Run the Application

Once installed and your virtual environment is activated:

```bash
python src/main.py
```

The GUI window will appear. You're ready to caption images.

### Command-Line Options

```bash
# Standard run
python src/main.py

# Enable verbose logging (helpful for debugging)
python src/main.py --verbose

# Check CUDA availability and installed version
python src/main.py --verbose
# Look for "CUDA available: True" in the output
```

---

## Usage Workflows: Real-World Scenarios

### Workflow 1: Caption a Folder of Photos

**Scenario**: You have 500 vacation photos and need quick descriptions for organizing them.

**Steps:**
1. Launch the app: `python src/main.py`
2. In the **Model Settings** panel (left sidebar):
   - Select **Device**: CUDA (if available) or CPU
   - Select **Quantization**: 4bit (recommended for most GPUs)
3. Click **Select Input Folder** and choose your vacation photos directory
4. (Optional) Customize the prompt in the **Prompt** panel if you want specific descriptions (e.g., "Describe the main objects and mood")
5. Click **Start Captioning**
6. Once complete, choose export format: **TXT Individual** (one file per image) or **CSV** (all in one table)
7. Results save to the output folder you specified

**Example Use Case**: Travel photographer with 5,000 images from a European trip. Setting quantization to 4-bit and batch size to 4 in the config generates captions in ~2 hours on a mid-range GPU. Exports to CSV for easy import into photo management software.

---

### Workflow 2: Create a Dataset with Consistent Captions

**Scenario**: You're building a machine learning dataset and need uniform caption formatting across 10,000 images.

**Steps:**
1. Launch the app: `python src/main.py`
2. In the **Prompt** panel, click **Create New Prompt** and define it:
   - Example: *"Provide a single-sentence caption describing the primary subject and action in this image."*
   - Save as: `dataset_captions`
3. In the **Config** panel, set:
   - **Temperature**: 0.1 (lower = more consistent)
   - **Max Tokens**: 50 (keeps captions short)
4. Select your full dataset folder
5. Choose **Select Custom Prompt**: `dataset_captions`
6. Click **Start Captioning**
7. Export as **JSON** (machine-learning friendly format)

**Example Use Case**: Research team with 15,000 labeled images for computer vision. Consistent prompts + low temperature = reproducible captions. JSON export integrates directly into PyTorch DataLoader.

---

### Workflow 3: Fine-Tune Captions with Trigger Words

**Scenario**: You're training a LoRA (fine-tuned model) and need captions that include a specific keyword.

**Steps:**
1. Launch the app and go to **Model Settings**
2. In the config file (`config/settings.yaml`), find `inference.trigger_word` and set it:
   ```yaml
   inference:
     trigger_word: "my_custom_concept "  # Note the trailing space
   ```
3. This word automatically prepends to every caption
4. Process images as normal
5. Captions will be: `"my_custom_concept [AI-generated description]"`

**Example Use Case**: Fine-tuning a Stable Diffusion model with a specific person's images. Add trigger word `"tiffani_thiessen_lora "` and all captions automatically include it, ready for training.

---

### Workflow 4: Export in Multiple Formats Simultaneously

**Scenario**: You need captions in three different formats for different tools.

**Steps:**
1. In the **Export** panel, select multiple formats:
   - ☑ **TXT Individual** (one .txt file per image)
   - ☑ **CSV** (spreadsheet format)
   - ☑ **JSON** (data interchange format)
2. Set the output directory
3. Process your images
4. All three formats generate in separate folders within the output directory

**Example Use Case**: Content team exports to CSV for internal review, TXT for CMS upload, and JSON for backup/archival.

---

## Development

### Project Structure

```
ImageCaptioner/
├── src/
│   ├── main.py                           # Entry point; initializes Qt app
│   ├── gui/
│   │   ├── main_window.py                # Main GUI window layout
│   │   ├── panels/
│   │   │   ├── config_panel.py           # Model & inference settings
│   │   │   ├── input_panel.py            # Folder/image selection
│   │   │   ├── output_panel.py           # Results display & export
│   │   │   ├── prompt_panel.py           # Prompt management UI
│   │   │   └── model_status_panel.py     # Model download/status
│   │   └── workers/
│   │       ├── inference_worker.py       # Async inference thread
│   │       └── model_checker_worker.py   # Async model status check
│   ├── models/
│   │   ├── base.py                       # Abstract vision-language model
│   │   ├── llava.py                      # LLaVA 1.5 implementation
│   │   └── downloader.py                 # Model caching & download
│   ├── processing/
│   │   ├── batch_processor.py            # Orchestrates batch captioning
│   │   ├── image_processor.py            # Image validation & preprocessing
│   │   └── export.py                     # TXT, CSV, JSON export logic
│   ├── config/
│   │   ├── app_config.py                 # Config file management
│   │   ├── defaults.py                   # Default settings
│   │   └── prompts.py                    # Prompt template library
│   └── utils/
│       ├── logger.py                     # Logging configuration
│       └── validators.py                 # Input validation helpers
├── config/
│   └── settings.yaml                     # User settings (auto-generated on first run)
├── requirements.txt                      # Python dependencies
└── README.md                             # Project overview
```

### Key Directories Explained

- **`src/gui/`**: PySide6 UI components. Edit panels here to change the interface.
- **`src/models/`**: Model wrappers. Swap `llava.py` to add a different vision-language model.
- **`src/processing/`**: Core captioning and export logic. This is where the heavy lifting happens.
- **`src/config/`**: Configuration management. `defaults.py` defines every possible setting.
- **`config/settings.yaml`**: Your instance's persistent settings. Edited via GUI or directly.

### Running Tests & Linting

**Python Linting** (check code style):
```bash
pip install pylint
pylint src/
```

**Type Checking** (validate type hints):
```bash
pip install mypy
mypy src/ --ignore-missing-imports
```

**Unit Tests** (when available):
```bash
pip install pytest
pytest tests/
```

*Note: Full test suite not yet implemented. Contributions welcome!*

---

## Dependencies & Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 | Deep learning framework |
| `torchvision` | ≥0.16.0 | Image utilities & transforms |
| `transformers` | ≥4.37.0 | Hugging Face model loading |
| `PySide6` | ≥6.6.0 | Cross-platform GUI framework |
| `accelerate` | ≥0.25.0 | Distributed training & optimization |
| `bitsandbytes` | ≥0.42.0 | 4-bit/8-bit quantization |
| `huggingface-hub` | ≥0.20.0 | Model downloading & caching |
| `pillow` | ≥10.0.0 | Image I/O (JPG, PNG, etc.) |
| `pyyaml` | ≥6.0 | Configuration file parsing |

### System Requirements

- **Python**: 3.9 or higher
- **GPU**: NVIDIA with CUDA 12.1 support (4GB+ VRAM recommended)
  - Alternatives: AMD (ROCm), CPU (not recommended—extremely slow)
- **RAM**: 8GB+ for optimal performance
- **Disk Space**: 15GB (13GB for model, 2GB for dependencies)
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Optional Dependencies (for development)

```bash
pip install pylint mypy pytest black
```

---

## Configuration: Advanced Settings

The `config/settings.yaml` file controls everything. Edit via GUI or directly:

### Model Settings

```yaml
model:
  name: llava-hf/llava-1.5-7b-hf        # Hugging Face model ID
  device: cuda                            # auto, cpu, cuda
  quantization: 4bit                      # auto, none, 4bit, 8bit
```

### Inference (Caption Generation)

```yaml
inference:
  temperature: 0.2                        # 0=deterministic, 1+=more random
  max_new_tokens: 512                     # Max caption length
  top_p: 0.9                              # Nucleus sampling
  top_k: 50                               # Top-k sampling
  repetition_penalty: 1.0                 # Penalize repeated words
  trigger_word: ""                        # Prefix for all captions
```

### Processing

```yaml
processing:
  batch_size: 4                           # Images per forward pass
  resize_before_inference: false          # Downscale images?
  max_dimension: 1024                     # Max width/height (pixels)
  skip_errors: true                       # Continue on corrupted images
```

### Export

```yaml
export:
  formats:
    - txt_individual                      # One .txt per image
    - csv                                 # Single CSV file
    - json                                # Single JSON file
    - txt_batch                           # One .txt with all captions
  csv_relative_paths: true                # Use relative paths in CSV
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Problem**: PyTorch not installed or virtual environment not activated.

**Solution**:
```bash
# Make sure venv is activated (you should see (venv) in your terminal)
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"

**Problem**: Image batch is too large for your GPU.

**Solution**:
1. Reduce batch size in **Model Settings** (try 2 or 1)
2. Enable 8-bit quantization (uses ~30% less VRAM)
3. Use CPU mode (slow but works)
4. Downscale images: Enable `resize_before_inference` in config

### Model download stuck or fails

**Problem**: Network timeout or Hugging Face Hub unreachable.

**Solution**:
```bash
# Check your connection and try manually logging in
huggingface-cli login  # Paste your Hugging Face token

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/
python src/main.py --verbose
```

### GUI doesn't launch or freezes

**Problem**: PySide6 installation issue or Qt platform plugin missing.

**Solution**:
```bash
# Reinstall PySide6
pip install --upgrade PySide6

# On Linux, you may need Qt libraries
sudo apt-get install libxkbcommon0 libdbus-1-3  # Debian/Ubuntu
```

### Captions are nonsensical or repetitive

**Problem**: Model settings are too aggressive or temperature too high.

**Solution**:
- Lower `temperature` to 0.1 (more predictable)
- Reduce `max_new_tokens` to 256 (shorter captions)
- Increase `repetition_penalty` to 1.5 (avoid repeats)

---

## Performance Tips

**Maximize Speed:**
- Use 8-bit quantization (faster than 4-bit, still memory-efficient)
- Increase batch size to 8 or 16 (if VRAM allows)
- Downscale images before processing (`max_dimension: 512`)
- Use GPU (NVIDIA CUDA is ~50x faster than CPU)

**Minimize Memory:**
- Use 4-bit quantization (half the VRAM of 8-bit)
- Reduce batch size to 1 or 2
- Enable image caching to avoid repeated preprocessing

**Best Quality:**
- Use full precision (quantization: none) if you have 24GB+ VRAM
- Lower temperature to 0.1–0.3
- Increase `max_new_tokens` to 1024

---

## What's Next?

- **Customize prompts**: Create prompts tailored to your image type
- **Explore the config**: Every setting is documented in `config/settings.yaml`
- **Check the logs**: Run with `--verbose` to see detailed inference info
- **Contribute**: Have an idea? [Open an issue](https://github.com/AdamMoses-GitHub/ImageCaptioner/issues)

---

## Getting Help

- **Issues & Bugs**: [GitHub Issues](https://github.com/AdamMoses-GitHub/ImageCaptioner/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/AdamMoses-GitHub/ImageCaptioner/discussions)
- **Documentation**: Check the [README.md](README.md) for overview
