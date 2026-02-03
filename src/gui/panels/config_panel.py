"""Configuration panel for model settings."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QGroupBox, QFormLayout, QCheckBox
)
from PySide6.QtCore import Qt, Signal
import logging

from config.defaults import DEFAULT_CONFIG
from utils.validators import (
    validate_temperature,
    validate_max_tokens,
    validate_top_p,
    validate_repetition_penalty
)

logger = logging.getLogger(__name__)


class ConfigPanel(QWidget):
    """Panel for configuring model and inference settings."""
    
    config_changed = Signal(dict)  # Emitted when configuration changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = DEFAULT_CONFIG.copy()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA (GPU)", "CPU"])
        self.device_combo.setCurrentText("Auto")
        self.device_combo.currentTextChanged.connect(self.on_config_changed)
        self.device_combo.setToolTip("Select processing device (Auto detects GPU availability)")
        model_layout.addRow("Device:", self.device_combo)
        
        # Quantization
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["Auto", "None (Full Precision)", "8-bit", "4-bit"])
        self.quant_combo.setCurrentText("Auto")
        self.quant_combo.currentTextChanged.connect(self.on_config_changed)
        self.quant_combo.currentTextChanged.connect(self.update_vram_info)
        self.quant_combo.setToolTip(
            "Quantization reduces memory usage:\n"
            "• Auto: Selects based on available VRAM\n"
            "• 4-bit: ~4GB VRAM (recommended for most GPUs)\n"
            "• 8-bit: ~7GB VRAM\n"
            "• None: ~14GB VRAM (best quality)"
        )
        model_layout.addRow("Quantization:", self.quant_combo)
        
        # VRAM requirements info
        self.vram_info_label = QLabel()
        self.vram_info_label.setStyleSheet("color: #7f8c8d; font-size: 9pt; padding-left: 5px;")
        self.vram_info_label.setWordWrap(True)
        model_layout.addRow("VRAM Required:", self.vram_info_label)
        self.update_vram_info()
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Inference Parameters Group
        inference_group = QGroupBox("Inference Parameters")
        inference_layout = QFormLayout()
        
        # Temperature
        temp_layout = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)  # 0.0 to 2.0
        self.temp_slider.setValue(20)  # 0.2
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(20)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.2)
        self.temp_spin.setDecimals(1)
        # Connect slider and spinbox
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_spin.setValue(v / 100.0)
        )
        self.temp_spin.valueChanged.connect(
            lambda v: self.temp_slider.setValue(int(v * 100))
        )
        self.temp_spin.valueChanged.connect(self.on_config_changed)
        temp_layout.addWidget(self.temp_slider, stretch=1)
        temp_layout.addWidget(self.temp_spin)
        inference_layout.addRow("Temperature:", temp_layout)
        
        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(16, 2048)
        self.max_tokens_spin.setSingleStep(16)
        self.max_tokens_spin.setValue(512)
        self.max_tokens_spin.valueChanged.connect(self.on_config_changed)
        self.max_tokens_spin.setToolTip("Maximum number of tokens to generate per caption")
        inference_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        # Top P
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.valueChanged.connect(self.on_config_changed)
        self.top_p_spin.setToolTip("Nucleus sampling threshold (0.0-1.0)")
        inference_layout.addRow("Top P:", self.top_p_spin)
        
        # Repetition penalty
        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setSingleStep(0.1)
        self.rep_penalty_spin.setValue(1.1)
        self.rep_penalty_spin.setDecimals(1)
        self.rep_penalty_spin.valueChanged.connect(self.on_config_changed)
        self.rep_penalty_spin.setToolTip("Penalty for repeating tokens (1.0=none, higher=more penalty)")
        inference_layout.addRow("Repetition Penalty:", self.rep_penalty_spin)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        # Image Pre-processing Group
        preprocessing_group = QGroupBox("Image Pre-processing")
        preprocessing_layout = QFormLayout()
        
        # Resize checkbox
        self.resize_enable_checkbox = QCheckBox("Resize images before inference")
        self.resize_enable_checkbox.setChecked(True)
        self.resize_enable_checkbox.setToolTip(
            "Resize large images for faster processing.\n"
            "Improves speed for high-resolution images without quality loss.\n"
            "Images smaller than max dimension are not resized."
        )
        self.resize_enable_checkbox.stateChanged.connect(self.on_config_changed)
        preprocessing_layout.addRow("", self.resize_enable_checkbox)
        
        # Max dimension spinbox
        self.max_dimension_spin = QSpinBox()
        self.max_dimension_spin.setRange(256, 4096)
        self.max_dimension_spin.setSingleStep(128)
        self.max_dimension_spin.setValue(1024)
        self.max_dimension_spin.setSuffix(" px")
        self.max_dimension_spin.valueChanged.connect(self.on_config_changed)
        self.max_dimension_spin.setToolTip(
            "Maximum width/height. Images larger than this will be resized.\n"
            "Recommended: 1024px (good balance), 512px (faster), 2048px (slower)"
        )
        preprocessing_layout.addRow("Max Dimension:", self.max_dimension_spin)
        
        # Cache resized images checkbox
        self.cache_resized_checkbox = QCheckBox("Cache resized images to disk")
        self.cache_resized_checkbox.setChecked(False)
        self.cache_resized_checkbox.setToolTip(
            "Save resized images in export directory for reuse.\n"
            "All images will be saved, even those not needing resize."
        )
        self.cache_resized_checkbox.stateChanged.connect(self.on_config_changed)
        preprocessing_layout.addRow("", self.cache_resized_checkbox)
        
        # Cache format dropdown
        self.cache_format_combo = QComboBox()
        self.cache_format_combo.addItems([
            "Original format",
            "PNG - Lossless",
            "JPEG - Quality 95"
        ])
        self.cache_format_combo.setToolTip(
            "Format for cached images:\n"
            "• Original: Keep source format\n"
            "• PNG: Lossless, larger files\n"
            "• JPEG Q95: 10x smaller, minimal quality loss"
        )
        self.cache_format_combo.currentIndexChanged.connect(self.on_config_changed)
        preprocessing_layout.addRow("Cache Format:", self.cache_format_combo)
        
        preprocessing_group.setLayout(preprocessing_layout)
        layout.addWidget(preprocessing_group)
        
        # Export Options Group
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        export_layout.addWidget(QLabel("Export formats (select one or more):"))
        
        self.export_txt_individual = QCheckBox("Individual .txt files (one per image)")
        self.export_txt_individual.setChecked(True)
        self.export_txt_individual.stateChanged.connect(self.on_config_changed)
        export_layout.addWidget(self.export_txt_individual)
        
        self.export_csv = QCheckBox("CSV file (all captions in one file)")
        self.export_csv.stateChanged.connect(self.on_config_changed)
        export_layout.addWidget(self.export_csv)
        
        self.export_json = QCheckBox("JSON file (structured format)")
        self.export_json.stateChanged.connect(self.on_config_changed)
        export_layout.addWidget(self.export_json)
        
        self.export_txt_batch = QCheckBox("Combined .txt file (all captions)")
        self.export_txt_batch.stateChanged.connect(self.on_config_changed)
        export_layout.addWidget(self.export_txt_batch)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
    
    def on_config_changed(self):
        """Handle configuration changes."""
        config = self.get_config()
        self.config = config
        self.config_changed.emit(config)
    
    def get_config(self) -> dict:
        """Get current configuration as dictionary."""
        # Map device combo to config value
        device_map = {
            "Auto": "auto",
            "CUDA (GPU)": "cuda",
            "CPU": "cpu"
        }
        
        # Map quantization combo to config value
        quant_map = {
            "Auto": "auto",
            "None (Full Precision)": "none",
            "8-bit": "8bit",
            "4-bit": "4bit"
        }
        
        # Get export formats
        formats = []
        if self.export_txt_individual.isChecked():
            formats.append("txt_individual")
        if self.export_csv.isChecked():
            formats.append("csv")
        if self.export_json.isChecked():
            formats.append("json")
        if self.export_txt_batch.isChecked():
            formats.append("txt_batch")
        
        return {
            "model": {
                "device": device_map[self.device_combo.currentText()],
                "quantization": quant_map[self.quant_combo.currentText()],
            },
            "inference": {
                "temperature": self.temp_spin.value(),
                "max_new_tokens": self.max_tokens_spin.value(),
                "top_p": self.top_p_spin.value(),
                "repetition_penalty": self.rep_penalty_spin.value(),
            },
            "processing": {
                "resize_before_inference": self.resize_enable_checkbox.isChecked(),
                "max_dimension": self.max_dimension_spin.value(),
                "cache_resized_images": self.cache_resized_checkbox.isChecked(),
                "cache_format": ["original", "png", "jpeg"][self.cache_format_combo.currentIndex()],
                "jpeg_quality": 95,
            },
            "export": {
                "formats": formats,
            }
        }
    
    def set_config(self, config: dict):
        """Set configuration from dictionary."""
        # Block signals during update
        self.blockSignals(True)
        
        # Model settings
        model_config = config.get("model", {})
        device = model_config.get("device", "auto")
        device_map_reverse = {"auto": "Auto", "cuda": "CUDA (GPU)", "cpu": "CPU"}
        self.device_combo.setCurrentText(device_map_reverse.get(device, "Auto"))
        
        quant = model_config.get("quantization", "auto")
        quant_map_reverse = {
            "auto": "Auto",
            "none": "None (Full Precision)",
            "8bit": "8-bit",
            "4bit": "4-bit"
        }
        self.quant_combo.setCurrentText(quant_map_reverse.get(quant, "Auto"))
        
        # Inference settings
        inference_config = config.get("inference", {})
        temp_value = inference_config.get("temperature", 0.2)
        if not validate_temperature(temp_value)[0]:
            logger.warning("Invalid temperature in config. Falling back to default.")
            temp_value = DEFAULT_CONFIG["inference"]["temperature"]
        self.temp_spin.setValue(temp_value)

        max_tokens_value = inference_config.get("max_new_tokens", 512)
        if not validate_max_tokens(max_tokens_value)[0]:
            logger.warning("Invalid max_new_tokens in config. Falling back to default.")
            max_tokens_value = DEFAULT_CONFIG["inference"]["max_new_tokens"]
        self.max_tokens_spin.setValue(max_tokens_value)

        top_p_value = inference_config.get("top_p", 0.9)
        if not validate_top_p(top_p_value)[0]:
            logger.warning("Invalid top_p in config. Falling back to default.")
            top_p_value = DEFAULT_CONFIG["inference"]["top_p"]
        self.top_p_spin.setValue(top_p_value)

        rep_penalty_value = inference_config.get("repetition_penalty", 1.1)
        if not validate_repetition_penalty(rep_penalty_value)[0]:
            logger.warning("Invalid repetition_penalty in config. Falling back to default.")
            rep_penalty_value = DEFAULT_CONFIG["inference"]["repetition_penalty"]
        self.rep_penalty_spin.setValue(rep_penalty_value)
        
        # Processing settings
        processing_config = config.get("processing", {})
        self.resize_enable_checkbox.setChecked(processing_config.get("resize_before_inference", True))
        self.max_dimension_spin.setValue(processing_config.get("max_dimension", 1024))
        self.cache_resized_checkbox.setChecked(processing_config.get("cache_resized_images", False))        
        # Cache format
        cache_format = processing_config.get("cache_format", "original")
        format_index = {"original": 0, "png": 1, "jpeg": 2}.get(cache_format, 0)
        self.cache_format_combo.setCurrentIndex(format_index)        
        # Export options
        export_config = config.get("export", {})
        formats = export_config.get("formats", ["txt_individual"])
        self.export_txt_individual.setChecked("txt_individual" in formats)
        self.export_csv.setChecked("csv" in formats)
        self.export_json.setChecked("json" in formats)
        self.export_txt_batch.setChecked("txt_batch" in formats)
        
        self.blockSignals(False)
        self.config = config
    
    def update_vram_info(self, quant_mode: str = None):
        """Update VRAM requirement information based on quantization."""
        if quant_mode is None:
            quant_mode = self.quant_combo.currentText()
        
        vram_info = {
            "Auto": "Auto-select based on available VRAM",
            "4-bit": "~4 GB (Recommended)",
            "8-bit": "~7 GB",
            "None (Full Precision)": "~14 GB (Best quality)"
        }
        
        info_text = vram_info.get(quant_mode, "Variable")
        self.vram_info_label.setText(info_text)
