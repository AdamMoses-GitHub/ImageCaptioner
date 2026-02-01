"""Model status panel showing cache status and download options."""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox
)
from PySide6.QtCore import Qt, Signal
import logging

logger = logging.getLogger(__name__)


class ModelStatusPanel(QWidget):
    """Panel displaying model cache status and download options."""
    
    download_requested = Signal()  # Emitted when user wants to download/verify
    status_changed = Signal(bool)  # Emitted when status changes (is_cached)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_cached = False
        self.is_checking = False
        self.model_name = "llava-hf/llava-1.5-7b-hf"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Status indicator
        self.status_icon = QLabel("⏳")
        self.status_icon.setStyleSheet("font-size: 18pt;")
        layout.addWidget(self.status_icon)
        
        # Status text
        status_layout = QVBoxLayout()
        status_layout.setSpacing(2)
        
        self.status_label = QLabel("Model Status: Checking...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        status_layout.addWidget(self.status_label)
        
        self.details_label = QLabel("Verifying model cache...")
        self.details_label.setStyleSheet("color: gray; font-size: 9pt;")
        status_layout.addWidget(self.details_label)
        
        layout.addLayout(status_layout, stretch=1)
        
        # Download/Verify button
        self.action_btn = QPushButton("Checking...")
        self.action_btn.setEnabled(False)
        self.action_btn.setMinimumWidth(150)
        self.action_btn.clicked.connect(self.on_action_clicked)
        layout.addWidget(self.action_btn)
        
        # Set panel background
        self.setStyleSheet(
            "QWidget { "
            "  background-color: #ecf0f1; "
            "  border: 1px solid #bdc3c7; "
            "  border-radius: 4px; "
            "}"
        )
        self.setMaximumHeight(70)
    
    def set_checking(self):
        """Set UI to checking state."""
        self.is_checking = True
        self.status_icon.setText("⏳")
        self.status_label.setText("Model Status: Checking...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #3498db;")
        self.details_label.setText("Verifying model cache...")
        self.action_btn.setText("Checking...")
        self.action_btn.setEnabled(False)
        logger.debug("Status: Checking")
    
    def set_cached(self, model_size: str = "~13 GB", cache_location: str = ""):
        """Set UI to cached (ready) state."""
        self.is_checking = False
        self.is_cached = True
        self.status_icon.setText("✓")
        self.status_label.setText("Model Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #27ae60;")
        
        details = f"LLaVA 1.5 7B ({model_size}) cached and ready"
        if cache_location:
            details += f" • {cache_location}"
        self.details_label.setText(details)
        
        self.action_btn.setText("Verify Model")
        self.action_btn.setEnabled(True)
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #2980b9; }"
        )
        
        self.status_changed.emit(True)
        logger.info("Status: Model ready")
    
    def set_not_cached(self, model_size: str = "~13 GB"):
        """Set UI to not cached state."""
        self.is_checking = False
        self.is_cached = False
        self.status_icon.setText("⚠️")
        self.status_label.setText("Model Status: Not Downloaded")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #e67e22;")
        self.details_label.setText(f"LLaVA 1.5 7B ({model_size}) needs to be downloaded")
        
        self.action_btn.setText("Download Model")
        self.action_btn.setEnabled(True)
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #d35400; }"
        )
        
        self.status_changed.emit(False)
        logger.warning("Status: Model not cached")
    
    def set_downloading(self):
        """Set UI to downloading state."""
        self.is_checking = False
        self.status_icon.setText("⬇")
        self.status_label.setText("Model Status: Downloading")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #3498db;")
        self.details_label.setText("Downloading model files...")
        
        self.action_btn.setText("Downloading...")
        self.action_btn.setEnabled(False)
        logger.info("Status: Downloading")
    
    def set_error(self, error_msg: str = "Unknown error"):
        """Set UI to error state."""
        self.is_checking = False
        self.is_cached = False
        self.status_icon.setText("✗")
        self.status_label.setText("Model Status: Error")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #e74c3c;")
        self.details_label.setText(f"Error: {error_msg}")
        
        self.action_btn.setText("Retry")
        self.action_btn.setEnabled(True)
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        
        self.status_changed.emit(False)
        logger.error(f"Status: Error - {error_msg}")
    
    def on_action_clicked(self):
        """Handle action button click."""
        logger.info("Download/verify action requested")
        self.download_requested.emit()
    
    def get_is_cached(self) -> bool:
        """Get current cached status."""
        return self.is_cached
