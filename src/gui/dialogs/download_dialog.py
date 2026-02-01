"""Dialog for model download progress."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from pathlib import Path
import logging

from models.downloader import ModelDownloader

logger = logging.getLogger(__name__)


class DownloadWorker(QThread):
    """Worker thread for downloading models."""
    
    progress_updated = Signal(int, int, str)  # current, total, message
    download_complete = Signal(bool)  # success
    
    def __init__(self, downloader: ModelDownloader):
        super().__init__()
        self.downloader = downloader
        self._is_cancelled = False
    
    def run(self):
        """Run the download in a background thread."""
        def progress_callback(current, total, message):
            if not self._is_cancelled:
                self.progress_updated.emit(current, total, message)
        
        success = self.downloader.download_model(progress_callback)
        self.download_complete.emit(success and not self._is_cancelled)
    
    def cancel(self):
        """Cancel the download."""
        self._is_cancelled = True
        self.downloader.cancel_download()


class DownloadDialog(QDialog):
    """Dialog showing model download progress and cache information."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.downloader = ModelDownloader(model_name)
        self.download_worker = None
        self.download_successful = False
        
        self.setWindowTitle("Model Download")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        self.setup_ui()
        self.update_cache_info()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("First-Time Setup: Model Download")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            f"The application needs to download the LLaVA 1.5 7B model "
            f"from Hugging Face.\n\n"
            f"This is a one-time download. The model will be cached locally "
            f"for future use."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(desc_label)
        
        # Cache Information Group
        cache_group = QGroupBox("Cache Information")
        cache_layout = QVBoxLayout()
        
        self.cache_location_label = QLabel()
        self.cache_location_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cache_location_label.setWordWrap(True)
        cache_layout.addWidget(self.cache_location_label)
        
        self.disk_space_label = QLabel()
        cache_layout.addWidget(self.disk_space_label)
        
        self.model_size_label = QLabel()
        cache_layout.addWidget(self.model_size_label)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        # Progress Group
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to download")
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # Progress details (text log)
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(100)
        self.progress_text.setStyleSheet("font-family: monospace; font-size: 9pt;")
        progress_layout.addWidget(self.progress_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Download")
        self.start_button.clicked.connect(self.start_download)
        self.start_button.setDefault(True)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_download)
        button_layout.addWidget(self.cancel_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def update_cache_info(self):
        """Update cache information display."""
        cache_info = self.downloader.get_cache_info()
        
        # Cache location
        cache_path = cache_info["cache_location"]
        self.cache_location_label.setText(
            f"<b>Cache Location:</b><br>{cache_path}"
        )
        
        # Available space
        available_space = cache_info["available_space_gb"]
        if available_space > 0:
            space_text = f"<b>Available Disk Space:</b> {available_space:.1f} GB"
            if available_space < 15:
                space_text += " ⚠️ <span style='color: orange;'>(Low disk space)</span>"
        else:
            space_text = "<b>Available Disk Space:</b> Unable to determine"
        self.disk_space_label.setText(space_text)
        
        # Model size
        model_size_gb, model_size_str = self.downloader.get_model_size()
        self.model_size_label.setText(
            f"<b>Download Size:</b> {model_size_str}"
        )
        
        # Check if sufficient space
        if available_space > 0 and available_space < model_size_gb + 2:
            self.log_message("⚠️ Warning: Insufficient disk space!", "orange")
            self.start_button.setEnabled(False)
    
    def start_download(self):
        """Start the model download."""
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.close_button.setEnabled(False)
        
        self.log_message(f"Starting download of {self.model_name}...")
        self.status_label.setText("Downloading model...")
        
        # Create and start worker thread
        self.download_worker = DownloadWorker(self.downloader)
        self.download_worker.progress_updated.connect(self.on_progress_update)
        self.download_worker.download_complete.connect(self.on_download_complete)
        self.download_worker.start()
    
    def cancel_download(self):
        """Cancel the ongoing download."""
        if self.download_worker and self.download_worker.isRunning():
            self.log_message("Cancelling download...", "orange")
            self.status_label.setText("Cancelling...")
            self.cancel_button.setEnabled(False)
            self.download_worker.cancel()
        else:
            self.reject()
    
    def on_progress_update(self, current: int, total: int, message: str):
        """
        Handle progress update from download worker.
        
        Args:
            current: Current progress value
            total: Total progress value
            message: Status message
        """
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            # Indeterminate progress
            self.progress_bar.setRange(0, 0)
        
        self.status_label.setText(message)
        self.log_message(message)
    
    def on_download_complete(self, success: bool):
        """
        Handle download completion.
        
        Args:
            success: Whether download was successful
        """
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        
        if success:
            self.download_successful = True
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.status_label.setText("✓ Download complete!")
            self.log_message("✓ Model downloaded successfully!", "green")
            
            # Auto-close after a short delay
            self.close_button.setText("Continue")
            self.close_button.setDefault(True)
            self.close_button.setFocus()
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("✗ Download failed or cancelled")
            self.log_message("✗ Download was not completed.", "red")
            self.close_button.setText("Close")
    
    def log_message(self, message: str, color: str = "black"):
        """
        Add a message to the progress log.
        
        Args:
            message: Message to log
            color: Text color
        """
        self.progress_text.append(f'<span style="color: {color};">{message}</span>')
        
        # Auto-scroll to bottom
        scrollbar = self.progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.download_worker and self.download_worker.isRunning():
            # Don't allow closing while download is in progress
            event.ignore()
            self.log_message("Please cancel the download first.", "orange")
        else:
            event.accept()
    
    def exec(self) -> bool:
        """
        Execute the dialog and return success status.
        
        Returns:
            True if model was downloaded successfully, False otherwise
        """
        result = super().exec()
        return self.download_successful


def show_download_dialog_if_needed(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    parent=None,
    force_check: bool = False
) -> bool:
    """
    Show download dialog if model is not cached.
    
    Args:
        model_name: Hugging Face model identifier
        parent: Parent widget
        force_check: If True, always show dialog for verification
        
    Returns:
        True if model is available (cached or downloaded), False otherwise
    """
    downloader = ModelDownloader(model_name)
    
    if not force_check and downloader.is_model_cached():
        logger.info(f"Model {model_name} is already cached")
        return True
    
    # Show download dialog
    dialog = DownloadDialog(model_name, parent)
    return dialog.exec()
