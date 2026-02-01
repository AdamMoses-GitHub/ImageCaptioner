"""Output panel for progress and results."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QTextEdit, QPushButton, QGroupBox
)
from PySide6.QtCore import Qt
import logging
import time
from datetime import timedelta

logger = logging.getLogger(__name__)


class OutputPanel(QWidget):
    """Panel for displaying progress and processing results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_time = None
        self.total_images = 0
        self.processed_count = 0
        self.error_count = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Progress Group
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready to process images")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Statistics layout
        stats_layout = QHBoxLayout()
        
        self.processed_label = QLabel("Processed: 0")
        stats_layout.addWidget(self.processed_label)
        
        self.errors_label = QLabel("Errors: 0")
        self.errors_label.setStyleSheet("color: red;")
        stats_layout.addWidget(self.errors_label)
        
        self.eta_label = QLabel("ETA: --:--")
        stats_layout.addWidget(self.eta_label)
        
        stats_layout.addStretch()
        progress_layout.addLayout(stats_layout)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Caption Log Group (fills remaining space)
        captions_group = QGroupBox("Caption Log")
        captions_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 10pt; background-color: #f5f5f5; padding: 5px;"
        )
        captions_layout.addWidget(self.log_text, stretch=1)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_log_btn)
        
        log_controls.addStretch()
        captions_layout.addLayout(log_controls)
        
        captions_group.setLayout(captions_layout)
        layout.addWidget(captions_group)
    
    def start_processing(self, total_images: int):
        """Initialize for processing."""
        self.start_time = time.time()
        self.total_images = total_images
        self.processed_count = 0
        self.error_count = 0
        
        self.progress_bar.setRange(0, total_images)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing images...")
        self.update_stats()
        
        logger.info(f"Started processing {total_images} images")
    
    def update_progress(self, current: int, total: int, image_name: str = ""):
        """Update progress display."""
        self.processed_count = current
        self.total_images = total
        
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        if image_name:
            self.status_label.setText(f"Processing: {image_name}")
        
        self.update_stats()
        self.update_eta()
    
    def add_caption_log(self, image_name: str, caption: str, is_error: bool = False, 
                        generation_time: float = 0.0, file_size: int = 0,
                        dimensions: str = "", img_type: str = ""):
        """Add a caption to the log with detailed statistics."""
        if is_error:
            self.error_count += 1
            color = "red"
            icon = "❌"
        else:
            color = "green"
            icon = "✓"
        
        # Format file size
        if file_size > 0:
            if file_size < 1024:
                size_str = f"{file_size}B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f}KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.2f}MB"
        else:
            size_str = "N/A"
        
        # Caption - show more text now
        display_caption = caption[:400] + "..." if len(caption) > 400 else caption
        
        # Build log entry with statistics
        log_entry = f'<div style="margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 8px;">'
        log_entry += f'<span style="color: {color}; font-weight: bold;">{icon} {image_name}</span>'
        
        if not is_error and generation_time > 0:
            log_entry += f' <span style="color: #666; font-size: 9pt;">'
            # Include dimensions and type in metadata
            metadata_parts = [f"{generation_time:.2f}s", size_str, f"{len(caption)} chars"]
            if dimensions:
                metadata_parts.append(dimensions)
            if img_type:
                metadata_parts.append(img_type)
            log_entry += f'[{", ".join(metadata_parts)}]</span>'
        
        log_entry += f'<br>'
        log_entry += f'<span style="color: #333; margin-left: 20px;">{display_caption}</span>'
        log_entry += f'</div>'
        
        self.log_text.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.update_stats()
    
    def update_stats(self):
        """Update statistics display."""
        self.processed_label.setText(f"Processed: {self.processed_count}/{self.total_images}")
        self.errors_label.setText(f"Errors: {self.error_count}")
    
    def update_eta(self):
        """Update estimated time remaining."""
        if not self.start_time or self.processed_count == 0:
            self.eta_label.setText("ETA: calculating...")
            return
        
        elapsed = time.time() - self.start_time
        avg_time_per_image = elapsed / self.processed_count
        remaining_images = self.total_images - self.processed_count
        eta_seconds = avg_time_per_image * remaining_images
        
        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            eta_str = f"{hours}h {minutes}m"
        
        self.eta_label.setText(f"ETA: {eta_str}")
    
    def finish_processing(self, success: bool = True):
        """Mark processing as finished."""
        if success:
            self.status_label.setText("✓ Processing complete!")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 11pt; color: green;"
            )
        else:
            self.status_label.setText("✗ Processing stopped")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 11pt; color: orange;"
            )
        
        self.eta_label.setText("ETA: --:--")
        
        # Calculate total time
        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            self.log_text.append(
                f'<br><span style="color: blue; font-weight: bold;">'
                f'Processing completed in {elapsed_str}</span><br>'
            )
        
        logger.info("Processing finished")
    
    def clear_log(self):
        """Clear the caption log."""
        self.log_text.clear()
    
    def reset(self):
        """Reset the panel to initial state."""
        self.start_time = None
        self.total_images = 0
        self.processed_count = 0
        self.error_count = 0
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to process images")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.processed_label.setText("Processed: 0")
        self.errors_label.setText("Errors: 0")
        self.eta_label.setText("ETA: --:--")
