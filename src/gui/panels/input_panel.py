"""Input panel for directory selection."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QGroupBox
)
from PySide6.QtCore import Signal
from pathlib import Path
import logging

from utils.validators import validate_directory, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


class InputPanel(QWidget):
    """Panel for selecting input directory and viewing image count."""
    
    directory_selected = Signal(Path)  # Emitted when valid directory is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = None
        self.image_count = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Group box
        group = QGroupBox("Input Directory")
        group_layout = QVBoxLayout()
        
        # Directory selection
        dir_layout = QHBoxLayout()
        
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select a directory containing images...")
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_edit, stretch=1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.browse_btn)
        
        group_layout.addLayout(dir_layout)
        
        # Image count display
        self.count_label = QLabel("No directory selected")
        self.count_label.setStyleSheet("color: gray; padding: 5px;")
        group_layout.addWidget(self.count_label)
        
        # Supported formats info
        formats_text = ", ".join(sorted(SUPPORTED_FORMATS))
        formats_label = QLabel(f"<b>Supported formats:</b> {formats_text}")
        formats_label.setWordWrap(True)
        formats_label.setStyleSheet("font-size: 9pt; color: #555; padding: 5px;")
        group_layout.addWidget(formats_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def browse_directory(self):
        """Open directory selection dialog."""
        # Start from current directory or home
        start_dir = str(self.current_directory) if self.current_directory else str(Path.home())
        
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dir_path:
            self.set_directory(Path(dir_path))
    
    def set_directory(self, dir_path: Path):
        """Set and validate the directory."""
        # Validate directory
        valid, error_msg = validate_directory(dir_path)
        if not valid:
            self.count_label.setText(f"❌ Error: {error_msg}")
            self.count_label.setStyleSheet("color: red; padding: 5px;")
            return
        
        self.current_directory = dir_path
        self.dir_edit.setText(str(dir_path))
        
        # Count images
        self.image_count = self._count_images(dir_path)
        
        if self.image_count == 0:
            self.count_label.setText("⚠️ No supported images found in this directory")
            self.count_label.setStyleSheet("color: orange; padding: 5px;")
        else:
            self.count_label.setText(f"✓ Found {self.image_count} image(s) ready for processing")
            self.count_label.setStyleSheet("color: green; padding: 5px;")
            
            # Emit signal
            self.directory_selected.emit(dir_path)
        
        logger.info(f"Directory selected: {dir_path} ({self.image_count} images)")
    
    def _count_images(self, dir_path: Path) -> int:
        """Count supported images in directory (non-recursive)."""
        count = 0
        try:
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix.lower() in SUPPORTED_FORMATS:
                    count += 1
        except Exception as e:
            logger.error(f"Error counting images: {e}")
        
        return count
    
    def get_directory(self) -> Path:
        """Get the currently selected directory."""
        return self.current_directory
    
    def get_image_count(self) -> int:
        """Get the number of images found."""
        return self.image_count
