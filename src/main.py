"""Main entry point for the Image Captioning GUI application."""

import sys
import argparse
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))


def main():
    """Initialize and run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Captioning Application")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    if args.verbose:
        logging.info("Verbose logging enabled")
    
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Image Captioning")
    app.setOrganizationName("ImageCaptioning")
    
    # Import here to avoid circular imports and ensure Qt is initialized
    from gui.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
