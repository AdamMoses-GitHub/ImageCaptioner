"""Main application window."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from pathlib import Path
from datetime import datetime
import logging

from .panels.input_panel import InputPanel
from .panels.config_panel import ConfigPanel
from .panels.prompt_panel import PromptPanel
from .panels.output_panel import OutputPanel
from .panels.model_status_panel import ModelStatusPanel
from .workers.inference_worker import InferenceWorker
from .workers.model_checker_worker import ModelCheckerWorker
from .dialogs.download_dialog import show_download_dialog_if_needed
from config.app_config import get_config, save_config
from processing.export import CaptionExporter

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Captioning Application")
        self.setMinimumSize(1200, 800)
        
        # Application state
        self.is_processing = False
        self.selected_directory = None
        self.inference_worker = None
        self.model_checker_worker = None
        self.processing_results = None
        self.app_config = get_config()
        
        self.setup_ui()
        self.connect_signals()
        self.load_configuration()
        
        # Check model status on startup
        self.check_model_status()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Compact title bar
        title = QLabel("Image Captioning with LLaVA")
        title.setStyleSheet(
            "font-size: 12pt; font-weight: bold; "
            "padding: 5px; background-color: #2c3e50; color: white; border-radius: 3px;"
        )
        title.setAlignment(Qt.AlignCenter)
        title.setMaximumHeight(35)
        main_layout.addWidget(title)
        
        # Model status panel
        self.model_status_panel = ModelStatusPanel()
        main_layout.addWidget(self.model_status_panel)
        
        # Create splitter for left and right panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (configuration)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Input panel
        self.input_panel = InputPanel()
        left_layout.addWidget(self.input_panel)
        
        # Scrollable area for config panels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        # Config panel
        self.config_panel = ConfigPanel()
        scroll_layout.addWidget(self.config_panel)
        
        # Prompt panel
        prompt_label = QLabel("Prompt Configuration")
        prompt_label.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 10px;")
        scroll_layout.addWidget(prompt_label)
        
        self.prompt_panel = PromptPanel()
        scroll_layout.addWidget(self.prompt_panel)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        left_layout.addWidget(scroll_area)
        
        splitter.addWidget(left_widget)
        
        # Right panel (output)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Output panel
        self.output_panel = OutputPanel()
        right_layout.addWidget(self.output_panel)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #4CAF50; "
            "  color: white; "
            "  font-weight: bold; "
            "  padding: 10px; "
            "  font-size: 11pt; "
            "} "
            "QPushButton:hover { background-color: #45a049; } "
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #f44336; "
            "  color: white; "
            "  font-weight: bold; "
            "  padding: 10px; "
            "  font-size: 11pt; "
            "} "
            "QPushButton:hover { background-color: #da190b; } "
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.stop_btn.clicked.connect(self.stop_processing)
        button_layout.addWidget(self.stop_btn)
        
        right_layout.addLayout(button_layout)
        
        splitter.addWidget(right_widget)
        
        # Set splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select a directory to begin")
    
    def connect_signals(self):
        """Connect signals between components."""
        # Input panel signals
        self.input_panel.directory_selected.connect(self.on_directory_selected)
        
        # Config panel signals
        self.config_panel.config_changed.connect(self.on_config_changed)
        
        # Prompt panel signals
        self.prompt_panel.prompt_changed.connect(self.on_prompt_changed)
        
        # Model status panel signals
        self.model_status_panel.download_requested.connect(self.on_download_requested)
    
    def on_directory_selected(self, directory):
        """Handle directory selection."""
        self.selected_directory = directory
        image_count = self.input_panel.get_image_count()
        
        if image_count > 0:
            self.start_btn.setEnabled(True)
            self.statusBar().showMessage(
                f"Ready to process {image_count} image(s) from {directory.name}"
            )
            logger.info(f"Directory selected: {directory}")
        else:
            self.start_btn.setEnabled(False)
            self.statusBar().showMessage("No images found in selected directory")
    
    def on_config_changed(self, config):
        """Handle configuration changes."""
        logger.debug(f"Configuration updated: {config}")
    
    def on_prompt_changed(self, prompt):
        """Handle prompt changes."""
        logger.debug(f"Prompt updated: {prompt[:50]}...")
    
    def load_configuration(self):
        """Load saved configuration into UI."""
        try:
            # Load panel configurations
            config_data = self.app_config.get_all()
            
            # Load config panel settings
            self.config_panel.set_config(config_data)
            
            # Load custom prompts
            custom_prompts = config_data.get("custom_prompts", [])
            self.prompt_panel.load_custom_prompts(custom_prompts)
            
            # Load trigger word
            trigger_word = config_data.get("inference", {}).get("trigger_word", "")
            self.prompt_panel.set_trigger_word(trigger_word)
            
            # Load last used directory
            last_dir = config_data.get("ui", {}).get("last_input_directory")
            if last_dir:
                last_dir_path = Path(last_dir)
                if last_dir_path.exists():
                    self.input_panel.set_directory(last_dir_path)
            
            logger.info("Loaded configuration")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_configuration(self):
        """Save current configuration."""
        try:
            # Get current settings
            config = self.config_panel.get_config()
            self.app_config.update(config)
            
            # Save custom prompts
            custom_prompts = self.prompt_panel.get_custom_prompts()
            self.app_config.set("custom_prompts", custom_prompts)
            
            # Save trigger word
            trigger_word = self.prompt_panel.get_trigger_word()
            self.app_config.set("inference.trigger_word", trigger_word)
            
            # Save last used directory
            if self.selected_directory:
                self.app_config.set("ui.last_input_directory", str(self.selected_directory))
            
            # Save to file
            self.app_config.save()
            logger.info("Saved configuration")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def start_processing(self):
        """Start image processing."""
        if not self.selected_directory:
            QMessageBox.warning(
                self,
                "No Directory Selected",
                "Please select a directory containing images first."
            )
            return
        
        # Check if at least one export format is selected
        config = self.config_panel.get_config()
        export_formats = config.get("export", {}).get("formats", [])
        
        if not export_formats:
            QMessageBox.warning(
                self,
                "No Export Format",
                "Please select at least one export format."
            )
            return
        
        # Check if model is cached, show download dialog if needed
        model_available = show_download_dialog_if_needed(parent=self)
        if not model_available:
            QMessageBox.warning(
                self,
                "Model Not Available",
                "The model is not available. Please download it first."
            )
            return
        
        # Get configuration
        model_config = config.get("model", {})
        inference_config = config.get("inference", {})
        processing_config = config.get("processing", {})
        prompt = self.prompt_panel.get_prompt()
        
        if not prompt.strip():
            QMessageBox.warning(
                self,
                "Empty Prompt",
                "Please enter a prompt for caption generation."
            )
            return
        
        # Create timestamped export directory early (for resize caching)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_export_dir = self.selected_directory / f"captions_{timestamp}"
        self.current_export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created export directory: {self.current_export_dir}")
        
        # Update UI state
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_panel.setEnabled(False)
        self.config_panel.setEnabled(False)
        self.prompt_panel.setEnabled(False)
        
        # Initialize output panel
        image_count = self.input_panel.get_image_count()
        self.output_panel.start_processing(image_count)
        
        # Get trigger word from prompt panel
        trigger_word = self.prompt_panel.get_trigger_word()
        
        # Create worker thread with full config
        self.inference_worker = InferenceWorker(
            directory=self.selected_directory,
            model_config=model_config,
            inference_config=inference_config,
            prompt=prompt,
            trigger_word=trigger_word,
            processing_config=processing_config
        )
        
        # Pass export directory to worker for resize caching
        self.inference_worker.export_dir = self.current_export_dir
        
        # Connect worker signals
        self.inference_worker.progress_updated.connect(self.on_progress_updated)
        self.inference_worker.caption_generated.connect(self.on_caption_generated)
        self.inference_worker.error_occurred.connect(self.on_error_occurred)
        self.inference_worker.status_message.connect(self.on_status_message)
        self.inference_worker.finished.connect(self.on_processing_finished)
        
        # Start processing
        self.inference_worker.start()
        
        self.statusBar().showMessage("Processing images...")
        logger.info(f"Started processing {image_count} images")
    
    def stop_processing(self):
        """Stop image processing."""
        if self.inference_worker and self.inference_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Stop",
                "Are you sure you want to stop processing?\n\n"
                "Progress will be saved for completed images.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.statusBar().showMessage("Stopping processing...")
                self.inference_worker.cancel()
                logger.info("User requested stop")
    
    def on_progress_updated(self, current: int, total: int, image_name: str):
        """Handle progress updates from worker."""
        self.output_panel.update_progress(current, total, image_name)
        self.statusBar().showMessage(f"Processing: {image_name} ({current}/{total})")
    
    def on_caption_generated(self, image_name: str, caption: str, generation_time: float = 0.0, 
                            file_size: int = 0, dimensions: str = "", img_type: str = ""):
        """Handle successful caption generation."""
        self.output_panel.add_caption_log(image_name, caption, is_error=False, 
                                         generation_time=generation_time, file_size=file_size,
                                         dimensions=dimensions, img_type=img_type)
    
    def on_error_occurred(self, image_name: str, error_message: str):
        """Handle processing errors."""
        self.output_panel.add_caption_log(image_name, error_message, is_error=True)
    
    def on_status_message(self, message: str):
        """Handle status messages from worker."""
        self.statusBar().showMessage(message)
    
    def on_processing_finished(self, success: bool, summary: dict):
        """Handle processing completion."""
        self.is_processing = False
        self.processing_results = summary
        
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_panel.setEnabled(True)
        self.config_panel.setEnabled(True)
        self.prompt_panel.setEnabled(True)
        
        # Update output panel
        self.output_panel.finish_processing(success)
        
        # Show summary
        if success:
            self.statusBar().showMessage("Processing completed successfully!")
            
            # Show completion dialog
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Image processing completed successfully!\n\n"
                f"• Processed: {summary.get('total_processed', 0)} images\n"
                f"• Errors: {summary.get('total_errors', 0)} images\n\n"
                f"Results are ready for export."
            )
        else:
            if summary.get('cancelled'):
                self.statusBar().showMessage("Processing cancelled by user")
                QMessageBox.information(
                    self,
                    "Processing Cancelled",
                    f"Processing was cancelled.\n\n"
                    f"• Processed: {summary.get('total_processed', 0)} images\n"
                    f"• Errors: {summary.get('total_errors', 0)} images\n\n"
                    f"Completed captions are ready for export."
                )
            else:
                error_msg = summary.get('error', 'Unknown error')
                self.statusBar().showMessage(f"Processing failed: {error_msg}")
                QMessageBox.critical(
                    self,
                    "Processing Failed",
                    f"Processing failed with error:\n\n{error_msg}"
                )
        
        # Show error details if any
        errors = summary.get('errors', [])
        if errors and len(errors) > 0:
            self.show_error_summary(errors)
        
        # Trigger export if processing was successful
        if success or summary.get('total_processed', 0) > 0:
            self.export_results(summary)
        
        logger.info(f"Processing finished: {summary}")
    
    def show_error_summary(self, errors: list):
        """Show a summary of processing errors."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Processing Errors ({len(errors)})")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel(f"{len(errors)} image(s) failed to process:")
        layout.addWidget(label)
        
        # List of errors
        error_list = QListWidget()
        for error_info in errors:
            path = error_info.get('path', 'Unknown')
            if hasattr(path, 'name'):
                name = path.name
            else:
                name = str(path)
            error_msg = error_info.get('error', 'Unknown error')
            error_list.addItem(f"{name}: {error_msg}")
        layout.addWidget(error_list)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def export_results(self, summary: dict):
        """Export processing results."""
        results = summary.get('results', [])
        if not results:
            logger.warning("No results to export")
            return
        
        # Get export formats
        config = self.config_panel.get_config()
        formats = config.get("export", {}).get("formats", [])
        
        if not formats:
            logger.warning("No export formats selected")
            return
        
        # Create exporter
        exporter = CaptionExporter()
        
        # Use the export directory created during processing
        if not hasattr(self, 'current_export_dir') or not self.current_export_dir:
            logger.error("Export directory not available")
            return
        
        output_dir = self.current_export_dir
        base_output_path = output_dir / "captions"
        
        # Prepare metadata
        metadata = {
            "model": "llava-1.5-7b",
            "total_images": len(results),
            "configuration": config
        }
        
        try:
            # Export in all selected formats
            export_status = exporter.export_all(
                results=results,
                formats=formats,
                base_output_path=base_output_path,
                metadata=metadata
            )
            
            # Show export results
            success_formats = [fmt for fmt, status in export_status.items() if status]
            failed_formats = [fmt for fmt, status in export_status.items() if not status]
            
            if success_formats:
                message = f"Successfully exported captions in {len(success_formats)} format(s):\n"
                for fmt in success_formats:
                    message += f"  • {fmt}\n"
                
                message += f"\nExport location:\n{output_dir}"
                
                if failed_formats:
                    message += f"\n\nFailed formats: {', '.join(failed_formats)}"
                
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Export Complete")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText(message)
                open_btn = msg_box.addButton("Open Export Folder", QMessageBox.ActionRole)
                msg_box.addButton(QMessageBox.Ok)
                msg_box.exec()

                if msg_box.clickedButton() == open_btn:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
                logger.info(f"Export complete: {export_status}")
            else:
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    "Failed to export captions in any format. Check logs for details."
                )
                
        except Exception as e:
            error_msg = f"Error during export: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(
                self,
                "Export Error",
                error_msg
            )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Save configuration before closing
        self.save_configuration()
        
        if self.is_processing:
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Processing is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Stop processing and clean up
                self.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def check_model_status(self):
        """Check model cache status in background thread."""
        logger.info("Starting model status check...")
        self.model_status_panel.set_checking()
        
        # Create and start worker thread
        self.model_checker_worker = ModelCheckerWorker("llava-hf/llava-1.5-7b-hf")
        self.model_checker_worker.check_complete.connect(self.on_model_check_complete)
        self.model_checker_worker.start()
    
    def on_model_check_complete(self, is_cached: bool, model_info: dict):
        """Handle completion of model status check."""
        try:
            if "error" in model_info:
                # Error occurred during check
                error_msg = model_info["error"]
                self.model_status_panel.set_error(error_msg)
                self.statusBar().showMessage(f"Model check error: {error_msg}")
            elif is_cached:
                # Model is cached and ready
                size_str = model_info.get("size_str", "~13 GB")
                cache_loc = model_info.get("cache_location", "")
                self.model_status_panel.set_cached(size_str, cache_loc)
                self.statusBar().showMessage("Model ready - Select a directory to begin")
            else:
                # Model is not cached
                size_str = model_info.get("size_str", "~13 GB")
                self.model_status_panel.set_not_cached(size_str)
                self.statusBar().showMessage("Model not downloaded - Click 'Download Model' to proceed")
            
            logger.info(f"Model check complete: cached={is_cached}")
            
        except Exception as e:
            logger.error(f"Error handling model check result: {e}")
            self.model_status_panel.set_error(str(e))
    
    def on_download_requested(self):
        """Handle download/verify button click from status panel."""
        from models.downloader import ModelDownloader
        
        downloader = ModelDownloader("llava-hf/llava-1.5-7b-hf")
        is_cached = downloader.is_model_cached()
        
        if is_cached:
            # Model is already cached - just verify
            logger.info("Verifying cached model...")
            self.model_status_panel.set_checking()
            self.statusBar().showMessage("Verifying model...")
            
            # Quick verification by re-checking cache
            QMessageBox.information(
                self,
                "Model Verification",
                "Model verification complete!\n\n"
                f"Model: llava-hf/llava-1.5-7b-hf\n"
                f"Status: ✓ Cached and ready\n"
                f"Location: {downloader.cache_dir / 'hub'}\n\n"
                "The model files are present and can be loaded."
            )
            
            # Re-check to update status panel
            self.check_model_status()
        else:
            # Model not cached - show download dialog
            logger.info("Model not cached, showing download dialog")
            self.model_status_panel.set_downloading()
            model_available = show_download_dialog_if_needed(parent=self, force_check=True)
            
            # Re-check status after download
            if model_available:
                self.check_model_status()
            else:
                self.model_status_panel.set_not_cached()
                QMessageBox.warning(
                    self,
                    "Download Failed",
                    "Model download was cancelled or failed. Please try again."
                )
