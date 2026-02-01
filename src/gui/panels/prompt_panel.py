"""Prompt panel for template selection and editing."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTextEdit, QPushButton, QMessageBox,
    QGroupBox, QLineEdit
)
from PySide6.QtCore import Signal
import logging

from config.prompts import get_preset_prompts, get_default_prompt

logger = logging.getLogger(__name__)


class PromptPanel(QWidget):
    """Panel for selecting and editing prompt templates."""
    
    prompt_changed = Signal(str)  # Emitted when prompt text changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preset_prompts = get_preset_prompts()
        self.custom_prompts = []  # Will be loaded from config
        self.current_prompt = get_default_prompt()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Preset selection
        preset_layout = QHBoxLayout()
        
        preset_label = QLabel("Template:")
        preset_layout.addWidget(preset_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Select a preset prompt template")
        self._populate_presets()
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo, stretch=1)
        
        layout.addLayout(preset_layout)
        
        # Prompt text editor
        prompt_label = QLabel("Prompt Text:")
        layout.addWidget(prompt_label)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        self.prompt_edit.setMaximumHeight(100)
        self.prompt_edit.setToolTip("Edit the prompt text. This will be sent to the model with each image.")
        self.prompt_edit.setText(self.current_prompt)
        self.prompt_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.prompt_edit)
        
        # Trigger word/prefix field
        trigger_label = QLabel("Trigger Word/Prefix:")
        layout.addWidget(trigger_label)
        
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("Optional: e.g., 'a photo of ' (note the trailing space)")
        self.trigger_edit.setToolTip(
            "Add a prefix/trigger word to prepend to all captions.\n"
            "The text will be used exactly as entered (no trimming).\n"
            "Include your own spacing if needed."
        )
        self.trigger_edit.textChanged.connect(self.on_trigger_word_changed)
        layout.addWidget(self.trigger_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_custom_btn = QPushButton("Save as Custom")
        self.save_custom_btn.setToolTip("Save this prompt as a custom template")
        self.save_custom_btn.clicked.connect(self.save_custom_prompt)
        button_layout.addWidget(self.save_custom_btn)
        
        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.setToolTip("Reset to the default prompt")
        self.reset_btn.clicked.connect(self.reset_to_default)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Info label
        info_label = QLabel(
            "💡 Tip: The prompt guides the model's captioning style. "
            "Try different templates for varying levels of detail."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 9pt; padding: 5px;")
        layout.addWidget(info_label)
    
    def _populate_presets(self):
        """Populate the preset combo box."""
        self.preset_combo.clear()
        
        # Add preset prompts
        for prompt_data in self.preset_prompts:
            self.preset_combo.addItem(
                prompt_data["name"],
                prompt_data["prompt"]
            )
        
        # Add separator before custom prompts if any exist
        if self.custom_prompts:
            self.preset_combo.insertSeparator(self.preset_combo.count())
            
            # Add custom prompts
            for idx, custom_prompt in enumerate(self.custom_prompts):
                name = custom_prompt.get("name", f"Custom {idx + 1}")
                prompt_text = custom_prompt.get("prompt", "")
                self.preset_combo.addItem(
                    f"✏️ {name}",
                    prompt_text
                )
    
    def on_preset_changed(self, index: int):
        """Handle preset selection change."""
        if index >= 0:
            prompt_text = self.preset_combo.itemData(index)
            if prompt_text:
                # Block signals to avoid triggering text_changed
                self.prompt_edit.blockSignals(True)
                self.prompt_edit.setText(prompt_text)
                self.prompt_edit.blockSignals(False)
                
                self.current_prompt = prompt_text
                self.prompt_changed.emit(prompt_text)
                logger.debug(f"Preset changed to: {self.preset_combo.currentText()}")
    
    def on_text_changed(self):
        """Handle manual text editing."""
        new_text = self.prompt_edit.toPlainText().strip()
        if new_text != self.current_prompt:
            self.current_prompt = new_text
            self.prompt_changed.emit(new_text)
            
            # Update combo to show "Custom" if text doesn't match any preset
            if not self._matches_preset(new_text):
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(-1)
                self.preset_combo.setEditText("Custom (Modified)")
                self.preset_combo.blockSignals(False)
    
    def _matches_preset(self, text: str) -> bool:
        """Check if text matches any preset."""
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemData(i) == text:
                return True
        return False
    
    def save_custom_prompt(self):
        """Save the current prompt as a custom template."""
        from PySide6.QtWidgets import QInputDialog
        
        current_text = self.prompt_edit.toPlainText().strip()
        if not current_text:
            QMessageBox.warning(
                self,
                "Empty Prompt",
                "Cannot save an empty prompt."
            )
            return
        
        # Ask for a name
        name, ok = QInputDialog.getText(
            self,
            "Save Custom Prompt",
            "Enter a name for this prompt template:",
            text=f"Custom Prompt"
        )
        
        if ok and name:
            # Add to custom prompts
            custom_prompt = {
                "name": name.strip(),
                "prompt": current_text
            }
            
            # Check if already exists
            exists = False
            for i, cp in enumerate(self.custom_prompts):
                if cp.get("prompt") == current_text:
                    # Update existing
                    self.custom_prompts[i] = custom_prompt
                    exists = True
                    break
            
            if not exists:
                self.custom_prompts.append(custom_prompt)
            
            # Refresh combo box
            self._populate_presets()
            
            # Select the new custom prompt
            for i in range(self.preset_combo.count()):
                if self.preset_combo.itemData(i) == current_text:
                    self.preset_combo.setCurrentIndex(i)
                    break
            
            logger.info(f"Saved custom prompt: {name}")
            
            QMessageBox.information(
                self,
                "Saved",
                f"Custom prompt '{name}' has been saved.\n\n"
                "Note: Custom prompts will be persisted when you save the configuration."
            )
    
    def reset_to_default(self):
        """Reset prompt to the default."""
        default_prompt = get_default_prompt()
        self.prompt_edit.setText(default_prompt)
        
        # Select the default in combo
        self.preset_combo.setCurrentIndex(0)
        
        logger.debug("Reset to default prompt")
    
    def get_prompt(self) -> str:
        """
        Get the current prompt text.
        
        Returns:
            Current prompt as string
        """
        return self.prompt_edit.toPlainText().strip()
    
    def get_trigger_word(self) -> str:
        """
        Get the trigger word/prefix text.
        
        Returns:
            Trigger word as string (exactly as entered, no stripping)
        """
        return self.trigger_edit.text()
    
    def on_trigger_word_changed(self):
        """Handle trigger word text change."""
        # Signal will be caught by main window to save to config
        pass
    
    def set_trigger_word(self, trigger_word: str):
        """
        Set the trigger word/prefix text.
        
        Args:
            trigger_word: Trigger word text to set
        """
        self.trigger_edit.setText(trigger_word)
    
    def set_prompt(self, prompt: str):
        """
        Set the prompt text.
        
        Args:
            prompt: Prompt text to set
        """
        self.prompt_edit.setText(prompt)
        
        # Try to find matching preset
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemData(i) == prompt:
                self.preset_combo.setCurrentIndex(i)
                return
        
        # If no match, just set the text
        self.current_prompt = prompt
    
    def load_custom_prompts(self, custom_prompts: list):
        """
        Load custom prompts from configuration.
        
        Args:
            custom_prompts: List of custom prompt dictionaries
        """
        self.custom_prompts = custom_prompts.copy() if custom_prompts else []
        self._populate_presets()
        logger.info(f"Loaded {len(self.custom_prompts)} custom prompts")
    
    def get_custom_prompts(self) -> list:
        """
        Get the list of custom prompts.
        
        Returns:
            List of custom prompt dictionaries
        """
        return self.custom_prompts.copy()

