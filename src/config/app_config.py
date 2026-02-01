"""Application configuration management."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from copy import deepcopy

from .defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class AppConfig:
    """Manage application configuration."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        if config_file is None:
            # Use default location
            config_file = Path("config") / "settings.yaml"
        
        self.config_file = Path(config_file)
        self.config = deepcopy(DEFAULT_CONFIG)
        
        # Load existing config if available
        if self.config_file.exists():
            self.load()
        else:
            logger.info("No existing configuration file, using defaults")
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config:
                # Merge with defaults to ensure all keys exist
                self.config = self._merge_configs(DEFAULT_CONFIG, loaded_config)
                logger.info(f"Loaded configuration from {self.config_file}")
                return True
            else:
                logger.warning("Configuration file is empty, using defaults")
                return False
                
        except FileNotFoundError:
            logger.info(f"Configuration file not found: {self.config_file}")
            return False
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.device')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self.config = self._merge_configs(self.config, updates)
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return deepcopy(self.config)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = deepcopy(DEFAULT_CONFIG)
        logger.info("Reset configuration to defaults")
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            updates: Updates to apply
            
        Returns:
            Merged configuration
        """
        result = deepcopy(base)
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


# Global configuration instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def save_config() -> bool:
    """Save the global configuration."""
    config = get_config()
    return config.save()
