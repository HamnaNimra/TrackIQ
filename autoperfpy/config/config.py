"""Configuration management system for AutoPerfPy."""

# TODO: Config/ConfigManager duplicate trackiq.config logic (get, update, to_dict,
# load_yaml, load_json, save_*). Differ in defaults (DEFAULT_CONFIG) and update
# semantics for nested dicts. Consider sharing a base or delegating to trackiq
# with app-specific overrides.

import os
from typing import Any, Dict, Optional
from dataclasses import asdict

from trackiq_core.config_io import (
    ensure_parent_dir,
    load_json_file,
    load_yaml_file,
    save_json_file,
    save_yaml_file,
)

from .defaults import (
    DEFAULT_CONFIG,
    BenchmarkConfig,
    LLMConfig,
    MonitoringConfig,
    AnalysisConfig,
    ProcessMonitorConfig,
)


class Config:
    """Unified configuration container for AutoPerfPy."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.

        Args:
            config_dict: Optional dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.update(config_dict)

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with provided values.

        Args:
            config_dict: Dictionary with configuration overrides
        """
        for key, value in config_dict.items():
            if key in self.config:
                if isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (e.g., 'benchmark.batch_sizes')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        result = {}
        for key, value in self.config.items():
            if hasattr(value, "__dataclass_fields__"):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result

    @property
    def benchmark(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        return self.config.get("benchmark")

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.get("llm")

    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.config.get("monitoring")

    @property
    def analysis(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self.config.get("analysis")

    @property
    def process(self) -> ProcessMonitorConfig:
        """Get process monitor configuration."""
        return self.config.get("process")


class ConfigManager:
    """Manages loading and saving configuration files."""

    @staticmethod
    def load_yaml(filepath: str) -> Config:
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Config object
        """
        config_dict = load_yaml_file(filepath)
        return Config(config_dict or {})

    @staticmethod
    def load_json(filepath: str) -> Config:
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            Config object
        """
        config_dict = load_json_file(filepath)
        return Config(config_dict or {})

    @staticmethod
    def save_yaml(config: Config, filepath: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Config object to save
            filepath: Output YAML file path
        """
        ensure_parent_dir(filepath)
        save_yaml_file(filepath, config.to_dict())

    @staticmethod
    def save_json(config: Config, filepath: str) -> None:
        """Save configuration to JSON file.

        Args:
            config: Config object to save
            filepath: Output JSON file path
        """
        ensure_parent_dir(filepath)
        save_json_file(filepath, config.to_dict())

    @staticmethod
    def load_or_default(filepath: Optional[str] = None) -> Config:
        """Load configuration from file or return defaults.

        Args:
            filepath: Optional path to configuration file

        Returns:
            Config object (loaded from file or defaults)
        """
        if filepath and os.path.exists(filepath):
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                return ConfigManager.load_yaml(filepath)
            elif filepath.endswith(".json"):
                return ConfigManager.load_json(filepath)
        return Config()
