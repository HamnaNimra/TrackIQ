"""Configuration management system for AutoPerfPy."""

import os
import json
from typing import Any, Dict, Optional
from dataclasses import asdict
import yaml

from .defaults import DEFAULT_CONFIG, BenchmarkConfig, LLMConfig, MonitoringConfig, AnalysisConfig, ProcessMonitorConfig


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
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config(config_dict)

    @staticmethod
    def load_json(filepath: str) -> Config:
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            Config object
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return Config(config_dict)

    @staticmethod
    def save_yaml(config: Config, filepath: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Config object to save
            filepath: Output YAML file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)

    @staticmethod
    def save_json(config: Config, filepath: str) -> None:
        """Save configuration to JSON file.

        Args:
            config: Config object to save
            filepath: Output JSON file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

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
