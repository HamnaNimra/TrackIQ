"""Configuration management system for AutoPerfPy.

Extends trackiq_core.configs.Config with automotive-specific defaults and property accessors.
"""

from typing import Any

from trackiq_core.configs.config import Config as TrackIQConfig

from .defaults import (
    DEFAULT_CONFIG,
    AnalysisConfig,
    BenchmarkConfig,
    LLMConfig,
    MonitoringConfig,
    ProcessMonitorConfig,
)


class Config(TrackIQConfig):
    """Unified configuration container for AutoPerfPy.

    Extends trackiq_core.configs.Config with automotive-specific defaults
    and property accessors for typed config sections.
    """

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """Initialize configuration with automotive defaults.

        Args:
            config_dict: Optional dictionary to override defaults
        """
        # Initialize with automotive defaults
        super().__init__(DEFAULT_CONFIG.copy())
        if config_dict:
            self.update(config_dict)

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
    """Manages loading and saving configuration files.

    Wraps trackiq_core.configs.ConfigManager to return AutoPerfPy Config instances
    with automotive defaults.
    """

    @staticmethod
    def load_yaml(filepath: str) -> Config:
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Config object with automotive defaults
        """
        from trackiq_core.configs.config_io import load_yaml_file

        config_dict = load_yaml_file(filepath)
        return Config(config_dict or {})

    @staticmethod
    def load_json(filepath: str) -> Config:
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            Config object with automotive defaults
        """
        from trackiq_core.configs.config_io import load_json_file

        config_dict = load_json_file(filepath)
        return Config(config_dict or {})

    @staticmethod
    def save_yaml(config: Config, filepath: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Config object to save
            filepath: Output YAML file path
        """
        from trackiq_core.configs.config import ConfigManager as TrackIQConfigManager

        TrackIQConfigManager.save_yaml(config, filepath)

    @staticmethod
    def save_json(config: Config, filepath: str) -> None:
        """Save configuration to JSON file.

        Args:
            config: Config object to save
            filepath: Output JSON file path
        """
        from trackiq_core.configs.config import ConfigManager as TrackIQConfigManager

        TrackIQConfigManager.save_json(config, filepath)

    @staticmethod
    def load_or_default(filepath: str | None = None) -> Config:
        """Load configuration from file or return automotive defaults.

        Args:
            filepath: Optional path to configuration file

        Returns:
            Config object (loaded from file or automotive defaults)
        """
        import os

        if filepath and os.path.exists(filepath):
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                return ConfigManager.load_yaml(filepath)
            elif filepath.endswith(".json"):
                return ConfigManager.load_json(filepath)
        return Config()
