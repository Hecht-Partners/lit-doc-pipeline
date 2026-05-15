"""
Configuration file loader for the litigation document pipeline.

Supports loading pipeline parameters from JSON or YAML config files.
CLI arguments override config file settings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and merge configuration from files and CLI arguments.

    Priority (highest to lowest):
    1. CLI arguments (explicit user input)
    2. Config file settings
    3. Default values
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config file (JSON or YAML)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        try:
            if path.suffix == '.json':
                with open(path) as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config from {config_path}")

            elif path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    with open(path) as f:
                        self.config = yaml.safe_load(f)
                    logger.info(f"Loaded config from {config_path}")
                except ImportError:
                    logger.error("YAML support requires PyYAML: pip install pyyaml")
                    return

            else:
                logger.warning(f"Unsupported config format: {path.suffix}")

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")

    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """
        Get config value by key, with optional section.

        Args:
            key: Config key to retrieve
            default: Default value if key not found
            section: Optional section name (for nested configs)

        Returns:
            Config value or default

        Examples:
            config.get("chunk_size", 1000)
            config.get("k1", 1.5, section="bm25")
        """
        if section:
            section_config = self.config.get(section, {})
            return section_config.get(key, default)
        return self.config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire config section.

        Args:
            section: Section name

        Returns:
            Dict of section config, or empty dict if not found
        """
        return self.config.get(section, {})

    def merge_cli_args(self, cli_args: Dict[str, Any], section: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge CLI arguments with config file settings.

        CLI arguments take precedence over config file.

        Args:
            cli_args: Dict of CLI arguments
            section: Optional section to merge from

        Returns:
            Merged configuration dict
        """
        # Start with config file settings
        if section:
            merged = self.get_section(section).copy()
        else:
            merged = self.config.copy()

        # Override with CLI args (only if explicitly set)
        for key, value in cli_args.items():
            if value is not None:
                merged[key] = value

        return merged


def load_default_config() -> Dict[str, Any]:
    """
    Load the default pipeline configuration.

    Checks for config files in this order:
    1. ./lit-pipeline.json
    2. ./lit-pipeline.yaml
    3. ./configs/default_config.json
    4. Built-in defaults

    Returns:
        Configuration dict
    """
    # Try to find config file
    config_paths = [
        "lit-pipeline.json",
        "lit-pipeline.yaml",
        "lit-pipeline.yml",
        "configs/default_config.json",
    ]

    for config_path in config_paths:
        if Path(config_path).exists():
            loader = ConfigLoader(config_path)
            return loader.config

    # Return built-in defaults
    return {
        "chunking": {
            "min_chunk_chars": 300,
            "max_chunk_chars": 15000,
            "target_chunk_chars": 8000,
            "overlap_paragraphs": 3
        },
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "max_features": 10000,
            "ngram_range": [1, 2]
        },
        "chroma": {
            "embedding_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434/api/embeddings"
        },
        "enrichment": {
            "backend": "ollama",
            "delay_between_calls": 0.1,
            "force_re_enrich": False
        },
        "docling": {
            "image_export_mode": "placeholder",
            "enrich_picture_classes": False,
            "enrich_picture_description": False,
            "enrich_chart_extraction": False,
            "enable_ocr": True
        }
    }
