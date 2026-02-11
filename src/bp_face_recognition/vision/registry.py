"""
Model Registry System for Vision Architecture

This module provides a configuration-driven plugin system for loading
face detection and recognition models dynamically based on YAML/JSON configuration.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union
import yaml
import json

from bp_face_recognition.vision.interfaces import FaceDetector, FaceRecognizer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Configuration-driven model registry for dynamic plugin loading.

    Supports both YAML and JSON configuration files with model versioning
    and runtime model switching capabilities.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize registry with configuration file.

        Args:
            config_path: Path to YAML or JSON config file.
                       Defaults to config/models.yaml
        """
        if config_path is None:
            # Default config location
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "models.yaml"

        self.config_path = Path(config_path) if config_path else None
        if self.config_path is None:
            raise ValueError("config_path cannot be None")
        self.config = self._load_config(self.config_path)
        self._validate_config()

        logger.info(
            f"Model registry initialized with {len(self.config.get('detectors', {}))} detectors and "
            f"{len(self.config.get('recognizers', {}))} recognizers"
        )

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            if config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {config_path.suffix}. Use .yaml or .json"
                )

            logger.info(f"Configuration loaded from {config_path}")
            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        required_sections = ["detectors", "recognizers"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

            if not isinstance(self.config[section], dict):
                raise ValueError(f"Section '{section}' must be a dictionary")

        # Validate each model configuration
        for model_type in ["detectors", "recognizers"]:
            for model_name, model_config in self.config[model_type].items():
                self._validate_model_config(model_name, model_config, model_type)

        logger.info("Configuration validation passed")

    def _validate_model_config(
        self, name: str, config: Dict[str, Any], model_type: str
    ) -> None:
        """Validate individual model configuration."""
        required_fields = ["class", "version"]

        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Model '{name}' in {model_type} missing required field: {field}"
                )

        # Validate class path format
        class_path = config["class"]
        if "." not in class_path:
            raise ValueError(f"Model '{name}' has invalid class path: {class_path}")

    def _dynamic_import(self, class_path: str) -> Type:
        """
        Dynamically import a class from string path.

        Args:
            class_path: Fully qualified class name (e.g., "vision.detection.mediapipe.MediaPipeDetector")

        Returns:
            Type: The imported class
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(f"bp_face_recognition.{module_path}")
            model_class = getattr(module, class_name)

            # Validate that it's a class
            if not inspect.isclass(model_class):
                raise ValueError(f"'{class_path}' is not a class")

            logger.debug(f"Successfully imported {class_path}")
            return model_class

        except ImportError as e:
            logger.error(f"Failed to import {class_path}: {e}")
            raise
        except AttributeError as e:
            logger.error(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            )
            raise

    def get_detector(self, name: str, **kwargs) -> FaceDetector:
        """
        Get detector instance by name.

        Args:
            name: Detector name from config (e.g., "mediapipe_v1")
            **kwargs: Additional configuration overrides

        Returns:
            FaceDetector: Configured detector instance
        """
        if name not in self.config["detectors"]:
            available = list(self.config["detectors"].keys())
            raise ValueError(f"Detector '{name}' not found. Available: {available}")

        detector_config = self.config["detectors"][name].copy()
        class_path = detector_config["class"]

        # Merge with default config
        default_config = detector_config.get("default_config", {})
        merged_config = {**default_config, **kwargs}

        # Add model file path if specified
        if "model_file" in detector_config:
            project_root = Path(__file__).parent.parent.parent
            model_file = project_root / detector_config["model_file"]
            merged_config["model_file"] = str(model_file)

        try:
            detector_class = self._dynamic_import(class_path)
            detector_instance = detector_class(**merged_config)

            logger.info(
                f"Created detector '{name}' ({class_path}) with config: {merged_config}"
            )
            return detector_instance

        except Exception as e:
            logger.error(f"Failed to create detector '{name}': {e}")
            raise

    def get_recognizer(self, name: str, **kwargs) -> FaceRecognizer:
        """
        Get recognizer instance by name.

        Args:
            name: Recognizer name from config (e.g., "custom_cnn_v1")
            **kwargs: Additional configuration overrides

        Returns:
            FaceRecognizer: Configured recognizer instance
        """
        if name not in self.config["recognizers"]:
            available = list(self.config["recognizers"].keys())
            raise ValueError(f"Recognizer '{name}' not found. Available: {available}")

        recognizer_config = self.config["recognizers"][name].copy()
        class_path = recognizer_config["class"]

        # Merge with default config
        default_config = recognizer_config.get("default_config", {})
        merged_config = {**default_config, **kwargs}

        # Add model file path if specified
        if "model_file" in recognizer_config:
            project_root = Path(__file__).parent.parent.parent
            model_file = project_root / recognizer_config["model_file"]
            merged_config["model_path"] = str(
                model_file
            )  # Most recognizers use model_path

        try:
            recognizer_class = self._dynamic_import(class_path)
            recognizer_instance = recognizer_class(**merged_config)

            logger.info(
                f"Created recognizer '{name}' ({class_path}) with config: {merged_config}"
            )
            return recognizer_instance

        except Exception as e:
            logger.error(f"Failed to create recognizer '{name}': {e}")
            raise

    def get_default_detector(self, **kwargs) -> FaceDetector:
        """Get default detector as specified in global settings."""
        default_name = self.config.get("global", {}).get(
            "default_detector", "mediapipe_v1"
        )
        return self.get_detector(default_name, **kwargs)

    def get_default_recognizer(self, **kwargs) -> FaceRecognizer:
        """Get default recognizer as specified in global settings."""
        default_name = self.config.get("global", {}).get(
            "default_recognizer", "custom_cnn_v1"
        )
        return self.get_recognizer(default_name, **kwargs)

    def list_detectors(self) -> Dict[str, Dict[str, Any]]:
        """List all available detectors with their configurations."""
        return self.config.get("detectors", {})

    def list_recognizers(self) -> Dict[str, Dict[str, Any]]:
        """List all available recognizers with their configurations."""
        return self.config.get("recognizers", {})

    def get_global_settings(self) -> Dict[str, Any]:
        """Get global configuration settings."""
        return self.config.get("global", {})

    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get performance optimization settings."""
        return self.config.get("optimization", {})


# Global registry instance
_registry = None


def get_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """
    Get global model registry instance.

    Args:
        config_path: Optional custom config path

    Returns:
        ModelRegistry: Singleton registry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry(config_path)
    return _registry


def reload_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """
    Force reload registry with new configuration.

    Args:
        config_path: Optional custom config path

    Returns:
        ModelRegistry: Fresh registry instance
    """
    global _registry
    _registry = ModelRegistry(config_path)
    return _registry
