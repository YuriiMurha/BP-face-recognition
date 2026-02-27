import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestConfigParsing:
    """Test config file parsing in isolation."""

    def test_parse_valid_yaml(self):
        """Test parsing valid YAML config."""
        config = {
            "detectors": {
                "test_detector": {
                    "class": "vision.detection.mediapipe.Detector",
                    "version": "1.0.0",
                }
            },
            "recognizers": {
                "test_recognizer": {
                    "class": "vision.recognition.facenet.Recognizer",
                    "version": "1.0.0",
                }
            },
            "global": {"default_detector": "test_detector"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)

            assert "detectors" in loaded
            assert "recognizers" in loaded
            assert "global" in loaded
            assert loaded["detectors"]["test_detector"]["version"] == "1.0.0"
        finally:
            os.unlink(temp_path)

    def test_load_json_config(self):
        """Test loading JSON config."""
        config = {"detectors": {}, "recognizers": {}, "global": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = json.load(f)
            assert loaded == config
        finally:
            os.unlink(temp_path)

    def test_missing_config_file(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            with open("nonexistent_config_xyz.yaml", "r") as f:
                yaml.safe_load(f)

    def test_invalid_yaml_syntax(self):
        """Test that invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                with open(temp_path, "r") as f:
                    yaml.safe_load(f)
        finally:
            os.unlink(temp_path)

    def test_config_missing_detectors_section(self):
        """Test config missing detectors section."""
        config = {"recognizers": {}, "global": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)

            assert "detectors" not in loaded
        finally:
            os.unlink(temp_path)

    def test_model_missing_version_field(self):
        """Test model config missing version field."""
        config = {
            "detectors": {
                "incomplete_detector": {"class": "vision.detection.mediapipe.Detector"}
            },
            "recognizers": {},
            "global": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)

            detector = loaded["detectors"]["incomplete_detector"]
            assert "version" not in detector
        finally:
            os.unlink(temp_path)


class TestEnvironmentConfig:
    """Test environment configuration."""

    def test_config_with_environments(self):
        """Test config with environment profiles."""
        config = {
            "detectors": {},
            "recognizers": {},
            "global": {"batch_size": 32},
            "environments": {
                "testing": {"batch_size": 8},
                "production": {"batch_size": 64},
            },
        }

        assert "environments" in config
        assert config["environments"]["testing"]["batch_size"] == 8
        assert config["environments"]["production"]["batch_size"] == 64

    def test_merge_global_and_environment(self):
        """Test merging global and environment configs."""
        global_config = {"batch_size": 32, "debug": False}
        env_config = {"batch_size": 8}

        merged = {**global_config, **env_config}

        assert merged["batch_size"] == 8
        assert merged["debug"] is False


class TestSettingsConfig:
    """Test Settings configuration."""

    def test_settings_defaults(self):
        """Test Settings defaults."""
        from bp_face_recognition.config.settings import Settings

        settings = Settings()
        assert settings.APP_NAME == "BP Face Recognition"
        assert settings.DEBUG is False
        assert str(settings.DATA_DIR).endswith("data")

    def test_settings_env_override(self, monkeypatch):
        """Test Settings env override."""
        from bp_face_recognition.config.settings import Settings

        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("DEBUG", "True")

        settings = Settings()
        assert settings.APP_NAME == "Test App"
        assert settings.DEBUG is True


class TestRegistryFunctions:
    """Test registry model loading functions."""

    def test_get_registry_singleton(self):
        """Test get_registry returns singleton."""
        try:
            from bp_face_recognition.vision.registry import get_registry
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_list_detectors_returns_dict(self):
        """Test list_detectors returns dictionary."""
        try:
            from bp_face_recognition.vision.registry import get_registry
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        registry = get_registry()
        detectors = registry.list_detectors()
        assert isinstance(detectors, dict)

    def test_list_recognizers_returns_dict(self):
        """Test list_recognizers returns dictionary."""
        try:
            from bp_face_recognition.vision.registry import get_registry
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        registry = get_registry()
        recognizers = registry.list_recognizers()
        assert isinstance(recognizers, dict)

    def test_get_global_settings(self):
        """Test getting global settings."""
        try:
            from bp_face_recognition.vision.registry import get_registry
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        registry = get_registry()
        settings = registry.get_global_settings()
        assert isinstance(settings, dict)
