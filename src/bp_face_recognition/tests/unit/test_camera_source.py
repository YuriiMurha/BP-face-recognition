import os
import sys
import cv2
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCameraConfig:
    def test_camera_config_defaults(self):
        from bp_face_recognition.utils.camera_source import CameraConfig

        config = CameraConfig(source_type="webcam")
        assert config.source_type == "webcam"
        assert config.device_index is None
        assert config.rtsp_url is None
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30

    def test_camera_config_with_rtsp(self):
        from bp_face_recognition.utils.camera_source import CameraConfig

        config = CameraConfig(
            source_type="rtsp",
            rtsp_url="rtsp://example.com/live",
            width=1920,
            height=1080,
            fps=60,
        )
        assert config.source_type == "rtsp"
        assert config.rtsp_url == "rtsp://example.com/live"
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60

    def test_camera_config_with_device(self):
        from bp_face_recognition.utils.camera_source import CameraConfig

        config = CameraConfig(source_type="webcam", device_index=1)
        assert config.source_type == "webcam"
        assert config.device_index == 1


class TestCameraSourceType:
    def test_camera_source_type_constants(self):
        from bp_face_recognition.config.settings import CameraSourceType

        assert CameraSourceType.WEBCAM == "webcam"
        assert CameraSourceType.RTSP == "rtsp"
        assert CameraSourceType.USB == "usb"


class TestSettingsCameraConfig:
    def test_settings_camera_defaults(self):
        from bp_face_recognition.config.settings import Settings

        settings = Settings()
        assert settings.CAMERA_SOURCE == "webcam"
        assert settings.CAMERA_DEVICE == 0
        assert settings.CAMERA_RTSP_URL is None
        assert settings.CAMERA_WIDTH == 1280
        assert settings.CAMERA_HEIGHT == 720
        assert settings.CAMERA_FPS == 30

    def test_settings_camera_env_override(self, monkeypatch):
        from bp_face_recognition.config.settings import Settings

        monkeypatch.setenv("CAMERA_SOURCE", "rtsp")
        monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.com/stream")
        monkeypatch.setenv("CAMERA_DEVICE", "2")
        monkeypatch.setenv("CAMERA_WIDTH", "1920")

        settings = Settings()
        assert settings.CAMERA_SOURCE == "rtsp"
        assert settings.CAMERA_RTSP_URL == "rtsp://test.com/stream"
        assert settings.CAMERA_DEVICE == 2
        assert settings.CAMERA_WIDTH == 1920


class TestWebcamSource:
    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_webcam_connect_success(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import WebcamSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        source = WebcamSource(device_index=0)
        result = source.connect()

        assert result is True
        mock_video_capture.assert_called()

    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_webcam_connect_failure(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import WebcamSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        source = WebcamSource(device_index=0)
        result = source.connect()

        assert result is False


class TestRTSPSource:
    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_rtsp_connect_success(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import RTSPSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        source = RTSPSource(rtsp_url="rtsp://example.com/stream")
        result = source.connect()

        assert result is True

    def test_rtsp_requires_url(self):
        from bp_face_recognition.utils.camera_source import RTSPSource

        source = RTSPSource(rtsp_url="")
        assert source.rtsp_url == ""


class TestUSBDeviceSource:
    @patch("bp_face_recognition.utils.camera_source.USBDeviceSource.list_adb_devices")
    @patch(
        "bp_face_recognition.utils.camera_source.USBDeviceSource.get_available_video_devices"
    )
    def test_usb_no_devices_found(self, mock_video_devices, mock_adb_devices):
        from bp_face_recognition.utils.camera_source import USBDeviceSource

        mock_adb_devices.return_value = []
        mock_video_devices.return_value = []

        source = USBDeviceSource(device_serial=None)
        result = source._connect_to_existing_device()

        assert result is False

    @patch("bp_face_recognition.utils.camera_source.subprocess.run")
    def test_list_adb_devices_no_adb(self, mock_subprocess):
        from bp_face_recognition.utils.camera_source import USBDeviceSource

        mock_subprocess.side_effect = FileNotFoundError("adb not found")

        devices = USBDeviceSource.list_adb_devices()

        assert devices == []


class TestCameraManager:
    @patch("bp_face_recognition.utils.camera_source.WebcamSource")
    def test_camera_manager_default_config(self, mock_webcam_source):
        from bp_face_recognition.utils.camera_source import CameraManager, CameraConfig

        mock_source = MagicMock()
        mock_source.connect.return_value = True
        mock_source.is_connected.return_value = True
        mock_webcam_source.return_value = mock_source

        with patch("bp_face_recognition.utils.camera_source.settings") as mock_settings:
            mock_settings.CAMERA_SOURCE = "webcam"
            mock_settings.CAMERA_DEVICE = 0
            mock_settings.CAMERA_RTSP_URL = None
            mock_settings.CAMERA_WIDTH = 1280
            mock_settings.CAMERA_HEIGHT = 720
            mock_settings.CAMERA_FPS = 30

            config = CameraConfig(source_type="webcam", device_index=0)
            manager = CameraManager(config)

            assert manager.config.source_type == "webcam"

    @patch("bp_face_recognition.utils.camera_source.WebcamSource")
    def test_create_camera_manager_helper(self, mock_webcam_source):
        from bp_face_recognition.utils.camera_source import create_camera_manager
        from bp_face_recognition.config.settings import CameraSourceType

        mock_source = MagicMock()
        mock_source.connect.return_value = True
        mock_source.is_connected.return_value = True
        mock_webcam_source.return_value = mock_source

        with patch("bp_face_recognition.utils.camera_source.settings") as mock_settings:
            mock_settings.CAMERA_SOURCE = CameraSourceType.WEBCAM
            mock_settings.CAMERA_DEVICE = 0
            mock_settings.CAMERA_RTSP_URL = None
            mock_settings.CAMERA_WIDTH = 1280
            mock_settings.CAMERA_HEIGHT = 720
            mock_settings.CAMERA_FPS = 30

            manager = create_camera_manager(source_type="webcam", device_index=1)

            assert manager.config.device_index == 1


class TestCameraSourceSwitching:
    def test_switch_source_changes_config(self):
        from bp_face_recognition.utils.camera_source import CameraManager, CameraConfig

        with patch.object(CameraManager, "_connect"):
            config1 = CameraConfig(source_type="webcam", device_index=0)
            manager = CameraManager(config1)

            new_config = CameraConfig(source_type="webcam", device_index=1)
            manager.config = new_config

            assert manager.config.device_index == 1


class TestWebcamSourceEdgeCases:
    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_webcam_read_frame_returns_none_on_failure(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import WebcamSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        source = WebcamSource(device_index=0)
        source.connect()
        frame = source.read_frame()

        assert frame is None
        source.release()

    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_webcam_is_connected(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import WebcamSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        source = WebcamSource(device_index=0)
        source.connect()
        assert source.is_connected() is True
        source.release()

        assert source.is_connected() is False


class TestRTSPSourceEdgeCases:
    @patch("bp_face_recognition.utils.camera_source.cv2.VideoCapture")
    def test_rtsp_read_frame_returns_none_on_failure(self, mock_video_capture):
        from bp_face_recognition.utils.camera_source import RTSPSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        source = RTSPSource(rtsp_url="rtsp://example.com/stream")
        source.connect()
        frame = source.read_frame()

        assert frame is None

    def test_rtsp_source_type_constant(self):
        from bp_face_recognition.config.settings import CameraSourceType

        assert CameraSourceType.RTSP == "rtsp"


class TestCameraManagerFullFlow:
    @patch("bp_face_recognition.utils.camera_source.WebcamSource")
    def test_camera_manager_with_all_sources(self, mock_webcam):
        from bp_face_recognition.utils.camera_source import (
            CameraManager,
            CameraConfig,
            CameraSourceType,
        )

        mock_source = MagicMock()
        mock_source.connect.return_value = True
        mock_source.is_connected.return_value = True
        mock_source.read_frame.return_value = MagicMock()
        mock_webcam.return_value = mock_source

        with patch("bp_face_recognition.utils.camera_source.settings") as mock_settings:
            mock_settings.CAMERA_SOURCE = CameraSourceType.WEBCAM
            mock_settings.CAMERA_DEVICE = 0
            mock_settings.CAMERA_RTSP_URL = None
            mock_settings.CAMERA_WIDTH = 1280
            mock_settings.CAMERA_HEIGHT = 720
            mock_settings.CAMERA_FPS = 30

            config = CameraConfig(source_type="webcam", device_index=0)
            manager = CameraManager(config)

            assert manager.config.source_type == "webcam"
            assert manager.is_connected() is True

            frame = manager.read_frame()
            assert frame is not None

            manager.release()
            mock_source.release.assert_called_once()

    @patch("bp_face_recognition.utils.camera_source.RTSPSource")
    def test_camera_manager_rtsp_source(self, mock_rtsp):
        from bp_face_recognition.utils.camera_source import (
            CameraManager,
            CameraConfig,
        )

        mock_source = MagicMock()
        mock_source.connect.return_value = True
        mock_source.is_connected.return_value = True
        mock_rtsp.return_value = mock_source

        config = CameraConfig(source_type="rtsp", rtsp_url="rtsp://test.com/stream")
        manager = CameraManager(config)

        assert manager.config.source_type == "rtsp"
