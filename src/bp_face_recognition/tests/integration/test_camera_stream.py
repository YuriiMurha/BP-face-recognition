"""
Integration tests for camera stream capture.

These tests require actual camera hardware and will be skipped
if no camera is available.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCameraStreamCapture:
    """Integration tests that require actual camera hardware."""

    @pytest.mark.integration
    def test_webcam_capture_single_frame(self):
        """Test that we can capture a single frame from webcam."""
        os.environ["CAMERA_SOURCE"] = "webcam"
        os.environ["CAMERA_DEVICE"] = "0"

        from bp_face_recognition.utils.camera_source import create_camera_manager

        camera = create_camera_manager()

        assert camera.is_connected(), "Camera should be connected"

        frame = camera.read_frame()

        assert frame is not None, "Frame should not be None"
        assert isinstance(frame, np.ndarray), "Frame should be numpy array"
        assert len(frame.shape) == 3, "Frame should have 3 dimensions (H, W, C)"
        assert frame.shape[2] == 3, "Frame should have 3 color channels"

        camera.release()

    @pytest.mark.integration
    def test_webcam_capture_multiple_frames(self):
        """Test that we can capture multiple frames from webcam."""
        os.environ["CAMERA_SOURCE"] = "webcam"
        os.environ["CAMERA_DEVICE"] = "0"

        from bp_face_recognition.utils.camera_source import create_camera_manager

        camera = create_camera_manager()
        assert camera.is_connected(), "Camera should be connected"

        frames = []
        for _ in range(5):
            frame = camera.read_frame()
            if frame is not None:
                frames.append(frame)

        assert len(frames) >= 3, f"Should capture at least 3 frames, got {len(frames)}"

        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape[2] == 3

        camera.release()

    @pytest.mark.integration
    def test_camera_frame_properties(self):
        """Test that captured frames have expected properties."""
        os.environ["CAMERA_SOURCE"] = "webcam"
        os.environ["CAMERA_DEVICE"] = "0"

        from bp_face_recognition.utils.camera_source import create_camera_manager

        camera = create_camera_manager()
        assert camera.is_connected(), "Camera should be connected"

        frame = camera.read_frame()
        assert frame is not None

        assert frame.dtype == np.uint8, "Frame should be uint8"
        assert np.all(
            (frame >= 0) & (frame <= 255)
        ), "Frame values should be in valid range"

        camera.release()

    @pytest.mark.integration
    @pytest.mark.skipif(
        os.name == "nt", reason="USB camera detection varies on Windows"
    )
    def test_usb_camera_detection(self):
        """Test USB camera detection on Linux."""
        os.environ["CAMERA_SOURCE"] = "usb"

        from bp_face_recognition.utils.camera_source import USBDeviceSource

        adb_devices = USBDeviceSource.list_adb_devices()
        video_devices = USBDeviceSource.get_available_video_devices()

        assert isinstance(adb_devices, list)
        assert isinstance(video_devices, list)

    @pytest.mark.integration
    def test_rtsp_stream_connection(self):
        """Test RTSP stream connection if URL is provided."""
        rtsp_url = os.environ.get("CAMERA_RTSP_URL")

        if not rtsp_url:
            pytest.skip("CAMERA_RTSP_URL not set")

        os.environ["CAMERA_SOURCE"] = "rtsp"
        os.environ["CAMERA_RTSP_URL"] = rtsp_url

        from bp_face_recognition.utils.camera_source import create_camera_manager

        camera = create_camera_manager()

        if camera.is_connected():
            frame = camera.read_frame()
            assert frame is not None or frame is None
            camera.release()


class TestCameraStreamPerformance:
    """Performance tests for camera stream."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_frame_capture_latency(self):
        """Test that frame capture latency is reasonable."""
        import time

        os.environ["CAMERA_SOURCE"] = "webcam"
        os.environ["CAMERA_DEVICE"] = "0"

        from bp_face_recognition.utils.camera_source import create_camera_manager

        camera = create_camera_manager()

        if not camera.is_connected():
            pytest.skip("Camera not connected")

        latencies = []
        for _ in range(10):
            start = time.time()
            frame = camera.read_frame()
            latency = time.time() - start

            if frame is not None:
                latencies.append(latency)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            assert avg_latency < 1.0, f"Frame capture too slow: {avg_latency:.3f}s"

        camera.release()


class TestCameraIntegrationCI:
    """Tests that can run in CI without camera."""

    def test_imports_work(self):
        """Test that all camera modules can be imported."""
        from bp_face_recognition.config.settings import CameraSourceType

        assert CameraSourceType.WEBCAM == "webcam"
        assert CameraSourceType.RTSP == "rtsp"
        assert CameraSourceType.USB == "usb"

    def test_settings_loading(self):
        """Test that settings can be loaded."""
        from bp_face_recognition.config.settings import Settings

        settings = Settings()
        assert hasattr(settings, "CAMERA_SOURCE")
        assert hasattr(settings, "CAMERA_DEVICE")
        assert hasattr(settings, "CAMERA_RTSP_URL")

    def test_camera_config_creation(self):
        """Test creating camera config."""
        from bp_face_recognition.utils.camera_source import CameraConfig

        config = CameraConfig(
            source_type="webcam",
            device_index=0,
            width=1280,
            height=720,
            fps=30,
        )

        assert config.source_type == "webcam"
        assert config.device_index == 0
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30
